"""End-to-end check that the real scenario suite runs cleanly against the real seeded local-fixture catalog through all three variants."""

from __future__ import annotations

from pathlib import Path

from data.data_source import get_configured_data_source
from data.loaders import connect_database, seed_database
from evaluation.runner import EvaluationRunner, load_scenarios
from llm.interface import TextGenerator
from retrieval.indexer import rebuild_index
from variants.hybrid_variant import HybridVariant
from variants.retrieval_variant import RetrievalVariant
from variants.tool_first_variant import ToolFirstVariant

_SUITE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "scenarios" / "context_categories_suite.json"


class _EchoGenerator(TextGenerator):
    """Echoes the prompt back, so any literal fact placed in the prompt (tool output, evidence content) trivially satisfies substring-based scoring -- this test cares about grounding/tool-execution correctness against the real DB, not generation quality."""

    def generate(self, prompt: str) -> str:
        return prompt


def _build_variants(db_path: str, generator: TextGenerator) -> dict[str, object]:
    return {
        "retrieval": RetrievalVariant(llm=generator, db_path=db_path),
        "tool_first": ToolFirstVariant(llm=generator, db_path=db_path),
        "hybrid": HybridVariant(llm=generator, db_path=db_path),
    }


def test_context_categories_suite_loads() -> None:
    scenarios = load_scenarios(_SUITE_PATH)

    assert len(scenarios) == 8
    ids = {scenario["id"] for scenario in scenarios}
    assert ids == {
        "policy-return-basic",
        "catalog-product-lookup",
        "catalog-compare",
        "catalog-lightest-under-budget",
        "document-explain-lightweight",
        "session-topic-carryover",
        "catalog-reviews",
        "policy-shipping-fallback",
    }
    categories = {scenario["category"] for scenario in scenarios}
    assert categories == {
        "structured_business_context",
        "catalog_context",
        "document_context",
        "session_context",
    }


def test_context_categories_suite_runs_without_failures(tmp_path: Path) -> None:
    data_source = get_configured_data_source({})  # defaults to local fixtures
    db_path = str(tmp_path / "eval_catalog.db")
    connection = connect_database(db_path)
    seed_database(connection, products=data_source.get_products(), policies=data_source.get_policies())
    connection.close()
    rebuild_index(db_path)

    variants = _build_variants(db_path, _EchoGenerator())
    runner = EvaluationRunner(variants=variants)
    scenarios = load_scenarios(_SUITE_PATH)

    rows = runner.run_scenarios(scenarios)

    assert len(rows) == len(scenarios) * 3
    assert all(row["failure"] is None for row in rows)


def test_tool_dependent_scenarios_hit_expected_tools(tmp_path: Path) -> None:
    """The tool_first/hybrid variants must actually execute the tool each tool-dependent scenario expects, proving the scenario IDs/arguments are grounded in the real seeded data, not just well-formed JSON."""

    data_source = get_configured_data_source({})
    db_path = str(tmp_path / "eval_catalog.db")
    connection = connect_database(db_path)
    seed_database(connection, products=data_source.get_products(), policies=data_source.get_policies())
    connection.close()
    rebuild_index(db_path)

    variants = _build_variants(db_path, _EchoGenerator())
    runner = EvaluationRunner(variants=variants)
    scenarios = load_scenarios(_SUITE_PATH)

    rows = runner.run_scenarios(scenarios)
    by_scenario_variant = {(row["scenario_id"], row["variant"]): row for row in rows}

    for scenario in scenarios:
        expected_tools = set(scenario.get("expected_tool_names", []))
        if not expected_tools:
            continue
        for variant_name in ("tool_first", "hybrid"):
            row = by_scenario_variant[(scenario["id"], variant_name)]
            actual_tools = {
                call["tool_name"]
                for call in row["tool_trace"]
                if call.get("status") == "completed"
            }
            assert expected_tools <= actual_tools, (
                f"{scenario['id']}/{variant_name}: expected {expected_tools}, got {actual_tools}"
            )
