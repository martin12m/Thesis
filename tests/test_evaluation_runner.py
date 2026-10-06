"""Integration-style tests for the evaluation runner with mocked variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.runner import EvaluationRunner, generate_simulated_scenarios, load_scenarios


class _StubSimulationGenerator:
    """Stand-in for a TextGenerator that returns a canned scenario-generation response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class MockVariant:
    """Mock variant with deterministic behavior for evaluation tests."""

    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self._result = result

    def run(
        self,
        user_turn: str,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del user_turn
        del session_context
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_evaluation_runner_loads_runs_and_exports_csv(tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "id": "shipping-1",
                    "user_turn": "What is the shipping policy?",
                    "session_context": {"turn_history": ["shipping context"]},
                    "expected_answer_contains": ["shipping", "3 to 5 business days"],
                    "expected_evidence_ids": ["policy:2"],
                    "required_context_terms": ["shipping"],
                    "expected_tool_names": ["get_shipping_policy_info"],
                }
            ]
        ),
        encoding="utf-8",
    )

    scenarios = load_scenarios(scenarios_path)
    runner = EvaluationRunner(
        variants={
            "tool_first": MockVariant(
                {
                    "answer": "The shipping policy says shipping takes 3 to 5 business days.",
                    "evidence": [{"document_id": "policy:2"}],
                    "trace": {
                        "tool_calls": [
                            {"tool_name": "get_shipping_policy_info", "status": "completed"}
                        ]
                    },
                }
            ),
            "broken": MockVariant(RuntimeError("boom")),
        }
    )

    rows = runner.run_scenarios(scenarios)

    assert len(rows) == 2
    assert rows[0]["variant"] == "tool_first"
    assert rows[0]["response_text"] == "The shipping policy says shipping takes 3 to 5 business days."
    assert rows[0]["evidence_ids"] == ["policy:2"]
    assert rows[0]["tool_trace"] == [
        {"tool_name": "get_shipping_policy_info", "status": "completed"}
    ]
    assert rows[0]["failure"] is None
    assert rows[0]["scores"]["total"] == 1.0

    assert rows[1]["variant"] == "broken"
    assert rows[1]["response_text"] == ""
    assert rows[1]["failure"] == "RuntimeError: boom"
    assert rows[1]["scores"]["validity"] == 0.0

    output_path = tmp_path / "results.csv"
    runner.export_results_to_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        exported_rows = list(csv.DictReader(handle))

    assert len(exported_rows) == 2
    assert exported_rows[0]["scenario_id"] == "shipping-1"
    assert exported_rows[0]["variant"] == "tool_first"
    assert exported_rows[0]["evidence_ids"] == "[\"policy:2\"]"
    assert "model" not in exported_rows[0]


def test_evaluation_runner_tags_rows_with_model_name(tmp_path: Path) -> None:
    scenarios = [{"id": "s1", "user_turn": "hi"}]
    runner = EvaluationRunner(
        variants={"retrieval": MockVariant({"answer": "hello", "evidence": [], "trace": {}})},
        model_name="llama",
    )

    rows = runner.run_scenarios(scenarios)

    assert rows[0]["model"] == "llama"

    output_path = tmp_path / "results.csv"
    runner.export_results_to_csv(rows, output_path)
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        exported_rows = list(csv.DictReader(handle))

    assert exported_rows[0]["model"] == "llama"


def test_generate_simulated_scenarios_parses_well_formed_response() -> None:
    response = json.dumps(
        [
            {"category": "structured_business_context", "user_turn": "What is your return policy?"},
            {
                "category": "session_context",
                "user_turn": "And how long do I have?",
                "prior_turn": "What is your return policy?",
            },
        ]
    )
    generator = _StubSimulationGenerator(response)

    scenarios = generate_simulated_scenarios(generator, n=2)

    assert len(scenarios) == 2
    assert scenarios[0]["category"] == "structured_business_context"
    assert scenarios[0]["user_turn"] == "What is your return policy?"
    assert "session_context" not in scenarios[0]

    assert scenarios[1]["category"] == "session_context"
    assert scenarios[1]["session_context"] == {"turn_history": ["What is your return policy?"]}
    assert scenarios[0]["id"] != scenarios[1]["id"]


def test_generate_simulated_scenarios_tolerates_markdown_fences() -> None:
    response = "```json\n" + json.dumps(
        [{"category": "catalog_context", "user_turn": "How much is product 2?"}]
    ) + "\n```"
    generator = _StubSimulationGenerator(response)

    scenarios = generate_simulated_scenarios(generator, n=1)

    assert scenarios[0]["user_turn"] == "How much is product 2?"


def test_generate_simulated_scenarios_drops_malformed_items() -> None:
    """A batch with some malformed items keeps only the valid ones -- and, since batching now retries to try to reach n, a generator that keeps producing nothing new on later batches must not loop forever or fabricate duplicates: max_batches bounds it deterministically here."""

    call_count = 0

    class _OneGoodBatchThenEmptyGenerator:
        def generate(self, prompt: str) -> str:  # noqa: ARG002
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(
                    [
                        {"category": "catalog_context", "user_turn": "Tell me about product 1."},
                        {"category": "not_a_real_category", "user_turn": "This should be dropped."},
                        {"category": "document_context"},  # missing user_turn
                        "not even an object",
                    ]
                )
            return json.dumps([])  # later batches produce nothing new

    scenarios = generate_simulated_scenarios(
        _OneGoodBatchThenEmptyGenerator(), n=4, batch_size=4, max_batches=3
    )

    assert len(scenarios) == 1
    assert scenarios[0]["user_turn"] == "Tell me about product 1."
    assert call_count == 3  # kept retrying up to max_batches, found nothing more


def test_generate_simulated_scenarios_raises_on_all_malformed() -> None:
    generator = _StubSimulationGenerator(json.dumps([{"category": "bogus", "user_turn": "x"}]))

    with pytest.raises(ValueError, match="no usable scenarios"):
        generate_simulated_scenarios(generator, n=1)


def test_generate_simulated_scenarios_raises_on_unparseable_response() -> None:
    """An unparseable response is retried (a batch failure, not fatal on its own) up to max_batches, then surfaces as 'no usable scenarios' once every attempt has failed -- not the raw per-batch parse error."""

    generator = _StubSimulationGenerator("this is not JSON at all")

    with pytest.raises(ValueError, match="no usable scenarios"):
        generate_simulated_scenarios(generator, n=1, max_batches=2)


def test_generate_simulated_scenarios_rejects_non_positive_n() -> None:
    generator = _StubSimulationGenerator("[]")

    with pytest.raises(ValueError, match="n must be at least 1"):
        generate_simulated_scenarios(generator, n=0)


def test_generate_simulated_scenarios_flow_through_run_scenarios() -> None:
    """Simulated scenarios must need no special-casing in the existing pipeline."""

    response = json.dumps(
        [{"category": "catalog_context", "user_turn": "What's the price of product 3?"}]
    )
    sim_generator = _StubSimulationGenerator(response)
    scenarios = generate_simulated_scenarios(sim_generator, n=1)

    runner = EvaluationRunner(
        variants={"tool_first": MockVariant({"answer": "It costs 50 EUR.", "evidence": [], "trace": {}})}
    )
    rows = runner.run_scenarios(scenarios)

    assert len(rows) == 1
    assert rows[0]["response_text"] == "It costs 50 EUR."
    assert rows[0]["scores"]["total"] == 1.0
