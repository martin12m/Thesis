"""Command-line entrypoint for running variant evaluations."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.data_source import get_configured_data_source
from data.loaders import connect_database, seed_database
from data.product_reviews_ingest import (
    external_reviews_products_from_json,
    load_external_reviews_json,
    merge_products,
)
from evaluation.runner import (
    EvaluationRunner,
    aggregate_latencies,
    aggregate_success_rate,
    export_summary_to_csv,
    generate_simulated_scenarios,
    load_scenarios,
)
from llm.interface import (
    TextGenerator,
    build_default_generator,
    build_generator_for_profile,
    list_configured_profiles,
)
from retrieval.indexer import rebuild_index
from variants.hybrid_variant import HybridVariant
from variants.retrieval_variant import RetrievalVariant
from variants.tool_first_variant import ToolFirstVariant

_BUILTIN_MODEL_NAMES = ("llama", "mistral")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for evaluation runs."""

    parser = argparse.ArgumentParser(description="Run evaluation scenarios.")
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="llama",
        help=(
            "Comma-separated model name(s) to evaluate: built-in local "
            "GGUF variants ('llama', 'mistral'), any custom profile name "
            "declared in LLM_PROFILES, 'both' (= llama,mistral), or 'all' "
            "(every built-in plus every configured profile). Default: llama."
        ),
    )
    parser.add_argument(
        "--judge",
        choices=["none", "cross", "self"],
        default="none",
        help=(
            "none (default): no LLM-as-a-judge scoring. "
            "cross: each model's answers are judged by a *different* "
            "selected model (requires 2+ models, avoids same-model judge "
            "bias). self: each model judges its own answers (documented "
            "bias risk)."
        ),
    )
    parser.add_argument(
        "--simulate",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Additionally generate N LLM-simulated user scenarios "
            "using the first selected --model, evaluated alongside "
            "--scenarios against every selected model. Lacks ground truth "
            "-- pair with --judge for a meaningful score on these rows."
        ),
    )
    parser.add_argument(
        "--simulate-batch-size",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Generate simulated scenarios N at a time per LLM call instead "
            "of all at once (default: 5) -- more reliable for larger "
            "--simulate counts, since one giant request is more likely to "
            "be truncated or degrade in quality. Use 1 for strictly "
            "one-at-a-time generation."
        ),
    )
    return parser


def _resolve_model_names(raw: str, configured_profiles: list[str]) -> list[str]:
    """Expand the --model argument into a concrete list of model names."""

    if raw == "both":
        return list(_BUILTIN_MODEL_NAMES)
    if raw == "all":
        return [*_BUILTIN_MODEL_NAMES, *configured_profiles]
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        msg = f"--model resolved to an empty list of models: {raw!r}"
        raise SystemExit(msg)
    return names


def _build_generator(name: str) -> TextGenerator:
    """Build the generator for one model name (built-in local variant or profile)."""

    if name in _BUILTIN_MODEL_NAMES:
        return build_default_generator({"LLM_MODEL_VARIANT": name})
    return build_generator_for_profile(name)


def _build_variants(generator: TextGenerator, db_path: str) -> dict[str, Any]:
    """Construct the three reasoning variants against one generator and DB."""

    return {
        "retrieval": RetrievalVariant(llm=generator, db_path=db_path),
        "tool_first": ToolFirstVariant(llm=generator, db_path=db_path),
        "hybrid": HybridVariant(llm=generator, db_path=db_path),
    }


_ENRICHMENT_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def _load_cross_domain_enrichment_products() -> list[dict[str, Any]]:
    """Load the radio/TV/vacuum/headphone/laptop mock+real fixtures as extra catalog products."""

    radio_doc = dict(
        load_external_reviews_json(_ENRICHMENT_FIXTURES_DIR / "external_reviews_real_example_radio.json")
    )
    radio_doc.setdefault("sku", "TEUFEL-RADIO-3SIXTY")  # the real capture has no top-level identifier

    documents = [radio_doc]
    for category in ("tvs", "vacuums", "headphones", "laptops"):
        documents.extend(
            load_external_reviews_json(_ENRICHMENT_FIXTURES_DIR / f"external_reviews_mock_{category}.json")
        )
    return external_reviews_products_from_json(documents)


def _prepare_evaluation_database() -> str:
    """Seed a temporary SQLite database from the configured commerce data source."""

    data_source = get_configured_data_source()
    with tempfile.NamedTemporaryFile(prefix="commerce-eval-", suffix=".db", delete=False) as handle:
        db_path = handle.name

    products = merge_products(data_source.get_products(), _load_cross_domain_enrichment_products())
    connection = connect_database(db_path)
    try:
        seed_database(
            connection,
            products=products,
            policies=data_source.get_policies(),
        )
    finally:
        connection.close()
    rebuild_index(db_path)
    return db_path


def _select_judge_generator(
    judge_mode: str,
    model_name: str,
    generators: dict[str, TextGenerator],
    model_names: list[str],
) -> TextGenerator | None:
    """Pick the judge generator for one model's rows, per ``--judge`` mode."""

    if judge_mode == "none":
        return None
    if judge_mode == "self":
        return generators[model_name]
    other = next(name for name in model_names if name != model_name)
    return generators[other]


def main() -> int:
    """Run the evaluation workflow and write CSV output."""

    args = build_parser().parse_args()
    scenarios = load_scenarios(args.scenarios)

    model_names = _resolve_model_names(args.model, list_configured_profiles())
    if args.judge == "cross" and len(model_names) < 2:
        msg = "--judge cross requires 2 or more --model names (to cross-judge against a different model)."
        raise SystemExit(msg)

    db_path = _prepare_evaluation_database()
    generators = {name: _build_generator(name) for name in model_names}

    if args.simulate > 0:
        simulated_scenarios = generate_simulated_scenarios(
            generators[model_names[0]],
            n=args.simulate,
            batch_size=args.simulate_batch_size,
        )
        print(f"Generated {len(simulated_scenarios)} simulated-user scenarios.")

        simulated_output = args.output.with_name(f"{args.output.stem}_simulated_scenarios.json")
        simulated_output.parent.mkdir(parents=True, exist_ok=True)
        simulated_output.write_text(json.dumps(simulated_scenarios, indent=2), encoding="utf-8")
        print(f"Wrote generated scenarios to {simulated_output}")

        scenarios = scenarios + simulated_scenarios

    all_rows: list[dict[str, Any]] = []
    for name in model_names:
        variants = _build_variants(generators[name], db_path)
        judge_generator = _select_judge_generator(args.judge, name, generators, model_names)
        model_label = name if len(model_names) > 1 else None
        runner = EvaluationRunner(
            variants=variants,
            judge_generator=judge_generator,
            model_name=model_label,
        )
        all_rows.extend(runner.run_scenarios(scenarios))

    EvaluationRunner.export_results_to_csv(all_rows, args.output)
    print(f"Wrote {len(all_rows)} evaluation rows to {args.output}")

    group_by = ("model", "variant") if len(model_names) > 1 else ("variant",)

    latency_summaries = aggregate_latencies(all_rows, group_by=group_by)
    latency_output = args.output.with_name(f"{args.output.stem}_latency_summary.csv")
    export_summary_to_csv(latency_summaries, latency_output)
    print(f"Wrote latency percentile summary to {latency_output}")

    success_summaries = aggregate_success_rate(all_rows, group_by=group_by)
    success_output = args.output.with_name(f"{args.output.stem}_success_rate.csv")
    export_summary_to_csv(success_summaries, success_output)
    print(f"Wrote success-rate (Reliability) summary to {success_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
