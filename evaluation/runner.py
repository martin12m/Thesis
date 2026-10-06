"""Scenario runner for deterministic variant evaluation."""

from __future__ import annotations

import csv
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from evaluation.llm_judge import JUDGE_DIMENSIONS, judge_result, strip_code_fences
from evaluation.scoring import score_result
from llm.interface import TextGenerator
from orchestration.variant_router import RunnableVariant

logger = logging.getLogger(__name__)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of a pre-sorted list (numpy's default method)."""

    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def aggregate_latencies(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...] = ("variant",),
) -> list[dict[str, Any]]:
    """Compute p50/p95 latency percentiles per group (e.g. per variant, per model)."""

    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_by)
        groups.setdefault(key, []).append(float(row["latency_ms"]))

    summaries: list[dict[str, Any]] = []
    for key, latencies in groups.items():
        sorted_latencies = sorted(latencies)
        summary: dict[str, Any] = dict(zip(group_by, key, strict=True))
        summary["count"] = len(sorted_latencies)
        summary["p50_ms"] = round(_percentile(sorted_latencies, 50), 3)
        summary["p95_ms"] = round(_percentile(sorted_latencies, 95), 3)
        summaries.append(summary)
    return summaries


def aggregate_success_rate(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...] = ("variant",),
) -> list[dict[str, Any]]:
    """Compute the Reliability metric (success rate) per group."""

    groups: dict[tuple[Any, ...], list[bool]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_by)
        succeeded = not row.get("failure") and bool(str(row.get("response_text", "")).strip())
        groups.setdefault(key, []).append(succeeded)

    summaries: list[dict[str, Any]] = []
    for key, outcomes in groups.items():
        summary: dict[str, Any] = dict(zip(group_by, key, strict=True))
        summary["count"] = len(outcomes)
        summary["success_rate"] = round(sum(outcomes) / len(outcomes), 4)
        summaries.append(summary)
    return summaries


def export_summary_to_csv(
    summaries: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Export a per-group summary (from `aggregate_latencies` or `aggregate_success_rate`) to a compact CSV of one row per group."""

    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys()) if summaries else []

    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation scenarios from a JSON file."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        msg = f"Expected a list of scenarios in {file_path}"
        raise ValueError(msg)
    if not all(isinstance(item, dict) for item in payload):
        msg = f"Expected each scenario in {file_path} to be an object"
        raise ValueError(msg)
    return payload


SIMULATION_CATEGORIES: tuple[str, ...] = (
    "structured_business_context",
    "catalog_context",
    "document_context",
    "session_context",
)

_SIMULATION_RUBRIC = """You are simulating realistic customer queries to evaluate an e-commerce chatbot.

Generate {n} varied, realistic user queries a real shopper might type. The \
store sells physical products (with prices, categories, weights) and has \
return/shipping/warranty policies and customer reviews.

Cover a mix of these query categories: {categories}
- structured_business_context: a question about a policy (returns, shipping, \
warranty), or a precise fact-lookup constrained by price/weight/category.
- catalog_context: a product lookup, a comparison between two named or \
numbered products, or a question about a product's reviews or rating.
- document_context: an open-ended "why"/"how"/"which is better" question \
that needs explanatory context, not just a single fact.
- session_context: a natural follow-up question that only makes sense after \
an earlier turn -- include that earlier turn as "prior_turn".

Respond with ONLY a JSON array, no other text, no markdown fences, using \
exactly this shape:
[{{"category": "<one of the categories above>", "user_turn": "<realistic \
query text>", "prior_turn": "<the earlier turn text -- ONLY for \
session_context items>"}}, ...]
"""


def generate_simulated_scenarios(
    generator: TextGenerator,
    *,
    n: int,
    categories: tuple[str, ...] = SIMULATION_CATEGORIES,
    batch_size: int = 5,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Generate up to `n` realistic simulated-user scenarios via an LLM agent."""

    if n < 1:
        msg = f"n must be at least 1, got {n}."
        raise ValueError(msg)
    if batch_size < 1:
        msg = f"batch_size must be at least 1, got {batch_size}."
        raise ValueError(msg)

    if max_batches is None:
        max_batches = max(4, 3 * -(-n // batch_size))  # ceil(n / batch_size) * 3, floor 4

    batch_id = uuid.uuid4().hex[:8]
    scenarios: list[dict[str, Any]] = []
    batches_attempted = 0

    while len(scenarios) < n and batches_attempted < max_batches:
        batches_attempted += 1
        wanted = min(batch_size, n - len(scenarios))
        prompt = _SIMULATION_RUBRIC.format(n=wanted, categories=", ".join(categories))
        try:
            raw_response = generator.generate(prompt)
            items = _extract_json_array(raw_response)
        except Exception as exc:  # noqa: BLE001 - one bad batch must not abort the run
            logger.warning("Simulated-scenario batch failed, skipping: %s", exc)
            continue

        for item in items:
            if len(scenarios) >= n:
                break
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).strip()
            user_turn = str(item.get("user_turn", "")).strip()
            if category not in categories or not user_turn:
                continue

            scenario: dict[str, Any] = {
                "id": f"sim-{batch_id}-{len(scenarios) + 1}",
                "category": category,
                "user_turn": user_turn,
            }
            prior_turn = item.get("prior_turn")
            if category == "session_context" and prior_turn:
                scenario["session_context"] = {"turn_history": [str(prior_turn)]}
            scenarios.append(scenario)

    if not scenarios:
        msg = "The simulated-user generator produced no usable scenarios."
        raise ValueError(msg)
    return scenarios


def _extract_json_array(text: str) -> list[Any]:
    """Pull the first JSON array out of a model response."""

    try:
        parsed = json.loads(strip_code_fences(text))
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return parsed

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        msg = "No JSON array found in the simulated-user generator's response."
        raise ValueError(msg)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        msg = f"Simulated-user generator response was not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, list):
        msg = f"Expected a JSON array from the simulated-user generator, got {type(parsed).__name__}."
        raise ValueError(msg)
    return parsed


class EvaluationRunner:
    """Run evaluation scenarios across multiple variants."""

    def __init__(
        self,
        variants: dict[str, RunnableVariant],
        *,
        judge_generator: TextGenerator | None = None,
        model_name: str | None = None,
    ) -> None:
        """Store the variants to evaluate and the optional judge generator."""

        self._variants = variants
        self._judge_generator = judge_generator
        self._model_name = model_name

    def run_scenarios(self, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run each scenario against each variant and score the outputs."""

        rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            for variant_name, variant in self._variants.items():
                rows.append(self._run_single(variant_name, variant, scenario))
        return rows

    @staticmethod
    def export_results_to_csv(
        rows: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """Export evaluation rows to CSV."""

        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "scenario_id",
            "variant",
            "response_text",
            "evidence_ids",
            "tool_trace",
            "latency_ms",
            "failure",
            "correctness",
            "grounding",
            "context_use",
            "validity",
            "total",
        ]
        include_model = any(row.get("model") is not None for row in rows)
        if include_model:
            fieldnames.insert(2, "model")
        include_judge = any("judge_scores" in row for row in rows)
        if include_judge:
            fieldnames += [f"judge_{dimension}" for dimension in JUDGE_DIMENSIONS]
            fieldnames += ["judge_average", "judge_error"]

        with file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                record = {
                    "scenario_id": row["scenario_id"],
                    "variant": row["variant"],
                    "response_text": row["response_text"],
                    "evidence_ids": json.dumps(row["evidence_ids"]),
                    "tool_trace": json.dumps(row["tool_trace"]),
                    "latency_ms": row["latency_ms"],
                    "failure": row["failure"] or "",
                    "correctness": row["scores"]["correctness"],
                    "grounding": row["scores"]["grounding"],
                    "context_use": row["scores"]["context_use"],
                    "validity": row["scores"]["validity"],
                    "total": row["scores"]["total"],
                }
                if include_model:
                    record["model"] = row.get("model") or ""
                if include_judge:
                    judge_scores = row.get("judge_scores") or {}
                    for dimension in JUDGE_DIMENSIONS:
                        record[f"judge_{dimension}"] = judge_scores.get(dimension)
                    record["judge_average"] = judge_scores.get("average")
                    record["judge_error"] = judge_scores.get("error") or ""
                writer.writerow(record)

    def _run_single(
        self,
        variant_name: str,
        variant: RunnableVariant,
        scenario: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one scenario on one variant and normalize the output."""

        user_turn = str(scenario["user_turn"])
        started = time.perf_counter()
        failure: str | None = None

        try:
            result = variant.run(user_turn, scenario.get("session_context"))
        except Exception as exc:  # noqa: BLE001
            result = {}
            failure = f"{type(exc).__name__}: {exc}"

        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        evidence_items = result.get("evidence", [])
        evidence_ids = [
            str(item.get("document_id", item.get("id", "")))
            for item in evidence_items
            if item.get("document_id", item.get("id"))
        ]
        tool_trace = result.get("trace", {}).get("tool_calls", [])
        normalized = {
            "scenario_id": str(scenario.get("id", "")),
            "variant": variant_name,
            "model": self._model_name,
            "response_text": str(result.get("answer", "")),
            "evidence_ids": evidence_ids,
            "tool_trace": tool_trace,
            "latency_ms": latency_ms,
            "failure": failure,
        }
        normalized["scores"] = score_result(scenario, normalized)

        if self._judge_generator is not None:
            normalized["judge_scores"] = judge_result(
                user_query=user_turn,
                response_text=normalized["response_text"],
                evidence=evidence_items,
                expected_facts=scenario.get("expected_answer_contains", []),
                generator=self._judge_generator,
            )

        return normalized
