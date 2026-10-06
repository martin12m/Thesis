"""Tests for the optional LLM-as-a-judge wiring inside EvaluationRunner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluation.runner import EvaluationRunner, load_scenarios


class _MockVariant:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def run(
        self,
        user_turn: str,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del user_turn
        del session_context
        return self._result


class _StubJudgeGenerator:
    def __init__(self, response: str | Exception) -> None:
        self._response = response

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


_SCENARIO_PAYLOAD = [
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

_VARIANT_RESULT = {
    "answer": "The shipping policy says shipping takes 3 to 5 business days.",
    "evidence": [{"document_id": "policy:2"}],
    "trace": {
        "tool_calls": [{"tool_name": "get_shipping_policy_info", "status": "completed"}]
    },
}

_VALID_JUDGE_JSON = (
    '{"groundedness": 5, "accuracy": 5, "completeness": 4, "clarity": 5, '
    '"rationale": "Matches evidence and query."}'
)


def _write_scenarios(tmp_path: Path) -> Path:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIO_PAYLOAD), encoding="utf-8")
    return scenarios_path


def test_runner_without_judge_has_no_judge_scores_key(tmp_path: Path) -> None:
    scenarios = load_scenarios(_write_scenarios(tmp_path))
    runner = EvaluationRunner(variants={"tool_first": _MockVariant(_VARIANT_RESULT)})

    rows = runner.run_scenarios(scenarios)

    assert "judge_scores" not in rows[0]
    assert rows[0]["scores"]["total"] == 1.0


def test_runner_with_judge_adds_judge_scores_without_altering_deterministic_scores(
    tmp_path: Path,
) -> None:
    scenarios = load_scenarios(_write_scenarios(tmp_path))

    baseline_runner = EvaluationRunner(variants={"tool_first": _MockVariant(_VARIANT_RESULT)})
    baseline_rows = baseline_runner.run_scenarios(scenarios)

    judged_runner = EvaluationRunner(
        variants={"tool_first": _MockVariant(_VARIANT_RESULT)},
        judge_generator=_StubJudgeGenerator(_VALID_JUDGE_JSON),
    )
    judged_rows = judged_runner.run_scenarios(scenarios)

    assert baseline_rows[0]["scores"] == judged_rows[0]["scores"]

    assert "judge_scores" in judged_rows[0]
    judge_scores = judged_rows[0]["judge_scores"]
    assert judge_scores["available"] is True
    assert judge_scores["groundedness"] == 5
    assert judge_scores["accuracy"] == 5
    assert judge_scores["completeness"] == 4
    assert judge_scores["clarity"] == 5
    assert judge_scores["average"] == 4.75


def test_runner_with_judge_fails_gracefully_when_local_model_offline(tmp_path: Path) -> None:
    scenarios = load_scenarios(_write_scenarios(tmp_path))

    runner = EvaluationRunner(
        variants={"tool_first": _MockVariant(_VARIANT_RESULT)},
        judge_generator=_StubJudgeGenerator(ConnectionError("model server offline")),
    )

    rows = runner.run_scenarios(scenarios)

    assert rows[0]["scores"]["total"] == 1.0  # deterministic pipeline unaffected
    assert rows[0]["judge_scores"]["available"] is False
    assert rows[0]["judge_scores"]["groundedness"] is None
    assert "offline" in rows[0]["judge_scores"]["error"]


def test_export_results_to_csv_without_judge_has_original_columns_only(
    tmp_path: Path,
) -> None:
    scenarios = load_scenarios(_write_scenarios(tmp_path))
    runner = EvaluationRunner(variants={"tool_first": _MockVariant(_VARIANT_RESULT)})
    rows = runner.run_scenarios(scenarios)

    output_path = tmp_path / "results.csv"
    runner.export_results_to_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        exported_rows = list(reader)

    assert fieldnames == [
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
    assert len(exported_rows) == 1


def test_export_results_to_csv_with_judge_appends_judge_columns(tmp_path: Path) -> None:
    scenarios = load_scenarios(_write_scenarios(tmp_path))
    runner = EvaluationRunner(
        variants={"tool_first": _MockVariant(_VARIANT_RESULT)},
        judge_generator=_StubJudgeGenerator(_VALID_JUDGE_JSON),
    )
    rows = runner.run_scenarios(scenarios)

    output_path = tmp_path / "results_judged.csv"
    runner.export_results_to_csv(rows, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        exported_rows = list(reader)

    assert fieldnames == [
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
        "judge_groundedness",
        "judge_accuracy",
        "judge_completeness",
        "judge_clarity",
        "judge_average",
        "judge_error",
    ]
    assert exported_rows[0]["judge_groundedness"] == "5.0"
    assert exported_rows[0]["judge_average"] == "4.75"
    assert exported_rows[0]["judge_error"] == ""
