"""Deterministic scoring helpers for evaluation results."""

from __future__ import annotations

from typing import Any


def score_result(
    scenario: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, float]:
    """Score a single variant result against a scenario."""

    response_text = str(result.get("response_text", result.get("answer", "")))

    correctness = _score_correctness(
        response_text=response_text,
        required_phrases=scenario.get("expected_answer_contains", []),
    )
    grounding = _score_grounding(
        evidence_ids=result.get("evidence_ids", []),
        expected_ids=scenario.get("expected_evidence_ids", []),
    )
    context_use = _score_context_use(
        response_text=response_text,
        required_terms=scenario.get("required_context_terms", []),
    )
    validity = _score_validity(
        response_text=response_text,
        tool_trace=result.get("tool_trace", []),
        failure=result.get("failure"),
        expected_tool_names=scenario.get("expected_tool_names", []),
    )

    total = round((correctness + grounding + context_use + validity) / 4.0, 4)
    return {
        "correctness": correctness,
        "grounding": grounding,
        "context_use": context_use,
        "validity": validity,
        "total": total,
    }


def _score_correctness(
    *,
    response_text: str,
    required_phrases: list[str],
) -> float:
    """Score correctness by exact substring coverage."""

    if not required_phrases:
        return 1.0 if response_text else 0.0

    lowered_response = response_text.lower()
    matched = sum(1 for phrase in required_phrases if phrase.lower() in lowered_response)
    return round(matched / len(required_phrases), 4)


def _score_grounding(
    *,
    evidence_ids: list[str],
    expected_ids: list[str],
) -> float:
    """Score grounding by overlap with expected evidence identifiers."""

    if not expected_ids:
        return 1.0

    expected_set = {str(item) for item in expected_ids}
    actual_set = {str(item) for item in evidence_ids}
    matched = len(expected_set & actual_set)
    return round(matched / len(expected_set), 4)


def _score_context_use(
    *,
    response_text: str,
    required_terms: list[str],
) -> float:
    """Score context use by coverage of required context terms."""

    if not required_terms:
        return 1.0

    lowered_response = response_text.lower()
    matched = sum(1 for term in required_terms if term.lower() in lowered_response)
    return round(matched / len(required_terms), 4)


def _score_validity(
    *,
    response_text: str,
    tool_trace: list[dict[str, Any]],
    failure: str | None,
    expected_tool_names: list[str],
) -> float:
    """Score result validity using failure state and expected tool usage."""

    if failure:
        return 0.0
    if not response_text.strip():
        return 0.0

    if not expected_tool_names:
        return 1.0

    actual_tools = {
        str(item.get("tool_name"))
        for item in tool_trace
        if item.get("status") == "completed"
    }
    expected_tools = {str(name) for name in expected_tool_names}
    matched = len(expected_tools & actual_tools)
    return round(matched / len(expected_tools), 4)
