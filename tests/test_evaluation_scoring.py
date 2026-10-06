"""Unit tests for deterministic evaluation scoring."""

from __future__ import annotations

from evaluation.scoring import score_result


def test_score_result_returns_full_credit_for_matching_output() -> None:
    scenario = {
        "expected_answer_contains": ["shipping", "3 to 5 business days"],
        "expected_evidence_ids": ["policy:2"],
        "required_context_terms": ["shipping"],
        "expected_tool_names": ["get_shipping_policy_info"],
    }
    result = {
        "answer": "The shipping policy says shipping takes 3 to 5 business days.",
        "evidence_ids": ["policy:2"],
        "tool_trace": [{"tool_name": "get_shipping_policy_info", "status": "completed"}],
        "failure": None,
    }

    scores = score_result(scenario, result)

    assert scores == {
        "correctness": 1.0,
        "grounding": 1.0,
        "context_use": 1.0,
        "validity": 1.0,
        "total": 1.0,
    }


def test_score_result_penalizes_missing_grounding_and_failures() -> None:
    scenario = {
        "expected_answer_contains": ["warranty"],
        "expected_evidence_ids": ["policy:3"],
        "required_context_terms": ["coverage"],
        "expected_tool_names": ["get_warranty_policy_info"],
    }
    result = {
        "answer": "Warranty details are unavailable.",
        "evidence_ids": [],
        "tool_trace": [],
        "failure": "LookupError: missing",
    }

    scores = score_result(scenario, result)

    assert scores == {
        "correctness": 1.0,
        "grounding": 0.0,
        "context_use": 0.0,
        "validity": 0.0,
        "total": 0.25,
    }
