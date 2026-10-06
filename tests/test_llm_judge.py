"""Unit tests for the optional LLM-as-a-judge side-car scorer."""

from __future__ import annotations

from typing import Any

import pytest

from evaluation.llm_judge import (
    JUDGE_DIMENSIONS,
    JudgeUnavailableError,
    build_judge_prompt,
    judge_result,
    parse_judge_response,
)


class _StubGenerator:
    """Minimal stand-in for llm.interface.TextGenerator."""

    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


VALID_JSON_RESPONSE = (
    '{"groundedness": 5, "accuracy": 4, "completeness": 4, "clarity": 5, '
    '"rationale": "Well grounded and accurate."}'
)


def test_build_judge_prompt_includes_all_sections() -> None:
    prompt = build_judge_prompt(
        user_query="What is the return policy?",
        response_text="Returns are accepted within 30 days.",
        evidence=[{"document_id": "policy:1", "text": "30-day return window."}],
        expected_facts=["30 days"],
    )

    assert "What is the return policy?" in prompt
    assert "Returns are accepted within 30 days." in prompt
    assert "30-day return window." in prompt
    assert "30 days" in prompt
    for dimension in JUDGE_DIMENSIONS:
        assert dimension in prompt


def test_build_judge_prompt_handles_missing_evidence_and_facts() -> None:
    prompt = build_judge_prompt(
        user_query="Hello?",
        response_text="Hi there.",
        evidence=None,
        expected_facts=None,
    )

    assert "(no evidence provided)" in prompt
    assert "(none specified)" in prompt


def test_parse_judge_response_valid_json() -> None:
    verdict = parse_judge_response(VALID_JSON_RESPONSE)

    assert verdict.available is True
    assert verdict.error is None
    assert verdict.groundedness == 5
    assert verdict.accuracy == 4
    assert verdict.completeness == 4
    assert verdict.clarity == 5
    assert verdict.average == pytest.approx(4.5)
    assert verdict.rationale == "Well grounded and accurate."


def test_parse_judge_response_strips_markdown_fences() -> None:
    fenced = f"```json\n{VALID_JSON_RESPONSE}\n```"
    verdict = parse_judge_response(fenced)

    assert verdict.available is True
    assert verdict.average == pytest.approx(4.5)


def test_parse_judge_response_extracts_json_with_preamble() -> None:
    noisy = f"Sure, here is my verdict:\n{VALID_JSON_RESPONSE}\nThank you."
    verdict = parse_judge_response(noisy)

    assert verdict.available is True
    assert verdict.groundedness == 5


def test_parse_judge_response_raises_on_non_json_text() -> None:
    with pytest.raises(JudgeUnavailableError):
        parse_judge_response("I think the response is pretty good overall.")


def test_parse_judge_response_raises_on_missing_dimension() -> None:
    incomplete = '{"groundedness": 5, "accuracy": 4, "completeness": 4}'
    with pytest.raises(JudgeUnavailableError):
        parse_judge_response(incomplete)


def test_parse_judge_response_raises_on_out_of_range_score() -> None:
    out_of_range = (
        '{"groundedness": 9, "accuracy": 4, "completeness": 4, "clarity": 5}'
    )
    with pytest.raises(JudgeUnavailableError):
        parse_judge_response(out_of_range)


def test_parse_judge_response_raises_on_non_numeric_score() -> None:
    bad_type = (
        '{"groundedness": "high", "accuracy": 4, "completeness": 4, "clarity": 5}'
    )
    with pytest.raises(JudgeUnavailableError):
        parse_judge_response(bad_type)


def test_judge_result_returns_scores_on_success() -> None:
    generator = _StubGenerator(VALID_JSON_RESPONSE)

    result = judge_result(
        user_query="What is the return policy?",
        response_text="Returns are accepted within 30 days.",
        evidence=[{"document_id": "policy:1"}],
        expected_facts=["30 days"],
        generator=generator,
    )

    assert result["available"] is True
    assert result["error"] is None
    assert result["groundedness"] == 5
    assert result["average"] == pytest.approx(4.5)
    assert generator.last_prompt is not None
    assert "What is the return policy?" in generator.last_prompt


def test_judge_result_fails_gracefully_when_generator_is_none() -> None:
    result = judge_result(
        user_query="q",
        response_text="a",
        evidence=None,
        expected_facts=None,
        generator=None,
    )

    assert result["available"] is False
    assert result["error"] == "No judge generator configured."
    for dimension in JUDGE_DIMENSIONS:
        assert result[dimension] is None
    assert result["average"] is None


def test_judge_result_fails_gracefully_when_generator_raises() -> None:
    generator = _StubGenerator(ConnectionError("local model server is offline"))

    result = judge_result(
        user_query="q",
        response_text="a",
        evidence=None,
        expected_facts=None,
        generator=generator,
    )

    assert result["available"] is False
    assert result["error"] is not None
    assert "ConnectionError" in result["error"]
    assert "offline" in result["error"]
    for dimension in JUDGE_DIMENSIONS:
        assert result[dimension] is None


def test_judge_result_fails_gracefully_on_unparsable_response() -> None:
    generator = _StubGenerator("I refuse to output JSON today.")

    result = judge_result(
        user_query="q",
        response_text="a",
        evidence=None,
        expected_facts=None,
        generator=generator,
    )

    assert result["available"] is False
    assert result["error"] is not None
    for dimension in JUDGE_DIMENSIONS:
        assert result[dimension] is None


def test_judge_result_fails_gracefully_on_out_of_range_scores() -> None:
    generator = _StubGenerator(
        '{"groundedness": 5, "accuracy": 4, "completeness": 4, "clarity": 42}'
    )

    result = judge_result(
        user_query="q",
        response_text="a",
        evidence=None,
        expected_facts=None,
        generator=generator,
    )

    assert result["available"] is False
    assert "clarity" in result["error"]


def test_judge_result_never_raises_for_arbitrary_generator_exceptions() -> None:
    class _ExplodingGenerator:
        def generate(self, prompt: str) -> str:  # noqa: ARG002
            raise RuntimeError("boom")

    result = judge_result(
        user_query="q",
        response_text="a",
        evidence=None,
        expected_facts=None,
        generator=_ExplodingGenerator(),
    )
    assert result["available"] is False


@pytest.mark.parametrize("payload_extra", [{}, {"rationale": None}])
def test_judge_result_rationale_optional(payload_extra: dict[str, Any]) -> None:
    payload = {
        "groundedness": 3,
        "accuracy": 3,
        "completeness": 3,
        "clarity": 3,
        **payload_extra,
    }
    import json as _json

    generator = _StubGenerator(_json.dumps(payload))

    result = judge_result(
        user_query="q",
        response_text="a",
        evidence=None,
        expected_facts=None,
        generator=generator,
    )

    assert result["available"] is True
    assert result["rationale"] is None
