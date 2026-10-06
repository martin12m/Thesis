"""Optional LLM-as-a-judge scorer."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from llm.interface import TextGenerator

JUDGE_DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "accuracy",
    "completeness",
    "clarity",
)

_SCORE_MIN = 1
_SCORE_MAX = 5

_RUBRIC = """You are an impartial evaluation judge for a customer-support chatbot.
Score the ASSISTANT RESPONSE on four dimensions, each on an integer scale from \
1 (very poor) to 5 (excellent):

1. groundedness: Is every claim in the response supported by the provided \
CONTEXT/EVIDENCE? A response that states facts absent from the evidence, or \
that contradicts it, scores low. A response with no unsupported claims \
scores high.
2. accuracy: Does the response correctly answer the USER QUERY, matching the \
EXPECTED FACTS where provided? Factual errors or contradictions of the \
expected facts score low.
3. completeness: Does the response address every part of the USER QUERY, \
covering all EXPECTED FACTS where provided? Partial or incomplete answers \
score low.
4. clarity: Is the response well-organized, concise, and easy to understand, \
free of redundancy or ambiguity?

Respond with ONLY a JSON object, no other text, no markdown fences, using \
exactly this shape:
{"groundedness": <int 1-5>, "accuracy": <int 1-5>, "completeness": <int 1-5>, \
"clarity": <int 1-5>, "rationale": "<one short sentence>"}
"""


class JudgeUnavailableError(Exception):
    """Raised internally when the judge cannot produce a usable verdict."""


@dataclass(frozen=True)
class JudgeVerdict:
    """Structured, immutable result of one LLM-as-a-judge evaluation."""

    groundedness: float | None
    accuracy: float | None
    completeness: float | None
    clarity: float | None
    average: float | None
    rationale: str | None
    available: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for CSV/JSON export."""

        return asdict(self)


def _unavailable(error: str) -> JudgeVerdict:
    """Build the standard fail-closed verdict shape."""

    return JudgeVerdict(
        groundedness=None,
        accuracy=None,
        completeness=None,
        clarity=None,
        average=None,
        rationale=None,
        available=False,
        error=error,
    )


def build_judge_prompt(
    *,
    user_query: str,
    response_text: str,
    evidence: list[dict[str, Any]] | None = None,
    expected_facts: list[str] | None = None,
) -> str:
    """Render the full judge prompt for one (query, response) pair."""

    evidence_text = "\n".join(
        f"- {item}" for item in (evidence or [])
    ) or "(no evidence provided)"
    expected_text = ", ".join(expected_facts or []) or "(none specified)"

    return (
        f"{_RUBRIC}\n"
        f"USER QUERY:\n{user_query}\n\n"
        f"CONTEXT/EVIDENCE:\n{evidence_text}\n\n"
        f"EXPECTED FACTS (if any, for reference only -- do not penalize "
        f"missing facts not relevant to the query): {expected_text}\n\n"
        f"ASSISTANT RESPONSE:\n{response_text}\n\n"
        f"JSON verdict:"
    )


def strip_code_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence (```json ... ```) from `text`."""

    stripped = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    return re.sub(r"```$", "", stripped).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""

    try:
        parsed = json.loads(strip_code_fences(text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            msg = "No JSON object found in judge response."
            raise JudgeUnavailableError(msg) from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            msg = f"Judge response was not valid JSON: {exc}"
            raise JudgeUnavailableError(msg) from exc

    if not isinstance(parsed, dict):
        msg = f"Judge response was not a JSON object: {type(parsed).__name__}"
        raise JudgeUnavailableError(msg)
    return parsed


def _coerce_score(payload: dict[str, Any], key: str) -> float:
    """Extract and validate one dimension score from the parsed payload."""

    if key not in payload:
        msg = f"Judge response missing required field '{key}'."
        raise JudgeUnavailableError(msg)
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        msg = f"Judge response field '{key}' was not numeric: {payload[key]!r}"
        raise JudgeUnavailableError(msg) from exc
    if not (_SCORE_MIN <= value <= _SCORE_MAX):
        msg = f"Judge response field '{key}' out of range 1-5: {value}"
        raise JudgeUnavailableError(msg)
    return value


def parse_judge_response(raw_response: str) -> JudgeVerdict:
    """Parse a raw model response into a validated ``JudgeVerdict``."""

    payload = _extract_json_object(raw_response)
    scores = {dimension: _coerce_score(payload, dimension) for dimension in JUDGE_DIMENSIONS}
    average = round(sum(scores.values()) / len(scores), 4)
    rationale = payload.get("rationale")

    return JudgeVerdict(
        groundedness=scores["groundedness"],
        accuracy=scores["accuracy"],
        completeness=scores["completeness"],
        clarity=scores["clarity"],
        average=average,
        rationale=str(rationale) if rationale is not None else None,
        available=True,
        error=None,
    )


def judge_result(
    *,
    user_query: str,
    response_text: str,
    evidence: list[dict[str, Any]] | None = None,
    expected_facts: list[str] | None = None,
    generator: TextGenerator | None,
) -> dict[str, Any]:
    """Score one response with the LLM judge, failing gracefully."""

    if generator is None:
        return _unavailable("No judge generator configured.").to_dict()

    prompt = build_judge_prompt(
        user_query=user_query,
        response_text=response_text,
        evidence=evidence,
        expected_facts=expected_facts,
    )

    try:
        raw_response = generator.generate(prompt)
    except Exception as exc:  # noqa: BLE001 - side-car must never crash the pipeline
        return _unavailable(f"Judge generation failed: {type(exc).__name__}: {exc}").to_dict()

    try:
        verdict = parse_judge_response(raw_response)
    except JudgeUnavailableError as exc:
        return _unavailable(str(exc)).to_dict()

    return verdict.to_dict()
