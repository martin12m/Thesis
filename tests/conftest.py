"""Shared pytest fixtures: stub text generators for tests and demos."""

from __future__ import annotations

import pytest

from llm.interface import TextGenerator


class StubAnswerLLM(TextGenerator):
    """Canned generator that verbalizes a fixed comparison answer."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return (
            "The Growth Plan (49.00 EUR) is better if you need analytics -- it "
            "extends the Starter Plan (19.00 EUR) with expanded analytics "
            "features. The Starter Plan is the right pick if you're a small "
            "team that just needs the entry-level basics at a lower price."
        )


class StubJudgeLLM(TextGenerator):
    """Canned generator that returns a fixed, well-formed judge verdict."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return (
            '{"groundedness": 5, "accuracy": 5, "completeness": 5, "clarity": 5, '
            '"rationale": "Cites both plan prices and the analytics '
            'differentiator directly from the retrieved evidence."}'
        )


@pytest.fixture
def stub_answer_llm() -> StubAnswerLLM:
    """Return a fresh `StubAnswerLLM` instance."""

    return StubAnswerLLM()


@pytest.fixture
def stub_judge_llm() -> StubJudgeLLM:
    """Return a fresh `StubJudgeLLM` instance."""

    return StubJudgeLLM()
