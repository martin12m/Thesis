"""Tests for the retrieval-grounded generation variant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from variants.retrieval_variant import RetrievalVariant


class MockGenerator:
    """Mock LLM that records prompts and returns fixed output."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_retrieval_variant_returns_answer_evidence_and_trace() -> None:
    generator = MockGenerator("Grounded answer.")

    def retrieval_backend(
        db_path: str | Path,
        query: str,
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        assert db_path == "/tmp/catalog.db"
        assert query == "What is the shipping policy?"
        assert top_k == 2
        return [
            {
                "document_id": "policy:2",
                "source_type": "policy",
                "source_record_id": 2,
                "title": "Shipping Policy",
                "content": "Standard shipping is available in 3 to 5 business days.",
                "score": -0.75,
            }
        ]

    variant = RetrievalVariant(
        llm=generator,
        db_path="/tmp/catalog.db",
        retrieval_backend=retrieval_backend,
        top_k=2,
    )

    result = variant.run(
        "What is the shipping policy?",
        session_context={
            "turn_history": ["Tell me about policy options."],
            "active_topic": "policy",
            "session_summary": "Tell me about policy options.",
        },
    )

    assert result["answer"] == "Grounded answer."
    assert result["evidence"] == [
        {
            "document_id": "policy:2",
            "source_type": "policy",
            "source_record_id": 2,
            "title": "Shipping Policy",
            "content": "Standard shipping is available in 3 to 5 business days.",
            "score": -0.75,
        }
    ]
    assert result["trace"]["variant"] == "retrieval"
    assert result["trace"]["active_topic"] == "policy"
    assert result["trace"]["evidence_count"] == 1
    assert "Shipping Policy" in generator.prompts[0]
    assert "User turn: What is the shipping policy?" in generator.prompts[0]


def test_retrieval_variant_omits_tools_and_handles_empty_evidence() -> None:
    generator = MockGenerator("Fallback answer.")

    def retrieval_backend(
        db_path: str | Path,
        query: str,
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return []

    variant = RetrievalVariant(
        llm=generator,
        db_path="/tmp/catalog.db",
        retrieval_backend=retrieval_backend,
    )

    result = variant.run("General question", session_context=None)

    assert result == {
        "answer": "Fallback answer.",
        "evidence": [],
        "trace": {
            "variant": "retrieval",
            "active_topic": "general",
            "evidence_count": 0,
            "prompt_chars": len(generator.prompts[0]),
            "session_summary": "General question",
        },
    }
    assert "Evidence:\n[none]" in generator.prompts[0]
