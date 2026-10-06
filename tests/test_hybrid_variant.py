"""Tests for the hybrid variant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from variants.hybrid_variant import HybridVariant


class MockGenerator:
    """Mock LLM that records prompts and returns fixed output."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_hybrid_variant_combines_tool_output_and_retrieval(monkeypatch: Any) -> None:
    generator = MockGenerator("Hybrid answer.")

    def fake_shipping_policy(db_path: str | Path) -> dict[str, Any]:
        return {
            "requested_type": "shipping",
            "resolved_type": "shipping",
            "title": "Shipping Policy",
            "body": "Standard shipping is available in 3 to 5 business days.",
            "source_table": "policies",
            "source_id": 2,
            "source_code": "SHIPPING",
            "used_fallback": False,
        }

    def retrieval_backend(
        db_path: str | Path,
        query: str,
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        assert query == "Explain the shipping policy and give context."
        return [
            {
                "document_id": "policy:2",
                "source_type": "policy",
                "source_record_id": 2,
                "title": "Shipping Policy",
                "content": "Shipping typically completes in 3 to 5 business days.",
                "score": -0.3,
            }
        ]

    monkeypatch.setattr(
        "orchestration.structured_routing.get_shipping_policy_info",
        fake_shipping_policy,
    )

    variant = HybridVariant(
        llm=generator,
        db_path="/tmp/catalog.db",
        retrieval_backend=retrieval_backend,
        top_k=1,
    )
    result = variant.run("Explain the shipping policy and give context.")

    assert result["answer"] == "Hybrid answer."
    assert result["trace"] == {
        "variant": "hybrid",
        "route": "combined",
        "active_topic": "policy",
        "session_summary": "Explain the shipping policy and give context.",
        "tool_calls": [
            {
                "tool_name": "get_shipping_policy_info",
                "arguments": {},
                "status": "completed",
            }
        ],
        "evidence_count": 1,
        "source_ids": ["policy:2"],
    }
    assert result["evidence"][0]["document_id"] == "policy:2"
    assert "Structured facts:" in generator.prompts[0]
    assert "Supporting evidence:" in generator.prompts[0]


def test_hybrid_variant_carries_over_session_history(monkeypatch: Any) -> None:
    """Session context must now shape hybrid prompts and traces too (RQ1 symmetry)."""

    generator = MockGenerator("Hybrid answer.")

    def fake_shipping_policy(db_path: str | Path) -> dict[str, Any]:
        return {
            "title": "Shipping Policy",
            "body": "Standard shipping is available in 3 to 5 business days.",
            "source_table": "policies",
            "source_id": 2,
        }

    def retrieval_backend(
        db_path: str | Path,
        query: str,
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        return [
            {
                "document_id": "policy:2",
                "source_type": "policy",
                "title": "Shipping Policy",
                "content": "Shipping typically completes in 3 to 5 business days.",
                "score": -0.3,
            }
        ]

    monkeypatch.setattr(
        "orchestration.structured_routing.get_shipping_policy_info",
        fake_shipping_policy,
    )

    variant = HybridVariant(llm=generator, db_path="/tmp/catalog.db", retrieval_backend=retrieval_backend)
    prior_session = {
        "turn_history": ["What is your return policy?"],
        "active_topic": "policy",
        "session_summary": "What is your return policy?",
    }
    result = variant.run("Explain the shipping policy and give context.", prior_session)

    assert result["trace"]["session_summary"] == (
        "What is your return policy? | Explain the shipping policy and give context."
    )
    assert (
        "What is your return policy? | Explain the shipping policy and give context."
        in generator.prompts[0]
    )
