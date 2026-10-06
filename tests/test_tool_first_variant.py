"""Tests for the rule-based tool-first variant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from variants.tool_first_variant import ToolFirstVariant


class MockGenerator:
    """Mock LLM that captures prompts and returns fixed output."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_tool_first_variant_routes_to_policy_tool(monkeypatch: Any) -> None:
    generator = MockGenerator("Policy answer.")

    def fake_shipping_policy(db_path: str | Path) -> dict[str, Any]:
        assert db_path == "/tmp/catalog.db"
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

    monkeypatch.setattr(
        "orchestration.structured_routing.get_shipping_policy_info",
        fake_shipping_policy,
    )

    variant = ToolFirstVariant(llm=generator, db_path="/tmp/catalog.db")
    result = variant.run("What is the shipping policy?")

    assert result == {
        "answer": "Policy answer.",
        "trace": {
            "variant": "tool_first",
            "route": "policy",
            "active_topic": "policy",
            "session_summary": "What is the shipping policy?",
            "tool_calls": [
                {
                    "tool_name": "get_shipping_policy_info",
                    "arguments": {},
                    "status": "completed",
                }
            ],
        },
    }
    assert "Tool: get_shipping_policy_info" in generator.prompts[0]
    assert "Shipping Policy" in generator.prompts[0]


def test_tool_first_variant_routes_to_catalog_compare(monkeypatch: Any) -> None:
    generator = MockGenerator("Catalog answer.")

    def fake_compare(
        db_path: str | Path,
        left_product_id: int,
        right_product_id: int,
    ) -> dict[str, Any]:
        assert db_path == "/tmp/catalog.db"
        assert left_product_id == 1
        assert right_product_id == 2
        return {
            "left_product": {"id": 1, "name": "Starter Plan"},
            "right_product": {"id": 2, "name": "Growth Plan"},
            "price_difference_cents": -3000,
            "common_policy_codes": ["PRIVACY"],
            "left_only_policy_codes": ["RETURNS"],
            "right_only_policy_codes": [],
        }

    monkeypatch.setattr("orchestration.structured_routing.compare_products", fake_compare)

    variant = ToolFirstVariant(llm=generator, db_path="/tmp/catalog.db")
    result = variant.run("Compare product 1 with product 2.")

    assert result["answer"] == "Catalog answer."
    assert result["trace"] == {
        "variant": "tool_first",
        "route": "catalog",
        "active_topic": "product",
        "session_summary": "Compare product 1 with product 2.",
        "tool_calls": [
            {
                "tool_name": "compare_products",
                "arguments": {"left_product_id": 1, "right_product_id": 2},
                "status": "completed",
            }
        ],
    }
    assert "Tool: compare_products" in generator.prompts[0]


def test_tool_first_variant_routes_to_product_reviews(monkeypatch: Any) -> None:
    """Reviews are the 4th dynamic-context category: external structured data."""

    generator = MockGenerator("Review answer.")

    def fake_get_product_reviews(
        db_path: str | Path,
        *,
        product_id: int,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        assert db_path == "/tmp/catalog.db"
        assert product_id == 3
        assert env == {"COMMERCE_PRODUCT_SOURCE": "local"}
        return {
            "product_id": 3,
            "sku": "RUN-003",
            "source": "local_fixture",
            "reviews": [{"stars": 5, "text": "Great shoe."}],
            "score": {"average_stars": 5.0, "count": 1},
        }

    monkeypatch.setattr(
        "orchestration.structured_routing.get_product_reviews",
        fake_get_product_reviews,
    )

    variant = ToolFirstVariant(llm=generator, db_path="/tmp/catalog.db")
    result = variant.run("What do the reviews say about product 3?")

    assert result["answer"] == "Review answer."
    assert result["trace"] == {
        "variant": "tool_first",
        "route": "catalog",
        "active_topic": "product",
        "session_summary": "What do the reviews say about product 3?",
        "tool_calls": [
            {
                "tool_name": "get_product_reviews",
                "arguments": {"product_id": 3},
                "status": "completed",
            }
        ],
    }
    assert "Tool: get_product_reviews" in generator.prompts[0]
    assert "RUN-003" in generator.prompts[0]


def test_tool_first_variant_carries_over_session_history(monkeypatch: Any) -> None:
    """Session context must now shape tool-first prompts and traces (RQ1 symmetry)."""

    generator = MockGenerator("Policy answer.")

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

    monkeypatch.setattr(
        "orchestration.structured_routing.get_shipping_policy_info",
        fake_shipping_policy,
    )

    variant = ToolFirstVariant(llm=generator, db_path="/tmp/catalog.db")
    prior_session = {
        "turn_history": ["What is your return policy?"],
        "active_topic": "policy",
        "session_summary": "What is your return policy?",
    }
    result = variant.run("And what about shipping?", prior_session)

    assert result["trace"]["session_summary"] == (
        "What is your return policy? | And what about shipping?"
    )
    assert "What is your return policy? | And what about shipping?" in generator.prompts[0]


def test_tool_first_variant_uses_fallback_when_no_tool_matches() -> None:
    generator = MockGenerator("Fallback answer.")

    def retrieval_backend(
        db_path: str | Path,
        query: str,
        *,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        assert db_path == "/tmp/catalog.db"
        assert query == "Tell me something unexpected."
        assert top_k == 2
        return [
            {
                "document_id": "policy:1",
                "source_type": "policy",
                "source_record_id": 1,
                "title": "Returns Policy",
                "content": "Returns are accepted within 30 days.",
                "score": -0.4,
            }
        ]

    variant = ToolFirstVariant(
        llm=generator,
        db_path="/tmp/catalog.db",
        retrieval_backend=retrieval_backend,
        top_k=2,
    )
    result = variant.run("Tell me something unexpected.")

    assert result == {
        "answer": "Fallback answer.",
        "trace": {
            "variant": "tool_first",
            "route": "fallback",
            "active_topic": "general",
            "session_summary": "Tell me something unexpected.",
            "tool_calls": [],
            "fallback_evidence_count": 1,
        },
    }
    assert "No structured tool matched the query." in generator.prompts[0]
    assert "Returns Policy" in generator.prompts[0]
