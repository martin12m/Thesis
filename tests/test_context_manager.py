"""Tests for the rule-based context manager."""

from __future__ import annotations

from orchestration.context_manager import ContextManager


def test_context_manager_carries_over_recent_product_context() -> None:
    manager = ContextManager()
    session: dict[str, object] | None = None

    session = manager.update_session("Tell me about the Starter Plan product.", session)
    session = manager.update_session("What is the price for that plan?", session)
    package = manager.build_context_package(
        "Compare that product with the Growth Plan.",
        session,
        retrieved_evidence=[
            {
                "document_id": "product:1",
                "source_type": "product",
                "title": "Starter Plan",
                "content": "Entry-level subscription and price details.",
                "score": 0.1,
            },
            {
                "document_id": "policy:1",
                "source_type": "policy",
                "title": "Returns Policy",
                "content": "Returns are accepted within 30 days.",
                "score": 0.2,
            },
        ],
        tool_outputs=[
            {"type": "product", "title": "Catalog comparison", "content": "Growth Plan is higher priced."},
            {"type": "policy", "title": "Policy lookup", "content": "Returns policy snippet."},
        ],
    )

    assert package["active_topic"] == "product"
    assert package["turns"] == [
        "Tell me about the Starter Plan product.",
        "What is the price for that plan?",
        "Compare that product with the Growth Plan.",
    ]
    assert [item["document_id"] for item in package["evidence"]] == ["product:1"]
    assert [item["title"] for item in package["tool_outputs"]] == ["Catalog comparison"]


def test_context_manager_prunes_on_explicit_context_switch() -> None:
    manager = ContextManager()
    session: dict[str, object] | None = None

    session = manager.update_session("Tell me about the Starter Plan product.", session)
    session = manager.update_session("What is the price for that plan?", session)
    package = manager.build_context_package(
        "Instead, switch to the shipping policy.",
        session,
        retrieved_evidence=[
            {
                "document_id": "product:1",
                "source_type": "product",
                "title": "Starter Plan",
                "content": "Entry-level subscription and price details.",
                "score": 0.1,
            },
            {
                "document_id": "policy:2",
                "source_type": "policy",
                "title": "Shipping Policy",
                "content": "Standard shipping is available in 3 to 5 business days.",
                "score": 0.2,
            },
        ],
        tool_outputs=[
            {"type": "product", "title": "Catalog lookup", "content": "Starter Plan details."},
            {"type": "policy", "title": "Shipping lookup", "content": "Shipping terms."},
        ],
    )

    assert package["active_topic"] == "policy"
    assert package["turns"] == ["Instead, switch to the shipping policy."]
    assert [item["document_id"] for item in package["evidence"]] == ["policy:2"]
    assert [item["title"] for item in package["tool_outputs"]] == ["Shipping lookup"]
