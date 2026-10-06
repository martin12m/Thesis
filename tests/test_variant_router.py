"""Tests for explicit variant routing decisions."""

from __future__ import annotations

from typing import Any

from orchestration.variant_router import VariantRouter


class StubVariant:
    """Simple runnable variant used to verify router dispatch."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def run(
        self,
        user_turn: str,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((user_turn, session_context))
        return {"answer": self.name, "trace": {"variant": self.name}}


def test_variant_router_chooses_tool_first_for_structured_fact_query() -> None:
    retrieval = StubVariant("retrieval")
    tool_first = StubVariant("tool_first")
    hybrid = StubVariant("hybrid")
    router = VariantRouter(
        retrieval_variant=retrieval,
        tool_first_variant=tool_first,
        hybrid_variant=hybrid,
    )

    result = router.run("What is the shipping policy?", {"turn_history": []})

    assert result["answer"] == "tool_first"
    assert result["trace"]["routing_decision"] == {
        "variant": "tool_first",
        "rule": "query requests structured facts that map to tools",
        "combined": False,
    }
    assert tool_first.calls == [("What is the shipping policy?", {"turn_history": []})]


def test_variant_router_chooses_hybrid_for_fact_plus_explanation_query() -> None:
    retrieval = StubVariant("retrieval")
    tool_first = StubVariant("tool_first")
    hybrid = StubVariant("hybrid")
    router = VariantRouter(
        retrieval_variant=retrieval,
        tool_first_variant=tool_first,
        hybrid_variant=hybrid,
    )

    result = router.run("Explain why product 1 is cheaper than product 2.", None)

    assert result["answer"] == "hybrid"
    assert result["trace"]["routing_decision"] == {
        "variant": "hybrid",
        "rule": "query combines multiple structured needs or asks for explanation",
        "combined": True,
    }
    assert hybrid.calls == [("Explain why product 1 is cheaper than product 2.", None)]


def test_variant_router_chooses_retrieval_for_open_ended_query() -> None:
    retrieval = StubVariant("retrieval")
    tool_first = StubVariant("tool_first")
    hybrid = StubVariant("hybrid")
    router = VariantRouter(
        retrieval_variant=retrieval,
        tool_first_variant=tool_first,
        hybrid_variant=hybrid,
    )

    result = router.run("Give me background on customer trust.", None)

    assert result["answer"] == "retrieval"
    assert result["trace"]["routing_decision"] == {
        "variant": "retrieval",
        "rule": "query is open-ended and better served by retrieval context",
        "combined": False,
    }
    assert retrieval.calls == [("Give me background on customer trust.", None)]


def test_variant_router_chooses_hybrid_for_catalog_and_policy_query() -> None:
    retrieval = StubVariant("retrieval")
    tool_first = StubVariant("tool_first")
    hybrid = StubVariant("hybrid")
    router = VariantRouter(
        retrieval_variant=retrieval,
        tool_first_variant=tool_first,
        hybrid_variant=hybrid,
    )

    result = router.run(
        "Which running shoe under 100 euros is the lightest, and what is the return period?",
        None,
    )

    assert result["answer"] == "hybrid"
    assert result["trace"]["routing_decision"] == {
        "variant": "hybrid",
        "rule": "query combines multiple structured needs or asks for explanation",
        "combined": True,
    }
