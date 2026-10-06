"""Explicit routing for conversation variants."""

from __future__ import annotations

from typing import Any, Protocol

_POLICY_MARKERS = ("return", "shipping", "warranty")
_CATALOG_MARKERS = ("compare", "product ", "plan ", "active products", "lightest", "running shoe")
_EXPLANATION_MARKERS = ("why", "how", "explain", "context", "tradeoff", "pros and cons")


class RunnableVariant(Protocol):
    """Protocol for variants with a standard run interface."""

    def run(
        self,
        user_turn: str,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the variant for the given user turn."""


class VariantRouter:
    """Choose between retrieval, tool-first, and hybrid variants."""

    def __init__(
        self,
        *,
        retrieval_variant: RunnableVariant,
        tool_first_variant: RunnableVariant,
        hybrid_variant: RunnableVariant,
    ) -> None:
        self._variants = {
            "retrieval": retrieval_variant,
            "tool_first": tool_first_variant,
            "hybrid": hybrid_variant,
        }

    def decide(self, user_turn: str) -> dict[str, Any]:
        """Return an explainable routing decision for the query."""

        lowered = user_turn.lower()
        has_policy = any(marker in lowered for marker in _POLICY_MARKERS)
        has_catalog = any(marker in lowered for marker in _CATALOG_MARKERS)
        structured = has_policy or has_catalog
        needs_explanation = any(marker in lowered for marker in _EXPLANATION_MARKERS)

        if (has_policy and has_catalog) or (structured and needs_explanation):
            return {
                "variant": "hybrid",
                "rule": "query combines multiple structured needs or asks for explanation",
                "combined": True,
            }
        if structured:
            return {
                "variant": "tool_first",
                "rule": "query requests structured facts that map to tools",
                "combined": False,
            }
        return {
            "variant": "retrieval",
            "rule": "query is open-ended and better served by retrieval context",
            "combined": False,
        }

    def run(
        self,
        user_turn: str,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch to the selected variant and attach the routing decision."""

        decision = self.decide(user_turn)
        result = self._variants[decision["variant"]].run(user_turn, session_context)
        result["trace"] = {**result.get("trace", {}), "routing_decision": decision}
        return result
