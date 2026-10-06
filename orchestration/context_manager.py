"""Rule-based helpers for managing downstream generation context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_SWITCH_MARKERS = (
    "instead",
    "switch",
    "different topic",
    "new topic",
    "now about",
)


@dataclass(slots=True)
class ContextPackage:
    """Compact context payload for downstream generation."""

    current_turn: str
    session_summary: str
    active_topic: str
    turns: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)


class ContextManager:
    """Build and maintain compact context using deterministic rules."""

    def update_session(
        self,
        current_turn: str,
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Update session state with the current turn and active topic."""

        session = dict(session_context or {})
        history = list(session.get("turn_history", []))
        history.append(current_turn)
        topic = self._derive_topic(current_turn)

        if self._is_context_switch(current_turn):
            history = [current_turn]

        session["turn_history"] = history[-5:]
        session["active_topic"] = topic
        session["session_summary"] = self._build_summary(session["turn_history"])
        return session

    def select_relevant_context(
        self,
        session_context: dict[str, Any] | None,
        retrieved_evidence: list[dict[str, Any]] | None = None,
        tool_outputs: list[dict[str, Any]] | None = None,
    ) -> ContextPackage:
        """Select compact context slices relevant to the active topic."""

        session = dict(session_context or {})
        active_topic = str(session.get("active_topic", "general"))
        turns = list(session.get("turn_history", []))

        selected_evidence = [
            item
            for item in (retrieved_evidence or [])
            if self._matches_topic(active_topic, item)
        ][:3]
        selected_tools = [
            item
            for item in (tool_outputs or [])
            if self._matches_topic(active_topic, item)
        ][:2]

        return ContextPackage(
            current_turn=turns[-1] if turns else "",
            session_summary=str(session.get("session_summary", "")),
            active_topic=active_topic,
            turns=turns,
            evidence=selected_evidence,
            tool_outputs=selected_tools,
        )

    def prune_stale_context(
        self,
        session_context: dict[str, Any] | None,
        *,
        max_turns: int = 3,
    ) -> dict[str, Any]:
        """Drop stale session history while preserving the active topic."""

        session = dict(session_context or {})
        turns = list(session.get("turn_history", []))
        session["turn_history"] = turns[-max_turns:]
        session["session_summary"] = self._build_summary(session["turn_history"])
        return session

    def build_context_package(
        self,
        current_turn: str,
        session_context: dict[str, Any] | None,
        retrieved_evidence: list[dict[str, Any]] | None = None,
        tool_outputs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Update, prune, and package context for downstream generation."""

        updated_session = self.update_session(current_turn, session_context)
        pruned_session = self.prune_stale_context(updated_session)
        package = self.select_relevant_context(
            pruned_session,
            retrieved_evidence=retrieved_evidence,
            tool_outputs=tool_outputs,
        )
        return asdict(package)

    @staticmethod
    def _build_summary(turns: list[str]) -> str:
        """Build a compact summary from recent turns."""

        return " | ".join(turns[-3:])

    @staticmethod
    def _derive_topic(current_turn: str) -> str:
        """Derive a compact active topic from the current turn."""

        lowered = current_turn.lower()
        if "policy" in lowered or "return" in lowered or "shipping" in lowered:
            return "policy"
        if (
            "product" in lowered
            or "plan" in lowered
            or "price" in lowered
            or "review" in lowered
            or "rating" in lowered
        ):
            return "product"
        return "general"

    @staticmethod
    def _is_context_switch(current_turn: str) -> bool:
        """Return whether the turn explicitly signals a topic change."""

        lowered = current_turn.lower()
        return any(marker in lowered for marker in _SWITCH_MARKERS)

    @staticmethod
    def _matches_topic(active_topic: str, item: dict[str, Any]) -> bool:
        """Return whether a retrieved item aligns with the active topic."""

        if active_topic == "general":
            return True

        item_type = str(item.get("source_type", item.get("type", ""))).lower()
        title = str(item.get("title", "")).lower()
        content = str(item.get("content", item.get("body", ""))).lower()

        if active_topic == "policy":
            return "policy" in item_type or "return" in title or "shipping" in title or "warranty" in title
        if active_topic == "product":
            return "product" in item_type or "plan" in title or "price" in content
        return False
