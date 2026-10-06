"""Tool-first generation variant with rule-based routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm.interface import TextGenerator
from orchestration.context_manager import ContextManager
from orchestration.structured_routing import (
    collect_structured_routes,
    run_catalog_tool,
    run_policy_tool,
    to_context_tool_output,
)
from retrieval.sparse import RetrievalBackend, search_hybrid


@dataclass(slots=True)
class ToolFirstVariant:
    """Route structured queries to tools before using the LLM."""

    llm: TextGenerator
    db_path: str | Path
    retrieval_backend: RetrievalBackend = search_hybrid
    context_manager: ContextManager = field(default_factory=ContextManager)
    top_k: int = 3

    def run(
        self,
        user_turn: str,
        session_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route the query to the first matching tool and verbalize the result."""

        routes = collect_structured_routes(user_turn)
        if not routes:
            return self._run_fallback(user_turn, session_context)

        route = routes[0]
        tool_name = str(route["tool_name"])
        if route["type"] == "policy":
            arguments: dict[str, Any] = {}
            structured_output = run_policy_tool(self.db_path, tool_name)
        else:
            arguments = route["arguments"]
            structured_output = run_catalog_tool(self.db_path, tool_name, arguments)

        context_package = self.context_manager.build_context_package(
            user_turn,
            session_context,
            tool_outputs=[to_context_tool_output(route["type"], tool_name, structured_output)],
        )
        prompt = self._build_tool_prompt(context_package, tool_name, structured_output)
        return {
            "answer": self.llm.generate(prompt),
            "trace": {
                "variant": "tool_first",
                "route": route["type"],
                "active_topic": context_package["active_topic"],
                "session_summary": context_package["session_summary"],
                "tool_calls": [
                    {"tool_name": tool_name, "arguments": arguments, "status": "completed"}
                ],
            },
        }

    def _run_fallback(
        self,
        user_turn: str,
        session_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Answer from retrieved evidence when no tool route matches."""

        evidence = self.retrieval_backend(self.db_path, user_turn, top_k=self.top_k)
        context_package = self.context_manager.build_context_package(
            user_turn,
            session_context,
            retrieved_evidence=evidence,
        )
        prompt = self._build_fallback_prompt(context_package)
        return {
            "answer": self.llm.generate(prompt),
            "trace": {
                "variant": "tool_first",
                "route": "fallback",
                "active_topic": context_package["active_topic"],
                "session_summary": context_package["session_summary"],
                "tool_calls": [],
                "fallback_evidence_count": len(context_package["evidence"]),
            },
        }

    @staticmethod
    def _build_tool_prompt(
        context_package: dict[str, Any],
        tool_name: str,
        structured_output: dict[str, Any] | list[dict[str, Any]],
    ) -> str:
        """Build a prompt that verbalizes structured tool output."""

        summary = str(context_package["session_summary"])
        return (
            "Answer the user using only the structured tool output.\n"
            f"Session summary: {summary or '[none]'}\n"
            f"User turn: {context_package['current_turn']}\n"
            f"Tool: {tool_name}\n"
            f"Structured output: {structured_output}\n"
            "Answer:"
        )

    @staticmethod
    def _build_fallback_prompt(context_package: dict[str, Any]) -> str:
        """Build a prompt for retrieval fallback when no tool route matches."""

        summary = str(context_package["session_summary"])
        return (
            "No structured tool matched the query. Answer using the fallback evidence.\n"
            f"Session summary: {summary or '[none]'}\n"
            f"User turn: {context_package['current_turn']}\n"
            f"Evidence: {context_package['evidence']}\n"
            "Answer:"
        )
