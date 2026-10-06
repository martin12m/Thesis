"""Hybrid variant that combines structured tool facts with retrieval context."""

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
class HybridVariant:
    """Use tools for exact facts and retrieval for added explanatory context."""

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
        """Run structured lookup first, then blend in retrieved evidence."""

        structured_outputs: list[dict[str, Any]] = []
        context_tool_outputs: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        routes = collect_structured_routes(user_turn)
        for route in routes:
            tool_name = str(route["tool_name"])
            if route["type"] == "policy":
                arguments: dict[str, Any] = {}
                output = run_policy_tool(self.db_path, tool_name)
            else:
                arguments = route["arguments"]
                output = run_catalog_tool(self.db_path, tool_name, arguments)

            structured_outputs.append({"tool_name": tool_name, "output": output})
            context_tool_outputs.append(to_context_tool_output(route["type"], tool_name, output))
            tool_calls.append({"tool_name": tool_name, "arguments": arguments, "status": "completed"})

        evidence = self.retrieval_backend(self.db_path, user_turn, top_k=self.top_k)
        context_package = self.context_manager.build_context_package(
            user_turn,
            session_context,
            retrieved_evidence=evidence,
            tool_outputs=context_tool_outputs,
        )
        prompt = self._build_hybrid_prompt(context_package, structured_output=structured_outputs or None)
        trace = {
            "variant": "hybrid",
            "route": "combined" if routes else "fallback",
            "active_topic": context_package["active_topic"],
            "session_summary": context_package["session_summary"],
            "tool_calls": tool_calls,
            "evidence_count": len(context_package["evidence"]),
        }
        if routes:
            trace["source_ids"] = _collect_source_ids(structured_outputs)
        return {
            "answer": self.llm.generate(prompt),
            "evidence": context_package["evidence"],
            "trace": trace,
        }

    @staticmethod
    def _build_hybrid_prompt(
        context_package: dict[str, Any],
        *,
        structured_output: list[dict[str, Any]] | None,
    ) -> str:
        """Build a prompt that combines exact facts with supporting context."""

        summary = str(context_package["session_summary"])
        return (
            "Answer the user using the structured facts and supporting evidence.\n"
            f"Session summary: {summary or '[none]'}\n"
            f"User turn: {context_package['current_turn']}\n"
            f"Structured facts: {structured_output}\n"
            f"Supporting evidence: {context_package['evidence']}\n"
            "Answer:"
        )


def _collect_source_ids(structured_outputs: list[dict[str, Any]]) -> list[str]:
    """Collect stable source identifiers from structured outputs."""

    source_ids: list[str] = []
    for item in structured_outputs:
        output = item["output"]
        if not isinstance(output, dict):
            continue
        if "source_table" in output and "source_id" in output:
            source_type = {
                "policies": "policy",
                "products": "product",
            }.get(str(output["source_table"]), str(output["source_table"]))
            source_ids.append(f"{source_type}:{output['source_id']}")
        elif "id" in output:
            source_ids.append(f"product:{output['id']}")
    return source_ids
