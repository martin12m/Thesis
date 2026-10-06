"""Retrieval-grounded generation variant without tool usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm.interface import TextGenerator
from orchestration.context_manager import ContextManager
from retrieval.sparse import RetrievalBackend, search_hybrid


@dataclass(slots=True)
class RetrievalVariant:
    """Run retrieval-grounded generation with a compact trace."""

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
        """Retrieve evidence, build a grounded prompt, and generate an answer."""

        evidence = self.retrieval_backend(self.db_path, user_turn, top_k=self.top_k)
        context_package = self.context_manager.build_context_package(
            user_turn,
            session_context,
            retrieved_evidence=evidence,
        )
        prompt = self._build_grounded_prompt(context_package)
        answer = self.llm.generate(prompt)
        return {
            "answer": answer,
            "evidence": context_package["evidence"],
            "trace": {
                "variant": "retrieval",
                "active_topic": context_package["active_topic"],
                "evidence_count": len(context_package["evidence"]),
                "prompt_chars": len(prompt),
                "session_summary": context_package["session_summary"],
            },
        }

    @staticmethod
    def _build_grounded_prompt(context_package: dict[str, Any]) -> str:
        """Build a compact grounded prompt from the selected context."""

        summary = str(context_package["session_summary"])
        evidence_lines = [
            (
                f"[{item['document_id']}] {item['title']} "
                f"(score={item['score']:.4f}): {item['content']}"
            )
            for item in context_package["evidence"]
        ]
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "[none]"
        return (
            "Use the provided evidence to answer the user.\n"
            f"Session summary: {summary or '[none]'}\n"
            f"User turn: {context_package['current_turn']}\n"
            f"Evidence:\n{evidence_block}\n"
            "Answer:"
        )
