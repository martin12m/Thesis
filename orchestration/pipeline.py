"""Framework-agnostic core chat pipeline shared by every entrypoint."""

from __future__ import annotations

import base64
from typing import Any, Protocol

from models import ChatRequest, ChatResponse, EvidenceItem, ToolCallRecord
from orchestration.variant_router import VariantRouter


class SupportsSynthesize(Protocol):
    """Protocol for TTS clients the pipeline can call for voice output."""

    def synthesize(self, text: str) -> Any:
        """Synthesize `text` to audio, returning an object with `audio_bytes: bytes` and `latency_ms: float` attributes."""


def _to_evidence_items(evidence: list[dict[str, Any]]) -> list[EvidenceItem]:
    """Map retrieval-shaped evidence dicts onto the API's `EvidenceItem` schema."""

    return [
        EvidenceItem(
            source=str(item.get("document_id", "unknown")),
            content=str(item.get("content") or item.get("title") or ""),
            metadata={key: item[key] for key in ("title", "score") if item.get(key) is not None},
        )
        for item in evidence
        if item.get("content") or item.get("title")
    ]


def _to_tool_call_records(tool_calls: list[dict[str, Any]]) -> list[ToolCallRecord]:
    """Map trace-shaped tool-call dicts onto the API's `ToolCallRecord` schema."""

    return [
        ToolCallRecord(
            tool_name=str(call["tool_name"]),
            arguments=call.get("arguments") or {},
            status=call.get("status", "completed"),
        )
        for call in tool_calls
    ]


def run_chat_pipeline(
    router: VariantRouter,
    request: ChatRequest,
    *,
    tts_client: SupportsSynthesize | None = None,
) -> ChatResponse:
    """Route `request` through `router` and shape the result into a `ChatResponse`."""

    session_context = request.session.model_dump() if request.session else None
    result = router.run(request.message, session_context)
    trace = result["trace"]
    message = str(result["answer"])

    audio_base64: str | None = None
    synthesis_latency_ms: float | None = None
    if request.synthesize_audio and tts_client is not None:
        synthesis = tts_client.synthesize(message)
        audio_base64 = base64.b64encode(synthesis.audio_bytes).decode("ascii")
        synthesis_latency_ms = synthesis.latency_ms

    return ChatResponse(
        message=message,
        session=request.session,
        evidence=_to_evidence_items(result.get("evidence", [])),
        tool_calls=_to_tool_call_records(trace.get("tool_calls", [])),
        metadata={"status": "ok", "trace": trace},
        audio_base64=audio_base64,
        synthesis_latency_ms=synthesis_latency_ms,
    )
