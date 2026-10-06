"""Tests for the framework-agnostic core chat pipeline."""

from __future__ import annotations

from typing import Any

from models import ChatRequest, SessionContext
from orchestration.pipeline import run_chat_pipeline
from orchestration.variant_router import VariantRouter


class _StubVariant:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def run(self, user_turn: str, session_context: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((user_turn, session_context))
        return self._result


def _build_router(result: dict[str, Any]) -> tuple[VariantRouter, _StubVariant]:
    variant = _StubVariant(result)
    router = VariantRouter(
        retrieval_variant=variant,  # type: ignore[arg-type]
        tool_first_variant=variant,  # type: ignore[arg-type]
        hybrid_variant=variant,  # type: ignore[arg-type]
    )
    return router, variant


def test_run_chat_pipeline_shapes_evidence_and_tool_calls() -> None:
    result = {
        "answer": "Returns are accepted within 30 days.",
        "evidence": [{"document_id": "policy:1", "title": "Returns", "content": "...", "score": -1.2}],
        "trace": {
            "tool_calls": [
                {"tool_name": "get_return_policy_info", "arguments": {}, "status": "completed"}
            ]
        },
    }
    router, variant = _build_router(result)
    request = ChatRequest(message="What is your return policy?")

    response = run_chat_pipeline(router, request)

    assert response.message == "Returns are accepted within 30 days."
    assert len(response.evidence) == 1
    assert response.evidence[0].source == "policy:1"
    assert response.evidence[0].metadata == {"title": "Returns", "score": -1.2}
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "get_return_policy_info"
    assert response.metadata["status"] == "ok"
    assert variant.calls == [("What is your return policy?", None)]


def test_run_chat_pipeline_forwards_session_context() -> None:
    router, variant = _build_router({"answer": "ok", "evidence": [], "trace": {}})
    session = SessionContext(session_id="sess-1")
    request = ChatRequest(message="hi", session=session)

    response = run_chat_pipeline(router, request)

    assert response.session == session
    assert variant.calls == [("hi", {"session_id": "sess-1", "user_id": None, "metadata": {}})]


def test_run_chat_pipeline_defaults_missing_trace_and_evidence() -> None:
    router, _ = _build_router({"answer": "ok"})
    request = ChatRequest(message="hi")

    response = run_chat_pipeline(router, request)

    assert response.evidence == []
    assert response.tool_calls == []
    assert "routing_decision" in response.metadata["trace"]


class _FakeSynthesisResult:
    def __init__(self, audio_bytes: bytes, latency_ms: float) -> None:
        self.audio_bytes = audio_bytes
        self.latency_ms = latency_ms


class _FakeTtsClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> _FakeSynthesisResult:
        self.calls.append(text)
        return _FakeSynthesisResult(audio_bytes=b"fake-wav-bytes", latency_ms=12.5)


def test_run_chat_pipeline_synthesizes_audio_when_requested() -> None:
    router, _ = _build_router({"answer": "Returns are accepted within 30 days."})
    tts_client = _FakeTtsClient()
    request = ChatRequest(message="What is your return policy?", synthesize_audio=True)

    response = run_chat_pipeline(router, request, tts_client=tts_client)

    assert tts_client.calls == ["Returns are accepted within 30 days."]
    assert response.synthesis_latency_ms == 12.5
    import base64

    assert base64.b64decode(response.audio_base64) == b"fake-wav-bytes"


def test_run_chat_pipeline_skips_synthesis_by_default() -> None:
    router, _ = _build_router({"answer": "ok"})
    tts_client = _FakeTtsClient()
    request = ChatRequest(message="hi")  # synthesize_audio defaults to False

    response = run_chat_pipeline(router, request, tts_client=tts_client)

    assert tts_client.calls == []
    assert response.audio_base64 is None
    assert response.synthesis_latency_ms is None


def test_run_chat_pipeline_skips_synthesis_without_tts_client() -> None:
    router, _ = _build_router({"answer": "ok"})
    request = ChatRequest(message="hi", synthesize_audio=True)

    response = run_chat_pipeline(router, request)  # no tts_client passed

    assert response.audio_base64 is None
    assert response.synthesis_latency_ms is None
