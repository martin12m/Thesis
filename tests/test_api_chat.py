"""API tests for the chat endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.routes_chat as routes_chat
from api.app import app
from llm.interface import TextGenerator
from orchestration.variant_router import VariantRouter
from variants.hybrid_variant import HybridVariant
from variants.retrieval_variant import RetrievalVariant
from variants.tool_first_variant import ToolFirstVariant

client = TestClient(app)


class _EchoGenerator(TextGenerator):
    """Deterministic generator that echoes a fixed reply for pipeline tests."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return "Stubbed answer for testing."


@pytest.fixture(autouse=True)
def _stub_variant_router(tmp_path: Any) -> Any:
    """Swap in a `VariantRouter` backed by a deterministic generator and empty DB."""

    db_path = tmp_path / "empty_catalog.db"
    from data.loaders import connect_database, seed_database
    from retrieval.indexer import rebuild_index

    connection = connect_database(db_path)
    seed_database(connection, products=[], policies=[])
    connection.close()
    rebuild_index(db_path)

    generator = _EchoGenerator()
    original = routes_chat._variant_router
    routes_chat._variant_router = VariantRouter(
        retrieval_variant=RetrievalVariant(llm=generator, db_path=db_path),
        tool_first_variant=ToolFirstVariant(llm=generator, db_path=db_path),
        hybrid_variant=HybridVariant(llm=generator, db_path=db_path),
    )
    yield
    routes_chat._variant_router = original


def test_chat_endpoint_returns_generated_response() -> None:
    response = client.post(
        "/chat",
        json={
            "message": "Hello world",
            "session": {"session_id": "sess-1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Stubbed answer for testing."
    assert body["session"] == {
        "session_id": "sess-1",
        "user_id": None,
        "metadata": {},
    }
    assert body["metadata"]["status"] == "ok"
    assert "trace" in body["metadata"]


def test_chat_endpoint_tool_calls_reflect_actual_execution(tmp_path: Any) -> None:
    """`tool_calls` in the response must be what the variant executed, not the request echo."""

    db_path = tmp_path / "policy_catalog.db"
    from data.loaders import connect_database, seed_database
    from retrieval.indexer import rebuild_index

    connection = connect_database(db_path)
    seed_database(
        connection,
        products=[],
        policies=[
            {
                "code": "RETURNS",
                "title": "Returns Policy",
                "body": "Returns are accepted within 30 days.",
            }
        ],
    )
    connection.close()
    rebuild_index(db_path)

    generator = _EchoGenerator()
    routes_chat._variant_router = VariantRouter(
        retrieval_variant=RetrievalVariant(llm=generator, db_path=db_path),
        tool_first_variant=ToolFirstVariant(llm=generator, db_path=db_path),
        hybrid_variant=HybridVariant(llm=generator, db_path=db_path),
    )

    response = client.post(
        "/chat",
        json={
            "message": "What is your return policy?",
            "tool_calls": [
                {"tool_name": "not_a_real_tool", "arguments": {}, "status": "pending"}
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"] == [
        {
            "tool_name": "get_return_policy_info",
            "arguments": {},
            "status": "completed",
            "result": None,
            "timestamp": None,
        }
    ]


class _StubTtsClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> Any:
        from types import SimpleNamespace

        self.calls.append(text)
        return SimpleNamespace(audio_bytes=b"fake-wav-bytes", latency_ms=7.5)


def test_chat_endpoint_synthesizes_audio_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_tts = _StubTtsClient()
    monkeypatch.setattr(routes_chat, "_tts_client", stub_tts)

    response = client.post("/chat", json={"message": "Hello world", "synthesize_audio": True})

    assert response.status_code == 200
    body = response.json()
    assert stub_tts.calls == ["Stubbed answer for testing."]
    assert body["synthesis_latency_ms"] == 7.5
    import base64

    assert base64.b64decode(body["audio_base64"]) == b"fake-wav-bytes"


def test_chat_endpoint_omits_audio_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_tts = _StubTtsClient()
    monkeypatch.setattr(routes_chat, "_tts_client", stub_tts)

    response = client.post("/chat", json={"message": "Hello world"})

    assert response.status_code == 200
    body = response.json()
    assert stub_tts.calls == []
    assert body["audio_base64"] is None
    assert body["synthesis_latency_ms"] is None


def test_chat_endpoint_synthesis_failure_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech.tts_client import SynthesisError

    class _BrokenTtsClient:
        def synthesize(self, text: str) -> Any:
            raise SynthesisError("voice model not found")

    monkeypatch.setattr(routes_chat, "_tts_client", _BrokenTtsClient())

    response = client.post("/chat", json={"message": "Hello world", "synthesize_audio": True})

    assert response.status_code == 400
    assert "voice model not found" in response.json()["detail"]


def test_chat_endpoint_rejects_invalid_payload() -> None:
    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Invalid request payload."
    assert body["path"] == "/chat"
