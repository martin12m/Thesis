"""Tests for the /chat/voice audio upload endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.routes_chat as routes_chat
from api.app import app
from llm.interface import TextGenerator
from orchestration.variant_router import VariantRouter
from speech.stt_client import TranscriptionResult
from variants.hybrid_variant import HybridVariant
from variants.retrieval_variant import RetrievalVariant
from variants.tool_first_variant import ToolFirstVariant

client = TestClient(app)


class _EchoGenerator(TextGenerator):
    """Deterministic generator that echoes a fixed reply for pipeline tests."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return "Stubbed answer for testing."


class _StubSttClient:
    def __init__(self, result: TranscriptionResult | Exception) -> None:
        self._result = result
        self.transcribe_calls: list[str] = []

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        self.transcribe_calls.append(str(audio_path))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.fixture(autouse=True)
def _restore_stt_client() -> Any:
    original = routes_chat._stt_client
    yield
    routes_chat._stt_client = original


@pytest.fixture(autouse=True)
def _stub_variant_router(tmp_path: Any) -> Any:
    """Swap in a `VariantRouter` backed by a deterministic generator and empty DB."""

    db_path = tmp_path / "empty_catalog.db"
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
                "body": "Returns are accepted within 30 days for eligible purchases.",
            }
        ],
    )
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


def test_voice_chat_transcribes_and_runs_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSttClient(
        TranscriptionResult(
            text="What is your return policy?",
            language="en",
            duration_seconds=1.2,
            latency_ms=42.0,
            segments=[{"start": 0.0, "end": 1.2, "text": "What is your return policy?"}],
        )
    )
    monkeypatch.setattr(routes_chat, "_stt_client", stub)

    response = client.post(
        "/chat/voice",
        files={"audio": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "What is your return policy?"
    assert body["transcription_latency_ms"] == pytest.approx(42.0)
    assert body["message"] == "Stubbed answer for testing."
    assert len(stub.transcribe_calls) == 1


def test_voice_chat_forwards_session_context(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSttClient(
        TranscriptionResult(
            text="What is your return policy?",
            language="en",
            duration_seconds=1.2,
            latency_ms=42.0,
            segments=[],
        )
    )
    monkeypatch.setattr(routes_chat, "_stt_client", stub)

    response = client.post(
        "/chat/voice",
        files={"audio": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
        data={"session": '{"session_id": "sess-42"}'},
    )

    assert response.status_code == 200
    assert response.json()["session"] == {
        "session_id": "sess-42",
        "user_id": None,
        "metadata": {},
    }


def test_voice_chat_synthesizes_audio_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    stub_stt = _StubSttClient(
        TranscriptionResult(
            text="What is your return policy?",
            language="en",
            duration_seconds=1.2,
            latency_ms=42.0,
            segments=[],
        )
    )
    monkeypatch.setattr(routes_chat, "_stt_client", stub_stt)

    class _StubTtsClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def synthesize(self, text: str) -> Any:
            self.calls.append(text)
            return SimpleNamespace(audio_bytes=b"fake-wav-bytes", latency_ms=9.0)

    stub_tts = _StubTtsClient()
    monkeypatch.setattr(routes_chat, "_tts_client", stub_tts)

    response = client.post(
        "/chat/voice",
        files={"audio": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
        data={"synthesize_audio": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert stub_tts.calls == ["Stubbed answer for testing."]
    assert body["synthesis_latency_ms"] == 9.0
    import base64

    assert base64.b64decode(body["audio_base64"]) == b"fake-wav-bytes"


def test_voice_chat_omits_audio_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_stt = _StubSttClient(
        TranscriptionResult(
            text="What is your return policy?",
            language="en",
            duration_seconds=1.2,
            latency_ms=42.0,
            segments=[],
        )
    )
    monkeypatch.setattr(routes_chat, "_stt_client", stub_stt)

    response = client.post(
        "/chat/voice",
        files={"audio": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["audio_base64"] is None
    assert body["synthesis_latency_ms"] is None


def test_voice_chat_rejects_invalid_session_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSttClient(
        TranscriptionResult(
            text="What is your return policy?",
            language="en",
            duration_seconds=1.2,
            latency_ms=42.0,
            segments=[],
        )
    )
    monkeypatch.setattr(routes_chat, "_stt_client", stub)

    response = client.post(
        "/chat/voice",
        files={"audio": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
        data={"session": "not-json"},
    )

    assert response.status_code == 400
    assert "session" in response.json()["detail"].lower()


def test_voice_chat_rejects_empty_audio() -> None:
    response = client.post(
        "/chat/voice",
        files={"audio": ("clip.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_voice_chat_surfaces_transcription_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from speech.stt_client import TranscriptionError

    stub = _StubSttClient(TranscriptionError("could not decode audio"))
    monkeypatch.setattr(routes_chat, "_stt_client", stub)

    response = client.post(
        "/chat/voice",
        files={"audio": ("clip.wav", b"fake-audio-bytes", "audio/wav")},
    )

    assert response.status_code == 400
    assert "could not decode audio" in response.json()["detail"]


def test_text_chat_endpoint_still_works() -> None:
    response = client.post(
        "/chat",
        json={"message": "Hello world", "session": {"session_id": "sess-1"}},
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Stubbed answer for testing."
