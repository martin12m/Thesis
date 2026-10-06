"""Tests for the Speech-to-Text adapter, using a mocked whisper-style engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from speech.stt_client import SpeechToTextClient, TranscriptionError


class _FakeSegment:
    def __init__(self, text: str, start: float, end: float) -> None:
        self.text = text
        self.start = start
        self.end = end


class _FakeInfo:
    def __init__(self, language: str = "en", duration: float = 1.5) -> None:
        self.language = language
        self.duration = duration


class _FakeWhisperModel:
    """Stand-in for faster_whisper.WhisperModel with canned output."""

    def __init__(self, segments: list[_FakeSegment], info: _FakeInfo) -> None:
        self._segments = segments
        self._info = info
        self.transcribe_calls: list[str] = []

    def transcribe(self, path: str) -> tuple[list[_FakeSegment], _FakeInfo]:
        self.transcribe_calls.append(path)
        return self._segments, self._info


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


def test_transcribe_returns_joined_text_and_latency(audio_file: Path) -> None:
    fake_model = _FakeWhisperModel(
        segments=[
            _FakeSegment("Hello there,", 0.0, 1.0),
            _FakeSegment("how can I help?", 1.0, 2.4),
        ],
        info=_FakeInfo(language="en", duration=2.4),
    )

    def factory(model_size: str, *, device: str, compute_type: str) -> Any:
        return fake_model

    client = SpeechToTextClient(model_size="tiny", model_factory=factory)
    result = client.transcribe(audio_file)

    assert result.text == "Hello there, how can I help?"
    assert result.language == "en"
    assert result.duration_seconds == pytest.approx(2.4)
    assert result.latency_ms >= 0.0
    assert len(result.segments) == 2
    assert fake_model.transcribe_calls == [str(audio_file)]


def test_transcribe_caches_model_across_calls(audio_file: Path) -> None:
    fake_model = _FakeWhisperModel(
        segments=[_FakeSegment("ok", 0.0, 0.5)],
        info=_FakeInfo(),
    )
    build_calls = 0

    def factory(model_size: str, *, device: str, compute_type: str) -> Any:
        nonlocal build_calls
        build_calls += 1
        return fake_model

    client = SpeechToTextClient(model_factory=factory)
    client.transcribe(audio_file)
    client.transcribe(audio_file)

    assert build_calls == 1


def test_model_size_env_var_takes_effect_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataclass default must not shadow WHISPER_MODEL_SIZE when unset explicitly."""

    monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v3")

    client = SpeechToTextClient(model_factory=lambda *a, **k: None)

    assert client.model_size == "large-v3"


def test_explicit_model_size_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v3")

    client = SpeechToTextClient(model_size="tiny", model_factory=lambda *a, **k: None)

    assert client.model_size == "tiny"


def test_transcribe_missing_file_raises() -> None:
    client = SpeechToTextClient(model_factory=lambda *a, **k: None)

    with pytest.raises(TranscriptionError, match="not found"):
        client.transcribe("/nonexistent/path/audio.wav")


def test_transcribe_empty_result_raises(audio_file: Path) -> None:
    fake_model = _FakeWhisperModel(segments=[], info=_FakeInfo())

    def factory(model_size: str, *, device: str, compute_type: str) -> Any:
        return fake_model

    client = SpeechToTextClient(model_factory=factory)

    with pytest.raises(TranscriptionError, match="no text"):
        client.transcribe(audio_file)


def test_transcribe_backend_failure_wrapped(audio_file: Path) -> None:
    class _BrokenModel:
        def transcribe(self, path: str) -> Any:
            raise RuntimeError("backend exploded")

    client = SpeechToTextClient(model_factory=lambda *a, **k: _BrokenModel())

    with pytest.raises(TranscriptionError, match="backend exploded"):
        client.transcribe(audio_file)


def test_missing_faster_whisper_dependency_raises(
    audio_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "faster_whisper":
            raise ImportError("no module named faster_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = SpeechToTextClient()
    with pytest.raises(TranscriptionError, match="faster-whisper is required"):
        client.transcribe(audio_file)
