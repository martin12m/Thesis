"""Tests for the Text-to-Speech adapter, using a mocked Piper-style engine."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

from speech.tts_client import SynthesisError, TTSClient


class _FakePiperVoice:
    """Stand-in for piper.voice.PiperVoice with canned WAV output."""

    def __init__(self, frame_count: int = 100, sample_rate: int = 22050) -> None:
        self._frame_count = frame_count
        self._sample_rate = sample_rate
        self.synthesize_calls: list[str] = []

    def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
        self.synthesize_calls.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(self._sample_rate)
        wav_file.writeframes(b"\x00\x00" * self._frame_count)


def test_synthesize_returns_audio_bytes_and_latency() -> None:
    fake_voice = _FakePiperVoice(sample_rate=22050)
    client = TTSClient(voice_factory=lambda voice_name: fake_voice)

    result = client.synthesize("Returns are accepted within thirty days.")

    assert result.audio_format == "wav"
    assert result.sample_rate == 22050
    assert result.latency_ms >= 0.0
    assert len(result.audio_bytes) > 0
    assert fake_voice.synthesize_calls == ["Returns are accepted within thirty days."]

    import io

    with wave.open(io.BytesIO(result.audio_bytes), "rb") as wav_reader:
        assert wav_reader.getnchannels() == 1
        assert wav_reader.getframerate() == 22050


def test_synthesize_caches_voice_across_calls() -> None:
    fake_voice = _FakePiperVoice()
    build_calls = 0

    def factory(voice_name: str) -> Any:
        nonlocal build_calls
        build_calls += 1
        return fake_voice

    client = TTSClient(voice_factory=factory)
    client.synthesize("First.")
    client.synthesize("Second.")

    assert build_calls == 1


def test_voice_env_vars_take_effect_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataclass default must not shadow TTS_VOICE/TTS_VOICES_DIR when unset explicitly."""

    monkeypatch.setenv("TTS_VOICE", "de_DE-thorsten-medium")
    monkeypatch.setenv("TTS_VOICES_DIR", "/custom/voices")

    client = TTSClient(voice_factory=lambda voice_name: _FakePiperVoice())

    assert client.voice_name == "de_DE-thorsten-medium"
    assert str(client.voices_dir) == "/custom/voices"


def test_explicit_voice_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTS_VOICE", "de_DE-thorsten-medium")

    client = TTSClient(voice_name="en_US-lessac-medium", voice_factory=lambda voice_name: _FakePiperVoice())

    assert client.voice_name == "en_US-lessac-medium"


def test_synthesize_empty_text_raises() -> None:
    client = TTSClient(voice_factory=lambda voice_name: _FakePiperVoice())

    with pytest.raises(SynthesisError, match="empty"):
        client.synthesize("   ")


def test_synthesize_missing_voice_model_raises(tmp_path: Path) -> None:
    client = TTSClient(voice_name="nonexistent-voice", voices_dir=tmp_path)

    with pytest.raises(SynthesisError, match="not found"):
        client.synthesize("Hello")


def test_synthesize_backend_failure_wrapped() -> None:
    class _BrokenVoice:
        def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
            raise RuntimeError("backend exploded")

    client = TTSClient(voice_factory=lambda voice_name: _BrokenVoice())

    with pytest.raises(SynthesisError, match="backend exploded"):
        client.synthesize("Hello")


def test_missing_piper_dependency_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    model_path = tmp_path / "en_US-lessac-medium.onnx"
    model_path.write_bytes(b"not a real model")

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "piper.voice":
            raise ImportError("no module named piper.voice")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = TTSClient(voices_dir=tmp_path)
    with pytest.raises(SynthesisError, match="piper-tts is required"):
        client.synthesize("Hello")
