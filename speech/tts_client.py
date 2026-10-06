"""Lightweight Text-to-Speech adapter wrapping ``piper-tts``."""

from __future__ import annotations

import io
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_VOICE = "en_US-lessac-medium"
DEFAULT_VOICES_DIR = "models/piper_voices"


class SynthesisError(RuntimeError):
    """Raised when text cannot be synthesized to audio."""


@dataclass(slots=True)
class SynthesisResult:
    """Structured synthesis output with raw audio bytes and basic timing info."""

    audio_bytes: bytes
    audio_format: str
    sample_rate: int | None
    latency_ms: float


@dataclass(slots=True)
class TTSClient:
    """Adapter around ``piper.voice.PiperVoice`` for local TTS."""

    voice_name: str | None = None
    voices_dir: str | Path | None = None
    voice_factory: Any = None
    _voice: Any = None

    def __post_init__(self) -> None:
        self.voice_name = self.voice_name or os.environ.get("TTS_VOICE", DEFAULT_VOICE)
        self.voices_dir = self.voices_dir or os.environ.get("TTS_VOICES_DIR", DEFAULT_VOICES_DIR)

    def _model_path(self) -> Path:
        """Return the ``.onnx`` model path for the configured voice."""

        return Path(self.voices_dir) / f"{self.voice_name}.onnx"

    def _get_voice(self) -> Any:
        """Return the cached Piper voice, constructing it on first use."""

        if self._voice is not None:
            return self._voice

        if self.voice_factory is not None:
            self._voice = self.voice_factory(self.voice_name)
            return self._voice

        model_path = self._model_path()
        if not model_path.exists():
            msg = (
                f"Piper voice model not found: {model_path}. Download it with "
                f"`python -m piper.download_voices {self.voice_name} "
                f"--download-dir {self.voices_dir}`."
            )
            raise SynthesisError(msg)

        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            msg = "piper-tts is required to use TTSClient."
            raise SynthesisError(msg) from exc

        self._voice = PiperVoice.load(str(model_path))
        return self._voice

    def synthesize(self, text: str) -> SynthesisResult:
        """Synthesize `text` to WAV audio bytes."""

        if not text.strip():
            msg = "Cannot synthesize empty text."
            raise SynthesisError(msg)

        voice = self._get_voice()

        start = time.perf_counter()
        buffer = io.BytesIO()
        wav_file = wave.open(buffer, "wb")
        try:
            voice.synthesize_wav(text, wav_file)
        except Exception as exc:  # noqa: BLE001 - surface any backend failure uniformly
            msg = f"Failed to synthesize audio for text ({len(text)} chars): {exc}"
            raise SynthesisError(msg) from exc
        finally:
            try:
                wav_file.close()
            except Exception:  # noqa: BLE001, S110
                pass
        latency_ms = (time.perf_counter() - start) * 1000.0

        audio_bytes = buffer.getvalue()
        if not audio_bytes:
            msg = "Synthesis produced no audio bytes."
            raise SynthesisError(msg)

        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_reader:
            sample_rate = wav_reader.getframerate()

        return SynthesisResult(
            audio_bytes=audio_bytes,
            audio_format="wav",
            sample_rate=sample_rate,
            latency_ms=latency_ms,
        )
