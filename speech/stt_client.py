"""Lightweight Speech-to-Text adapter wrapping ``faster-whisper``."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_SIZE = "base"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"


class TranscriptionError(RuntimeError):
    """Raised when an audio file cannot be transcribed."""


@dataclass(slots=True)
class TranscriptionResult:
    """Structured transcription output with basic timing/confidence info."""

    text: str
    language: str | None
    duration_seconds: float | None
    latency_ms: float
    segments: list[dict[str, Any]]


@dataclass(slots=True)
class SpeechToTextClient:
    """Adapter around ``faster_whisper.WhisperModel`` for local STT."""

    model_size: str | None = None
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    model_factory: Any = None
    _model: Any = None

    def __post_init__(self) -> None:
        self.model_size = self.model_size or os.environ.get(
            "WHISPER_MODEL_SIZE", DEFAULT_MODEL_SIZE
        )

    def _get_model(self) -> Any:
        """Return the cached Whisper model, constructing it on first use."""

        if self._model is None:
            factory = self.model_factory
            if factory is None:
                try:
                    from faster_whisper import WhisperModel as factory
                except ImportError as exc:
                    msg = "faster-whisper is required to use SpeechToTextClient."
                    raise TranscriptionError(msg) from exc

            self._model = factory(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe a local audio file (.wav, .mp3, ...) to text."""

        path = Path(audio_path)
        if not path.exists():
            msg = f"Audio file not found: {path}"
            raise TranscriptionError(msg)

        model = self._get_model()

        start = time.perf_counter()
        try:
            segments, info = model.transcribe(str(path))
        except Exception as exc:  # noqa: BLE001 - surface any backend failure uniformly
            msg = f"Failed to transcribe audio file {path}: {exc}"
            raise TranscriptionError(msg) from exc
        segment_list = list(segments)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        if not text:
            msg = f"Transcription produced no text for {path}"
            raise TranscriptionError(msg)

        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_seconds=info.duration,
            latency_ms=latency_ms,
            segments=[
                {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
                for segment in segment_list
            ],
        )
