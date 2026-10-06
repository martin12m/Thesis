"""Pydantic models for chat requests, responses, and related records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model with consistent validation settings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SessionContext(StrictModel):
    """Context carried across related chat interactions."""

    session_id: str = Field(min_length=1)
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(StrictModel):
    """Supporting evidence attached to a request or response."""

    source: str = Field(min_length=1)
    content: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(StrictModel):
    """A single tool invocation and its outcome."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "completed", "failed"] = "completed"
    result: dict[str, Any] | list[Any] | str | None = None
    timestamp: datetime | None = None


class ChatRequest(StrictModel):
    """Incoming payload for a chat turn."""

    message: str = Field(min_length=1)
    session: SessionContext | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    synthesize_audio: bool = False


class ChatResponse(StrictModel):
    """Validated response payload for a chat turn."""

    message: str = Field(min_length=1)
    session: SessionContext | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    audio_base64: str | None = None
    synthesis_latency_ms: float | None = Field(default=None, ge=0.0)


class VoiceChatResponse(ChatResponse):
    """Chat response returned for the voice modality, with transcript metadata."""

    transcript: str = Field(min_length=1)
    transcription_latency_ms: float = Field(ge=0.0)
