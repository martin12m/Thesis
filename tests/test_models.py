"""Basic validation tests for the Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import ChatRequest, ChatResponse, EvidenceItem, SessionContext, ToolCallRecord


def test_chat_request_parses_nested_payloads() -> None:
    request = ChatRequest(
        message="Summarize this",
        session={"session_id": "sess-1", "user_id": "user-1"},
        evidence=[
            {
                "source": "knowledge_base",
                "content": "Important context",
                "confidence": 0.9,
            }
        ],
        tool_calls=[
            {
                "tool_name": "search",
                "arguments": {"query": "topic"},
                "status": "completed",
                "result": {"hits": 3},
            }
        ],
    )

    assert isinstance(request.session, SessionContext)
    assert isinstance(request.evidence[0], EvidenceItem)
    assert isinstance(request.tool_calls[0], ToolCallRecord)
    assert request.evidence[0].confidence == pytest.approx(0.9)


def test_chat_response_defaults_are_independent() -> None:
    first = ChatResponse(message="First")
    second = ChatResponse(message="Second")

    first.metadata["trace_id"] = "abc123"
    first.evidence.append(EvidenceItem(source="doc", content="snippet"))

    assert second.metadata == {}
    assert second.evidence == []


def test_evidence_confidence_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(source="doc", content="snippet", confidence=1.5)


def test_blank_message_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        ToolCallRecord(tool_name="search", unexpected=True)
