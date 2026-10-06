"""Chat API routes."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from llm.interface import TextGenerator, build_default_generator
from models import ChatRequest, ChatResponse, SessionContext, VoiceChatResponse
from orchestration.pipeline import run_chat_pipeline
from orchestration.variant_router import VariantRouter
from speech.stt_client import SpeechToTextClient, TranscriptionError
from speech.tts_client import SynthesisError, TTSClient
from variants.hybrid_variant import HybridVariant
from variants.retrieval_variant import RetrievalVariant
from variants.tool_first_variant import ToolFirstVariant

logger = logging.getLogger(__name__)

router = APIRouter()

_stt_client = SpeechToTextClient()
_tts_client = TTSClient()
_variant_router: VariantRouter | None = None
_generator: TextGenerator | None = None


def _get_variant_router() -> VariantRouter:
    """Return the module-level `VariantRouter`, building it on first use."""

    global _variant_router, _generator  # noqa: PLW0603
    if _variant_router is None:
        _generator = build_default_generator()
        db_path = os.environ.get("CATALOG_DB_PATH", "data/catalog.db")
        _variant_router = VariantRouter(
            retrieval_variant=RetrievalVariant(llm=_generator, db_path=db_path),
            tool_first_variant=ToolFirstVariant(llm=_generator, db_path=db_path),
            hybrid_variant=HybridVariant(llm=_generator, db_path=db_path),
        )
    return _variant_router


def shutdown_variant_router() -> None:
    """Release the generator's resources, if any, on application shutdown."""

    global _variant_router, _generator  # noqa: PLW0603
    close = getattr(_generator, "close", None)
    if close is not None:
        close()
    _generator = None
    _variant_router = None


def _run_pipeline(request: ChatRequest) -> ChatResponse:
    """Run the shared pipeline, mapping its client-caused failures to HTTP 400."""

    try:
        return run_chat_pipeline(_get_variant_router(), request, tts_client=_tts_client)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except SynthesisError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not synthesize audio: {exc}",
        ) from exc


@router.post("/chat", response_model=ChatResponse)
async def create_chat(request: ChatRequest) -> ChatResponse:
    """Process a text chat request and run it through the shared pipeline."""

    return _run_pipeline(request)


@router.post("/chat/voice", response_model=VoiceChatResponse)
async def create_voice_chat(
    audio: UploadFile = File(...),
    session: str | None = Form(default=None),
    synthesize_audio: bool = Form(default=False),
) -> VoiceChatResponse:
    """Transcribe an uploaded audio file and run it through the chat pipeline."""

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    suffix = Path(audio.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()

        start = time.perf_counter()
        try:
            transcription = _stt_client.transcribe(tmp_file.name)
        except TranscriptionError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not transcribe audio: {exc}",
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

    logger.info(
        "Transcribed voice chat input in %.1fms (reported model latency %.1fms)",
        latency_ms,
        transcription.latency_ms,
    )

    session_context: SessionContext | None = None
    if session is not None:
        try:
            session_context = SessionContext.model_validate_json(session)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid session payload: {exc}",
            ) from exc

    chat_response = _run_pipeline(
        ChatRequest(
            message=transcription.text,
            session=session_context,
            synthesize_audio=synthesize_audio,
        )
    )
    return VoiceChatResponse(
        **chat_response.model_dump(),
        transcript=transcription.text,
        transcription_latency_ms=transcription.latency_ms,
    )
