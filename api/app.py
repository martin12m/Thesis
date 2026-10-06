"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes_chat import router as chat_router
from api.routes_chat import shutdown_variant_router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release any resources owned by the runtime LLM generator on shutdown."""

    yield
    shutdown_variant_router()


app = FastAPI(title="Chat API", lifespan=_lifespan)
app.include_router(chat_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a compact JSON payload for invalid request bodies."""

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "message": "Invalid request payload.",
            "path": str(request.url.path),
        },
    )
