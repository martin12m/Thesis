"""Common interfaces and backend selection for text generation clients."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class GenerationConfig:
    """Configuration for a single text generation client."""

    model_path: str
    context_window: int = 4096
    temperature: float = 0.2
    max_tokens: int = 256


@dataclass(slots=True)
class OpenAICompatibleConfig:
    """Configuration for an OpenAI-compatible HTTP endpoint (e.g. Ollama, Groq)."""

    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.2
    max_tokens: int = 256
    timeout_seconds: float = 60.0
    max_retries: int = 3
    """Retries on HTTP 429 (rate limited) only -- a generic HTTP mechanism
    (RFC 6585), not specific to any provider. Honors the response's
    `Retry-After` header when present, else backs off exponentially
    (1s, 2s, 4s, ...). Set to 0 to disable retrying."""
    extra_params: dict[str, Any] = field(default_factory=dict)
    """Provider-specific chat-completions fields merged into every request.

    Not hardcoded to any provider: e.g. Groq's reasoning-model endpoints
    (Qwen3, DeepSeek-R1-distill, GPT-OSS) accept `reasoning_effort` /
    `reasoning_format` to control or hide a `<think>...</think>` block that
    otherwise eats the `max_tokens` budget before the real answer appears.
    Configured via `LLM_EXTRA_PARAMS` (a JSON object) in `build_default_generator`.
    """


class TextGenerator(Protocol):
    """Protocol for components that generate text from a prompt."""

    def generate(self, prompt: str) -> str:
        """Generate text for the given prompt."""


MODEL_VARIANT_PATHS: dict[str, str] = {
    "llama": "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "mistral": "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
}


def build_default_generator(env: Mapping[str, str] | None = None) -> TextGenerator:
    """Build the configured runtime `TextGenerator` from environment variables."""

    from llm.llama_cpp_client import LlamaCppClient, OpenAICompatibleClient  # noqa: PLC0415

    environment = _resolve_environment(env)
    backend = environment.get("LLM_BACKEND", "llama_cpp").lower()

    if backend == "openai_compatible":
        config = OpenAICompatibleConfig(
            base_url=environment.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            model=environment.get("LLM_MODEL", "llama3"),
            api_key=environment.get("LLM_API_KEY", "not-needed"),
            temperature=float(environment.get("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(environment.get("LLM_MAX_TOKENS", "256")),
            timeout_seconds=float(environment.get("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(environment.get("LLM_MAX_RETRIES", "3")),
            extra_params=_parse_extra_params(environment.get("LLM_EXTRA_PARAMS")),
        )
        return OpenAICompatibleClient(config)

    if backend == "llama_cpp":
        config = GenerationConfig(
            model_path=environment.get("MODEL_PATH", _resolve_model_variant_path(environment)),
            context_window=int(environment.get("LLM_CONTEXT_WINDOW", "4096")),
            temperature=float(environment.get("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(environment.get("LLM_MAX_TOKENS", "256")),
        )
        return LlamaCppClient(config)

    msg = f"Unknown LLM_BACKEND '{backend}'; expected 'llama_cpp' or 'openai_compatible'."
    raise ValueError(msg)


def list_configured_profiles(env: Mapping[str, str] | None = None) -> list[str]:
    """Return the model profile names declared in `LLM_PROFILES` (comma-separated)."""

    raw = _resolve_environment(env).get("LLM_PROFILES", "")
    return [name.strip() for name in raw.split(",") if name.strip()]


def build_generator_for_profile(name: str, env: Mapping[str, str] | None = None) -> TextGenerator:
    """Build a `TextGenerator` for one named model profile."""

    environment = _resolve_environment(env)
    prefix = f"LLM_PROFILE_{name.upper()}_"
    resolved = dict(environment)
    for key, value in environment.items():
        if key.startswith(prefix):
            field = key[len(prefix) :]
            resolved[field if field == "MODEL_PATH" else f"LLM_{field}"] = value

    return build_default_generator(resolved)


def _resolve_environment(env: Mapping[str, str] | None) -> Mapping[str, str]:
    """Return `env`, or the process environment after loading any `.env` file."""

    if env is not None:
        return env
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
    return os.environ


def _parse_extra_params(raw: str | None) -> dict[str, Any]:
    """Parse `LLM_EXTRA_PARAMS` (a JSON object string) into a dict."""

    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"LLM_EXTRA_PARAMS must be a JSON object, got invalid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"LLM_EXTRA_PARAMS must be a JSON object, got {type(parsed).__name__}."
        raise ValueError(msg)
    return parsed


def _resolve_model_variant_path(environment: Mapping[str, str]) -> str:
    """Resolve the default GGUF path for ``LLM_MODEL_VARIANT``, if set."""

    variant = environment.get("LLM_MODEL_VARIANT", "llama").lower()
    try:
        return MODEL_VARIANT_PATHS[variant]
    except KeyError:
        msg = (
            f"Unknown LLM_MODEL_VARIANT '{variant}'; expected one of "
            f"{sorted(MODEL_VARIANT_PATHS)}."
        )
        raise ValueError(msg) from None
