"""Text generation clients: local llama.cpp and OpenAI-compatible HTTP."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from llm.interface import GenerationConfig, OpenAICompatibleConfig, TextGenerator

logger = logging.getLogger(__name__)


class LlamaCppClient(TextGenerator):
    """In-process generation via a llama.cpp-compatible backend."""

    def __init__(
        self,
        config: GenerationConfig,
        *,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the adapter with static generation settings."""

        self._config = config
        factory = backend_factory or self._default_backend_factory()
        self._backend = factory(model_path=config.model_path, n_ctx=config.context_window)

    @property
    def config(self) -> GenerationConfig:
        """Return the active client configuration."""

        return self._config

    def generate(self, prompt: str) -> str:
        """Generate text from the configured llama.cpp backend."""

        response = self._backend(
            prompt,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        return str(response["choices"][0]["text"])

    @staticmethod
    def _default_backend_factory() -> Callable[..., Any]:
        """Import and return the default llama.cpp backend factory."""

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            msg = "llama_cpp is required to use LlamaCppClient."
            raise RuntimeError(msg) from exc
        return Llama


class OpenAICompatibleClient(TextGenerator):
    """Generation via any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the adapter, optionally injecting an httpx client for tests."""

        self._config = config
        self._client = http_client or httpx.Client(timeout=config.timeout_seconds)
        self._owns_client = http_client is None

    @property
    def config(self) -> OpenAICompatibleConfig:
        """Return the active client configuration."""

        return self._config

    def generate(self, prompt: str) -> str:
        """Generate text via a chat completion request."""

        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            **self._config.extra_params,
        }
        headers = {"Authorization": f"Bearer {self._config.api_key}"}

        for attempt in range(self._config.max_retries + 1):
            response = self._client.post(url, json=payload, headers=headers)
            if response.status_code != 429 or attempt == self._config.max_retries:
                break
            delay = _retry_delay_seconds(response, attempt)
            logger.warning(
                "Rate limited (429) by %s, retrying in %.1fs (attempt %d/%d)",
                url,
                delay,
                attempt + 1,
                self._config.max_retries,
            )
            time.sleep(delay)

        if response.is_error:
            msg = (
                f"OpenAI-compatible endpoint returned {response.status_code} "
                f"for {url}: {response.text}"
            )
            raise RuntimeError(msg)
        body = response.json()
        return str(body["choices"][0]["message"]["content"])

    def close(self) -> None:
        """Release the underlying HTTP client if this instance owns it."""

        if self._owns_client:
            self._client.close()


def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    """Compute the backoff delay for a rate-limited (429) retry."""

    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return float(2**attempt)
