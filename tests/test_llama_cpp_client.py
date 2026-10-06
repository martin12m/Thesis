"""Tests for the text generation interface, llama.cpp, and OpenAI-compatible adapters."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm.interface import GenerationConfig, OpenAICompatibleConfig
from llm.llama_cpp_client import LlamaCppClient, OpenAICompatibleClient


class MockGenerator:
    """Callable mock backend for deterministic generation tests."""

    def __init__(self, *, model_path: str, n_ctx: int) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, list[dict[str, str]]]:
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return {"choices": [{"text": f"echo:{prompt}"}]}


def test_llama_cpp_client_uses_configured_backend_settings() -> None:
    created_backends: list[MockGenerator] = []

    def factory(*, model_path: str, n_ctx: int) -> MockGenerator:
        backend = MockGenerator(model_path=model_path, n_ctx=n_ctx)
        created_backends.append(backend)
        return backend

    config = GenerationConfig(
        model_path="/models/test.gguf",
        context_window=8192,
        temperature=0.7,
        max_tokens=128,
    )
    client = LlamaCppClient(config, backend_factory=factory)

    result = client.generate("Hello")

    assert result == "echo:Hello"
    assert client.config == config
    assert len(created_backends) == 1
    assert created_backends[0].model_path == "/models/test.gguf"
    assert created_backends[0].n_ctx == 8192
    assert created_backends[0].calls == [
        {
            "prompt": "Hello",
            "max_tokens": 128,
            "temperature": 0.7,
        }
    ]


def test_llama_cpp_client_raises_clear_error_without_dependency() -> None:
    config = GenerationConfig(model_path="/models/test.gguf")

    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        pass
    else:
        return

    try:
        LlamaCppClient(config)
    except RuntimeError as exc:
        assert str(exc) == "llama_cpp is required to use LlamaCppClient."
    else:
        raise AssertionError("Expected RuntimeError when llama_cpp is unavailable.")


def test_openai_compatible_client_sends_expected_chat_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hello there"}}]},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)

    config = OpenAICompatibleConfig(
        base_url="http://localhost:11434/v1",
        model="llama3",
        api_key="test-key",
        temperature=0.5,
        max_tokens=64,
    )
    client = OpenAICompatibleClient(config, http_client=http_client)

    result = client.generate("Hi")

    assert result == "hello there"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["body"] == {
        "model": "llama3",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
        "max_tokens": 64,
    }
    assert captured["headers"]["authorization"] == "Bearer test-key"


def test_openai_compatible_client_merges_extra_params_into_payload() -> None:
    """extra_params (e.g. Groq's reasoning_effort for Qwen3) must reach the request body."""

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "4"}}]},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    config = OpenAICompatibleConfig(
        base_url="http://localhost:11434/v1",
        model="qwen/qwen3.6-27b",
        extra_params={"reasoning_effort": "none"},
    )
    client = OpenAICompatibleClient(config, http_client=http_client)

    client.generate("What is 2+2?")

    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["model"] == "qwen/qwen3.6-27b"


def test_openai_compatible_client_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(500, json={"error": "boom"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    config = OpenAICompatibleConfig(base_url="http://localhost:11434/v1", model="llama3")
    client = OpenAICompatibleClient(config, http_client=http_client)

    try:
        client.generate("Hi")
    except RuntimeError as exc:
        assert "500" in str(exc)
        assert "boom" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError on a 500 response.")


def test_openai_compatible_client_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("llm.llama_cpp_client.time.sleep", lambda seconds: sleeps.append(seconds))

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "rate limited"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    config = OpenAICompatibleConfig(base_url="http://localhost:11434/v1", model="llama3")
    client = OpenAICompatibleClient(config, http_client=http_client)

    result = client.generate("Hi")

    assert result == "ok"
    assert call_count == 2
    assert sleeps == [2.0]  # honored the Retry-After header exactly


def test_openai_compatible_client_backs_off_exponentially_without_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("llm.llama_cpp_client.time.sleep", lambda seconds: sleeps.append(seconds))

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(429, json={"error": "rate limited"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    config = OpenAICompatibleConfig(base_url="http://localhost:11434/v1", model="llama3", max_retries=3)
    client = OpenAICompatibleClient(config, http_client=http_client)

    try:
        client.generate("Hi")
    except RuntimeError as exc:
        assert "429" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError after exhausting retries.")

    assert sleeps == [1.0, 2.0, 4.0]  # exponential backoff, one sleep per retried attempt


def test_openai_compatible_client_max_retries_zero_disables_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("llm.llama_cpp_client.time.sleep", lambda seconds: sleeps.append(seconds))

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, json={"error": "rate limited"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    config = OpenAICompatibleConfig(base_url="http://localhost:11434/v1", model="llama3", max_retries=0)
    client = OpenAICompatibleClient(config, http_client=http_client)

    with pytest.raises(RuntimeError, match="429"):
        client.generate("Hi")

    assert call_count == 1
    assert sleeps == []
