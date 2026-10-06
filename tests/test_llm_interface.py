"""Tests for LLM backend/model selection in llm/interface.py."""

from __future__ import annotations

import pytest

from llm.interface import (
    GenerationConfig,
    MODEL_VARIANT_PATHS,
    OpenAICompatibleConfig,
    build_default_generator,
    build_generator_for_profile,
    list_configured_profiles,
)


class _StubLlamaCppClient:
    """Stand-in for LlamaCppClient that records its config without loading a model."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config


class _StubOpenAICompatibleClient:
    """Stand-in for OpenAICompatibleClient that records its config without an HTTP call."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config


@pytest.fixture(autouse=True)
def _stub_llama_cpp_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm.llama_cpp_client.LlamaCppClient", _StubLlamaCppClient)
    monkeypatch.setattr("llm.llama_cpp_client.OpenAICompatibleClient", _StubOpenAICompatibleClient)


def test_build_default_generator_defaults_to_llama_variant() -> None:
    generator = build_default_generator({})

    assert isinstance(generator, _StubLlamaCppClient)
    assert generator.config.model_path == MODEL_VARIANT_PATHS["llama"]


def test_empty_dict_env_does_not_fall_through_to_real_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit empty env dict means use defaults, not os.environ."""

    monkeypatch.setenv("LLM_BACKEND", "openai_compatible")

    generator = build_default_generator({})

    assert isinstance(generator, _StubLlamaCppClient)


def test_build_default_generator_selects_mistral_variant() -> None:
    generator = build_default_generator({"LLM_MODEL_VARIANT": "mistral"})

    assert isinstance(generator, _StubLlamaCppClient)
    assert generator.config.model_path == MODEL_VARIANT_PATHS["mistral"]


def test_explicit_model_path_overrides_variant_selection() -> None:
    generator = build_default_generator(
        {"LLM_MODEL_VARIANT": "mistral", "MODEL_PATH": "models/custom.gguf"}
    )

    assert generator.config.model_path == "models/custom.gguf"


def test_unknown_model_variant_raises() -> None:
    with pytest.raises(ValueError, match="gpt4"):
        build_default_generator({"LLM_MODEL_VARIANT": "gpt4"})


def test_generation_config_carries_variant_model_path() -> None:
    config = GenerationConfig(model_path=MODEL_VARIANT_PATHS["mistral"])

    assert config.model_path == "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"


def test_openai_compatible_backend_defaults_to_empty_extra_params() -> None:
    generator = build_default_generator(
        {"LLM_BACKEND": "openai_compatible", "LLM_BASE_URL": "https://api.groq.com/openai/v1"}
    )

    assert isinstance(generator, _StubOpenAICompatibleClient)
    assert generator.config.extra_params == {}


def test_openai_compatible_backend_defaults_max_retries_to_three() -> None:
    generator = build_default_generator({"LLM_BACKEND": "openai_compatible"})

    assert generator.config.max_retries == 3


def test_openai_compatible_backend_reads_llm_max_retries_env_var() -> None:
    generator = build_default_generator({"LLM_BACKEND": "openai_compatible", "LLM_MAX_RETRIES": "0"})

    assert generator.config.max_retries == 0


def test_openai_compatible_backend_parses_extra_params_json() -> None:
    generator = build_default_generator(
        {
            "LLM_BACKEND": "openai_compatible",
            "LLM_EXTRA_PARAMS": '{"reasoning_effort": "none"}',
        }
    )

    assert generator.config.extra_params == {"reasoning_effort": "none"}


def test_invalid_extra_params_json_raises() -> None:
    with pytest.raises(ValueError, match="LLM_EXTRA_PARAMS"):
        build_default_generator(
            {"LLM_BACKEND": "openai_compatible", "LLM_EXTRA_PARAMS": "not-json"}
        )


def test_non_object_extra_params_raises() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        build_default_generator(
            {"LLM_BACKEND": "openai_compatible", "LLM_EXTRA_PARAMS": "[1, 2, 3]"}
        )


def test_list_configured_profiles_parses_comma_separated_names() -> None:
    assert list_configured_profiles({"LLM_PROFILES": "qwen, gptoss20b ,llama70b"}) == [
        "qwen",
        "gptoss20b",
        "llama70b",
    ]


def test_list_configured_profiles_empty_when_unset() -> None:
    assert list_configured_profiles({}) == []


def test_build_generator_for_profile_inherits_base_config() -> None:
    """A profile only overrides what it specifies; everything else comes from the shared base LLM_* config (e.g. base_url, api_key)."""

    env = {
        "LLM_BACKEND": "openai_compatible",
        "LLM_BASE_URL": "https://api.groq.com/openai/v1",
        "LLM_API_KEY": "shared-key",
        "LLM_EXTRA_PARAMS": '{"reasoning_effort": "none"}',
        "LLM_PROFILE_QWEN_MODEL": "qwen/qwen3.6-27b",
    }

    generator = build_generator_for_profile("qwen", env)

    assert isinstance(generator, _StubOpenAICompatibleClient)
    assert generator.config.base_url == "https://api.groq.com/openai/v1"
    assert generator.config.api_key == "shared-key"
    assert generator.config.model == "qwen/qwen3.6-27b"
    assert generator.config.extra_params == {"reasoning_effort": "none"}


def test_build_generator_for_profile_can_override_extra_params() -> None:
    """A profile must be able to fully replace inherited extra_params -- e.g. GPT-OSS models on Groq reject reasoning_effort='none' (only low/medium/high), unlike Qwen3, so a mixed-model comparison needs per-profile overrides, not just per-profile model names."""

    env = {
        "LLM_BACKEND": "openai_compatible",
        "LLM_BASE_URL": "https://api.groq.com/openai/v1",
        "LLM_EXTRA_PARAMS": '{"reasoning_effort": "none"}',
        "LLM_PROFILE_GPTOSS_MODEL": "openai/gpt-oss-20b",
        "LLM_PROFILE_GPTOSS_EXTRA_PARAMS": "{}",
    }

    generator = build_generator_for_profile("gptoss", env)

    assert generator.config.model == "openai/gpt-oss-20b"
    assert generator.config.extra_params == {}


def test_build_generator_for_profile_can_select_a_different_backend() -> None:
    env = {
        "LLM_BACKEND": "openai_compatible",  # base config is Groq...
        "LLM_PROFILE_LOCALLLAMA_BACKEND": "llama_cpp",  # ...but this profile is local
        "LLM_PROFILE_LOCALLLAMA_MODEL_PATH": "models/custom.gguf",
    }

    generator = build_generator_for_profile("localllama", env)

    assert isinstance(generator, _StubLlamaCppClient)
    assert generator.config.model_path == "models/custom.gguf"
