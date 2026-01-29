"""Tests for the dual-provider ComputeClient."""

import pytest

from acheron.config import Settings
from acheron.rag.compute import ComputeClient, ComputeUnavailableError


def _make_settings(**overrides) -> Settings:
    """Build a Settings instance with specific overrides (no .env file)."""
    defaults = {
        "ACHERON_LLM_PROVIDER": "openai",
        "ACHERON_LLM_API_KEY": "",
        "ACHERON_LLM_BASE_URL": "https://api.openai.com/v1",
        "ACHERON_LLM_MODEL": "gpt-4o",
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": "claude-sonnet-4-20250514",
        "ANTHROPIC_MAX_TOKENS": "4096",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ------------------------------------------------------------------
# Provider routing
# ------------------------------------------------------------------
class TestProviderRouting:
    def test_openai_provider_selected(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="openai", ACHERON_LLM_API_KEY="sk-test")
        client = ComputeClient(settings=s)
        assert client.provider == "openai"
        assert client.model == "gpt-4o"
        assert client.available is True

    def test_anthropic_provider_selected(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-test")
        client = ComputeClient(settings=s)
        assert client.provider == "anthropic"
        assert client.model == "claude-sonnet-4-20250514"
        assert client.available is True

    def test_anthropic_custom_model(self):
        s = _make_settings(
            ACHERON_LLM_PROVIDER="anthropic",
            ANTHROPIC_API_KEY="sk-ant-test",
            ANTHROPIC_MODEL="claude-opus-4-20250514",
        )
        client = ComputeClient(settings=s)
        assert client.model == "claude-opus-4-20250514"


# ------------------------------------------------------------------
# Unavailability
# ------------------------------------------------------------------
class TestComputeUnavailable:
    def test_openai_no_key(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="openai", ACHERON_LLM_API_KEY="")
        client = ComputeClient(settings=s)
        assert client.available is False
        with pytest.raises(ComputeUnavailableError, match="no API key"):
            client.generate("system", "user")

    def test_anthropic_no_key(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="")
        client = ComputeClient(settings=s)
        assert client.available is False
        with pytest.raises(ComputeUnavailableError, match="no API key"):
            client.generate("system", "user")


# ------------------------------------------------------------------
# active_model / compute_available on Settings directly
# ------------------------------------------------------------------
class TestSettingsProperties:
    def test_active_model_openai(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="openai")
        assert s.active_model == "gpt-4o"

    def test_active_model_anthropic(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="anthropic", ANTHROPIC_MODEL="claude-sonnet-4-20250514")
        assert s.active_model == "claude-sonnet-4-20250514"

    def test_compute_available_false(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="openai", ACHERON_LLM_API_KEY="")
        assert s.compute_available is False

    def test_compute_available_true(self):
        s = _make_settings(ACHERON_LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-xyz")
        assert s.compute_available is True
