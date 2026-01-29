"""Central configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings, populated from env vars or .env file."""

    # LLM provider selection: "openai" or "anthropic"
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai", alias="ACHERON_LLM_PROVIDER"
    )

    # OpenAI-compatible LLM settings
    llm_api_key: str = Field(default="", alias="ACHERON_LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="ACHERON_LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4o", alias="ACHERON_LLM_MODEL")

    # Anthropic-native LLM settings
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=4096, alias="ANTHROPIC_MAX_TOKENS")

    # Embeddings
    embedding_api_key: str = Field(default="", alias="ACHERON_EMBEDDING_API_KEY")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="ACHERON_EMBEDDING_MODEL")

    # NCBI / PubMed
    ncbi_api_key: str = Field(default="", alias="NCBI_API_KEY")
    ncbi_email: str = Field(default="", alias="NCBI_EMAIL")
    ncbi_tool: str = Field(default="AcheronNexus", alias="NCBI_TOOL")

    # Paths
    data_dir: Path = Field(default=_PROJECT_ROOT / "data", alias="ACHERON_DATA_DIR")
    vectorstore_dir: Path = Field(
        default=_PROJECT_ROOT / "data" / "vectorstore", alias="ACHERON_VECTORSTORE_DIR"
    )

    # Server
    host: str = Field(default="127.0.0.1", alias="ACHERON_HOST")
    port: int = Field(default=8000, alias="ACHERON_PORT")

    # Logging
    log_level: str = Field(default="INFO", alias="ACHERON_LOG_LEVEL")

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    # Derived — active model name regardless of provider
    @property
    def active_model(self) -> str:
        """Return the model name for the currently selected provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.llm_model

    @property
    def compute_available(self) -> bool:
        """True when the selected provider has an API key configured."""
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return bool(self.llm_api_key)

    # Derived paths
    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "papers" / "pdf"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "papers" / "raw"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "papers" / "metadata"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()  # type: ignore[attr-defined]
    return get_settings._instance  # type: ignore[attr-defined]
