"""Tests for configuration."""

from pathlib import Path

from acheron.config import Settings, get_settings


def test_default_settings():
    s = Settings()
    assert s.llm_model == "gpt-4o"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.port == 8000
    assert isinstance(s.data_dir, Path)


def test_derived_paths():
    s = Settings()
    assert s.pdf_dir == s.data_dir / "papers" / "pdf"
    assert s.raw_dir == s.data_dir / "papers" / "raw"
    assert s.metadata_dir == s.data_dir / "papers" / "metadata"


def test_get_settings_cached():
    # Clear the cache
    if hasattr(get_settings, "_instance"):
        delattr(get_settings, "_instance")
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
