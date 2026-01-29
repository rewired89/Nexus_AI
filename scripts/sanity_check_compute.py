#!/usr/bin/env python3
"""Acheron Nexus — Compute-layer sanity check.

Verifies that:
 1. Retrieve-only (Layers 1-2) works WITHOUT any Compute provider.
 2. Compute (Layer 3) works when the selected provider key is set.
 3. Compute gracefully reports unavailability when misconfigured.

Usage (any OS):
    python scripts/sanity_check_compute.py

Cross-platform — works on Windows, macOS, Linux.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project src is importable regardless of install status
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _banner(label: str) -> None:
    width = 68
    pad = (width - len(label) - 2) // 2
    print(f"\n{'─' * pad} {label} {'─' * pad}")


def check_config() -> dict:
    """Display and return current provider config."""
    from acheron.config import get_settings

    s = get_settings()
    _banner("CONFIGURATION")
    print(f"  Provider           : {s.llm_provider}")
    print(f"  Active model       : {s.active_model}")
    print(f"  Compute available  : {s.compute_available}")
    if s.llm_provider == "anthropic":
        key_display = s.anthropic_api_key[:8] + "..." if s.anthropic_api_key else "(not set)"
        print(f"  ANTHROPIC_API_KEY  : {key_display}")
        print(f"  ANTHROPIC_MODEL    : {s.anthropic_model}")
        print(f"  ANTHROPIC_MAX_TOKENS: {s.anthropic_max_tokens}")
    else:
        key_display = s.llm_api_key[:8] + "..." if s.llm_api_key else "(not set)"
        print(f"  ACHERON_LLM_API_KEY: {key_display}")
        print(f"  ACHERON_LLM_BASE_URL: {s.llm_base_url}")
        print(f"  ACHERON_LLM_MODEL  : {s.llm_model}")
    return {"provider": s.llm_provider, "available": s.compute_available}


def check_retrieve_only() -> bool:
    """Test 1: retrieve-only must work without Compute."""
    _banner("TEST 1 — Retrieve-Only (no Compute needed)")
    try:
        from acheron.vectorstore.store import VectorStore

        store = VectorStore()
        results = store.search(query="bioelectricity membrane voltage", n_results=3)
        if results:
            print(f"  PASS  Retrieved {len(results)} chunks from index.")
            for i, r in enumerate(results[:3], 1):
                print(f"    [{i}] {r.paper_title[:60]}... (score={r.relevance_score:.3f})")
        else:
            print("  PASS  Index is empty (no papers collected yet) — but retrieval works.")
            print("        Run 'acheron collect' to add papers, then re-test.")
        return True
    except Exception as exc:
        print(f"  FAIL  Retrieval error: {exc}")
        return False


def check_compute_unavailable() -> bool:
    """Test 2: Compute must report 'unavailable' gracefully when misconfigured."""
    _banner("TEST 2 — Compute Unavailable (graceful degradation)")
    try:
        from acheron.rag.compute import ComputeClient, ComputeUnavailableError
        from acheron.config import Settings

        # Build a Settings object with an intentionally blank key
        fake = Settings(
            **{
                "ACHERON_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "",
            }
        )
        client = ComputeClient(settings=fake)
        assert not client.available, "Expected available=False with blank key"

        try:
            client.generate("system", "user")
            print("  FAIL  Expected ComputeUnavailableError, got success.")
            return False
        except ComputeUnavailableError as exc:
            print(f"  PASS  ComputeUnavailableError raised correctly.")
            print(f"        Message: {exc}")
            return True
    except Exception as exc:
        print(f"  FAIL  Unexpected error: {exc}")
        return False


def check_compute_live() -> bool:
    """Test 3: Live Compute call (only runs if key is set)."""
    _banner("TEST 3 — Live Compute Call")
    try:
        from acheron.rag.compute import ComputeClient

        client = ComputeClient()
        if not client.available:
            print(f"  SKIP  No API key for provider '{client.provider}'. Set the key to test.")
            return True  # Not a failure — just skipped

        print(f"  Calling {client.provider} ({client.model}) ...")
        result = client.generate(
            system_prompt="You are a bioelectric research assistant. Be concise.",
            user_prompt="In one sentence, what is Vmem?",
            max_tokens=100,
            temperature=0.0,
        )
        if result and len(result) > 5:
            print(f"  PASS  Response ({len(result)} chars):")
            print(f"        {result[:200]}")
            return True
        else:
            print(f"  FAIL  Empty or near-empty response: {result!r}")
            return False
    except Exception as exc:
        print(f"  FAIL  Compute call raised: {exc}")
        return False


def main() -> None:
    _banner("ACHERON NEXUS — COMPUTE SANITY CHECK")
    config = check_config()

    results = {
        "retrieve_only": check_retrieve_only(),
        "compute_unavailable": check_compute_unavailable(),
        "compute_live": check_compute_live(),
    }

    _banner("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status:4s}  {name}")

    print()
    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks failed — see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
