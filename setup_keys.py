#!/usr/bin/env python3
"""One-time setup: creates .env with your API keys.

Run this once on your machine:
    python setup_keys.py
"""
import os
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"

def main():
    print("=" * 50)
    print("  Nexus AI — Setup")
    print("=" * 50)
    print()

    # Check if .env already exists.
    existing = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                existing[key.strip()] = val.strip()
        print(f"Found existing .env with {len(existing)} key(s).")
        print()

    # Anthropic API key (required).
    current_ant = existing.get("ANTHROPIC_API_KEY", "")
    if current_ant:
        print(f"ANTHROPIC_API_KEY: already set ({current_ant[:15]}...)")
        ant_key = current_ant
    else:
        print("ANTHROPIC_API_KEY (required for AI responses)")
        print("  Get one at: https://console.anthropic.com/settings/keys")
        ant_key = input("  Paste your key: ").strip()
        if not ant_key:
            print("  Skipped — Nexus won't be able to answer questions.")

    # ElevenLabs API key (optional).
    current_el = existing.get("ELEVENLABS_API_KEY", "")
    if current_el:
        print(f"ELEVENLABS_API_KEY: already set ({current_el[:8]}...)")
        el_key = current_el
    else:
        print()
        print("ELEVENLABS_API_KEY (optional — for premium voice)")
        print("  Skip this if you want to use Edge TTS (free, sounds great)")
        el_key = input("  Paste your key (or press Enter to skip): ").strip()

    # Write .env
    lines = [
        "# Nexus AI configuration",
        "# This file is NOT tracked by git — safe for API keys.",
        "",
    ]
    if ant_key:
        lines.append(f"ANTHROPIC_API_KEY={ant_key}")
    if el_key:
        lines.append(f"ELEVENLABS_API_KEY={el_key}")
    lines.append("")

    ENV_PATH.write_text("\n".join(lines))
    print()
    print(f"Saved to {ENV_PATH}")
    print()
    print("You're all set! Run 'nexus' to start the interface.")


if __name__ == "__main__":
    main()
