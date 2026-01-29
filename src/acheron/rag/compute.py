"""Dual-provider Compute client (OpenAI / Anthropic).

The Compute layer is Layer 3 of the Nexus architecture.  This module
provides a unified ``ComputeClient`` that dispatches to either the
OpenAI-compatible SDK or the Anthropic Messages API based on the
``ACHERON_LLM_PROVIDER`` setting.

Evidence / retrieval (Layers 1-2) never depend on this module.
"""

from __future__ import annotations

import logging
from typing import Optional

from acheron.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ComputeUnavailableError(RuntimeError):
    """Raised when the Compute layer cannot be reached."""


class ComputeClient:
    """Unified Compute interface over OpenAI and Anthropic backends.

    Usage::

        client = ComputeClient()           # reads provider from Settings
        text = client.generate(system_prompt, user_prompt)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._openai_client: Optional[object] = None
        self._anthropic_client: Optional[object] = None

    @property
    def provider(self) -> str:
        return self.settings.llm_provider

    @property
    def model(self) -> str:
        return self.settings.active_model

    @property
    def available(self) -> bool:
        return self.settings.compute_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        """Send a system+user message pair and return the assistant text.

        Raises ``ComputeUnavailableError`` when the provider is not
        configured or the API call fails.
        """
        if not self.available:
            raise ComputeUnavailableError(
                f"Compute layer unavailable: no API key for provider "
                f"'{self.provider}'.  Set the appropriate env var "
                f"(ACHERON_LLM_API_KEY or ANTHROPIC_API_KEY)."
            )

        if self.provider == "anthropic":
            return self._generate_anthropic(
                system_prompt, user_prompt, max_tokens, temperature
            )
        return self._generate_openai(
            system_prompt, user_prompt, max_tokens, temperature
        )

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------
    def _get_openai_client(self):  # -> openai.OpenAI
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(
                api_key=self.settings.llm_api_key or "not-set",
                base_url=self.settings.llm_base_url,
            )
        return self._openai_client

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("OpenAI Compute call failed")
            raise ComputeUnavailableError(
                f"OpenAI Compute call failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Anthropic backend
    # ------------------------------------------------------------------
    def _get_anthropic_client(self):  # -> anthropic.Anthropic
        if self._anthropic_client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ComputeUnavailableError(
                    "anthropic SDK not installed.  Run: pip install anthropic"
                ) from exc
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key,
            )
        return self._anthropic_client

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        try:
            client = self._get_anthropic_client()
            # Use configured max_tokens ceiling if caller requests less
            effective_max = min(max_tokens, self.settings.anthropic_max_tokens)
            response = client.messages.create(
                model=self.settings.anthropic_model,
                max_tokens=effective_max,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            # Anthropic response: list of content blocks
            parts = []
            for block in response.content:
                if block.type == "text":
                    parts.append(block.text)
            return "\n".join(parts)
        except Exception as exc:
            logger.exception("Anthropic Compute call failed")
            raise ComputeUnavailableError(
                f"Anthropic Compute call failed: {exc}"
            ) from exc
