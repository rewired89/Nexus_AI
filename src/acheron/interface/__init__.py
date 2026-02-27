"""Nexus-Interface — Conversational presence layer for Acheron.

Provides voice I/O (Whisper STT + Piper/ElevenLabs TTS), avatar state
management, WebSocket bridge to the RAG pipeline, and a kiosk-mode
browser frontend.  Modeled after Arthur from *Second Chance* (Fox, 2016).

Data flow::

    Microphone → Whisper STT → query text
      → Nexus session context enrichment
      → Acheron RAGPipeline (retrieve → generate)
      → Response text → Piper/ElevenLabs TTS → audio
      → Avatar lip-sync → Touchscreen display
"""

from acheron.interface.nexus import NexusContext, SessionMemory

__all__ = [
    "NexusContext",
    "SessionMemory",
]
