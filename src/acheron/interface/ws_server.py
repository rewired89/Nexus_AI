"""WebSocket server bridging voice I/O to Acheron's RAG pipeline.

Runs as a FastAPI application.  The kiosk frontend connects over a
single WebSocket and exchanges JSON + binary audio frames.

Message protocol (client → server)
-----------------------------------
Text frames are JSON::

    {"type": "audio_start"}          # begin recording
    {"type": "audio_end"}            # stop recording → trigger STT
    {"type": "text", "query": "..."}  # typed query (bypass STT)
    {"type": "mode", "mode": "discover"}  # switch query mode
    {"type": "ping"}                  # keep-alive

Binary frames are raw 16-bit PCM audio chunks (16 kHz, mono).

Message protocol (server → client)
-----------------------------------
Text frames are JSON::

    {"type": "status", "message": "..."}
    {"type": "transcription", "text": "..."}
    {"type": "response", "answer": "...", "sources": [...], ...}
    {"type": "avatar", ...animation params...}
    {"type": "error", "message": "..."}

Binary frames are WAV audio of the TTS response.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from acheron.interface.avatar.renderer import (
    AvatarController,
    AvatarState,
    LipSyncTimeline,
)
from acheron.interface.nexus import SessionMemory
from acheron.interface.voice.stt import WhisperSTT
from acheron.interface.voice.tts import (
    LipSyncFrame,
    PiperTTS,
    ElevenLabsTTS,
    extract_lip_sync,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_KIOSK_DIR = Path(__file__).parent / "kiosk"


def create_interface_app(
    stt_model: str = "base",
    stt_device: str = "auto",
    piper_model: str = "",
    elevenlabs_key: str = "",
    elevenlabs_voice: str = "21m00Tcm4TlvDq8ikWAM",
    session_path: str = "data/nexus_session.json",
) -> FastAPI:
    """Build the Nexus interface FastAPI application.

    This app serves the kiosk frontend and exposes a WebSocket at
    ``/ws`` for the full voice ↔ pipeline bridge.
    """
    app = FastAPI(title="Nexus Interface", version="0.1.0")

    # ---- shared state (created once, shared across connections) ----
    stt = WhisperSTT(model_size=stt_model, device=stt_device)
    memory = SessionMemory(path=session_path)
    avatar = AvatarController()

    # TTS: prefer Piper, fall back to ElevenLabs, allow none.
    tts_engine: Optional[object] = None
    if piper_model:
        piper = PiperTTS(model_path=piper_model)
        if piper.available:
            tts_engine = piper
    if tts_engine is None and elevenlabs_key:
        tts_engine = ElevenLabsTTS(
            api_key=elevenlabs_key, voice_id=elevenlabs_voice
        )

    # Lazy-loaded pipeline to avoid import cost at startup.
    _pipeline_cache: dict[str, object] = {}

    def get_pipeline():  # type: ignore[return]
        if "p" not in _pipeline_cache:
            from acheron.rag.pipeline import RAGPipeline
            _pipeline_cache["p"] = RAGPipeline()
            logger.info("RAG pipeline initialized")
        return _pipeline_cache["p"]

    # ---- kiosk static files ----
    if _KIOSK_DIR.is_dir():
        app.mount(
            "/kiosk",
            StaticFiles(directory=str(_KIOSK_DIR)),
            name="kiosk",
        )

    @app.get("/")
    async def index():
        """Serve the kiosk frontend."""
        index_path = _KIOSK_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
        return HTMLResponse("<h1>Nexus Interface</h1><p>Kiosk frontend not found.</p>")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "stt_available": stt.available,
            "tts_available": tts_engine is not None,
            "session_turns": memory.turn_count,
        }

    # ---- WebSocket handler ----

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        audio_buffer = bytearray()
        query_mode = "analyze"  # default: hypothesis engine
        explicit_mode: Optional[str] = None

        async def send_json(payload: dict) -> None:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                pass

        async def send_avatar_state() -> None:
            params = avatar.current_params()
            await send_json({"type": "avatar", **params.to_dict()})

        try:
            await send_json({
                "type": "status",
                "message": "Nexus interface connected.",
                "stt_available": stt.available,
                "tts_available": tts_engine is not None,
            })
            await send_avatar_state()

            while True:
                msg = await ws.receive()

                # -- binary: audio chunk --
                if "bytes" in msg and msg["bytes"]:
                    audio_buffer.extend(msg["bytes"])
                    continue

                # -- text: JSON command --
                if "text" not in msg or not msg["text"]:
                    continue

                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await send_json({"type": "pong"})
                    continue

                if msg_type == "mode":
                    query_mode = data.get("mode", "analyze")
                    explicit_mode = data.get("explicit_mode")
                    await send_json({
                        "type": "status",
                        "message": f"Mode set to {query_mode}",
                    })
                    continue

                if msg_type == "audio_start":
                    audio_buffer.clear()
                    avatar.transition(AvatarState.LISTENING)
                    await send_avatar_state()
                    continue

                if msg_type == "audio_end":
                    # Transcribe collected audio.
                    avatar.transition(AvatarState.THINKING)
                    await send_avatar_state()

                    pcm_data = bytes(audio_buffer)
                    audio_buffer.clear()

                    if not stt.available:
                        await send_json({
                            "type": "error",
                            "message": "STT not available — type your query instead.",
                        })
                        avatar.transition(AvatarState.IDLE)
                        await send_avatar_state()
                        continue

                    # Run STT in threadpool to avoid blocking.
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(
                        None, stt.transcribe, pcm_data
                    )

                    if not text.strip():
                        await send_json({
                            "type": "status",
                            "message": "No speech detected.",
                        })
                        avatar.transition(AvatarState.IDLE)
                        await send_avatar_state()
                        continue

                    await send_json({"type": "transcription", "text": text})
                    # Fall through to process the query.
                    await _process_query(
                        ws, text, query_mode, explicit_mode,
                        get_pipeline, memory, avatar, tts_engine,
                        send_json, send_avatar_state,
                    )
                    continue

                if msg_type == "text":
                    query = data.get("query", "").strip()
                    if not query:
                        continue
                    avatar.transition(AvatarState.THINKING)
                    await send_avatar_state()
                    await _process_query(
                        ws, query, query_mode, explicit_mode,
                        get_pipeline, memory, avatar, tts_engine,
                        send_json, send_avatar_state,
                    )
                    continue

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("WebSocket error")

    return app


# ---------------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------------

async def _process_query(
    ws: WebSocket,
    query: str,
    query_mode: str,
    explicit_mode: Optional[str],
    get_pipeline,  # type: ignore[type-arg]
    memory: SessionMemory,
    avatar: AvatarController,
    tts_engine: Optional[object],
    send_json,  # type: ignore[type-arg]
    send_avatar_state,  # type: ignore[type-arg]
) -> None:
    """Run a query through the pipeline and send results back."""
    loop = asyncio.get_event_loop()

    try:
        pipeline = get_pipeline()

        # Enrich with session context.
        enriched = memory.enrich_query(query)

        await send_json({"type": "status", "message": "Querying Acheron..."})

        # Run the pipeline in a threadpool (it's synchronous).
        if query_mode == "discover":
            result = await loop.run_in_executor(
                None, lambda: pipeline.discover(enriched)
            )
            answer = _format_discovery(result)
            mode_used = getattr(result, "detected_mode", "discovery")
            sources = [
                _source_to_dict(s) for s in getattr(result, "sources", [])
            ]
        elif query_mode == "query":
            result = await loop.run_in_executor(
                None, lambda: pipeline.query(enriched)
            )
            answer = result.answer
            mode_used = "evidence"
            sources = [_source_to_dict(s) for s in result.sources]
        else:
            # Default: analyze (hypothesis engine).
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.analyze(enriched, mode=explicit_mode),
            )
            answer = result.raw_output
            mode_used = result.mode.value if hasattr(result.mode, "value") else str(result.mode)
            sources = [
                _source_to_dict(s) for s in getattr(result, "sources", [])
            ]

        # Record in session memory.
        topics = _extract_topics(query)
        memory.record_turn(
            query=query,
            mode=mode_used,
            response_summary=answer[:300],
            topics=topics,
        )

        # Send response.
        avatar.transition(AvatarState.SPEAKING)
        await send_avatar_state()

        await send_json({
            "type": "response",
            "answer": answer,
            "sources": sources,
            "mode": mode_used,
            "session_turns": memory.turn_count,
        })

        # TTS if available.
        if tts_engine is not None:
            try:
                # Truncate for TTS — long answers would take forever.
                tts_text = _truncate_for_speech(answer, max_chars=1500)
                wav = await loop.run_in_executor(
                    None, tts_engine.synthesize, tts_text  # type: ignore[union-attr]
                )

                # Extract lip-sync data and send it first.
                lip_frames = extract_lip_sync(wav)
                timeline = LipSyncTimeline(
                    frames=[f.amplitude for f in lip_frames],
                    frame_duration_ms=50.0,
                )
                avatar.set_lip_sync(timeline)
                await send_avatar_state()

                # Send audio as binary frame.
                await ws.send_bytes(wav)

            except Exception:
                logger.exception("TTS synthesis failed")

        # Return to idle after a delay (lip-sync will handle transition
        # if TTS is active; otherwise go idle immediately).
        if tts_engine is None:
            avatar.transition(AvatarState.IDLE)
            await send_avatar_state()

    except Exception as exc:
        logger.exception("Pipeline query failed")
        avatar.transition(AvatarState.ERROR)
        await send_avatar_state()
        await send_json({
            "type": "error",
            "message": f"Query failed: {exc}",
        })
        avatar.transition(AvatarState.IDLE)
        await send_avatar_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_to_dict(src: object) -> dict:
    """Convert a QueryResult to a JSON-safe dict."""
    return {
        "title": getattr(src, "paper_title", ""),
        "authors": getattr(src, "authors", []),
        "doi": getattr(src, "doi", ""),
        "section": getattr(src, "section", ""),
        "score": round(getattr(src, "relevance_score", 0.0), 3),
        "excerpt": getattr(src, "excerpt", ""),
    }


def _format_discovery(result: object) -> str:
    """Format a DiscoveryResult into readable text."""
    parts = []
    evidence = getattr(result, "evidence", [])
    if evidence:
        parts.append("## Evidence\n" + "\n".join(f"- {e}" for e in evidence))
    inference = getattr(result, "inference", [])
    if inference:
        parts.append("## Inference\n" + "\n".join(f"- {i}" for i in inference))
    speculation = getattr(result, "speculation", [])
    if speculation:
        parts.append("## Speculation\n" + "\n".join(f"- {s}" for s in speculation))
    hypotheses = getattr(result, "hypotheses", [])
    if hypotheses:
        parts.append("## Hypotheses\n" + "\n".join(
            f"- {getattr(h, 'statement', str(h))}" for h in hypotheses
        ))
    schematic = getattr(result, "bioelectric_schematic", "")
    if schematic:
        parts.append(f"## Bioelectric Schematic\n{schematic}")
    raw = getattr(result, "raw_output", "")
    if raw and not parts:
        return raw
    return "\n\n".join(parts) if parts else "No results."


def _truncate_for_speech(text: str, max_chars: int = 1500) -> str:
    """Truncate text for TTS, breaking at sentence boundaries."""
    if len(text) <= max_chars:
        return text
    # Find the last sentence boundary before the limit.
    truncated = text[:max_chars]
    for sep in [". ", ".\n", "! ", "? "]:
        idx = truncated.rfind(sep)
        if idx > max_chars // 2:
            return truncated[: idx + 1]
    return truncated


def _extract_topics(query: str) -> list[str]:
    """Extract rough topic keywords from a query string."""
    # Simple keyword extraction — not NLP, just the most relevant terms.
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "of", "in", "to", "for", "with", "on", "at", "from", "by",
        "about", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "and", "but", "or",
        "if", "what", "which", "who", "whom", "this", "that", "these",
        "those", "i", "me", "my", "we", "our", "you", "your", "it",
        "its", "they", "them", "their",
    }
    words = query.lower().split()
    topics = []
    for w in words:
        clean = w.strip("?,!.;:'\"()[]{}").lower()
        if len(clean) > 2 and clean not in stopwords:
            topics.append(clean)
    return topics[:8]
