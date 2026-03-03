"""WebSocket server bridging voice I/O to the Nexus RAG pipeline.

Runs as a FastAPI application.  The kiosk frontend connects over a
single WebSocket and exchanges JSON + binary audio frames.

Designed for continuous, conversational interaction — the user speaks
naturally, the system detects speech segments via client-side VAD,
transcribes, responds, and immediately resumes listening.  Saying
"stop" or "Nexus stop" interrupts the current response.

Message protocol (client → server)
-----------------------------------
Text frames are JSON::

    {"type": "speech_segment"}       # next binary frame is a complete WAV segment
    {"type": "text", "query": "..."}  # typed query (bypass STT)
    {"type": "mode", "mode": "discover"}  # switch query mode
    {"type": "voice", "profile": "male"|"female"}  # switch voice
    {"type": "interrupt"}            # user spoke during playback — cancel TTS
    {"type": "ping"}                  # keep-alive

Binary frames: complete WAV audio of one speech segment (from client VAD).

Message protocol (server → client)
-----------------------------------
Text frames are JSON::

    {"type": "status", "message": "..."}
    {"type": "listening"}             # server ready for next utterance
    {"type": "transcription", "text": "..."}
    {"type": "response_chunk", "text": "..."}  # streaming LLM text fragment
    {"type": "response", "answer": "...", "sources": [...], ...}  # final complete response
    {"type": "interrupted"}           # response was cancelled
    {"type": "avatar", ...animation params...}
    {"type": "error", "message": "..."}

Binary frames are WAV audio of the TTS response (sent in sentence-sized
chunks for progressive playback).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware import Middleware

from acheron.interface.avatar.renderer import (
    AvatarController,
    AvatarState,
    LipSyncTimeline,
)
from acheron.interface.emotion.detector import EmotionDetector
from acheron.interface.nexus import SessionMemory
from acheron.interface.voice.stt import WhisperSTT
from acheron.interface.voice.tts import (
    DEFAULT_VOICE,
    EdgeTTS,
    ElevenLabsTTS,
    LipSyncFrame,
    PiperTTS,
    VOICE_PROFILES,
    extract_lip_sync,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interrupt detection
# ---------------------------------------------------------------------------

_STOP_PATTERNS = re.compile(
    r"\b(stop|nexus stop|shut up|be quiet|quiet|enough|cancel|"
    r"hold on|wait|pause|never\s?mind|nevermind)\b",
    re.IGNORECASE,
)


def is_interrupt_command(text: str) -> bool:
    """Check if transcribed text is a stop/interrupt command."""
    cleaned = text.strip().lower()
    # Short utterances that are purely stop commands.
    if len(cleaned.split()) <= 4 and _STOP_PATTERNS.search(cleaned):
        return True
    return False



# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_KIOSK_DIR = Path(__file__).parent / "kiosk"


def create_interface_app(
    stt_model: str = "",
    stt_device: str = "",
    piper_model: str = "",
    elevenlabs_key: str = "",
    elevenlabs_voice: str = "",
    session_path: str = "",
) -> FastAPI:
    """Build the Nexus interface FastAPI application.

    This app serves the kiosk frontend and exposes a WebSocket at
    ``/ws`` for the full voice <-> pipeline bridge.

    All parameters auto-read from .env / config if not explicitly provided.
    """
    # Auto-read from .env / config for any unset parameter.
    from acheron.config import get_settings
    settings = get_settings()
    stt_model = stt_model or settings.whisper_model
    stt_device = stt_device or settings.whisper_device
    piper_model = piper_model or settings.piper_model_path
    elevenlabs_key = elevenlabs_key or settings.elevenlabs_api_key
    elevenlabs_voice = elevenlabs_voice or settings.elevenlabs_voice_id
    session_path = session_path or settings.nexus_session_path

    app = FastAPI(title="Nexus Interface", version="0.2.0")

    # ---- shared state (created once, shared across connections) ----
    stt = WhisperSTT(model_size=stt_model, device=stt_device)
    logger.info(
        "Nexus STT status: backend=%s, available=%s",
        stt._backend, stt.available,
    )
    if not stt.available:
        logger.info(
            "Server-side Whisper STT not installed — this is fine. "
            "The browser handles speech recognition via the Web Speech API. "
            "Use Chrome or Edge for best support."
        )
    memory = SessionMemory(path=session_path)
    avatar = AvatarController()
    emotion_detector = EmotionDetector()

    # Dedicated thread pool for emotion processing so it never starves
    # the main query executor (which uses the default thread pool).
    _emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")

    # TTS: build engines for both voice profiles.
    # Priority: Piper (local) → ElevenLabs (cloud, API key) → Edge TTS (free).
    tts_engines: dict[str, object] = {}
    for profile_id, profile in VOICE_PROFILES.items():
        if piper_model:
            engine = PiperTTS(model_path=piper_model)
            if engine.available:
                tts_engines[profile_id] = engine
                continue
        if elevenlabs_key:
            # Always use the profile-specific voice, not the global override,
            # so male/female actually sound different.
            vid = profile.elevenlabs_voice_id
            tts_engines[profile_id] = ElevenLabsTTS(
                api_key=elevenlabs_key, voice_id=vid
            )
            continue
        # Free fallback: Edge TTS (no API key needed).
        edge = EdgeTTS(voice=profile.edge_tts_voice or "en-US-AriaNeural")
        if edge.available:
            tts_engines[profile_id] = edge

    tts_engine_type = "none"
    if tts_engines:
        sample = next(iter(tts_engines.values()))
        tts_engine_type = type(sample).__name__
        logger.info("TTS enabled (%s): %s", tts_engine_type, ", ".join(tts_engines.keys()))
    else:
        logger.warning(
            "No TTS engine available. Install edge-tts (pip install edge-tts) "
            "or set ELEVENLABS_API_KEY."
        )

    # Lazy-loaded pipeline to avoid import cost at startup.
    _pipeline_cache: dict[str, object] = {}

    def get_pipeline():  # type: ignore[return]
        if "p" not in _pipeline_cache:
            from acheron.rag.pipeline import RAGPipeline
            _pipeline_cache["p"] = RAGPipeline()
            logger.info("RAG pipeline initialized")
        return _pipeline_cache["p"]

    # ---- no-cache middleware for kiosk static assets ----
    @app.middleware("http")
    async def no_cache_kiosk_assets(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/kiosk/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    # ---- kiosk static files ----
    if _KIOSK_DIR.is_dir():
        app.mount(
            "/kiosk",
            StaticFiles(directory=str(_KIOSK_DIR)),
            name="kiosk",
        )

    # Cache-bust token — changes every server restart so browsers
    # never serve stale app.js / style.css after a code update.
    _cache_bust = str(int(time.time()))

    @app.get("/")
    async def index():
        """Serve the kiosk frontend with cache-busting asset URLs."""
        index_path = _KIOSK_DIR / "index.html"
        if index_path.exists():
            html = index_path.read_text()
            # Replace static version tags with a per-restart timestamp
            html = html.replace("app.js?v=3", f"app.js?v={_cache_bust}")
            html = html.replace("style.css?v=3", f"style.css?v={_cache_bust}")
            return HTMLResponse(
                content=html,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return HTMLResponse("<h1>Nexus Interface</h1><p>Kiosk frontend not found.</p>")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "stt_available": stt.available,
            "stt_backend": stt._backend,
            "tts_available": len(tts_engines) > 0,
            "voices": list(tts_engines.keys()),
            "session_turns": memory.turn_count,
        }

    # ---- WebSocket handler ----

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()

        # Per-connection state.
        query_mode = "analyze"
        explicit_mode: Optional[str] = None
        voice_profile = DEFAULT_VOICE
        interrupted = asyncio.Event()
        processing = asyncio.Event()  # set while a query is in flight
        expect_audio = False  # True after receiving "speech_segment" header

        async def send_json(payload: dict) -> None:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                pass

        async def send_avatar_state() -> None:
            params = avatar.current_params()
            await send_json({"type": "avatar", **params.to_dict()})

        async def signal_listening() -> None:
            avatar.transition(AvatarState.IDLE)
            await send_avatar_state()
            await send_json({"type": "listening"})

        try:
            profiles_info = [
                {"id": p.id, "label": p.label, "description": p.description}
                for p in VOICE_PROFILES.values()
            ]
            await send_json({
                "type": "status",
                "message": "Nexus online.",
                "stt_available": stt.available,
                "tts_available": len(tts_engines) > 0,
                "voices": profiles_info,
                "active_voice": voice_profile,
            })

            # Warn user about missing configuration at connect time.
            from acheron.config import get_settings as _get_settings
            _cfg = _get_settings()
            _warnings = []
            if not _cfg.compute_available:
                _warnings.append(
                    "No LLM API key configured. Restart 'nexus interface' "
                    "and it will prompt you, or run: python setup_keys.py"
                )
            if not _cfg.vectorstore_dir.exists() or not any(
                _cfg.vectorstore_dir.iterdir()
            ) if _cfg.vectorstore_dir.exists() else True:
                _warnings.append(
                    "Knowledge base is empty. Run 'acheron collect' or "
                    "'acheron add <pdf>' to ingest papers."
                )
            if not tts_engines:
                _warnings.append(
                    "No voice engine available. Run: pip install edge-tts"
                )
            if _warnings:
                await send_json({
                    "type": "response",
                    "answer": "**Setup needed:**\n\n- " + "\n- ".join(_warnings),
                    "sources": [],
                    "mode": "setup",
                    "session_turns": 0,
                })
            await signal_listening()

            while True:
                msg = await ws.receive()

                # -- binary: complete speech segment WAV --
                if "bytes" in msg and msg["bytes"]:
                    if not expect_audio:
                        continue
                    expect_audio = False

                    wav_data = bytes(msg["bytes"])

                    # If we're currently processing/speaking, this is an
                    # interrupt — the user started talking over Nexus.
                    if processing.is_set():
                        interrupted.set()
                        await send_json({"type": "interrupted"})
                        await signal_listening()
                        # Still transcribe to see if it's a real query.

                    avatar.transition(AvatarState.THINKING)
                    await send_avatar_state()

                    if not stt.available:
                        await send_json({
                            "type": "error",
                            "message": (
                                "Your browser is sending audio to the server, but "
                                "server-side Whisper is not installed. This means "
                                "you're running a cached version of the interface. "
                                "Please hard-refresh (Ctrl+Shift+R / Cmd+Shift+R) "
                                "to load the current version, which uses your "
                                "browser's built-in speech recognition instead."
                            ),
                        })
                        await signal_listening()
                        continue

                    # Transcribe in threadpool.
                    loop = asyncio.get_event_loop()
                    text = await loop.run_in_executor(
                        None, stt.transcribe, wav_data
                    )

                    if not text.strip():
                        await signal_listening()
                        continue

                    await send_json({"type": "transcription", "text": text})

                    # Check for stop commands.
                    if is_interrupt_command(text):
                        interrupted.set()
                        await send_json({
                            "type": "status",
                            "message": "Understood — stopped.",
                        })
                        await signal_listening()
                        continue

                    # Also run voice emotion analysis on the audio.
                    if emotion_detector.voice_available:
                        asyncio.ensure_future(
                            _process_emotion_audio(
                                emotion_detector, wav_data,
                                send_json, _emotion_executor,
                            )
                        )

                    # Process as a real query.
                    interrupted.clear()
                    processing.set()
                    try:
                        tts_engine = tts_engines.get(voice_profile)
                        await _process_query(
                            ws, text, query_mode, explicit_mode,
                            get_pipeline, memory, avatar, tts_engine,
                            send_json, send_avatar_state, interrupted,
                            emotion_detector,
                        )
                    finally:
                        processing.clear()
                    await signal_listening()
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
                        "message": f"Mode: {query_mode}",
                    })
                    continue

                if msg_type == "voice":
                    profile_id = data.get("profile", DEFAULT_VOICE)
                    if profile_id in VOICE_PROFILES:
                        voice_profile = profile_id
                        label = VOICE_PROFILES[profile_id].label
                        await send_json({
                            "type": "status",
                            "message": f"Voice: {label}",
                        })
                    continue

                if msg_type == "speech_segment":
                    # Next binary frame will be the WAV data.
                    expect_audio = True
                    continue

                if msg_type == "interrupt":
                    interrupted.set()
                    if processing.is_set():
                        await send_json({"type": "interrupted"})
                    avatar.transition(AvatarState.IDLE)
                    await send_avatar_state()
                    continue

                if msg_type == "video_frame":
                    # Camera frame for emotion detection — process in
                    # background so we don't block the message loop.
                    # Throttle: _process_emotion_frame skips if already busy.
                    frame_data = data.get("data", "")
                    if frame_data:
                        asyncio.ensure_future(
                            _process_emotion_frame(
                                emotion_detector, frame_data,
                                send_json, _emotion_executor,
                            )
                        )
                    continue

                if msg_type == "text":
                    query = data.get("query", "").strip()
                    if not query:
                        continue

                    # Accept optional inline mode (avoids two-message race).
                    inline_mode = data.get("mode")
                    if inline_mode and inline_mode in (
                        "analyze", "discover", "query",
                    ):
                        query_mode = inline_mode
                        explicit_mode = data.get("explicit_mode")

                    # Check for stop commands from text input too.
                    if is_interrupt_command(query):
                        interrupted.set()
                        await send_json({
                            "type": "status",
                            "message": "Understood — stopped.",
                        })
                        await signal_listening()
                        continue

                    avatar.transition(AvatarState.THINKING)
                    await send_avatar_state()

                    interrupted.clear()
                    processing.set()
                    try:
                        tts_engine = tts_engines.get(voice_profile)
                        await _process_query(
                            ws, query, query_mode, explicit_mode,
                            get_pipeline, memory, avatar, tts_engine,
                            send_json, send_avatar_state, interrupted,
                            emotion_detector,
                        )
                    finally:
                        processing.clear()
                    await signal_listening()
                    continue

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("WebSocket error")

    return app


# ---------------------------------------------------------------------------
# Query processing  (streaming)
# ---------------------------------------------------------------------------

# Sentence-boundary pattern used to split TTS chunks at natural pauses.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=\.)\n")

# Maximum characters to speak aloud (prevents 5-minute monologues).
_MAX_SPEECH_CHARS = 1500


def _iter_stream_in_thread(generator):
    """Consume a blocking generator and return items as a list.

    Runs inside ``run_in_executor`` so the async loop is never blocked.
    Separates string text chunks from the final metadata object that
    the pipeline yields last.
    """
    chunks: list[str] = []
    metadata = None
    for item in generator:
        if isinstance(item, str):
            chunks.append(item)
        else:
            metadata = item
    return chunks, metadata


async def _process_emotion_frame(
    detector: EmotionDetector,
    base64_data: str,
    send_json,  # type: ignore[type-arg]
    executor: ThreadPoolExecutor,
) -> None:
    """Decode a base64 JPEG frame and run facial emotion analysis.

    Uses a dedicated single-thread executor so it can never starve the
    default pool (which handles LLM queries).  A module-level throttle
    flag prevents frames from piling up.
    """
    # Import nonlocal throttle flag from the enclosing create_interface_app.
    # We don't have direct access, so we use the detector as a flag carrier.
    if getattr(detector, "_frame_in_flight", False):
        return
    detector._frame_in_flight = True  # type: ignore[attr-defined]
    try:
        import base64
        frame_bytes = base64.b64decode(base64_data)

        loop = asyncio.get_event_loop()
        reading = await loop.run_in_executor(
            executor, detector.process_video_frame, frame_bytes,
        )
        if reading:
            state = detector.current_state
            await send_json({
                "type": "emotion",
                "state": state.to_dict(),
            })
    except Exception:
        logger.debug("Emotion frame processing failed", exc_info=True)
    finally:
        detector._frame_in_flight = False  # type: ignore[attr-defined]


async def _process_emotion_audio(
    detector: EmotionDetector,
    audio_bytes: bytes,
    send_json,  # type: ignore[type-arg]
    executor: ThreadPoolExecutor,
) -> None:
    """Run voice emotion analysis on a WAV audio segment."""
    try:
        loop = asyncio.get_event_loop()
        reading = await loop.run_in_executor(
            executor, detector.process_audio, audio_bytes,
        )
        if reading:
            state = detector.current_state
            await send_json({
                "type": "emotion",
                "state": state.to_dict(),
            })
    except Exception:
        logger.debug("Emotion audio processing failed", exc_info=True)


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
    interrupted: asyncio.Event,
    emotion_detector: Optional[EmotionDetector] = None,
) -> None:
    """Run a query through the pipeline, streaming results back.

    Text chunks are sent to the client as they arrive from the LLM so
    the user sees the response being "typed".  TTS is synthesised in
    sentence-sized chunks and audio frames are sent progressively —
    Nexus starts speaking within ~1-2 s of the first sentence.

    If an :class:`EmotionDetector` is provided and has readings, the
    current emotional state is injected into the pipeline so the LLM
    can adapt its tone and style to the user's mood.
    """
    loop = asyncio.get_event_loop()

    try:
        pipeline = get_pipeline()

        # Build multi-turn messages for the LLM.
        memory.record_message("user", query)
        messages = memory.get_messages_for_llm(query)

        await send_json({"type": "status", "message": "Thinking..."})

        if interrupted.is_set():
            return

        # -----------------------------------------------------------
        # Emotional context — inject user mood into the pipeline.
        # -----------------------------------------------------------
        emotional_context = ""
        if emotion_detector and emotion_detector.available:
            emotional_context = (
                emotion_detector.current_state.to_prompt_context()
            )

        # -----------------------------------------------------------
        # Kick off the streaming pipeline in a thread.
        # -----------------------------------------------------------
        _QUERY_TIMEOUT = 90  # seconds

        try:
            if query_mode == "discover":
                # discover() doesn't have a streaming variant yet — use
                # the batch path and fall back to non-streaming.
                await send_json({"type": "response_chunk", "text": ""})
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: pipeline.discover(query)
                    ),
                    timeout=_QUERY_TIMEOUT,
                )
                text_chunks = [_format_discovery(result)]
                metadata = result
                mode_used = getattr(result, "detected_mode", "discovery")
                sources = [
                    _source_to_dict(s) for s in getattr(result, "sources", [])
                ]
            elif query_mode == "query":
                gen = pipeline.query_stream(query, messages=messages)
                text_chunks, metadata = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, _iter_stream_in_thread, gen
                    ),
                    timeout=_QUERY_TIMEOUT,
                )
                mode_used = "evidence"
                sources = [
                    _source_to_dict(s)
                    for s in getattr(metadata, "sources", [])
                ] if metadata else []
            else:
                # Use the ReAct agent for intelligent evidence
                # gathering.  Falls back to analyze_stream internally
                # if the agent encounters any errors.
                await send_json({
                    "type": "status",
                    "message": "Gathering evidence...",
                })
                gen = pipeline.agent_stream(
                    query, mode=explicit_mode, messages=messages,
                    emotional_context=emotional_context,
                )
                text_chunks, metadata = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, _iter_stream_in_thread, gen
                    ),
                    timeout=_QUERY_TIMEOUT,
                )
                mode_used = (
                    metadata.mode.value
                    if metadata and hasattr(getattr(metadata, "mode", None), "value")
                    else str(getattr(metadata, "mode", "analyze"))
                )
                sources = [
                    _source_to_dict(s)
                    for s in getattr(metadata, "sources", [])
                ] if metadata else []

        except asyncio.TimeoutError:
            logger.warning("Query timed out after %ds: %s", _QUERY_TIMEOUT, query[:80])
            avatar.transition(AvatarState.ERROR)
            await send_avatar_state()
            await send_json({
                "type": "response",
                "answer": (
                    "The query took too long (over 90 seconds). "
                    "This usually happens when I need to search external databases "
                    "like PubMed for papers. Try switching to Query mode "
                    "(faster, uses only local knowledge) or rephrase your question."
                ),
                "sources": [],
                "mode": "timeout",
                "session_turns": memory.turn_count,
            })
            avatar.transition(AvatarState.IDLE)
            await send_avatar_state()
            return

        if interrupted.is_set():
            return

        answer = "".join(text_chunks)

        # -----------------------------------------------------------
        # Stream text chunks to the client.
        # -----------------------------------------------------------
        avatar.transition(AvatarState.SPEAKING)
        await send_avatar_state()

        for chunk in text_chunks:
            if interrupted.is_set():
                return
            await send_json({"type": "response_chunk", "text": chunk})

        # Final complete response with metadata.
        await send_json({
            "type": "response",
            "answer": answer,
            "sources": sources,
            "mode": mode_used,
            "session_turns": memory.turn_count,
        })

        if interrupted.is_set():
            return

        # Record in session memory (both legacy + multi-turn).
        topics = _extract_topics(query)
        memory.record_turn(
            query=query,
            mode=mode_used,
            response_summary=answer[:300],
            topics=topics,
        )
        memory.record_message("assistant", answer)

        # -----------------------------------------------------------
        # Chunked TTS — synthesise sentence by sentence.
        # -----------------------------------------------------------
        if tts_engine is not None:
            try:
                clean_text = _strip_markdown_for_speech(answer)
                # Split into sentence-sized chunks for progressive TTS.
                sentences = _SENTENCE_END.split(clean_text)
                # Merge very short fragments with the previous sentence.
                merged: list[str] = []
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if merged and len(merged[-1]) < 40:
                        merged[-1] = merged[-1] + " " + s
                    else:
                        merged.append(s)

                spoken_chars = 0
                for sentence in merged:
                    if interrupted.is_set():
                        return
                    if spoken_chars >= _MAX_SPEECH_CHARS:
                        break

                    sentence = sentence[:_MAX_SPEECH_CHARS - spoken_chars]
                    spoken_chars += len(sentence)

                    wav = await loop.run_in_executor(
                        None,
                        tts_engine.synthesize,  # type: ignore[union-attr]
                        sentence,
                    )

                    if interrupted.is_set():
                        return

                    # Lip-sync for this audio chunk.
                    lip_frames = extract_lip_sync(wav)
                    timeline = LipSyncTimeline(
                        frames=[f.amplitude for f in lip_frames],
                        frame_duration_ms=50.0,
                    )
                    avatar.set_lip_sync(timeline)
                    await send_avatar_state()

                    # Send audio chunk — client appends to playback queue.
                    await ws.send_bytes(wav)

                avatar.transition(AvatarState.IDLE)
                await send_avatar_state()
            except Exception:
                logger.exception("TTS synthesis failed")
                avatar.transition(AvatarState.IDLE)
                await send_avatar_state()
        else:
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


def _strip_markdown_for_speech(text: str) -> str:
    """Convert markdown-formatted text into clean, natural prose for TTS.

    Removes headers, bullet points, bold/italic markers, links, code
    fences, and other formatting that a TTS engine would read literally
    (e.g. "hashtag hashtag Evidence" instead of just "Evidence").
    """
    # Remove code fences  ```...```
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code  `...`
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove markdown images  ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Convert links  [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Remove header markers  ## Header → Header
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bold/italic  ***text***, **text**, *text*, __text__, _text_
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove strikethrough  ~~text~~
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Convert bullet lines  "- item" or "* item" → "item."
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Convert numbered lists  "1. item" → "item."
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Remove horizontal rules  --- or ***
    text = re.sub(r"^[\s]*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove blockquote markers  > text → text
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple newlines into a pause-friendly double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


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
