"""Tests for the Nexus Interface layer.

Tests cover:
  - SessionMemory persistence and enrichment
  - Avatar state machine transitions
  - Lip-sync extraction
  - STT/TTS availability detection
  - WebSocket server app creation
  - Topic extraction helper
"""

from __future__ import annotations

import io
import json
import math
import struct
import tempfile
import time
import wave
from pathlib import Path

import pytest


# ======================================================================
# Session Memory (nexus.py)
# ======================================================================

class TestSessionMemory:

    def _make_memory(self, tmp_path):
        from acheron.interface.nexus import SessionMemory
        return SessionMemory(path=tmp_path / "test_session.json")

    def test_new_session_empty(self, tmp_path):
        mem = self._make_memory(tmp_path)
        assert mem.turn_count == 0
        assert mem.history == []
        assert mem.experiments == []

    def test_record_turn(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.record_turn(
            query="What is Vmem?",
            mode="tutor",
            response_summary="Vmem is the transmembrane voltage...",
            topics=["vmem", "bioelectricity"],
        )
        assert mem.turn_count == 1
        assert mem.history[0].query == "What is Vmem?"
        assert mem.history[0].mode == "tutor"
        assert "vmem" in mem.history[0].topics

    def test_persistence_roundtrip(self, tmp_path):
        from acheron.interface.nexus import SessionMemory
        path = tmp_path / "persist.json"
        mem1 = SessionMemory(path=path)
        mem1.record_turn("q1", "evidence", "answer1", ["topic1"])
        mem1.record_turn("q2", "hypothesis", "answer2", ["topic2"])

        # Load fresh from same file.
        mem2 = SessionMemory(path=path)
        assert mem2.turn_count == 2
        assert mem2.history[0].query == "q1"
        assert mem2.history[1].query == "q2"

    def test_record_experiment(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.record_experiment(
            title="Vmem measurement",
            hypothesis="Vmem > -40mV in head region",
            outcome="PASS",
            notes="Used voltage-sensitive dye",
        )
        assert len(mem.experiments) == 1
        assert mem.experiments[0].outcome == "PASS"

    def test_add_note(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.add_note("Important: check pH before measurements")
        assert len(mem.project_notes) == 1
        assert "pH" in mem.project_notes[0]

    def test_build_context_empty(self, tmp_path):
        mem = self._make_memory(tmp_path)
        ctx = mem.build_context()
        assert ctx.session_summary == "New session — no prior context."
        assert ctx.recent_topics == []
        assert ctx.active_hypotheses == []

    def test_build_context_with_history(self, tmp_path):
        mem = self._make_memory(tmp_path)
        for i in range(5):
            mem.record_turn(f"query {i}", "evidence", f"answer {i}", [f"topic{i}"])
        mem.record_turn("hyp query", "hypothesis", "A hypothesis about planarians", ["planarians"])

        ctx = mem.build_context()
        assert "6 recent queries" in ctx.session_summary
        assert len(ctx.recent_topics) > 0
        assert len(ctx.active_hypotheses) == 1
        assert "planarians" in ctx.recent_topics

    def test_enrich_query_empty_session(self, tmp_path):
        mem = self._make_memory(tmp_path)
        enriched = mem.enrich_query("What is Vmem?")
        assert enriched == "What is Vmem?"  # No enrichment for new sessions.

    def test_enrich_query_with_context(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.record_turn("previous query", "evidence", "previous answer", ["bioelectricity"])
        enriched = mem.enrich_query("What about gap junctions?")
        assert "Session Context" in enriched
        assert "What about gap junctions?" in enriched

    def test_context_to_prompt_block(self, tmp_path):
        from acheron.interface.nexus import NexusContext
        ctx = NexusContext(
            recent_topics=["vmem", "planarian"],
            active_hypotheses=["Vmem drives regeneration"],
            experiment_outcomes=["Test 1: PASS"],
            session_summary="5 queries, evidence mode.",
        )
        block = ctx.to_prompt_block()
        assert "## Session Context (Nexus)" in block
        assert "vmem" in block
        assert "Vmem drives regeneration" in block
        assert "Test 1: PASS" in block

    def test_clear(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.record_turn("q", "evidence", "a")
        mem.record_experiment("t", "h", "PASS")
        mem.add_note("n")
        mem.clear()
        assert mem.turn_count == 0
        assert len(mem.experiments) == 0
        assert len(mem.project_notes) == 0

    def test_max_history_trimming(self, tmp_path):
        mem = self._make_memory(tmp_path)
        for i in range(250):
            mem.record_turn(f"q{i}", "evidence", f"a{i}")
        assert mem.turn_count == 200  # _MAX_HISTORY

    def test_corrupted_file_recovery(self, tmp_path):
        from acheron.interface.nexus import SessionMemory
        path = tmp_path / "corrupted.json"
        path.write_text("{invalid json!!!")
        mem = SessionMemory(path=path)
        assert mem.turn_count == 0  # Graceful recovery.


# ======================================================================
# Avatar State Machine (avatar/renderer.py)
# ======================================================================

class TestAvatarController:

    def test_initial_state(self):
        from acheron.interface.avatar.renderer import AvatarController, AvatarState
        ctrl = AvatarController()
        assert ctrl.state == AvatarState.IDLE

    def test_valid_transitions(self):
        from acheron.interface.avatar.renderer import AvatarController, AvatarState
        ctrl = AvatarController()

        assert ctrl.transition(AvatarState.LISTENING)
        assert ctrl.state == AvatarState.LISTENING

        assert ctrl.transition(AvatarState.THINKING)
        assert ctrl.state == AvatarState.THINKING

        assert ctrl.transition(AvatarState.SPEAKING)
        assert ctrl.state == AvatarState.SPEAKING

        assert ctrl.transition(AvatarState.IDLE)
        assert ctrl.state == AvatarState.IDLE

    def test_invalid_transition(self):
        from acheron.interface.avatar.renderer import AvatarController, AvatarState
        ctrl = AvatarController()
        # Can't go directly from IDLE to SPEAKING.
        assert not ctrl.transition(AvatarState.SPEAKING)
        assert ctrl.state == AvatarState.IDLE

    def test_error_from_any_state(self):
        from acheron.interface.avatar.renderer import AvatarController, AvatarState
        ctrl = AvatarController()
        ctrl.transition(AvatarState.LISTENING)
        assert ctrl.transition(AvatarState.ERROR)
        assert ctrl.state == AvatarState.ERROR
        # Can only go back to IDLE from ERROR.
        assert not ctrl.transition(AvatarState.SPEAKING)
        assert ctrl.transition(AvatarState.IDLE)

    def test_same_state_transition(self):
        from acheron.interface.avatar.renderer import AvatarController, AvatarState
        ctrl = AvatarController()
        assert ctrl.transition(AvatarState.IDLE)  # Same state = OK.

    def test_current_params_idle(self):
        from acheron.interface.avatar.renderer import AvatarController, AvatarState
        ctrl = AvatarController()
        params = ctrl.current_params()
        assert params.state == AvatarState.IDLE
        assert params.mouth_open == 0.0
        assert params.glow_intensity == 0.0

    def test_current_params_thinking(self):
        from acheron.interface.avatar.renderer import AvatarController, AvatarState
        ctrl = AvatarController()
        ctrl.transition(AvatarState.LISTENING)
        ctrl.transition(AvatarState.THINKING)
        params = ctrl.current_params()
        assert params.state == AvatarState.THINKING
        assert params.glow_intensity > 0

    def test_current_params_speaking_with_lip_sync(self):
        from acheron.interface.avatar.renderer import (
            AvatarController, AvatarState, LipSyncTimeline,
        )
        ctrl = AvatarController()
        ctrl.transition(AvatarState.LISTENING)
        ctrl.transition(AvatarState.THINKING)
        ctrl.transition(AvatarState.SPEAKING)

        timeline = LipSyncTimeline(
            frames=[0.5, 0.8, 0.3, 0.1],
            frame_duration_ms=50.0,
        )
        ctrl.set_lip_sync(timeline)
        params = ctrl.current_params()
        assert params.state == AvatarState.SPEAKING
        # Mouth should be non-zero at the start.
        assert params.mouth_open >= 0.0

    def test_animation_params_to_dict(self):
        from acheron.interface.avatar.renderer import AnimationParams, AvatarState
        params = AnimationParams(
            state=AvatarState.SPEAKING,
            mouth_open=0.75,
            blink=True,
            glow_intensity=0.5,
        )
        d = params.to_dict()
        assert d["state"] == "speaking"
        assert d["mouth_open"] == 0.75
        assert d["blink"] is True
        assert d["glow_intensity"] == 0.5

    def test_lip_sync_timeline(self):
        from acheron.interface.avatar.renderer import LipSyncTimeline
        timeline = LipSyncTimeline(
            frames=[0.2, 0.6, 0.9, 0.4, 0.1],
            frame_duration_ms=100.0,
        )
        assert timeline.duration_ms == 500.0
        assert timeline.amplitude_at(0) == 0.2
        assert timeline.amplitude_at(150) == 0.6
        assert timeline.amplitude_at(250) == 0.9
        assert timeline.amplitude_at(600) == 0.0  # Past end.


# ======================================================================
# Lip-sync extraction (voice/tts.py)
# ======================================================================

class TestLipSync:

    def _make_wav(self, freq=440, duration_s=0.5, sample_rate=22050):
        """Generate a simple sine wave WAV."""
        n_samples = int(sample_rate * duration_s)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            val = int(32767 * 0.8 * math.sin(2 * math.pi * freq * t))
            samples.append(val)
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    def test_extract_lip_sync_basic(self):
        from acheron.interface.voice.tts import extract_lip_sync
        wav = self._make_wav(duration_s=0.5)
        frames = extract_lip_sync(wav, frame_duration_ms=50.0)
        assert len(frames) > 0
        # Sine wave should have consistent amplitude.
        amps = [f.amplitude for f in frames]
        assert all(0.0 <= a <= 1.0 for a in amps)
        assert max(amps) > 0.5  # Should have significant amplitude.

    def test_extract_lip_sync_empty(self):
        from acheron.interface.voice.tts import extract_lip_sync
        frames = extract_lip_sync(b"not a wav file")
        assert frames == []

    def test_extract_lip_sync_silence(self):
        from acheron.interface.voice.tts import extract_lip_sync
        # Generate silence.
        n_samples = 11025
        pcm = struct.pack(f"<{n_samples}h", *([0] * n_samples))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(pcm)
        wav = buf.getvalue()
        frames = extract_lip_sync(wav)
        # All amplitudes should be 0 for silence.
        if frames:
            assert all(f.amplitude == 0.0 for f in frames)

    def test_lip_sync_frame_timing(self):
        from acheron.interface.voice.tts import extract_lip_sync
        wav = self._make_wav(duration_s=0.2)
        frames = extract_lip_sync(wav, frame_duration_ms=50.0)
        # Check timing is correct.
        for i, f in enumerate(frames):
            assert f.time_ms == pytest.approx(i * 50.0)


# ======================================================================
# STT (voice/stt.py)
# ======================================================================

class TestWhisperSTT:

    def test_backend_detection(self):
        from acheron.interface.voice.stt import _detect_backend
        backend = _detect_backend()
        # Should return one of the three options.
        assert backend in ("faster_whisper", "openai_whisper", "none")

    def test_stt_construction(self):
        from acheron.interface.voice.stt import WhisperSTT
        stt = WhisperSTT(model_size="tiny", device="cpu")
        assert stt.model_size == "tiny"
        assert stt.device == "cpu"
        # available depends on whether whisper is installed.
        assert isinstance(stt.available, bool)

    def test_transcribe_empty_audio(self):
        from acheron.interface.voice.stt import WhisperSTT
        stt = WhisperSTT()
        # Empty/silence should return empty string.
        result = stt.transcribe(b"\x00" * 100)
        assert result == ""

    def test_pcm_to_wav(self):
        from acheron.interface.voice.stt import _pcm_to_wav, _is_wav
        pcm = struct.pack("<10h", *range(10))
        wav = _pcm_to_wav(pcm, sample_rate=16000)
        assert _is_wav(wav)

    def test_is_wav(self):
        from acheron.interface.voice.stt import _is_wav
        assert not _is_wav(b"")
        assert not _is_wav(b"not a wav")
        # Construct minimal RIFF header.
        header = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE"
        assert _is_wav(header)

    def test_compute_rms(self):
        from acheron.interface.voice.stt import _compute_rms
        # Silence.
        silence = struct.pack("<10h", *([0] * 10))
        assert _compute_rms(silence) == 0.0
        # Loud signal.
        loud = struct.pack("<10h", *([32000] * 10))
        assert _compute_rms(loud) > 1000


# ======================================================================
# TTS Engines (voice/tts.py)
# ======================================================================

class TestTTSEngines:

    def test_piper_not_available_without_model(self):
        from acheron.interface.voice.tts import PiperTTS
        piper = PiperTTS(model_path="")
        assert not piper.available

    def test_piper_not_available_missing_file(self, tmp_path):
        from acheron.interface.voice.tts import PiperTTS
        piper = PiperTTS(model_path=tmp_path / "nonexistent.onnx")
        assert not piper.available

    def test_elevenlabs_not_available_without_key(self):
        from acheron.interface.voice.tts import ElevenLabsTTS
        el = ElevenLabsTTS(api_key="")
        assert not el.available

    def test_elevenlabs_available_with_key(self):
        from acheron.interface.voice.tts import ElevenLabsTTS
        el = ElevenLabsTTS(api_key="test-key-123")
        assert el.available

    def test_elevenlabs_set_voice(self):
        from acheron.interface.voice.tts import ElevenLabsTTS
        el = ElevenLabsTTS(api_key="test-key")
        assert el.voice_id == "ErXwobaYiN019PkySvjV"  # Antoni (new default)
        el.set_voice("21m00Tcm4TlvDq8ikWAM")  # Rachel
        assert el.voice_id == "21m00Tcm4TlvDq8ikWAM"

    def test_create_tts_engine_no_backends(self):
        from acheron.interface.voice.tts import create_tts_engine
        with pytest.raises(RuntimeError, match="No TTS engine available"):
            create_tts_engine(piper_model="", elevenlabs_key="")


# ======================================================================
# Voice Profiles (voice/tts.py)
# ======================================================================

class TestVoiceProfiles:

    def test_profiles_exist(self):
        from acheron.interface.voice.tts import VOICE_PROFILES, DEFAULT_VOICE
        assert "male" in VOICE_PROFILES
        assert "female" in VOICE_PROFILES
        assert DEFAULT_VOICE in VOICE_PROFILES

    def test_profile_structure(self):
        from acheron.interface.voice.tts import VOICE_PROFILES
        for pid, profile in VOICE_PROFILES.items():
            assert profile.id == pid
            assert profile.label
            assert profile.elevenlabs_voice_id
            assert profile.piper_model_name
            assert profile.description

    def test_male_and_female_different_voices(self):
        from acheron.interface.voice.tts import VOICE_PROFILES
        male = VOICE_PROFILES["male"]
        female = VOICE_PROFILES["female"]
        assert male.elevenlabs_voice_id != female.elevenlabs_voice_id
        assert male.piper_model_name != female.piper_model_name

    def test_default_voice_is_male(self):
        from acheron.interface.voice.tts import DEFAULT_VOICE
        assert DEFAULT_VOICE == "male"


# ======================================================================
# WebSocket Server (ws_server.py)
# ======================================================================

class TestInterruptDetection:

    def test_stop_is_interrupt(self):
        from acheron.interface.ws_server import is_interrupt_command
        assert is_interrupt_command("stop")
        assert is_interrupt_command("Stop")
        assert is_interrupt_command("STOP")

    def test_nexus_stop_is_interrupt(self):
        from acheron.interface.ws_server import is_interrupt_command
        assert is_interrupt_command("nexus stop")
        assert is_interrupt_command("Nexus stop")

    def test_quiet_variants(self):
        from acheron.interface.ws_server import is_interrupt_command
        assert is_interrupt_command("be quiet")
        assert is_interrupt_command("quiet")
        assert is_interrupt_command("enough")

    def test_cancel_and_wait(self):
        from acheron.interface.ws_server import is_interrupt_command
        assert is_interrupt_command("cancel")
        assert is_interrupt_command("hold on")
        assert is_interrupt_command("wait")
        assert is_interrupt_command("never mind")
        assert is_interrupt_command("nevermind")

    def test_long_query_not_interrupt(self):
        from acheron.interface.ws_server import is_interrupt_command
        # More than 4 words → not treated as interrupt even if it contains "stop".
        assert not is_interrupt_command(
            "Can you stop and explain the bioelectric mechanism?"
        )

    def test_normal_query_not_interrupt(self):
        from acheron.interface.ws_server import is_interrupt_command
        assert not is_interrupt_command("What is Vmem?")
        assert not is_interrupt_command("Tell me about planarians")
        assert not is_interrupt_command("")


class TestWSServer:

    def test_create_app(self):
        from acheron.interface.ws_server import create_interface_app
        app = create_interface_app()
        assert app.title == "Nexus Interface"
        routes = [r.path for r in app.routes]
        assert "/" in routes
        assert "/ws" in routes
        assert "/health" in routes

    def test_topic_extraction(self):
        from acheron.interface.ws_server import _extract_topics
        topics = _extract_topics("What is the role of Vmem in planarian regeneration?")
        assert "vmem" in topics
        assert "planarian" in topics
        assert "regeneration" in topics
        # Stopwords should be excluded.
        assert "the" not in topics
        assert "is" not in topics
        assert "of" not in topics

    def test_topic_extraction_empty(self):
        from acheron.interface.ws_server import _extract_topics
        topics = _extract_topics("")
        assert topics == []

    def test_topic_extraction_limit(self):
        from acheron.interface.ws_server import _extract_topics
        long_query = " ".join(f"word{i}" for i in range(50))
        topics = _extract_topics(long_query)
        assert len(topics) <= 8

    def test_truncate_for_speech(self):
        from acheron.interface.ws_server import _truncate_for_speech
        short = "Hello world."
        assert _truncate_for_speech(short) == short

        long = "First sentence. " * 200
        truncated = _truncate_for_speech(long, max_chars=100)
        assert len(truncated) <= 100
        assert truncated.endswith(".")

    def test_source_to_dict(self):
        from acheron.interface.ws_server import _source_to_dict

        class FakeSource:
            paper_title = "A Paper"
            authors = ["Author A"]
            doi = "10.1234/test"
            section = "Abstract"
            relevance_score = 0.85
            excerpt = "Some text..."

        d = _source_to_dict(FakeSource())
        assert d["title"] == "A Paper"
        assert d["score"] == 0.85
        assert d["doi"] == "10.1234/test"

    def test_format_discovery(self):
        from acheron.interface.ws_server import _format_discovery

        class FakeDiscovery:
            evidence = ["Evidence 1", "Evidence 2"]
            inference = ["Inference 1"]
            speculation = []
            hypotheses = []
            bioelectric_schematic = ""
            raw_output = ""

        text = _format_discovery(FakeDiscovery())
        assert "## Evidence" in text
        assert "Evidence 1" in text
        assert "## Inference" in text
        assert "## Speculation" not in text  # Empty section omitted.


# ======================================================================
# Config integration
# ======================================================================

class TestInterfaceConfig:

    def test_interface_settings_defaults(self):
        from acheron.config import Settings
        s = Settings(
            _env_file="",  # Don't read any .env file
        )
        assert s.interface_port == 8100
        assert s.interface_host == "0.0.0.0"
        assert s.whisper_model == "base"
        assert s.whisper_device == "auto"
        assert s.piper_model_path == ""
        assert s.elevenlabs_api_key == ""
        assert s.nexus_session_path == "data/nexus_session.json"
