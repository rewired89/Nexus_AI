"""Multimodal emotion detector with late fusion and trajectory tracking.

Orchestrates :class:`FaceEmotionAnalyzer` and :class:`VoiceEmotionAnalyzer`,
fuses their outputs into a single :class:`EmotionalState`, and tracks
how the user's emotional valence changes over time.

Usage (inside the WebSocket handler)::

    detector = EmotionDetector()
    # lazy-load heavy models on first use
    detector.initialize()

    # from camera frames (every ~500 ms)
    detector.process_video_frame(jpeg_bytes)

    # from audio segments (when available)
    detector.process_audio(wav_bytes)

    # read the current fused state
    state = detector.current_state
    prompt_ctx = state.to_prompt_context()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

from acheron.interface.emotion.state import (
    EMOTION_AROUSAL,
    EMOTION_VALENCE,
    Emotion,
    EmotionalState,
    EmotionReading,
)

logger = logging.getLogger(__name__)

# How many recent readings to keep for trajectory analysis.
_HISTORY_SIZE = 30

# Minimum readings before we compute a trajectory.
_MIN_TRAJECTORY_READINGS = 4

# Readings older than this (seconds) are ignored in fusion.
_READING_MAX_AGE = 30.0

# Modality weights (face is generally more reliable for basic emotions).
_FACE_WEIGHT = 0.6
_VOICE_WEIGHT = 0.4


class EmotionDetector:
    """Fuses face + voice emotion readings into a single state."""

    def __init__(self) -> None:
        self._face = None       # type: ignore[assignment]
        self._voice = None      # type: ignore[assignment]
        self._history: deque[EmotionReading] = deque(maxlen=_HISTORY_SIZE)
        self._state = EmotionalState()
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Lazy-load the heavy backends (DeepFace, transformers, etc.)."""
        if self._initialized:
            return
        self._initialized = True

        from acheron.interface.emotion.face import FaceEmotionAnalyzer
        from acheron.interface.emotion.voice_emotion import (
            VoiceEmotionAnalyzer,
        )

        self._face = FaceEmotionAnalyzer()
        self._voice = VoiceEmotionAnalyzer()

        parts = []
        if self._face.available:
            parts.append("face")
        if self._voice.available:
            parts.append("voice")

        if parts:
            logger.info("EmotionDetector ready: %s", " + ".join(parts))
        else:
            logger.info(
                "EmotionDetector: no backends installed. "
                "Emotion detection will be disabled."
            )

    @property
    def available(self) -> bool:
        if not self._initialized:
            return False
        face_ok = self._face is not None and self._face.available
        voice_ok = self._voice is not None and self._voice.available
        return face_ok or voice_ok

    @property
    def face_available(self) -> bool:
        return (
            self._initialized
            and self._face is not None
            and self._face.available
        )

    @property
    def voice_available(self) -> bool:
        return (
            self._initialized
            and self._voice is not None
            and self._voice.available
        )

    @property
    def current_state(self) -> EmotionalState:
        return self._state

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_video_frame(
        self, frame_bytes: bytes,
    ) -> Optional[EmotionReading]:
        """Analyse a JPEG frame for facial emotion."""
        if not self._initialized:
            self.initialize()
        if not self._face or not self._face.available:
            return None

        reading = self._face.analyze_frame(frame_bytes)
        if reading:
            self._history.append(reading)
            self._update_state()
        return reading

    def process_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> Optional[EmotionReading]:
        """Analyse WAV/PCM bytes for voice emotion."""
        if not self._initialized:
            self.initialize()
        if not self._voice or not self._voice.available:
            return None

        reading = self._voice.analyze_audio(audio_bytes, sample_rate)
        if reading:
            self._history.append(reading)
            self._update_state()
        return reading

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _update_state(self) -> None:
        """Re-compute the fused emotional state from recent readings."""
        now = time.time()

        recent = [
            r for r in self._history
            if (now - r.timestamp) < _READING_MAX_AGE
        ]
        if not recent:
            self._state = EmotionalState()
            return

        # Most recent reading per modality.
        latest_face: Optional[EmotionReading] = None
        latest_voice: Optional[EmotionReading] = None
        for r in reversed(recent):
            if r.source == "face" and latest_face is None:
                latest_face = r
            elif r.source == "voice" and latest_voice is None:
                latest_voice = r
            if latest_face and latest_voice:
                break

        # Weighted fusion.
        primary, confidence = self._fuse(latest_face, latest_voice)
        valence = EMOTION_VALENCE.get(primary, 0.0)
        arousal = EMOTION_AROUSAL.get(primary, 0.0)
        trajectory = self._compute_trajectory(recent)

        self._state = EmotionalState(
            primary_emotion=primary,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            trajectory=trajectory,
            face_emotion=latest_face.emotion if latest_face else None,
            voice_emotion=latest_voice.emotion if latest_voice else None,
        )

    @staticmethod
    def _fuse(
        face: Optional[EmotionReading],
        voice: Optional[EmotionReading],
    ) -> tuple[Emotion, float]:
        """Return (primary_emotion, confidence) from available readings."""
        if face and voice:
            fc = face.confidence * _FACE_WEIGHT
            vc = voice.confidence * _VOICE_WEIGHT

            # Pick the stronger signal.
            if fc >= vc:
                primary = face.emotion
            else:
                primary = voice.emotion

            total = fc + vc
            confidence = total / (total + 0.01)

            # Boost when both modalities agree.
            if face.emotion == voice.emotion:
                confidence = min(1.0, confidence * 1.3)

            return primary, confidence

        if face:
            return face.emotion, face.confidence * 0.8

        if voice:
            return voice.emotion, voice.confidence * 0.7

        return Emotion.NEUTRAL, 0.0

    @staticmethod
    def _compute_trajectory(readings: list[EmotionReading]) -> str:
        """Compare older vs. newer readings to detect valence drift."""
        if len(readings) < _MIN_TRAJECTORY_READINGS:
            return "stable"

        mid = len(readings) // 2
        older = readings[:mid]
        newer = readings[mid:]

        def _avg_valence(rs: list[EmotionReading]) -> float:
            vals = [EMOTION_VALENCE.get(r.emotion, 0.0) for r in rs]
            return sum(vals) / len(vals) if vals else 0.0

        diff = _avg_valence(newer) - _avg_valence(older)
        if diff > 0.15:
            return "improving"
        if diff < -0.15:
            return "deteriorating"
        return "stable"

    def reset(self) -> None:
        """Clear all history (e.g., on session change)."""
        self._history.clear()
        self._state = EmotionalState()
