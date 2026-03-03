"""Voice / speech emotion recognition.

Two backends, tried in order:

1. **transformers** — wav2vec2-based audio-classification pipeline
   (most accurate, ~80 % on IEMOCAP/RAVDESS).
2. **librosa** — rule-based prosodic-feature analysis (pitch, energy,
   speech-rate).  Less accurate but very lightweight.

Both are optional; ``available`` is ``False`` when neither is installed.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Optional

from acheron.interface.emotion.state import Emotion, EmotionReading

logger = logging.getLogger(__name__)

# HuggingFace model labels → our Emotion enum.
_HF_LABEL_MAP = {
    "angry": Emotion.ANGRY,
    "anger": Emotion.ANGRY,
    "calm": Emotion.NEUTRAL,
    "disgust": Emotion.DISGUSTED,
    "fearful": Emotion.FEARFUL,
    "fear": Emotion.FEARFUL,
    "happy": Emotion.HAPPY,
    "happiness": Emotion.HAPPY,
    "neutral": Emotion.NEUTRAL,
    "sad": Emotion.SAD,
    "sadness": Emotion.SAD,
    "surprised": Emotion.SURPRISED,
    "surprise": Emotion.SURPRISED,
    "ps": Emotion.SURPRISED,         # some models use "ps" for surprise
}


class VoiceEmotionAnalyzer:
    """Detect emotions from raw audio (WAV bytes)."""

    def __init__(self) -> None:
        self._backend = "none"
        self._pipeline = None  # type: ignore[assignment]
        self._available = False

        # --- try transformers first ---
        try:
            from transformers import pipeline  # type: ignore[import-untyped]

            self._pipeline = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=-1,  # CPU — avoids CUDA dependency
            )
            self._backend = "transformers"
            self._available = True
            logger.info(
                "VoiceEmotionAnalyzer: transformers wav2vec2 loaded"
            )
            return
        except Exception:
            pass

        # --- fallback: librosa prosody ---
        try:
            import librosa  # noqa: F401

            self._backend = "librosa"
            self._available = True
            logger.info("VoiceEmotionAnalyzer: librosa prosody fallback")
        except ImportError:
            logger.info(
                "Voice emotion detection unavailable. Install with: "
                "pip install transformers torch   OR   pip install librosa"
            )

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
    ) -> Optional[EmotionReading]:
        """Analyse WAV/PCM bytes and return an EmotionReading, or *None*."""
        if not self._available:
            return None

        if self._backend == "transformers":
            return self._analyze_transformers(audio_bytes, sample_rate)
        if self._backend == "librosa":
            return self._analyze_prosody(audio_bytes, sample_rate)
        return None

    # ------------------------------------------------------------------
    # Transformers backend
    # ------------------------------------------------------------------

    def _analyze_transformers(
        self, audio_bytes: bytes, sample_rate: int,
    ) -> Optional[EmotionReading]:
        try:
            audio = _bytes_to_float(audio_bytes)
            if audio is None or len(audio) < sample_rate:
                return None  # need at least 1 s

            results = self._pipeline(
                {"raw": audio, "sampling_rate": sample_rate},
                top_k=7,
            )
            if not results:
                return None

            top = results[0]
            label = top["label"].lower()
            mapped = _HF_LABEL_MAP.get(label, Emotion.NEUTRAL)
            confidence = float(top["score"])

            raw_scores: dict[str, float] = {}
            for r in results:
                em = _HF_LABEL_MAP.get(r["label"].lower())
                if em:
                    raw_scores[em.value] = float(r["score"])

            return EmotionReading(
                emotion=mapped,
                confidence=confidence,
                source="voice",
                raw_scores=raw_scores,
            )
        except Exception as exc:
            logger.debug("Transformers voice emotion failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Librosa prosody fallback
    # ------------------------------------------------------------------

    def _analyze_prosody(
        self, audio_bytes: bytes, sample_rate: int,
    ) -> Optional[EmotionReading]:
        """Rule-based emotion estimation from pitch / energy / tempo."""
        try:
            import librosa
            import numpy as np

            audio = _bytes_to_float(audio_bytes)
            if audio is None or len(audio) < sample_rate:
                return None

            # Pitch contour.
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=sample_rate,
            )
            pitch_values = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if not pitch_values:
                return EmotionReading(
                    emotion=Emotion.NEUTRAL,
                    confidence=0.3,
                    source="voice",
                )

            pitch_arr = np.array(pitch_values)
            pitch_std = float(np.std(pitch_arr))
            pitch_range = float(np.ptp(pitch_arr))

            # Energy (RMS).
            rms = librosa.feature.rms(y=audio)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))

            # Speech rate (onset events / duration).
            onsets = librosa.onset.onset_detect(y=audio, sr=sample_rate)
            duration = len(audio) / sample_rate
            speech_rate = len(onsets) / duration if duration > 0 else 0

            # Simple heuristic mapping.
            emotion = Emotion.NEUTRAL
            confidence = 0.35

            if energy_mean > 0.08 and pitch_std > 50 and speech_rate > 4:
                emotion = Emotion.ANGRY
                confidence = 0.50
            elif energy_mean < 0.02 and speech_rate < 2:
                emotion = Emotion.SAD
                confidence = 0.45
            elif pitch_range > 200:
                emotion = Emotion.SURPRISED
                confidence = 0.40
            elif pitch_std > 40 and energy_std > 0.03:
                emotion = Emotion.FRUSTRATED
                confidence = 0.45

            return EmotionReading(
                emotion=emotion,
                confidence=confidence,
                source="voice",
                raw_scores={emotion.value: confidence},
            )
        except Exception as exc:
            logger.debug("Librosa prosody analysis failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _bytes_to_float(audio_bytes: bytes):
    """Convert WAV or raw-PCM bytes to a float32 numpy array."""
    try:
        import numpy as np

        # WAV container.
        if audio_bytes[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                sw = wf.getsampwidth()
                channels = wf.getnchannels()

                if sw == 2:
                    audio = np.frombuffer(
                        frames, dtype=np.int16,
                    ).astype(np.float32) / 32768.0
                elif sw == 4:
                    audio = np.frombuffer(
                        frames, dtype=np.int32,
                    ).astype(np.float32) / 2147483648.0
                else:
                    audio = np.frombuffer(
                        frames, dtype=np.int16,
                    ).astype(np.float32) / 32768.0

                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                return audio

        # Raw PCM int16 fallback.
        return np.frombuffer(
            audio_bytes, dtype=np.int16,
        ).astype(np.float32) / 32768.0

    except Exception:
        return None
