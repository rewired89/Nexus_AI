"""Text-to-speech for the Nexus interface.

Two engines:

1. **Piper TTS** — Fast, local, no API key.  Requires the ``piper-tts``
   package and a downloaded voice model (``.onnx`` + ``.json``).
2. **ElevenLabs** — Cloud API, higher quality, requires API key.

Both produce 16-bit PCM WAV audio suitable for browser playback and
avatar lip-sync amplitude extraction.
"""

from __future__ import annotations

import io
import logging
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------

class TTSEngine(Protocol):
    """Protocol that all TTS backends implement."""

    def synthesize(self, text: str) -> bytes:
        """Return WAV audio bytes for the given text."""
        ...

    @property
    def available(self) -> bool:
        """Whether this engine is usable."""
        ...


# ---------------------------------------------------------------------------
# Lip-sync utilities
# ---------------------------------------------------------------------------

@dataclass
class LipSyncFrame:
    """Single frame of amplitude data for avatar mouth animation."""

    time_ms: float
    amplitude: float  # 0.0 – 1.0 normalized


def extract_lip_sync(
    wav_data: bytes,
    frame_duration_ms: float = 50.0,
) -> list[LipSyncFrame]:
    """Extract per-frame RMS amplitudes from WAV audio.

    The frontend uses these to drive mouth-open animation on the avatar.

    Parameters
    ----------
    wav_data : bytes
        Complete WAV file bytes.
    frame_duration_ms : float
        Duration of each analysis frame in milliseconds.

    Returns
    -------
    list[LipSyncFrame]
        Amplitude envelope sampled at ``frame_duration_ms`` intervals.
    """
    buf = io.BytesIO(wav_data)
    try:
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except wave.Error:
        return []

    if sample_width != 2:
        return []

    # Unpack to mono samples.
    n_samples = len(raw) // 2
    samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * 2])
    if n_channels == 2:
        # Average stereo to mono.
        mono = [
            (samples[i] + samples[i + 1]) // 2
            for i in range(0, len(samples) - 1, 2)
        ]
    else:
        mono = list(samples)

    if not mono:
        return []

    # Frame parameters.
    samples_per_frame = max(1, int(sample_rate * frame_duration_ms / 1000))

    # Compute per-frame RMS.
    frames: list[LipSyncFrame] = []
    peak_rms = 0.0
    raw_rms: list[float] = []

    for start in range(0, len(mono), samples_per_frame):
        chunk = mono[start : start + samples_per_frame]
        if not chunk:
            break
        mean_sq = sum(s * s for s in chunk) / len(chunk)
        rms = math.sqrt(mean_sq)
        raw_rms.append(rms)
        if rms > peak_rms:
            peak_rms = rms

    # Normalize to 0–1.
    for i, rms in enumerate(raw_rms):
        amp = rms / peak_rms if peak_rms > 0 else 0.0
        frames.append(LipSyncFrame(
            time_ms=i * frame_duration_ms,
            amplitude=min(1.0, amp),
        ))

    return frames


# ---------------------------------------------------------------------------
# Piper TTS (local)
# ---------------------------------------------------------------------------

class PiperTTS:
    """Local text-to-speech via Piper.

    Parameters
    ----------
    model_path : str or Path
        Path to the Piper ONNX voice model file.
    config_path : str or Path or None
        Path to the model's JSON config.  If None, assumes
        ``{model_path}.json`` exists alongside the model.
    """

    def __init__(
        self,
        model_path: str | Path = "",
        config_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.config_path = Path(config_path) if config_path else None
        self._voice: object = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import piper  # noqa: F401  # type: ignore[import-untyped]
            self._available = self.model_path is not None and self.model_path.exists()
        except ImportError:
            self._available = False
        return self._available

    def _ensure_voice(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice  # type: ignore[import-untyped]

        config = self.config_path or (
            self.model_path.with_suffix(self.model_path.suffix + ".json")
            if self.model_path
            else None
        )
        self._voice = PiperVoice.load(str(self.model_path), config_path=str(config))
        logger.info("Loaded Piper voice: %s", self.model_path)

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV bytes."""
        if not self.available:
            raise RuntimeError("Piper TTS not available — check model path and installation")

        self._ensure_voice()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            self._voice.synthesize(text, wf)  # type: ignore[union-attr]
        return buf.getvalue()


# ---------------------------------------------------------------------------
# ElevenLabs TTS (cloud)
# ---------------------------------------------------------------------------

class ElevenLabsTTS:
    """Cloud text-to-speech via ElevenLabs API.

    Parameters
    ----------
    api_key : str
        ElevenLabs API key.
    voice_id : str
        Voice ID to use.  Defaults to "Rachel" (a neutral female voice).
    model_id : str
        ElevenLabs model.  ``eleven_monolingual_v1`` is fastest.
    """

    def __init__(
        self,
        api_key: str = "",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
        model_id: str = "eleven_monolingual_v1",
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is not None:
            return self._available
        self._available = bool(self.api_key)
        return self._available

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV bytes via ElevenLabs API."""
        if not self.available:
            raise RuntimeError(
                "ElevenLabs TTS not available — set ELEVENLABS_API_KEY"
            )
        import httpx

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/wav",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tts_engine(
    piper_model: str = "",
    elevenlabs_key: str = "",
    elevenlabs_voice: str = "21m00Tcm4TlvDq8ikWAM",
) -> TTSEngine:
    """Create the best available TTS engine.

    Prefers Piper (local, no latency) if a model is provided and
    the library is installed.  Falls back to ElevenLabs if an API key
    is set.  Raises RuntimeError if neither is available.
    """
    if piper_model:
        piper = PiperTTS(model_path=piper_model)
        if piper.available:
            logger.info("Using Piper TTS (local)")
            return piper  # type: ignore[return-value]

    if elevenlabs_key:
        el = ElevenLabsTTS(api_key=elevenlabs_key, voice_id=elevenlabs_voice)
        if el.available:
            logger.info("Using ElevenLabs TTS (cloud)")
            return el  # type: ignore[return-value]

    raise RuntimeError(
        "No TTS engine available. Provide a Piper model path or "
        "set ELEVENLABS_API_KEY."
    )
