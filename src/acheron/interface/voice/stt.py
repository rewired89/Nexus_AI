"""Whisper-based speech-to-text for the Nexus interface.

Supports two backends:

1. **openai-whisper** (``pip install openai-whisper``) — Reference
   implementation from OpenAI.  Requires ``ffmpeg``.
2. **faster-whisper** (``pip install faster-whisper``) — CTranslate2-based,
   lower VRAM, faster inference on GPU.  Preferred for production.

The module auto-detects which backend is available and falls back
gracefully.  All public methods accept raw audio bytes (16-bit PCM or
WAV) and return transcribed text.
"""

from __future__ import annotations

import io
import logging
import struct
import tempfile
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_BACKEND: Optional[str] = None


def _detect_backend() -> str:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import faster_whisper  # noqa: F401
        _BACKEND = "faster_whisper"
        return _BACKEND
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        _BACKEND = "openai_whisper"
        return _BACKEND
    except ImportError:
        pass
    _BACKEND = "none"
    return _BACKEND


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _is_wav(data: bytes) -> bool:
    """Check if data starts with a RIFF/WAVE header."""
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _compute_rms(audio_bytes: bytes) -> float:
    """Compute RMS amplitude of 16-bit PCM samples."""
    if len(audio_bytes) < 2:
        return 0.0
    n_samples = len(audio_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", audio_bytes[:n_samples * 2])
    if not samples:
        return 0.0
    mean_sq = sum(s * s for s in samples) / n_samples
    return mean_sq ** 0.5


# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------

# Default silence threshold — audio below this RMS is skipped.
_SILENCE_RMS_THRESHOLD = 50.0


class WhisperSTT:
    """Local Whisper speech-to-text engine.

    Parameters
    ----------
    model_size : str
        Whisper model size: ``tiny``, ``base``, ``small``, ``medium``,
        ``large-v3``.  Smaller = faster / less VRAM.
    device : str
        ``cuda`` or ``cpu``.  ``auto`` picks GPU if available.
    language : str
        ISO language code.  ``en`` for English-only (faster).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: str = "en",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model: object = None
        self._backend = _detect_backend()

        if self._backend == "none":
            logger.warning(
                "No Whisper backend found. Install faster-whisper or "
                "openai-whisper. STT will return empty strings."
            )

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        if self._backend == "faster_whisper":
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(
                self.model_size, device=device, compute_type=compute_type
            )
            logger.info(
                "Loaded faster-whisper %s on %s (%s)",
                self.model_size, device, compute_type,
            )

        elif self._backend == "openai_whisper":
            import whisper  # type: ignore[import-untyped]

            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            self._model = whisper.load_model(self.model_size, device=device)
            logger.info("Loaded openai-whisper %s on %s", self.model_size, device)

    def transcribe(
        self,
        audio: bytes,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe audio bytes to text.

        Parameters
        ----------
        audio : bytes
            Raw 16-bit PCM or WAV-encoded audio.
        sample_rate : int
            Sample rate of the audio (used only for raw PCM).

        Returns
        -------
        str
            Transcribed text, or empty string on failure / silence.
        """
        if self._backend == "none":
            return ""

        # Ensure we have WAV data.
        if _is_wav(audio):
            wav_data = audio
        else:
            wav_data = _pcm_to_wav(audio, sample_rate=sample_rate)

        # Skip silence.
        pcm_payload = audio if not _is_wav(audio) else audio[44:]
        if _compute_rms(pcm_payload) < _SILENCE_RMS_THRESHOLD:
            return ""

        self._ensure_model()

        # Write to temp file — both backends prefer file paths.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_data)
            tmp_path = Path(tmp.name)

        try:
            if self._backend == "faster_whisper":
                segments, _ = self._model.transcribe(  # type: ignore[union-attr]
                    str(tmp_path), language=self.language, beam_size=5
                )
                text = " ".join(seg.text.strip() for seg in segments)
            else:
                result = self._model.transcribe(  # type: ignore[union-attr]
                    str(tmp_path), language=self.language
                )
                text = result.get("text", "").strip()
            return text
        except Exception:
            logger.exception("Whisper transcription failed")
            return ""
        finally:
            tmp_path.unlink(missing_ok=True)

    @property
    def available(self) -> bool:
        """Whether a Whisper backend is installed."""
        return self._backend != "none"
