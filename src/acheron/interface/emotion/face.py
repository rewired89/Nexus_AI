"""Facial expression recognition using DeepFace.

Wraps the ``deepface`` library to analyse single video frames
(JPEG bytes from the kiosk camera) and return an ``EmotionReading``.

Falls back gracefully when ``deepface`` or ``cv2`` are not installed —
``available`` will simply be ``False``.
"""

from __future__ import annotations

import logging
from typing import Optional

from acheron.interface.emotion.state import Emotion, EmotionReading

logger = logging.getLogger(__name__)

# DeepFace label → our Emotion enum
_LABEL_MAP = {
    "angry": Emotion.ANGRY,
    "disgust": Emotion.DISGUSTED,
    "fear": Emotion.FEARFUL,
    "happy": Emotion.HAPPY,
    "sad": Emotion.SAD,
    "surprise": Emotion.SURPRISED,
    "neutral": Emotion.NEUTRAL,
}


class FaceEmotionAnalyzer:
    """Detect emotions from a single camera frame using DeepFace."""

    def __init__(self) -> None:
        self._available = False
        self._deepface = None  # type: ignore[assignment]
        self._cv2 = None       # type: ignore[assignment]

        try:
            import cv2
            from deepface import DeepFace

            self._cv2 = cv2
            self._deepface = DeepFace
            self._available = True
            logger.info("FaceEmotionAnalyzer: DeepFace + OpenCV loaded")
        except ImportError as exc:
            logger.info(
                "Facial emotion detection unavailable (%s). "
                "Install with:  pip install deepface tf-keras opencv-python",
                exc,
            )

    @property
    def available(self) -> bool:
        return self._available

    def analyze_frame(self, frame_bytes: bytes) -> Optional[EmotionReading]:
        """Analyse JPEG bytes and return an EmotionReading, or *None*."""
        if not self._available:
            return None

        try:
            import numpy as np

            # Decode JPEG → BGR numpy array.
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
            if img is None:
                return None

            results = self._deepface.analyze(
                img,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            if not results:
                return None

            result = results[0]
            emotions: dict = result.get("emotion", {})
            if not emotions:
                return None

            dominant_label = result.get("dominant_emotion", "neutral")
            mapped = _LABEL_MAP.get(dominant_label, Emotion.NEUTRAL)
            confidence = emotions.get(dominant_label, 0.0) / 100.0

            raw_scores = {}
            for label, score in emotions.items():
                em = _LABEL_MAP.get(label)
                if em:
                    raw_scores[em.value] = score / 100.0

            return EmotionReading(
                emotion=mapped,
                confidence=confidence,
                source="face",
                raw_scores=raw_scores,
            )

        except Exception as exc:
            logger.debug("Face emotion analysis failed: %s", exc)
            return None
