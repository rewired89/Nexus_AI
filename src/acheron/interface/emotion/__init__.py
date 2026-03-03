"""Multimodal emotion detection for the Nexus interface.

Detects user emotional state from facial expressions (camera) and
voice prosody (audio) to enable empathetic, human-like responses.

Components
----------
- ``EmotionDetector``  — orchestrates face + voice analysis, fuses signals
- ``FaceEmotionAnalyzer`` — facial expression recognition (DeepFace)
- ``VoiceEmotionAnalyzer`` — speech emotion recognition (transformers / librosa)
- ``EmotionalState``  — fused state with valence, arousal, trajectory

All backends are optional and lazy-loaded.  The system degrades
gracefully when dependencies (deepface, transformers, librosa) are
not installed.
"""

from acheron.interface.emotion.detector import EmotionDetector
from acheron.interface.emotion.state import Emotion, EmotionalState, EmotionReading

__all__ = [
    "EmotionDetector",
    "Emotion",
    "EmotionalState",
    "EmotionReading",
]
