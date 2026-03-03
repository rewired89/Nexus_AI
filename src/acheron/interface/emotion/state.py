"""Emotion data types and prompt generation.

Defines the core vocabulary of emotions, dimensional mappings
(valence / arousal), and the logic for turning a detected emotional
state into natural-language guidance that can be injected into an
LLM system prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Emotion taxonomy
# ---------------------------------------------------------------------------

class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"


# Valence: how positive (>0) or negative (<0) the emotion is.
EMOTION_VALENCE: Dict[Emotion, float] = {
    Emotion.NEUTRAL: 0.0,
    Emotion.HAPPY: 0.8,
    Emotion.SAD: -0.6,
    Emotion.ANGRY: -0.8,
    Emotion.FEARFUL: -0.5,
    Emotion.DISGUSTED: -0.7,
    Emotion.SURPRISED: 0.1,
    Emotion.CONFUSED: -0.2,
    Emotion.FRUSTRATED: -0.7,
}

# Arousal: how activated / intense the emotion is (0 = calm, 1 = intense).
EMOTION_AROUSAL: Dict[Emotion, float] = {
    Emotion.NEUTRAL: 0.1,
    Emotion.HAPPY: 0.6,
    Emotion.SAD: 0.2,
    Emotion.ANGRY: 0.9,
    Emotion.FEARFUL: 0.8,
    Emotion.DISGUSTED: 0.5,
    Emotion.SURPRISED: 0.8,
    Emotion.CONFUSED: 0.4,
    Emotion.FRUSTRATED: 0.7,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EmotionReading:
    """A single emotion measurement from one modality (face or voice)."""

    emotion: Emotion
    confidence: float          # 0–1
    source: str                # "face" or "voice"
    timestamp: float = field(default_factory=time.time)
    raw_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EmotionalState:
    """Fused emotional state from all available modalities.

    This is the object that gets passed to the LLM pipeline so Nexus
    can adapt its behaviour to the user's mood.
    """

    primary_emotion: Emotion = Emotion.NEUTRAL
    confidence: float = 0.0
    valence: float = 0.0       # -1 (negative) … +1 (positive)
    arousal: float = 0.0       # 0 (calm) … 1 (intense)
    trajectory: str = "stable" # "improving" | "stable" | "deteriorating"
    face_emotion: Optional[Emotion] = None
    voice_emotion: Optional[Emotion] = None
    timestamp: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def to_prompt_context(self) -> str:
        """Return natural-language guidance for the LLM system prompt.

        Returns an empty string when confidence is too low to be useful,
        so callers can simply check ``if ctx: …``.
        """
        if self.confidence < 0.30:
            return ""

        lines: list[str] = []

        # ---- primary observation ----
        qual = "high" if self.confidence > 0.70 else "moderate"
        lines.append(
            f"The user appears to be feeling {self.primary_emotion.value} "
            f"({qual} confidence, {self.confidence:.0%})."
        )

        # ---- trajectory ----
        if self.trajectory == "deteriorating":
            lines.append(
                "Their emotional state has been worsening over the last "
                "few exchanges."
            )
        elif self.trajectory == "improving":
            lines.append(
                "Their emotional state has been improving recently."
            )

        # ---- behavioural guidance ----
        guidance = _RESPONSE_GUIDANCE.get(self.primary_emotion)
        if guidance:
            lines.append(f"Adapt your response: {guidance}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-safe dictionary for sending to the client."""
        return {
            "emotion": self.primary_emotion.value,
            "confidence": round(self.confidence, 2),
            "valence": round(self.valence, 2),
            "arousal": round(self.arousal, 2),
            "trajectory": self.trajectory,
            "face": self.face_emotion.value if self.face_emotion else None,
            "voice": self.voice_emotion.value if self.voice_emotion else None,
        }


# ---------------------------------------------------------------------------
# Response guidance per emotion
# ---------------------------------------------------------------------------

_RESPONSE_GUIDANCE: Dict[Emotion, str] = {
    Emotion.ANGRY: (
        "Be concise and direct. Acknowledge their frustration briefly "
        "without being patronizing. Avoid lengthy preambles."
    ),
    Emotion.FRUSTRATED: (
        "Be concise and direct. Acknowledge their frustration briefly "
        "without being patronizing. Offer to approach the problem "
        "differently if what you tried before isn't working."
    ),
    Emotion.SAD: (
        "Be gentle and supportive. Show genuine empathy. Offer clear "
        "help without being overbearing."
    ),
    Emotion.CONFUSED: (
        "Break your explanation into simpler steps. Use analogies when "
        "appropriate. Ask if they need clarification on any specific part."
    ),
    Emotion.HAPPY: (
        "Match their positive energy. Be warm and enthusiastic while "
        "remaining professional and accurate."
    ),
    Emotion.FEARFUL: (
        "Be reassuring and calm. Provide clear, confident guidance. "
        "Avoid alarming language."
    ),
    Emotion.DISGUSTED: (
        "Be respectful and neutral. Stay factual and avoid "
        "judgmental language."
    ),
    Emotion.SURPRISED: (
        "Provide clear context. The user may not have expected this — "
        "give them a moment and explain calmly."
    ),
}
