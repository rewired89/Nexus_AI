"""Avatar state management for the Nexus kiosk display.

The actual 3D/2D rendering happens in the browser via Three.js (for
Ready Player Me GLB models) or a 2D canvas fallback.  This module
manages the *state machine* that drives the frontend:

    IDLE → LISTENING → THINKING → SPEAKING → IDLE

Each state carries animation parameters that the frontend consumes
via the WebSocket message stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AvatarState(str, Enum):
    """Visual states of the avatar."""

    IDLE = "idle"          # Ambient breathing / subtle motion
    LISTENING = "listening"  # Microphone active, ears "perked"
    THINKING = "thinking"    # Processing query, loading indicator
    SPEAKING = "speaking"    # Delivering response, mouth animated
    ERROR = "error"        # Something went wrong


@dataclass
class AnimationParams:
    """Parameters sent to the frontend for the current animation frame.

    The frontend maps these to Three.js morph targets or CSS keyframes
    depending on the renderer mode (3D vs 2D).
    """

    state: AvatarState
    mouth_open: float = 0.0    # 0.0 – 1.0 (lip-sync amplitude)
    blink: bool = False        # trigger a blink on this frame
    glow_intensity: float = 0.0  # 0.0 – 1.0 (thinking indicator glow)
    head_nod: float = 0.0      # -1.0 – 1.0 (listening acknowledgment)
    transition_ms: int = 300    # CSS/GSAP transition duration

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "mouth_open": round(self.mouth_open, 3),
            "blink": self.blink,
            "glow_intensity": round(self.glow_intensity, 3),
            "head_nod": round(self.head_nod, 3),
            "transition_ms": self.transition_ms,
        }


@dataclass
class LipSyncTimeline:
    """Pre-computed amplitude envelope for a spoken response."""

    frames: list[float] = field(default_factory=list)  # amplitude per frame
    frame_duration_ms: float = 50.0
    _start_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        return len(self.frames) * self.frame_duration_ms

    def start(self) -> None:
        self._start_time = time.time()

    def amplitude_at(self, elapsed_ms: float) -> float:
        """Get the amplitude at a given elapsed time."""
        idx = int(elapsed_ms / self.frame_duration_ms)
        if 0 <= idx < len(self.frames):
            return self.frames[idx]
        return 0.0

    def current_amplitude(self) -> float:
        """Get amplitude based on wall-clock time since start()."""
        elapsed = (time.time() - self._start_time) * 1000
        return self.amplitude_at(elapsed)

    @property
    def finished(self) -> bool:
        if self._start_time == 0:
            return True
        elapsed = (time.time() - self._start_time) * 1000
        return elapsed >= self.duration_ms


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

# Valid transitions.
_TRANSITIONS: dict[AvatarState, set[AvatarState]] = {
    AvatarState.IDLE: {AvatarState.LISTENING, AvatarState.THINKING, AvatarState.ERROR},
    AvatarState.LISTENING: {AvatarState.THINKING, AvatarState.IDLE, AvatarState.ERROR},
    AvatarState.THINKING: {AvatarState.SPEAKING, AvatarState.IDLE, AvatarState.ERROR},
    AvatarState.SPEAKING: {AvatarState.IDLE, AvatarState.LISTENING, AvatarState.ERROR},
    AvatarState.ERROR: {AvatarState.IDLE},
}


class AvatarController:
    """Manages avatar state transitions and animation parameters.

    The WebSocket server reads ``current_params()`` each tick and
    sends the result to the frontend.
    """

    def __init__(self) -> None:
        self._state = AvatarState.IDLE
        self._lip_sync: Optional[LipSyncTimeline] = None
        self._last_blink: float = 0.0
        self._blink_interval: float = 4.0  # seconds between blinks
        self._thinking_start: float = 0.0

    @property
    def state(self) -> AvatarState:
        return self._state

    def transition(self, target: AvatarState) -> bool:
        """Attempt a state transition.  Returns True if successful."""
        if target == self._state:
            return True
        if target not in _TRANSITIONS.get(self._state, set()):
            return False
        self._state = target
        if target == AvatarState.THINKING:
            self._thinking_start = time.time()
        if target == AvatarState.IDLE:
            self._lip_sync = None
            self._thinking_start = 0.0
        return True

    def set_lip_sync(self, timeline: LipSyncTimeline) -> None:
        """Attach a lip-sync timeline and start playback."""
        self._lip_sync = timeline
        self._lip_sync.start()

    def current_params(self) -> AnimationParams:
        """Compute the current animation parameters."""
        now = time.time()

        # Blink logic — periodic, state-independent.
        blink = False
        if now - self._last_blink >= self._blink_interval:
            blink = True
            self._last_blink = now

        if self._state == AvatarState.SPEAKING and self._lip_sync:
            amp = self._lip_sync.current_amplitude()
            if self._lip_sync.finished:
                self.transition(AvatarState.IDLE)
                return AnimationParams(
                    state=AvatarState.IDLE, blink=blink, transition_ms=500
                )
            return AnimationParams(
                state=AvatarState.SPEAKING,
                mouth_open=amp,
                blink=blink,
                transition_ms=50,  # fast updates for lip-sync
            )

        if self._state == AvatarState.THINKING:
            # Pulsing glow that intensifies over time (capped at 1.0).
            elapsed = now - self._thinking_start
            import math
            glow = min(1.0, 0.3 + 0.4 * abs(math.sin(elapsed * 1.5)))
            return AnimationParams(
                state=AvatarState.THINKING,
                glow_intensity=glow,
                blink=blink,
                transition_ms=200,
            )

        if self._state == AvatarState.LISTENING:
            return AnimationParams(
                state=AvatarState.LISTENING,
                head_nod=0.2,  # subtle acknowledgment pose
                blink=blink,
                transition_ms=300,
            )

        if self._state == AvatarState.ERROR:
            return AnimationParams(
                state=AvatarState.ERROR,
                glow_intensity=0.8,
                blink=blink,
                transition_ms=500,
            )

        # IDLE
        return AnimationParams(
            state=AvatarState.IDLE,
            blink=blink,
            transition_ms=500,
        )
