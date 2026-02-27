"""Nexus — Predictive behavioral context and session memory.

Nexus sits between the voice pipeline and Acheron's RAG engine.  It
maintains longitudinal project state, tracks experiment outcomes, and
enriches each query with relevant prior context before it hits the
pipeline.  This is *not* a separate service — it is a module inside the
interface layer.

Persistence is via a JSON file so state survives restarts without
requiring a database.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """Single query/response pair."""

    timestamp: float
    query: str
    mode: str  # evidence, hypothesis, synthesis, decision, tutor
    response_summary: str  # first ~300 chars of the answer
    topics: list[str] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    """Logged experiment outcome from the ledger or manual entry."""

    timestamp: float
    title: str
    hypothesis: str
    outcome: str  # PASS, FAIL, INCONCLUSIVE
    notes: str = ""


@dataclass
class NexusContext:
    """Enrichment payload injected into the pipeline prompt."""

    recent_topics: list[str]
    active_hypotheses: list[str]
    experiment_outcomes: list[str]
    session_summary: str

    def to_prompt_block(self) -> str:
        """Format as a context block for the LLM system prompt."""
        lines = ["## Session Context (Nexus)"]
        if self.session_summary:
            lines.append(f"Summary: {self.session_summary}")
        if self.recent_topics:
            lines.append(f"Recent topics: {', '.join(self.recent_topics)}")
        if self.active_hypotheses:
            lines.append("Active hypotheses:")
            for h in self.active_hypotheses:
                lines.append(f"  - {h}")
        if self.experiment_outcomes:
            lines.append("Recent experiment outcomes:")
            for o in self.experiment_outcomes:
                lines.append(f"  - {o}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------

_MAX_HISTORY = 200  # max conversation turns to keep
_CONTEXT_WINDOW = 20  # turns to consider for enrichment
_TOPIC_DEDUP_WINDOW = 50  # recent topics to de-duplicate


class SessionMemory:
    """Persistent session memory backed by a JSON file.

    Parameters
    ----------
    path : Path or str
        Location of the JSON persistence file.  Created on first save
        if it does not exist.
    """

    def __init__(self, path: Path | str = "data/nexus_session.json") -> None:
        self.path = Path(path)
        self.history: list[ConversationTurn] = []
        self.experiments: list[ExperimentRecord] = []
        self.project_notes: list[str] = []
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            self.history = [
                ConversationTurn(**t) for t in raw.get("history", [])
            ]
            self.experiments = [
                ExperimentRecord(**e) for e in raw.get("experiments", [])
            ]
            self.project_notes = raw.get("project_notes", [])
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupted file — start fresh rather than crash.
            self.history = []
            self.experiments = []
            self.project_notes = []

    def save(self) -> None:
        """Persist current state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "history": [asdict(t) for t in self.history[-_MAX_HISTORY:]],
            "experiments": [asdict(e) for e in self.experiments],
            "project_notes": self.project_notes,
        }
        self.path.write_text(json.dumps(payload, indent=2))

    # -- recording ----------------------------------------------------------

    def record_turn(
        self,
        query: str,
        mode: str,
        response_summary: str,
        topics: Optional[list[str]] = None,
    ) -> None:
        """Record a completed query/response exchange."""
        self.history.append(
            ConversationTurn(
                timestamp=time.time(),
                query=query,
                mode=mode,
                response_summary=response_summary[:300],
                topics=topics or [],
            )
        )
        # Trim oldest if over limit.
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_MAX_HISTORY:]
        self.save()

    def record_experiment(
        self,
        title: str,
        hypothesis: str,
        outcome: str,
        notes: str = "",
    ) -> None:
        """Log an experiment result."""
        self.experiments.append(
            ExperimentRecord(
                timestamp=time.time(),
                title=title,
                hypothesis=hypothesis,
                outcome=outcome,
                notes=notes,
            )
        )
        self.save()

    def add_note(self, note: str) -> None:
        """Add a free-form project note."""
        self.project_notes.append(note)
        self.save()

    # -- context enrichment -------------------------------------------------

    def build_context(self) -> NexusContext:
        """Build the enrichment payload from recent session state."""
        recent = self.history[-_CONTEXT_WINDOW:]

        # Collect unique recent topics, preserving order.
        seen: set[str] = set()
        recent_topics: list[str] = []
        for turn in reversed(recent):
            for topic in turn.topics:
                t_lower = topic.lower()
                if t_lower not in seen:
                    seen.add(t_lower)
                    recent_topics.append(topic)
                if len(recent_topics) >= 10:
                    break
            if len(recent_topics) >= 10:
                break

        # Active hypotheses: most recent hypothesis-mode answers.
        active_hypotheses: list[str] = []
        for turn in reversed(recent):
            if turn.mode == "hypothesis" and turn.response_summary:
                active_hypotheses.append(turn.response_summary)
            if len(active_hypotheses) >= 3:
                break

        # Experiment outcomes: last 5.
        experiment_outcomes = [
            f"{e.title}: {e.outcome}" + (f" — {e.notes}" if e.notes else "")
            for e in self.experiments[-5:]
        ]

        # Session summary: what the user has been working on.
        if recent:
            modes = [t.mode for t in recent]
            mode_counts = {m: modes.count(m) for m in set(modes)}
            dominant = max(mode_counts, key=mode_counts.get)  # type: ignore[arg-type]
            session_summary = (
                f"{len(recent)} recent queries, primarily {dominant} mode. "
                f"Topics: {', '.join(recent_topics[:5]) or 'general'}."
            )
        else:
            session_summary = "New session — no prior context."

        return NexusContext(
            recent_topics=recent_topics,
            active_hypotheses=active_hypotheses,
            experiment_outcomes=experiment_outcomes,
            session_summary=session_summary,
        )

    def enrich_query(self, query: str) -> str:
        """Prepend session context to a query for the pipeline.

        Returns the original query with a context preamble that the LLM
        can use to provide continuity across the session.
        """
        ctx = self.build_context()
        if ctx.session_summary == "New session — no prior context.":
            return query
        block = ctx.to_prompt_block()
        return f"{block}\n\n---\nUser query: {query}"

    # -- utilities ----------------------------------------------------------

    def clear(self) -> None:
        """Reset all session state."""
        self.history.clear()
        self.experiments.clear()
        self.project_notes.clear()
        self.save()

    @property
    def turn_count(self) -> int:
        return len(self.history)
