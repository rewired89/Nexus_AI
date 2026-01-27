"""Experiment ledger — persistent log of discovery loop findings."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from acheron.config import get_settings
from acheron.models import DiscoveryResult, Hypothesis, LedgerEntry, StructuredVariable

logger = logging.getLogger(__name__)


class ExperimentLedger:
    """Append-only ledger for recording discovery loop outputs.

    Stores entries as individual JSON files in a dedicated directory,
    enabling both programmatic access and human review.
    """

    def __init__(self, ledger_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.ledger_dir = ledger_dir or settings.data_dir / "ledger"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        discovery: DiscoveryResult,
        notes: str = "",
        tags: Optional[list[str]] = None,
    ) -> LedgerEntry:
        """Record a discovery result as a ledger entry."""
        entry_id = f"ledger-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        entry = LedgerEntry(
            entry_id=entry_id,
            query=discovery.query,
            evidence_summary="\n".join(discovery.evidence) if discovery.evidence else "",
            hypotheses=discovery.hypotheses,
            variables=discovery.variables,
            source_ids=[s.paper_id for s in discovery.sources],
            notes=notes,
            tags=tags or [],
        )

        dest = self.ledger_dir / f"{entry_id}.json"
        dest.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Ledger entry recorded: %s", entry_id)
        return entry

    def list_entries(self, tag: Optional[str] = None) -> list[LedgerEntry]:
        """List all ledger entries, optionally filtered by tag."""
        entries = []
        for path in sorted(self.ledger_dir.glob("ledger-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entry = LedgerEntry(**data)
                if tag and tag not in entry.tags:
                    continue
                entries.append(entry)
            except Exception:
                logger.debug("Skipping malformed ledger entry: %s", path.name)
        return entries

    def get_entry(self, entry_id: str) -> Optional[LedgerEntry]:
        """Retrieve a single ledger entry by ID."""
        path = self.ledger_dir / f"{entry_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LedgerEntry(**data)

    def search_entries(self, keyword: str) -> list[LedgerEntry]:
        """Search ledger entries by keyword in query or evidence summary."""
        keyword_lower = keyword.lower()
        results = []
        for entry in self.list_entries():
            if (
                keyword_lower in entry.query.lower()
                or keyword_lower in entry.evidence_summary.lower()
                or keyword_lower in entry.notes.lower()
            ):
                results.append(entry)
        return results

    def count(self) -> int:
        """Return the number of ledger entries."""
        return len(list(self.ledger_dir.glob("ledger-*.json")))

    def export_all(self, dest: Path) -> int:
        """Export all ledger entries to a single JSON file."""
        entries = self.list_entries()
        data = [e.model_dump(mode="json") for e in entries]
        dest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return len(data)
