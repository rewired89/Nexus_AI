"""Tests for the experiment ledger."""

import tempfile
from pathlib import Path

from acheron.models import DiscoveryResult, Hypothesis, StructuredVariable
from acheron.rag.ledger import ExperimentLedger


def _make_discovery() -> DiscoveryResult:
    return DiscoveryResult(
        query="What voltage gradients drive planarian head regeneration?",
        evidence=["Vmem gradients of -45mV establish AP polarity [1]."],
        inference=["Gap junctions likely propagate these signals [1][2]."],
        speculation=["Bioelectric patterns may encode morphological memory."],
        variables=[
            StructuredVariable(name="Vmem", value="-45", unit="mV", source_ref="[1]"),
        ],
        hypotheses=[
            Hypothesis(
                statement="Depolarization triggers blastema formation",
                supporting_refs=["[1]"],
                confidence="medium",
                validation_strategy="Re-analyze PhysioNet EEG datasets",
            ),
        ],
        uncertainty_notes=["No time-course data for Vmem changes in first 6h."],
        model_used="gpt-4o",
        total_chunks_searched=50,
    )


def test_ledger_record_and_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ExperimentLedger(ledger_dir=Path(tmpdir))
        assert ledger.count() == 0

        discovery = _make_discovery()
        entry = ledger.record(discovery, notes="Test run", tags=["planarian", "voltage"])

        assert entry.entry_id.startswith("ledger-")
        assert entry.query == discovery.query
        assert len(entry.hypotheses) == 1
        assert entry.tags == ["planarian", "voltage"]

        entries = ledger.list_entries()
        assert len(entries) == 1
        assert entries[0].entry_id == entry.entry_id


def test_ledger_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ExperimentLedger(ledger_dir=Path(tmpdir))
        ledger.record(_make_discovery(), notes="planarian experiment")
        ledger.record(
            DiscoveryResult(query="EEG oscillation patterns", model_used="test"),
            notes="eeg analysis",
        )

        planarian_results = ledger.search_entries("planarian")
        assert len(planarian_results) == 1

        eeg_results = ledger.search_entries("EEG")
        assert len(eeg_results) == 1


def test_ledger_filter_by_tag():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ExperimentLedger(ledger_dir=Path(tmpdir))
        ledger.record(_make_discovery(), tags=["planarian"])
        ledger.record(
            DiscoveryResult(query="EEG patterns", model_used="test"),
            tags=["eeg"],
        )

        planarian_entries = ledger.list_entries(tag="planarian")
        assert len(planarian_entries) == 1

        eeg_entries = ledger.list_entries(tag="eeg")
        assert len(eeg_entries) == 1

        all_entries = ledger.list_entries()
        assert len(all_entries) == 2


def test_ledger_get_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ExperimentLedger(ledger_dir=Path(tmpdir))
        entry = ledger.record(_make_discovery())

        retrieved = ledger.get_entry(entry.entry_id)
        assert retrieved is not None
        assert retrieved.entry_id == entry.entry_id
        assert retrieved.query == entry.query

        missing = ledger.get_entry("nonexistent-id")
        assert missing is None


def test_ledger_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = ExperimentLedger(ledger_dir=Path(tmpdir))
        ledger.record(_make_discovery())
        ledger.record(_make_discovery())

        export_path = Path(tmpdir) / "export.json"
        count = ledger.export_all(export_path)
        assert count == 2
        assert export_path.exists()
