"""Tests for live retrieval module.

Tests evidence weakness detection and persistence logic — all offline.
"""

from acheron.models import QueryResult
from acheron.rag.live_retrieval import evidence_is_weak


# ======================================================================
# Evidence weakness detection
# ======================================================================
def _make_results(n: int, score: float = 0.5) -> list[QueryResult]:
    """Create n dummy QueryResults with the given score."""
    return [
        QueryResult(
            text=f"Chunk {i}",
            paper_id=f"paper_{i}",
            paper_title=f"Paper {i}",
            relevance_score=score,
        )
        for i in range(n)
    ]


def test_evidence_weak_few_chunks():
    results = _make_results(1, score=0.8)
    assert evidence_is_weak(results, min_chunks=3, min_score=0.3) is True


def test_evidence_weak_low_score():
    results = _make_results(10, score=0.1)
    assert evidence_is_weak(results, min_chunks=3, min_score=0.35) is True


def test_evidence_strong():
    results = _make_results(10, score=0.8)
    assert evidence_is_weak(results, min_chunks=3, min_score=0.35) is False


def test_evidence_weak_empty():
    assert evidence_is_weak([], min_chunks=3, min_score=0.35) is True


def test_evidence_weak_exact_threshold():
    results = _make_results(3, score=0.35)
    # 3 chunks (not fewer than 3) and score 0.35 (not below 0.35)
    assert evidence_is_weak(results, min_chunks=3, min_score=0.35) is False


def test_evidence_weak_one_below_threshold():
    results = _make_results(2, score=0.5)
    assert evidence_is_weak(results, min_chunks=3, min_score=0.35) is True


def test_evidence_weak_score_just_below():
    results = _make_results(5, score=0.34)
    assert evidence_is_weak(results, min_chunks=3, min_score=0.35) is True
