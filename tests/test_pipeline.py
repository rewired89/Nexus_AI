"""Tests for RAG pipeline parsing and discovery logic."""

from acheron.rag.pipeline import (
    RAGPipeline,
    _try_parse_hypothesis,
    _try_parse_variable,
)


def test_parse_epistemic_sections():
    text = """
EVIDENCE
- Planarian regeneration involves bioelectric signals [1].
- Vmem gradients establish anterior-posterior polarity [2].

INFERENCE
- Gap junction connectivity likely encodes morphological targets [1][2].

SPECULATION
- Bioelectric patterns may serve as a computational substrate (medium confidence).

UNCERTAINTY
- No data on ion channel expression kinetics during first 6 hours.
"""
    evidence, inference, speculation = RAGPipeline._parse_epistemic_sections(text)
    assert len(evidence) == 2
    assert "bioelectric signals" in evidence[0]
    assert len(inference) == 1
    assert "Gap junction" in inference[0]
    assert len(speculation) == 1
    assert "computational substrate" in speculation[0]


def test_parse_epistemic_sections_markdown_headers():
    text = """
## EVIDENCE
- Fact one [1].

## INFERENCE
- Derived conclusion [1][2].

## SPECULATION
- Guess [low confidence].
"""
    evidence, inference, speculation = RAGPipeline._parse_epistemic_sections(text)
    assert len(evidence) >= 1
    assert len(inference) >= 1
    assert len(speculation) >= 1


def test_try_parse_variable_full():
    var = _try_parse_variable("Vmem=-45 (mV) [1]")
    assert var is not None
    assert var.name == "Vmem"
    assert var.value == "-45"
    assert var.unit == "mV"
    assert var.source_ref == "[1]"


def test_try_parse_variable_no_unit():
    var = _try_parse_variable("organism=planaria [2]")
    assert var is not None
    assert var.name == "organism"
    assert var.value == "planaria"
    assert var.unit == ""
    assert var.source_ref == "[2]"


def test_try_parse_variable_no_source():
    var = _try_parse_variable("conductance=12 (pS)")
    assert var is not None
    assert var.name == "conductance"
    assert var.value == "12"
    assert var.unit == "pS"


def test_try_parse_variable_not_a_variable():
    result = _try_parse_variable("This is just a sentence.")
    assert result is None


def test_try_parse_hypothesis_high_confidence():
    hyp = _try_parse_hypothesis(
        "Depolarization of wound-adjacent cells triggers regeneration (high confidence) [1][3]."
    )
    assert hyp is not None
    assert hyp.confidence == "high"
    assert "[1]" in hyp.supporting_refs
    assert "[3]" in hyp.supporting_refs


def test_try_parse_hypothesis_medium():
    hyp = _try_parse_hypothesis(
        "Gap junction blockers may impair anterior-posterior patterning (medium confidence)."
    )
    assert hyp is not None
    assert hyp.confidence == "medium"


def test_try_parse_hypothesis_default_low():
    hyp = _try_parse_hypothesis("Ion channels could store positional information.")
    assert hyp is not None
    assert hyp.confidence == "low"


def test_try_parse_hypothesis_too_short():
    result = _try_parse_hypothesis("Short.")
    assert result is None
