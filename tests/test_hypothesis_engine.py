"""Tests for the Evidence-Bound Hypothesis Engine.

Tests mode detection, evidence graph parsing, hypothesis parsing (IBE),
next query extraction, uncertainty parsing, and decision mode — all offline.
"""

from acheron.models import ClaimStatus, NexusMode
from acheron.rag.hypothesis_engine import (
    build_engine_result,
    detect_mode,
    get_mode_prompt,
    parse_evidence_graph,
    parse_hypotheses,
    parse_next_queries,
    parse_overall_confidence,
    parse_uncertainty_notes,
    parse_verdict,
    validate_decision_output,
)


# ======================================================================
# Mode detection tests
# ======================================================================
def test_detect_mode_default():
    assert detect_mode("What is Vmem in planaria?") == NexusMode.EVIDENCE


def test_detect_mode_hypothesis_trigger():
    assert detect_mode("Why do planaria regenerate heads?") == NexusMode.HYPOTHESIS
    assert detect_mode("Hypothesize about Vmem role") == NexusMode.HYPOTHESIS
    assert detect_mode("What could explain the regeneration?") == NexusMode.HYPOTHESIS
    assert detect_mode("What mechanism drives head formation?") == NexusMode.HYPOTHESIS


def test_detect_mode_synthesis_trigger():
    assert detect_mode("Design a protocol for Vmem measurement") == NexusMode.SYNTHESIS
    assert detect_mode("Build a threat model for bioelectric tampering") == NexusMode.SYNTHESIS
    assert detect_mode("Propose a system for automated ion measurement") == NexusMode.SYNTHESIS


def test_detect_mode_explicit_overrides():
    assert detect_mode("What is Vmem?", explicit_mode="hypothesis") == NexusMode.HYPOTHESIS
    assert detect_mode("Why does X happen?", explicit_mode="evidence") == NexusMode.EVIDENCE
    assert detect_mode("Design a thing", explicit_mode="evidence") == NexusMode.EVIDENCE


def test_detect_mode_invalid_explicit():
    # Invalid explicit mode falls through to auto-detection
    assert detect_mode("What is Vmem?", explicit_mode="invalid") == NexusMode.EVIDENCE


def test_get_mode_prompt_returns_different_prompts():
    p1 = get_mode_prompt(NexusMode.EVIDENCE)
    p2 = get_mode_prompt(NexusMode.HYPOTHESIS)
    p3 = get_mode_prompt(NexusMode.SYNTHESIS)
    assert "MODE 1" in p1 or "EVIDENCE-GROUNDED" in p1
    assert "MODE 2" in p2 or "HYPOTHESIS" in p2
    assert "MODE 3" in p3 or "SYNTHESIS" in p3
    assert p1 != p2 != p3


# ======================================================================
# Evidence graph parsing tests
# ======================================================================
SAMPLE_EVIDENCE_OUTPUT = """\
EVIDENCE CLAIMS

- CLAIM: Vmem regulates planarian head-tail polarity
  REFS: [1], [3]
  SUPPORT_COUNT: 3
  CONTRADICTION_COUNT: 0
  STATUS: supported
  STUDY_TYPES: primary, review

- CLAIM: Gap junctions mediate bioelectric signaling in regeneration
  REFS: [2], [4]
  SUPPORT_COUNT: 2
  CONTRADICTION_COUNT: 1
  STATUS: mixed
  STUDY_TYPES: primary

- CLAIM: Depolarization blocks head regeneration
  REFS: [1]
  SUPPORT_COUNT: 1
  CONTRADICTION_COUNT: 0
  STATUS: supported
  STUDY_TYPES: primary

CLAIM RELATIONSHIPS
- Vmem supports Gap junctions
- Depolarization contradicts head regeneration

UNCERTAINTY
- No data on specific Vmem values in planarian stem cells
"""


def test_parse_evidence_graph_claims():
    graph = parse_evidence_graph(SAMPLE_EVIDENCE_OUTPUT)
    assert len(graph.claims) == 3
    c0 = graph.claims[0]
    assert "Vmem" in c0.subject or "Vmem" in c0.object or "Vmem" in c0.predicate
    assert c0.status == ClaimStatus.SUPPORTED
    assert c0.support_count == 3
    assert c0.contradiction_count == 0
    assert c0.agreement_score == 1.0
    assert "[1]" in c0.source_refs


def test_parse_evidence_graph_mixed_status():
    graph = parse_evidence_graph(SAMPLE_EVIDENCE_OUTPUT)
    mixed = [c for c in graph.claims if c.status == ClaimStatus.MIXED]
    assert len(mixed) == 1
    assert mixed[0].support_count == 2
    assert mixed[0].contradiction_count == 1


def test_parse_evidence_graph_edges():
    graph = parse_evidence_graph(SAMPLE_EVIDENCE_OUTPUT)
    assert len(graph.edges) >= 1
    relations = [e.relation for e in graph.edges]
    assert "supports" in relations or "contradicts" in relations


def test_parse_evidence_graph_empty():
    graph = parse_evidence_graph("No structured output here.")
    assert len(graph.claims) == 0
    assert len(graph.edges) == 0


def test_evidence_graph_supported_claims():
    graph = parse_evidence_graph(SAMPLE_EVIDENCE_OUTPUT)
    supported = graph.supported_claims()
    assert len(supported) == 2
    for c in supported:
        assert c.status == ClaimStatus.SUPPORTED


def test_evidence_graph_contested_claims():
    graph = parse_evidence_graph(SAMPLE_EVIDENCE_OUTPUT)
    contested = graph.contested_claims()
    assert len(contested) == 1


def test_evidence_graph_summary():
    graph = parse_evidence_graph(SAMPLE_EVIDENCE_OUTPUT)
    summary = graph.to_summary()
    assert "claims" in summary
    assert "supported" in summary


# ======================================================================
# Hypothesis parsing tests (IBE + falsification)
# ======================================================================
SAMPLE_HYPOTHESIS_OUTPUT = """\
HYPOTHESES

HYPOTHESIS: [H1] Vmem gradients encode positional information for anterior-posterior axis
RANK: 1
EXPLANATORY_POWER: 0.8
SIMPLICITY: 0.7
CONSISTENCY: 0.9
MECHANISTIC_PLAUSIBILITY: 0.85
RATIONALE: Explains how head vs tail regeneration decision is made
PREDICTIONS: Vmem values should differ systematically between anterior and posterior
FALSIFIERS: Finding no Vmem gradient across AP axis despite normal regeneration
MINIMAL_TEST: Map Vmem using voltage-sensitive dyes across AP axis of amputated fragments
CONFIDENCE: 72
CONFIDENCE_JUSTIFICATION: Strong evidence from multiple species, planarian-specific values missing
ASSUMPTIONS: A1. Vmem is measured intracellularly; A2. Gap junctions relay positional info
KNOWN_UNKNOWNS: Exact Vmem thresholds; downstream transcription targets
FAILURE_MODES: Vmem may be a readout rather than a cause; other signals may dominate
REFS: [1], [3]

HYPOTHESIS: [H2] Gap junction network topology determines regeneration outcome
RANK: 2
EXPLANATORY_POWER: 0.6
SIMPLICITY: 0.5
CONSISTENCY: 0.7
MECHANISTIC_PLAUSIBILITY: 0.6
RATIONALE: Network structure could carry pattern information
PREDICTIONS: Disrupting specific junctions should alter regeneration outcomes predictably
FALSIFIERS: Random junction disruption producing identical phenotypes to targeted disruption
MINIMAL_TEST: Compare regeneration after targeted vs random gap junction inhibition
CONFIDENCE: 45
CONFIDENCE_JUSTIFICATION: Limited direct evidence for topology vs. general connectivity
ASSUMPTIONS: Gap junctions form organized networks rather than random connections
KNOWN_UNKNOWNS: Network topology in planarian tissue
FAILURE_MODES: Connectivity may be uniform and non-informational
REFS: [2], [4]

UNCERTAINTY
- Lack of single-cell resolution Vmem data in planarians
"""


def test_parse_hypotheses_count():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    assert len(hyps) >= 2


def test_parse_hypothesis_ibe_scores():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    h1 = hyps[0]
    assert h1.explanatory_power == 0.8
    assert h1.simplicity == 0.7
    assert h1.consistency == 0.9
    assert h1.mechanistic_plausibility == 0.85
    assert h1.overall_score > 0


def test_parse_hypothesis_falsification():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    h1 = hyps[0]
    assert len(h1.predictions) >= 1
    assert len(h1.falsifiers) >= 1
    assert h1.minimal_test != ""


def test_parse_hypothesis_confidence():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    h1 = hyps[0]
    assert h1.confidence == 72
    assert h1.confidence_justification != ""


def test_parse_hypothesis_assumptions():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    h1 = hyps[0]
    assert len(h1.assumptions) >= 1


def test_parse_hypothesis_failure_modes():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    h1 = hyps[0]
    assert len(h1.failure_modes) >= 1


def test_parse_hypothesis_refs():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    h1 = hyps[0]
    assert "[1]" in h1.supporting_refs or "[3]" in h1.supporting_refs


def test_parse_hypothesis_known_unknowns():
    hyps = parse_hypotheses(SAMPLE_HYPOTHESIS_OUTPUT)
    h1 = hyps[0]
    assert len(h1.known_unknowns) >= 1


def test_parse_hypotheses_empty():
    hyps = parse_hypotheses("No hypotheses here.")
    assert len(hyps) == 0


# ======================================================================
# Next queries parsing
# ======================================================================
def test_parse_next_queries():
    text = """\
NEXT_QUERIES
- "Vmem gradient planarian regeneration anterior posterior"
- "gap junction topology planarian bioelectric"
- "voltage-sensitive dye planarian Vmem mapping"
"""
    queries = parse_next_queries(text)
    assert len(queries) == 3
    assert "Vmem" in queries[0]


def test_parse_next_queries_empty():
    queries = parse_next_queries("No queries section here.")
    assert len(queries) == 0


# ======================================================================
# Overall confidence parsing
# ======================================================================
def test_parse_overall_confidence():
    text = """\
OVERALL_CONFIDENCE: 65
OVERALL_JUSTIFICATION: Multiple sources support key claims but quantitative data is missing
"""
    conf, justification = parse_overall_confidence(text)
    assert conf == 65
    assert "quantitative" in justification.lower() or "sources" in justification.lower()


def test_parse_overall_confidence_missing():
    conf, justification = parse_overall_confidence("No confidence here.")
    assert conf == 0
    assert justification == ""


# ======================================================================
# Uncertainty parsing
# ======================================================================
def test_parse_uncertainty_notes():
    text = """\
UNCERTAINTY
- No data on specific Vmem values in planarian stem cells
- Conflicting results between planarian and Xenopus models
- Limited temporal resolution in existing studies
"""
    notes = parse_uncertainty_notes(text)
    assert len(notes) == 3
    assert "Vmem" in notes[0]


# ======================================================================
# Full engine result
# ======================================================================
def test_build_engine_result():
    full_output = SAMPLE_EVIDENCE_OUTPUT + "\n" + SAMPLE_HYPOTHESIS_OUTPUT + """\
OVERALL_CONFIDENCE: 60
OVERALL_JUSTIFICATION: Moderate evidence

NEXT_QUERIES
- "planarian Vmem single cell"
"""
    result = build_engine_result(
        raw_output=full_output,
        query="test query",
        mode=NexusMode.HYPOTHESIS,
        sources=[],
        total_searched=10,
        model_used="test-model",
        live_sources_fetched=5,
    )
    assert result.query == "test query"
    assert result.mode == NexusMode.HYPOTHESIS
    assert len(result.evidence_graph.claims) >= 3
    assert len(result.hypotheses) >= 2
    # Hypotheses should be sorted by overall_score descending
    if len(result.hypotheses) >= 2:
        assert result.hypotheses[0].overall_score >= result.hypotheses[1].overall_score
    assert result.confidence == 60
    assert result.live_sources_fetched == 5
    assert len(result.next_queries) >= 1


# ======================================================================
# Decision mode detection tests
# ======================================================================
def test_detect_mode_decision_triggers():
    assert detect_mode("Should I use planarian substrate?") == NexusMode.DECISION
    assert detect_mode("Is this viable for bioelectric memory?") == NexusMode.DECISION
    assert detect_mode("Is it feasible to store data in Vmem?") == NexusMode.DECISION
    assert detect_mode("Give me a yes or no verdict") == NexusMode.DECISION
    assert detect_mode("Can this work with innexin channels?") == NexusMode.DECISION
    assert detect_mode("Go/no-go on planarian biocomputation") == NexusMode.DECISION
    assert detect_mode("Should we switch substrate to Xenopus?") == NexusMode.DECISION


def test_detect_mode_decision_explicit():
    assert detect_mode("What is Vmem?", explicit_mode="decision") == NexusMode.DECISION
    result = detect_mode("Why do planaria regenerate?", explicit_mode="decision")
    assert result == NexusMode.DECISION


def test_detect_mode_decision_priority_over_hypothesis():
    """Decision triggers should take priority over hypothesis triggers."""
    # "should I" is decision, even though "what if" might match hypothesis
    assert detect_mode("Should I hypothesize about Vmem?") == NexusMode.DECISION


def test_get_mode_prompt_decision():
    prompt = get_mode_prompt(NexusMode.DECISION)
    assert "VERDICT" in prompt
    assert "MODE 4" in prompt or "ENGINEERING VERDICT" in prompt
    # Should be distinct from other modes
    assert prompt != get_mode_prompt(NexusMode.EVIDENCE)
    assert prompt != get_mode_prompt(NexusMode.HYPOTHESIS)
    assert prompt != get_mode_prompt(NexusMode.SYNTHESIS)


# ======================================================================
# Verdict parsing tests
# ======================================================================
SAMPLE_DECISION_OUTPUT = """\
VERDICT: YES — planarian bioelectric memory is viable for Phase-0 testing.

CONFIDENCE: 68 — sufficient evidence from Levin lab studies on Vmem-encoded \
anterior-posterior polarity, though planarian-specific T_hold is unmeasured.

RATIONALE:
Multiple independent studies demonstrate that Vmem gradients in planaria encode \
positional information that persists through regeneration [EVIDENCE] [1]. \
Gap junction (innexin) connectivity patterns survive decapitation and guide \
head-vs-tail fate decisions [EVIDENCE] [2]. The BIGR framework maps \
naturally: innexin connectivity = SSD (non-volatile), Vmem gradient = RAM \
(volatile read/write) [INFERENCE]. However, T_hold and BER remain unmeasured \
for planarian gap junctions specifically [DATA GAP].

KEY EVIDENCE:
- Vmem gradients encode anterior-posterior polarity in planaria [EVIDENCE] [1]
- Innexin-6 knockout produces two-headed phenotype [EVIDENCE] [2]
- Gap junction connectivity survives regeneration cycles [EVIDENCE] [3]

KEY UNKNOWNS:
- T_hold for planarian Vmem states (requires patch clamp)
- BER for innexin-mediated signaling (requires single-channel recording)

ACTION:
Run a Phase-0 voltage-sensitive dye (DiBAC) imaging experiment on \
amputated planaria to establish baseline Vmem maps. Cost: ~$200. Timeline: 1 week.

KILL CRITERIA:
If patch-clamp measurement yields T_hold < 100 ms, abandon planarian Vmem \
as information storage substrate and pivot to innexin connectivity (SSD-analog).

PIVOT:
If planarian Vmem proves too volatile, switch to Physarum polycephalum \
tube-network conductance as the storage substrate — demonstrated stable \
memory traces over 24+ hours [3].
"""


def test_parse_verdict_yes():
    verdict, rationale = parse_verdict(SAMPLE_DECISION_OUTPUT)
    assert verdict == "YES"
    assert "planarian" in rationale.lower() or len(rationale) > 0


def test_parse_verdict_no():
    text = "VERDICT: NO — insufficient evidence for bioelectric data storage."
    verdict, rationale = parse_verdict(text)
    assert verdict == "NO"


def test_parse_verdict_conditional():
    text = "VERDICT: CONDITIONAL — viable if T_hold > 500 ms."
    verdict, rationale = parse_verdict(text)
    assert verdict == "CONDITIONAL"


def test_parse_verdict_missing():
    verdict, rationale = parse_verdict("No verdict here, just a report.")
    assert verdict == ""
    assert rationale == ""


def test_validate_decision_output_good():
    warnings = validate_decision_output(SAMPLE_DECISION_OUTPUT)
    assert len(warnings) == 0


def test_validate_decision_output_no_verdict():
    bad_output = "Evidence Extracted\n- Some fact [EVIDENCE]\nHypothesis\n..."
    warnings = validate_decision_output(bad_output)
    assert any("VERDICT" in w for w in warnings)


def test_validate_decision_output_no_kill_criteria():
    output = "VERDICT: YES\nSome rationale."
    warnings = validate_decision_output(output)
    assert any("KILL CRITERIA" in w for w in warnings)


def test_build_engine_result_decision_mode():
    result = build_engine_result(
        raw_output=SAMPLE_DECISION_OUTPUT,
        query="Should I use planarian substrate?",
        mode=NexusMode.DECISION,
        sources=[],
        total_searched=10,
        model_used="test-model",
    )
    assert result.mode == NexusMode.DECISION
    # Good decision output should have no validation warnings
    validation_warnings = [
        n for n in result.uncertainty_notes if n.startswith("[VALIDATION]")
    ]
    assert len(validation_warnings) == 0


def test_build_engine_result_decision_mode_bad_output():
    """Decision mode with report-style output should produce validation warnings."""
    report_output = """\
EVIDENCE EXTRACTED
- Vmem gradients encode polarity [EVIDENCE] [1]

HYPOTHESIS
Vmem gradients store positional memory...

UNCERTAINTY
- T_hold is unknown
"""
    result = build_engine_result(
        raw_output=report_output,
        query="Should I use planarian substrate?",
        mode=NexusMode.DECISION,
        sources=[],
        total_searched=10,
        model_used="test-model",
    )
    assert result.mode == NexusMode.DECISION
    validation_warnings = [
        n for n in result.uncertainty_notes if n.startswith("[VALIDATION]")
    ]
    # Should have at least a warning about missing VERDICT
    assert len(validation_warnings) >= 1
