"""Tests for Science-First Mode modules.

Covers:
  - query_parser: entity extraction, measurement detection, constraint parsing
  - science_filter: evidence scoring function, organism matching, ranking
  - experiment_designer: template experiments, experiment selection
  - config: science-first flags
"""

from __future__ import annotations

from acheron.models import QueryResult


# ======================================================================
# Query Parser tests
# ======================================================================
class TestQueryParser:
    """Tests for Stage A — Query Understanding."""

    def test_parse_planarian_query(self):
        from acheron.rag.query_parser import parse_query

        parsed = parse_query(
            "How does Vmem change in planarian neoblasts after amputation?"
        )
        assert "planarian" in parsed.species
        assert "vmem" in parsed.required_measurements
        assert "regeneration" in parsed.required_measurements
        assert "neoblast" in parsed.tissues

    def test_parse_cross_species(self):
        from acheron.rag.query_parser import parse_query

        parsed = parse_query(
            "Compare gap junction signaling across species in regeneration"
        )
        assert parsed.cross_species_allowed is True
        assert "gj" in parsed.required_measurements

    def test_parse_gene_entities(self):
        from acheron.rag.query_parser import parse_query

        parsed = parse_query("What role does Wnt and beta-catenin play?")
        assert any("wnt" in g for g in parsed.genes)
        assert any("beta" in g for g in parsed.genes)

    def test_parse_no_entities(self):
        from acheron.rag.query_parser import parse_query

        parsed = parse_query("What is bioelectricity?")
        assert parsed.species == []
        assert parsed.organism_constraint == "planarian"  # default

    def test_generate_collection_queries(self):
        from acheron.rag.query_parser import generate_collection_queries, parse_query

        parsed = parse_query("planarian Vmem regeneration gap junction")
        queries = generate_collection_queries(parsed)
        assert len(queries) >= 2
        assert all(isinstance(q, str) for q in queries)
        assert any("planarian" in q.lower() for q in queries)

    def test_summary(self):
        from acheron.rag.query_parser import parse_query

        parsed = parse_query("planarian Vmem measurement")
        summary = parsed.summary()
        assert "planarian" in summary
        assert "vmem" in summary


# ======================================================================
# Science Filter tests
# ======================================================================
def _make_result(
    text: str = "",
    title: str = "",
    doi: str = "",
    pmid: str = "",
    pmcid: str = "",
    authors: list[str] | None = None,
) -> QueryResult:
    """Helper to build a QueryResult for testing."""
    return QueryResult(
        text=text,
        paper_id=f"test:{pmid or 'none'}",
        paper_title=title,
        authors=authors or [],
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        relevance_score=0.8,
    )


class TestEvidenceScoring:
    """Tests for Stage C — Evidence Filtering."""

    def test_organism_match_planarian(self):
        from acheron.rag.science_filter import score_organism_match

        assert score_organism_match("planarian neoblast Vmem") == 1.0
        assert score_organism_match("Schmidtea mediterranea regeneration") == 1.0

    def test_organism_match_xenopus(self):
        from acheron.rag.science_filter import score_organism_match

        assert score_organism_match("Xenopus laevis tadpole tail") == 0.4

    def test_organism_match_vertebrate(self):
        from acheron.rag.science_filter import score_organism_match

        assert score_organism_match("mouse embryonic fibroblast") == 0.2

    def test_organism_match_no_organism(self):
        from acheron.rag.science_filter import score_organism_match

        assert score_organism_match("bioelectric signaling overview") == 0.0

    def test_primary_data_detection(self):
        from acheron.rag.science_filter import score_primary_data

        primary = score_primary_data(
            "We measured membrane potential using patch clamp. "
            "Figure 3 shows the results (p < 0.01, n = 15)."
        )
        assert primary >= 0.7

    def test_review_detection(self):
        from acheron.rag.science_filter import score_primary_data

        review = score_primary_data(
            "This review summarizes recent advances in bioelectric signaling."
        )
        assert review <= 0.5

    def test_measurement_specificity_vmem(self):
        from acheron.rag.science_filter import score_measurement_specificity

        score = score_measurement_specificity(
            "Resting potential was -45 mV in neoblasts, measured via "
            "patch clamp. Gap junction conductance was 2.3 nS."
        )
        assert score >= 0.7

    def test_measurement_specificity_qualitative(self):
        from acheron.rag.science_filter import score_measurement_specificity

        score = score_measurement_specificity(
            "Depolarization of the membrane potential was observed."
        )
        assert 0.2 <= score <= 0.6

    def test_citation_quality(self):
        from acheron.rag.science_filter import score_citation_quality

        result = _make_result(
            doi="10.1234/test", pmid="12345678", pmcid="PMC123",
            authors=["Smith A", "Jones B"],
            title="A test paper about planarians",
        )
        score = score_citation_quality(result)
        assert score >= 0.8

    def test_composite_score(self):
        from acheron.rag.science_filter import score_evidence

        result = _make_result(
            text="We measured planarian neoblast Vmem at -42 mV using "
                 "patch clamp (n=20, p<0.01). Figure 2 shows results.",
            title="Bioelectric control in S. mediterranea 2023",
            doi="10.1234/test",
            pmid="12345678",
        )
        score = score_evidence(result, target_organism="planarian")
        assert score.organism_match == 1.0
        assert score.primary_data >= 0.7
        assert score.total >= 0.5

    def test_rank_and_filter(self):
        from acheron.rag.science_filter import rank_and_filter

        results = [
            _make_result(
                text="planarian neoblast Vmem was -42 mV",
                title="Planarian electrophysiology 2023",
                pmid="111",
            ),
            _make_result(
                text="mouse fibroblast culture voltage",
                title="Mouse cell voltage 2020",
                pmid="222",
            ),
            _make_result(
                text="Xenopus tadpole bioelectric pattern",
                title="Xenopus V-ATPase 2022",
                pmid="333",
            ),
        ]
        ranked = rank_and_filter(
            results, target_organism="planarian", top_k=3, strict_organism=False
        )
        assert len(ranked) >= 1
        # Planarian result should rank first
        assert "planarian" in ranked[0].result.text.lower()

    def test_strict_organism_filter(self):
        from acheron.rag.science_filter import rank_and_filter

        results = [
            _make_result(
                text="mouse fibroblast voltage signaling",
                title="Mouse cell 2020",
            ),
        ]
        ranked = rank_and_filter(
            results, target_organism="planarian", strict_organism=True
        )
        # Mouse result should be filtered out in strict mode
        assert len(ranked) == 0

    def test_contradiction_check(self):
        from acheron.rag.science_filter import (
            ScoredResult,
            check_contradictions,
            score_evidence,
        )

        r1 = _make_result(text="depolarization drives head regeneration")
        r2 = _make_result(text="hyperpolarization is required for head formation")
        scored = [
            ScoredResult(result=r1, score=score_evidence(r1)),
            ScoredResult(result=r2, score=score_evidence(r2)),
        ]
        contradictions = check_contradictions(scored)
        assert len(contradictions) == 1
        assert "conflict" in contradictions[0][2].lower()


# ======================================================================
# Experiment Designer tests
# ======================================================================
class TestExperimentDesigner:
    """Tests for experiment template generation."""

    def test_vmem_experiment(self):
        from acheron.rag.experiment_designer import vmem_imaging_post_amputation

        exp = vmem_imaging_post_amputation()
        assert "Vmem" in exp.title
        assert len(exp.steps) == 6
        assert len(exp.materials) >= 5
        assert len(exp.expected_outcomes) >= 3
        assert len(exp.failure_modes) >= 3
        assert exp.organism == "Schmidtea mediterranea (asexual CIW4)"

    def test_gap_junction_experiment(self):
        from acheron.rag.experiment_designer import gap_junction_modulation_regeneration

        exp = gap_junction_modulation_regeneration()
        assert "Gap Junction" in exp.title
        assert len(exp.steps) == 6
        assert "octanol" in exp.materials[1].lower()

    def test_propose_experiment_vmem(self):
        from acheron.rag.experiment_designer import propose_experiment

        exp = propose_experiment(["vmem", "regeneration"])
        assert exp is not None
        assert "Vmem" in exp.title

    def test_propose_experiment_gj(self):
        from acheron.rag.experiment_designer import propose_experiment

        exp = propose_experiment(["gj"])
        assert exp is not None
        assert "Gap Junction" in exp.title

    def test_propose_experiment_no_match(self):
        from acheron.rag.experiment_designer import propose_experiment

        exp = propose_experiment(["unrelated_measurement"])
        assert exp is None

    def test_experiment_to_text(self):
        from acheron.rag.experiment_designer import vmem_imaging_post_amputation

        exp = vmem_imaging_post_amputation()
        text = exp.to_text()
        assert "MINIMAL WET-LAB TEST" in text
        assert "MATERIALS:" in text
        assert "PROTOCOL:" in text
        assert "Step 1:" in text
        assert "EXPECTED OUTCOMES:" in text
        assert "FAILURE MODES:" in text


# ======================================================================
# Config flag tests
# ======================================================================
class TestScienceFirstConfig:
    """Tests for Science-First configuration flags."""

    def test_default_flags(self):
        from acheron.config import Settings

        settings = Settings(
            llm_api_key="test",
            llm_model="test",
            llm_base_url="",
        )
        assert settings.science_first_mode is True
        assert settings.fetch_mode == "off"
        assert settings.organism_strict == "planarian"
        assert settings.max_speculation == 3

    def test_custom_flags(self):
        from acheron.config import Settings

        settings = Settings(
            llm_api_key="test",
            llm_model="test",
            llm_base_url="",
            science_first_mode=False,
            fetch_mode="on",
            organism_strict="any",
            max_speculation=1,
        )
        assert settings.science_first_mode is False
        assert settings.fetch_mode == "on"
        assert settings.organism_strict == "any"
        assert settings.max_speculation == 1
