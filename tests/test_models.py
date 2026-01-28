"""Tests for domain models."""

from datetime import date

from acheron.models import (
    DiscoveryResult,
    EpistemicTag,
    Hypothesis,
    LedgerEntry,
    Paper,
    PaperSection,
    PaperSource,
    QueryResult,
    RAGResponse,
    StructuredVariable,
    TableData,
)


def test_paper_display_citation():
    paper = Paper(
        paper_id="10.1234/test",
        title="Bioelectric signaling in planarian regeneration",
        authors=["Michael Levin", "Tal Shomrat", "Maya Bhatt", "Patrick McMillen"],
        publication_date=date(2023, 6, 15),
        doi="10.1234/test",
        journal="Nature Communications",
        source=PaperSource.PUBMED,
    )
    citation = paper.display_citation()
    assert "Michael Levin" in citation
    assert "et al." in citation
    assert "2023" in citation
    assert "10.1234/test" in citation


def test_paper_display_citation_no_date():
    paper = Paper(
        paper_id="no-date-paper",
        title="Test",
        authors=["Author One"],
        source=PaperSource.MANUAL,
    )
    assert "n.d." in paper.display_citation()


def test_table_data_to_text():
    table = TableData(
        caption="Ion channel conductance",
        headers=["Channel", "Conductance (pS)", "Selectivity"],
        rows=[
            ["Kv1.1", "12", "K+"],
            ["Nav1.5", "24", "Na+"],
        ],
    )
    text = table.to_text()
    assert "Ion channel conductance" in text
    assert "Kv1.1" in text
    assert "Nav1.5" in text


def test_query_result_format_citation():
    qr = QueryResult(
        text="Sample passage",
        paper_id="10.1234/test",
        paper_title="Bioelectric Signals in Morphogenesis",
        authors=["Michael Levin"],
        doi="10.1234/test",
        relevance_score=0.92,
    )
    citation = qr.format_citation()
    assert "Michael Levin" in citation
    assert "Bioelectric Signals" in citation
    assert "10.1234/test" in citation


def test_rag_response_format_full():
    response = RAGResponse(
        query="How do planaria regenerate?",
        answer="Planaria regenerate through bioelectric signaling [1].",
        sources=[
            QueryResult(
                text="Passage about regeneration",
                paper_id="10.1234/a",
                paper_title="Planarian Regeneration",
                authors=["Author A"],
                doi="10.1234/a",
                relevance_score=0.9,
            ),
            QueryResult(
                text="Another passage",
                paper_id="10.1234/b",
                paper_title="Bioelectric Signals",
                authors=["Author B"],
                doi="10.1234/b",
                relevance_score=0.85,
            ),
        ],
        model_used="gpt-4o",
        total_chunks_searched=100,
    )
    full = response.format_full()
    assert "Planaria regenerate" in full
    assert "[1]" in full
    assert "[2]" in full
    assert "Planarian Regeneration" in full


def test_paper_section_model():
    section = PaperSection(heading="Introduction", text="The body text.", order=0)
    assert section.heading == "Introduction"
    assert section.order == 0


def test_rag_response_epistemic_fields():
    response = RAGResponse(
        query="test",
        answer="test answer",
        evidence_statements=["Fact one [1]"],
        inference_statements=["Derived two [1][2]"],
        speculation_statements=["Hypothesis three"],
    )
    assert len(response.evidence_statements) == 1
    assert len(response.inference_statements) == 1
    assert len(response.speculation_statements) == 1


def test_epistemic_tag_enum():
    assert EpistemicTag.EVIDENCE == "evidence"
    assert EpistemicTag.INFERENCE == "inference"
    assert EpistemicTag.SPECULATION == "speculation"


def test_structured_variable():
    var = StructuredVariable(
        name="Vmem",
        value="-45",
        unit="mV",
        context="planarian anterior",
        source_ref="[1]",
        variable_type="vmem",
    )
    assert var.name == "Vmem"
    assert var.unit == "mV"
    assert var.variable_type == "vmem"


def test_structured_variable_default_type():
    var = StructuredVariable(name="Gj", value="0.5", unit="nS")
    assert var.variable_type == ""


def test_hypothesis_model():
    hyp = Hypothesis(
        statement="Depolarization triggers regeneration",
        supporting_refs=["[1]", "[3]"],
        confidence="high",
        predicted_impact="Would confirm Vmem as master regulator of blastema formation",
        assumptions=["Vmem is measurable in vivo", "No confounding chemical signals"],
        validation_strategy="Re-analyze existing voltage imaging data",
    )
    assert hyp.confidence == "high"
    assert len(hyp.supporting_refs) == 2
    assert "master regulator" in hyp.predicted_impact
    assert len(hyp.assumptions) == 2


def test_hypothesis_defaults():
    hyp = Hypothesis(statement="Test hypothesis")
    assert hyp.confidence == "low"
    assert hyp.predicted_impact == ""
    assert hyp.assumptions == []


def test_discovery_result():
    result = DiscoveryResult(
        query="voltage gradients in planarian regeneration",
        evidence=["Fact one", "Fact two"],
        inference=["Pattern A"],
        speculation=["Hypothesis X"],
        variables=[StructuredVariable(name="Vmem", value="-45", unit="mV")],
        hypotheses=[
            Hypothesis(statement="Test hypothesis", confidence="medium"),
        ],
        uncertainty_notes=["No kinetics data available"],
        model_used="gpt-4o",
        total_chunks_searched=50,
    )
    assert len(result.evidence) == 2
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].confidence == "medium"


def test_discovery_result_new_fields():
    result = DiscoveryResult(
        query="gap junction role in planarian regeneration",
        evidence=["Gj blockers impair head regeneration [1]"],
        bioelectric_schematic=(
            "Amputation -> Vmem depolarization at wound -> "
            "Gj-mediated signal propagation -> Anterior gene activation -> "
            "Head regeneration"
        ),
        validation_path=[
            "Re-analyze existing Vmem imaging datasets",
            "Simulate Gj network in computational model",
        ],
        cross_species_notes=[
            "Planarian Gj (innexins) vs Xenopus Gj (connexins) — conserved role in patterning",
        ],
    )
    assert "Amputation" in result.bioelectric_schematic
    assert len(result.validation_path) == 2
    assert len(result.cross_species_notes) == 1


def test_rag_response_bioelectric_schematic():
    response = RAGResponse(
        query="test",
        answer="test answer",
        bioelectric_schematic="Vmem shift -> Gj change -> morphological outcome",
    )
    assert "Vmem shift" in response.bioelectric_schematic


def test_ledger_entry():
    entry = LedgerEntry(
        entry_id="ledger-test-001",
        query="test query",
        evidence_summary="Evidence here",
        tags=["test", "planarian"],
    )
    assert entry.entry_id == "ledger-test-001"
    assert "planarian" in entry.tags


# ======================================================================
# Evidence span tests
# ======================================================================
from acheron.models import TextChunk


def test_text_chunk_evidence_spans():
    chunk = TextChunk(
        chunk_id="test::0::abc123",
        paper_id="pmid:12345",
        text="Bioelectric signals regulate regeneration in planaria.",
        section="Abstract",
        source_file="pubmed_12345.json",
        span_start=0,
        span_end=54,
        excerpt="Bioelectric signals regulate regeneration...",
        xpath="abstract",
    )
    assert chunk.source_file == "pubmed_12345.json"
    assert chunk.span_start == 0
    assert chunk.span_end == 54
    assert chunk.excerpt
    assert chunk.xpath == "abstract"


def test_query_result_evidence_spans():
    result = QueryResult(
        text="Vmem gradients control head regeneration in planaria.",
        paper_id="pmid:67890",
        paper_title="Bioelectric Control of Regeneration",
        authors=["Levin M", "Adams DS"],
        pmid="67890",
        pmcid="PMC123456",
        source_file="pmc_PMC123456.nxml",
        span_start=1024,
        span_end=1200,
        excerpt="Vmem gradients control head regeneration...",
        xpath="body/sec[@title='Results']",
    )
    assert result.pmid == "67890"
    assert result.pmcid == "PMC123456"
    assert result.source_file == "pmc_PMC123456.nxml"
    assert result.xpath == "body/sec[@title='Results']"

    # Test format_evidence_span
    span_str = result.format_evidence_span()
    assert "pmc_PMC123456.nxml" in span_str
    assert "Results" in span_str


def test_query_result_format_citation_with_pmid():
    result = QueryResult(
        text="Test text",
        paper_id="pmid:12345",
        paper_title="Test Paper",
        authors=["Author A", "Author B"],
        doi="10.1234/test",
        pmid="12345",
    )
    citation = result.format_citation()
    assert "PMID:12345" in citation
    assert "DOI: 10.1234/test" in citation
