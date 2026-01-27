"""Tests for domain models."""

from datetime import date

from acheron.models import Paper, PaperSection, PaperSource, QueryResult, RAGResponse, TableData


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
