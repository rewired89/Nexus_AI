"""Tests for the text chunker."""

from acheron.extraction.chunker import TextChunker
from acheron.models import Paper, PaperSection, PaperSource, TableData


def _make_paper(**kwargs) -> Paper:
    defaults = {
        "paper_id": "test/paper-1",
        "title": "Test Paper",
        "authors": ["Author One"],
        "source": PaperSource.MANUAL,
    }
    defaults.update(kwargs)
    return Paper(**defaults)


def test_chunk_abstract_only():
    paper = _make_paper(abstract="This is a test abstract about bioelectricity in planaria.")
    chunker = TextChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)
    assert len(chunks) >= 1
    assert chunks[0].section == "Abstract"
    assert "bioelectricity" in chunks[0].text


def test_chunk_with_sections():
    paper = _make_paper(
        sections=[
            PaperSection(heading="Introduction", text="A " * 300, order=0),
            PaperSection(heading="Methods", text="B " * 300, order=1),
        ]
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)
    assert len(chunks) >= 2
    # Sections should be preserved
    sections_found = {c.section for c in chunks}
    assert "Introduction" in sections_found
    assert "Methods" in sections_found


def test_chunk_with_tables():
    paper = _make_paper(
        abstract="Short abstract.",
        tables=[
            TableData(
                caption="Results",
                headers=["A", "B"],
                rows=[["1", "2"], ["3", "4"]],
            )
        ],
    )
    chunker = TextChunker(chunk_size=500, min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)
    table_chunks = [c for c in chunks if "Table" in c.section]
    assert len(table_chunks) >= 1


def test_chunk_full_text_fallback():
    # Use actual sentences so the chunker can split at sentence boundaries
    sentence = "Ion channels regulate membrane voltage in planarian cells. "
    paper = _make_paper(full_text=sentence * 50)
    chunker = TextChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)
    assert len(chunks) >= 2


def test_chunk_ids_are_unique():
    paper = _make_paper(full_text="Word " * 500)
    chunker = TextChunker(chunk_size=50, chunk_overlap=5, min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_metadata():
    paper = _make_paper(
        abstract="Test abstract.",
        doi="10.1234/test",
        authors=["Author One", "Author Two"],
    )
    chunker = TextChunker(min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)
    assert chunks[0].metadata["doi"] == "10.1234/test"
    assert "Author One" in chunks[0].metadata["authors"]


def test_chunk_evidence_spans():
    """Test that chunks include evidence span fields."""
    paper = _make_paper(
        abstract="Bioelectric signals regulate regeneration. This is a key finding.",
        pmid="12345",
        pmcid="PMC67890",
    )
    chunker = TextChunker(min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)

    assert len(chunks) >= 1
    # Abstract chunk should have evidence spans
    abstract_chunk = chunks[0]
    assert abstract_chunk.section == "Abstract"
    assert abstract_chunk.span_end > 0
    assert abstract_chunk.excerpt
    assert "Bioelectric" in abstract_chunk.excerpt
    assert abstract_chunk.xpath == "abstract"


def test_chunk_with_offsets():
    """Test _chunk_text_with_offsets returns correct offsets."""
    chunker = TextChunker(chunk_size=20, chunk_overlap=5, min_chunk_size=5)
    text = "First sentence here. Second sentence there. Third sentence now."

    chunks, offsets = chunker._chunk_text_with_offsets(text)

    assert len(chunks) == len(offsets)
    # Offsets should be valid ranges
    for chunk, (start, end) in zip(chunks, offsets):
        assert start >= 0
        assert end >= start
        assert end <= len(text)


def test_chunk_excerpt_generation():
    """Test that excerpt is generated for chunks."""
    long_text = "This is a very long sentence that contains important information. " * 20
    paper = _make_paper(abstract=long_text)
    chunker = TextChunker(min_chunk_size=1)
    chunks = chunker.chunk_paper(paper)

    for chunk in chunks:
        assert chunk.excerpt
        # Excerpt should be shorter than 400 chars
        assert len(chunk.excerpt) <= 400
