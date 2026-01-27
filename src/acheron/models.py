"""Core domain models used across the application."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PaperSource(str, Enum):
    PUBMED = "pubmed"
    BIORXIV = "biorxiv"
    ARXIV = "arxiv"
    PHYSIONET = "physionet"
    MANUAL = "manual"


class Paper(BaseModel):
    """Represents a single research paper with full metadata."""

    paper_id: str = Field(..., description="Unique identifier (DOI or source-specific ID)")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    publication_date: Optional[date] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    arxiv_id: Optional[str] = None
    source: PaperSource = PaperSource.MANUAL
    journal: str = ""
    keywords: list[str] = Field(default_factory=list)
    url: str = ""
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None
    sections: list[PaperSection] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    citation_count: Optional[int] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)

    def display_citation(self) -> str:
        """Return a human-readable citation string."""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        year = self.publication_date.year if self.publication_date else "n.d."
        doi_str = f" DOI: {self.doi}" if self.doi else ""
        return f"{author_str} ({year}). {self.title}. {self.journal}.{doi_str}"


class PaperSection(BaseModel):
    """A section from a parsed paper."""

    heading: str = ""
    text: str = ""
    order: int = 0


class TableData(BaseModel):
    """A table extracted from a paper."""

    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_text(self) -> str:
        """Render the table as a readable text block."""
        lines = []
        if self.caption:
            lines.append(f"Table: {self.caption}")
        if self.headers:
            lines.append(" | ".join(self.headers))
            lines.append("-" * (len(" | ".join(self.headers))))
        for row in self.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)


class TextChunk(BaseModel):
    """A chunk of text ready for embedding and storage."""

    chunk_id: str
    paper_id: str
    text: str
    section: str = ""
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)


class QueryResult(BaseModel):
    """A single retrieval result with source attribution."""

    text: str
    paper_id: str
    paper_title: str
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    section: str = ""
    relevance_score: float = 0.0

    def format_citation(self) -> str:
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        doi_part = f" (DOI: {self.doi})" if self.doi else ""
        return f"[{author_str}. \"{self.paper_title}\"{doi_part}]"


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""

    query: str
    answer: str
    sources: list[QueryResult] = Field(default_factory=list)
    model_used: str = ""
    total_chunks_searched: int = 0

    def format_full(self) -> str:
        """Format the response with inline citations for display."""
        lines = [self.answer, "", "--- Sources ---"]
        seen = set()
        for i, src in enumerate(self.sources, 1):
            key = src.paper_id
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  [{i}] {src.format_citation()}")
        return "\n".join(lines)


# Forward-reference resolution
Paper.model_rebuild()
