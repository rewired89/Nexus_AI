"""Core domain models used across the application."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ======================================================================
# Enumerations
# ======================================================================
class PaperSource(str, Enum):
    PUBMED = "pubmed"
    BIORXIV = "biorxiv"
    ARXIV = "arxiv"
    PHYSIONET = "physionet"
    MANUAL = "manual"


class EpistemicTag(str, Enum):
    """Classification for separating evidence, inference, and speculation."""

    EVIDENCE = "evidence"
    INFERENCE = "inference"
    SPECULATION = "speculation"


# ======================================================================
# Paper models
# ======================================================================
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


# ======================================================================
# Query and response models
# ======================================================================
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


class StructuredVariable(BaseModel):
    """A variable extracted from source material during the discovery loop."""

    name: str
    value: str = ""
    unit: str = ""
    context: str = ""
    source_ref: str = ""


class Hypothesis(BaseModel):
    """A testable hypothesis generated from pattern comparison."""

    statement: str
    supporting_refs: list[str] = Field(default_factory=list)
    confidence: str = "low"  # low, medium, high — based on evidence density
    validation_strategy: str = ""


class DiscoveryResult(BaseModel):
    """Structured output from the discovery loop.

    Separates evidence (from sources), inference (logical derivation),
    and speculation (hypotheses beyond direct evidence).
    """

    query: str
    evidence: list[str] = Field(
        default_factory=list,
        description="Statements directly supported by retrieved sources",
    )
    inference: list[str] = Field(
        default_factory=list,
        description="Logical derivations from the evidence",
    )
    speculation: list[str] = Field(
        default_factory=list,
        description="Hypotheses that go beyond what the evidence directly states",
    )
    variables: list[StructuredVariable] = Field(
        default_factory=list,
        description="Structured variables extracted from sources",
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="Testable hypotheses generated from pattern comparison",
    )
    sources: list[QueryResult] = Field(default_factory=list)
    model_used: str = ""
    total_chunks_searched: int = 0
    uncertainty_notes: list[str] = Field(
        default_factory=list,
        description="Explicit declarations of uncertainty or missing data",
    )


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""

    query: str
    answer: str
    sources: list[QueryResult] = Field(default_factory=list)
    model_used: str = ""
    total_chunks_searched: int = 0
    # Optional structured layers — populated when discovery mode is used
    evidence_statements: list[str] = Field(default_factory=list)
    inference_statements: list[str] = Field(default_factory=list)
    speculation_statements: list[str] = Field(default_factory=list)

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


# ======================================================================
# Experiment ledger
# ======================================================================
class LedgerEntry(BaseModel):
    """A single entry in the experiment ledger."""

    entry_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query: str
    evidence_summary: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    variables: list[StructuredVariable] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


# Forward-reference resolution
Paper.model_rebuild()
