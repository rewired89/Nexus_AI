"""Split paper text into overlapping chunks for vector embedding."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from acheron.models import Paper, TextChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk paper text with configurable size, overlap, and boundary respect."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ) -> None:
        self.chunk_size = chunk_size  # in tokens (approx words)
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_paper(self, paper: Paper) -> list[TextChunk]:
        """Create text chunks from a paper, preserving section structure."""
        chunks: list[TextChunk] = []
        chunk_index = 0

        # If we have structured sections, chunk by section
        if paper.sections:
            for section in paper.sections:
                if not section.text.strip():
                    continue
                section_chunks = self._chunk_text(section.text)
                for text in section_chunks:
                    chunks.append(self._make_chunk(
                        paper=paper,
                        text=text,
                        section=section.heading,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1

        # Also chunk the abstract separately (high-value content)
        if paper.abstract:
            chunks.insert(0, self._make_chunk(
                paper=paper,
                text=paper.abstract,
                section="Abstract",
                chunk_index=-1,
            ))

        # If no sections but we have full_text, chunk the whole thing
        if not paper.sections and paper.full_text:
            for text in self._chunk_text(paper.full_text):
                chunks.append(self._make_chunk(
                    paper=paper,
                    text=text,
                    section="",
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

        # Include tables as separate chunks
        for i, table in enumerate(paper.tables):
            table_text = table.to_text()
            if table_text.strip():
                chunks.append(self._make_chunk(
                    paper=paper,
                    text=table_text,
                    section=f"Table {i + 1}",
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

        logger.debug(
            "Chunked paper '%s' into %d chunks", paper.title[:60], len(chunks)
        )
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks respecting sentence boundaries."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0

        for sentence in sentences:
            words = sentence.split()
            sent_len = len(words)

            if current_len + sent_len > self.chunk_size and current_len > 0:
                chunk_text = " ".join(current_words)
                if len(current_words) >= self.min_chunk_size:
                    chunks.append(chunk_text.strip())

                # Keep overlap
                overlap_words = current_words[-self.chunk_overlap :] if self.chunk_overlap else []
                current_words = overlap_words
                current_len = len(current_words)

            current_words.extend(words)
            current_len += sent_len

        # Final chunk
        if current_words and len(current_words) >= self.min_chunk_size:
            chunks.append(" ".join(current_words).strip())

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex heuristics."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r"(Dr|Mr|Mrs|Ms|Prof|Fig|Eq|Ref|Vol|No|et al)\.", r"\1<DOT>", text)
        text = re.sub(r"(\d)\.", r"\1<DOT>", text)  # decimal numbers

        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Restore dots
        return [s.replace("<DOT>", ".") for s in sentences if s.strip()]

    def _make_chunk(
        self,
        paper: Paper,
        text: str,
        section: str,
        chunk_index: int,
    ) -> TextChunk:
        content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        chunk_id = f"{paper.paper_id}::{chunk_index}::{content_hash}"

        return TextChunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            text=text,
            section=section,
            chunk_index=chunk_index,
            metadata={
                "title": paper.title,
                "authors": paper.authors,
                "doi": paper.doi or "",
                "source": paper.source,
                "date": str(paper.publication_date) if paper.publication_date else "",
            },
        )
