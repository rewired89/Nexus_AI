"""Extract structured text from PDFs using PyMuPDF and pdfplumber."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from acheron.models import Paper, PaperSection, TableData

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract clean text, sections, and tables from academic PDFs."""

    # Common section headings in biomedical papers
    SECTION_PATTERNS = [
        r"^(?:abstract|introduction|background|methods?|materials?\s+and\s+methods?)",
        r"^(?:results?|discussion|conclusions?|acknowledgements?|references?)",
        r"^(?:supplementary|supporting\s+information|data\s+availability)",
        r"^(?:figure|table)\s+\d+",
        r"^\d+\.?\s+\w+",  # Numbered sections (1. Introduction, 2. Methods, etc.)
    ]

    def parse(self, pdf_path: str | Path) -> Paper | None:
        """Parse a PDF file and return structured content.

        Tries PyMuPDF first (faster, better for text-heavy PDFs),
        falls back to pdfplumber for complex layouts.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error("PDF not found: %s", pdf_path)
            return None

        try:
            return self._parse_with_pymupdf(pdf_path)
        except Exception:
            logger.debug("PyMuPDF failed for %s, trying pdfplumber", pdf_path.name)
            try:
                return self._parse_with_pdfplumber(pdf_path)
            except Exception:
                logger.exception("All PDF parsers failed for %s", pdf_path.name)
                return None

    def extract_text(self, pdf_path: str | Path) -> str:
        """Extract plain text from a PDF — simpler interface."""
        pdf_path = Path(pdf_path)
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            pages = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()
            return "\n\n".join(pages)
        except Exception:
            try:
                import pdfplumber

                with pdfplumber.open(str(pdf_path)) as pdf:
                    return "\n\n".join(
                        page.extract_text() or "" for page in pdf.pages
                    )
            except Exception:
                logger.exception("Failed to extract text from %s", pdf_path.name)
                return ""

    def _parse_with_pymupdf(self, pdf_path: Path) -> Paper:
        import fitz

        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata or {}

        full_text_parts: list[str] = []
        all_blocks: list[dict] = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            full_text_parts.append(text)

            # Extract text blocks with position info for section detection
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        spans = line.get("spans", [])
                        if spans:
                            text_content = " ".join(s["text"] for s in spans)
                            font_size = max(s.get("size", 10) for s in spans)
                            is_bold = any("bold" in s.get("font", "").lower() for s in spans)
                            all_blocks.append({
                                "text": text_content.strip(),
                                "size": font_size,
                                "bold": is_bold,
                                "page": page_num,
                            })

        doc.close()

        full_text = "\n\n".join(full_text_parts)
        sections = self._detect_sections(all_blocks)
        tables = self._extract_tables_pdfplumber(pdf_path)

        return Paper(
            paper_id=f"pdf:{pdf_path.stem}",
            title=metadata.get("title", pdf_path.stem) or pdf_path.stem,
            authors=self._parse_author_string(metadata.get("author", "")),
            source="manual",
            full_text=self._clean_text(full_text),
            sections=sections,
            tables=tables,
            pdf_path=str(pdf_path),
        )

    def _parse_with_pdfplumber(self, pdf_path: Path) -> Paper:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages_text = []
            tables: list[TableData] = []

            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)

                # Extract tables
                for table in page.extract_tables():
                    if table and len(table) > 1:
                        headers = [str(c) if c else "" for c in table[0]]
                        rows = [
                            [str(c) if c else "" for c in row]
                            for row in table[1:]
                        ]
                        tables.append(TableData(headers=headers, rows=rows))

            full_text = "\n\n".join(pages_text)
            return Paper(
                paper_id=f"pdf:{pdf_path.stem}",
                title=pdf_path.stem,
                source="manual",
                full_text=self._clean_text(full_text),
                tables=tables,
                pdf_path=str(pdf_path),
            )

    def _detect_sections(self, blocks: list[dict]) -> list[PaperSection]:
        """Detect section boundaries from font size and bold patterns."""
        if not blocks:
            return []

        # Find the median font size — headings are typically larger
        sizes = [b["size"] for b in blocks if b["text"]]
        if not sizes:
            return []
        median_size = sorted(sizes)[len(sizes) // 2]

        sections: list[PaperSection] = []
        current_heading = ""
        current_text_parts: list[str] = []
        order = 0

        for block in blocks:
            text = block["text"]
            if not text:
                continue

            is_heading = (
                (block["size"] > median_size * 1.15 or block["bold"])
                and len(text) < 200
                and self._looks_like_heading(text)
            )

            if is_heading:
                # Save previous section
                if current_text_parts:
                    sections.append(PaperSection(
                        heading=current_heading,
                        text="\n".join(current_text_parts).strip(),
                        order=order,
                    ))
                    order += 1
                current_heading = text.strip()
                current_text_parts = []
            else:
                current_text_parts.append(text)

        # Last section
        if current_text_parts:
            sections.append(PaperSection(
                heading=current_heading,
                text="\n".join(current_text_parts).strip(),
                order=order,
            ))

        return sections

    def _looks_like_heading(self, text: str) -> bool:
        text_lower = text.strip().lower()
        for pattern in self.SECTION_PATTERNS:
            if re.match(pattern, text_lower):
                return True
        # Short lines that are likely headings
        if len(text) < 100 and not text.endswith("."):
            return True
        return False

    def _extract_tables_pdfplumber(self, pdf_path: Path) -> list[TableData]:
        """Use pdfplumber to extract tables (works better than PyMuPDF for tables)."""
        tables: list[TableData] = []
        try:
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        if table and len(table) > 1:
                            headers = [str(c) if c else "" for c in table[0]]
                            rows = [
                                [str(c) if c else "" for c in row]
                                for row in table[1:]
                            ]
                            tables.append(TableData(headers=headers, rows=rows))
        except Exception:
            logger.debug("pdfplumber table extraction failed for %s", pdf_path.name)
        return tables

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text."""
        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        # Fix hyphenation at line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Collapse spaces
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _parse_author_string(author_str: str) -> list[str]:
        if not author_str:
            return []
        # Handle various separators
        for sep in [";", " and ", "&", ","]:
            if sep in author_str:
                return [a.strip() for a in author_str.split(sep) if a.strip()]
        return [author_str.strip()] if author_str.strip() else []
