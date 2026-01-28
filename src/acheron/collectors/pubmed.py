"""Collector for PubMed Central via NCBI E-Utilities.

Uses the nexus_ingest.pmc_pubmed module for verified ingestion with:
  - Full metadata from PubMed esearch/efetch
  - Full text NXML from PMC when available
  - Provenance tracking for verification
  - Evidence span support
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSection, PaperSource, SourceProvenance

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_PDF_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"


class PubMedCollector(BaseCollector):
    """Harvest papers from PubMed / PubMed Central.

    Uses enhanced ingestion from nexus_ingest for PMC full text when available.
    Saves raw NXML alongside JSON metadata for verification.
    """

    source_name = "pubmed"

    def search(
        self,
        query: str,
        max_results: int = 50,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
        use_enhanced: bool = True,
    ) -> list[Paper]:
        """Search PubMed and return enriched Paper objects.

        Args:
            query: PubMed search query
            max_results: Maximum results to return
            mindate: Minimum date (YYYY or YYYY/MM)
            maxdate: Maximum date
            use_enhanced: Use enhanced ingestion with PMC full text (default True)
        """
        if use_enhanced:
            return self._search_enhanced(query, max_results, mindate, maxdate)

        # Fallback to basic search
        ids = self._esearch(query, max_results)
        if not ids:
            logger.info("PubMed: no results for '%s'", query)
            return []
        logger.info("PubMed: fetching metadata for %d articles", len(ids))
        return self._efetch(ids)

    def _search_enhanced(
        self,
        query: str,
        max_results: int,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
    ) -> list[Paper]:
        """Enhanced search using nexus_ingest for PMC full text."""
        try:
            from nexus_ingest.pmc_pubmed import PMCPubMedFetcher, save_record_to_library
        except ImportError:
            logger.warning("nexus_ingest not available, falling back to basic search")
            ids = self._esearch(query, max_results)
            return self._efetch(ids) if ids else []

        papers = []
        nxml_dir = self.settings.data_dir / "nxml"
        nxml_dir.mkdir(parents=True, exist_ok=True)

        with PMCPubMedFetcher(
            api_key=self.settings.ncbi_api_key,
            email=getattr(self.settings, "ncbi_email", None),
        ) as fetcher:
            records = fetcher.search(
                query,
                retmax=max_results,
                mindate=mindate,
                maxdate=maxdate,
            )

            for record in records:
                # Save raw NXML if available
                nxml_path = None
                if record.fulltext_nxml and record.pmcid:
                    pmcid = record.pmcid.upper()
                    if not pmcid.startswith("PMC"):
                        pmcid = f"PMC{pmcid}"
                    nxml_path = nxml_dir / f"pmc_{pmcid}.nxml"
                    nxml_path.write_text(record.fulltext_nxml, encoding="utf-8")
                    logger.debug("Saved NXML: %s", nxml_path.name)

                # Convert to Paper model
                paper = self._record_to_paper(record, nxml_path)
                papers.append(paper)

        logger.info(
            "PubMed enhanced: %d papers (%d with full text)",
            len(papers),
            sum(1 for p in papers if p.full_text),
        )
        return papers

    def _record_to_paper(self, record, nxml_path: Optional[Path] = None) -> Paper:
        """Convert a PubMedRecord to a Paper model."""
        # Build sections from NXML if available
        sections = []
        full_text = None
        if record.fulltext_sections:
            for sec in record.fulltext_sections:
                sections.append(PaperSection(
                    heading=sec.get("heading", ""),
                    text=sec.get("text", ""),
                ))
            full_text = "\n\n".join(s.text for s in sections if s.text)

        paper_id = record.doi if record.doi else f"pmid:{record.pmid}"
        url = PMC_PDF_URL.format(pmcid=record.pmcid) if record.pmcid else ""

        pub_date = None
        if record.pub_date:
            try:
                parts = record.pub_date.split("-")
                pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                pass
        elif record.year:
            pub_date = date(record.year, 1, 1)

        # Convert nexus_ingest provenance to models.SourceProvenance
        provenance = None
        if record.provenance:
            provenance = SourceProvenance(
                provider=record.provenance.provider,
                database=record.provenance.database,
                fetched_at_utc=record.provenance.fetched_at_utc,
                request_query=record.provenance.request_query,
                url_used=record.provenance.url_used,
            )

        return Paper(
            paper_id=paper_id,
            title=record.title,
            authors=record.authors,
            abstract=record.abstract,
            publication_date=pub_date,
            doi=record.doi,
            pmid=record.pmid,
            pmcid=record.pmcid,
            source=PaperSource.PUBMED,
            journal=record.journal,
            keywords=record.keywords + record.mesh_terms,
            url=url,
            full_text=full_text,
            sections=sections,
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    def _esearch(self, query: str, max_results: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "sort": "relevance",
        }
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        resp = self._get(ESEARCH_URL, params=params)
        root = ET.fromstring(resp.text)
        return [id_el.text for id_el in root.findall(".//Id") if id_el.text]

    def _efetch(self, pmids: list[str]) -> list[Paper]:
        papers: list[Paper] = []
        batch_size = 100
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "full",
            }
            if self.settings.ncbi_api_key:
                params["api_key"] = self.settings.ncbi_api_key

            resp = self._get(EFETCH_URL, params=params)
            root = ET.fromstring(resp.text)

            for article_el in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_article(article_el)
                    if paper:
                        papers.append(paper)
                except Exception:
                    logger.exception("Failed to parse PubMed article")

            # Respect NCBI rate limits
            if not self.settings.ncbi_api_key:
                time.sleep(0.4)
            else:
                time.sleep(0.15)

        return papers

    def _parse_article(self, article_el: ET.Element) -> Optional[Paper]:
        medline = article_el.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        art = medline.find("Article")
        if art is None:
            return None

        # Title
        title_el = art.find("ArticleTitle")
        title = self._text(title_el)

        # Abstract
        abs_el = art.find("Abstract")
        abstract = ""
        if abs_el is not None:
            parts = []
            for abs_text in abs_el.findall("AbstractText"):
                label = abs_text.get("Label", "")
                text = self._text(abs_text)
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(text)
            abstract = "\n".join(parts)

        # Authors
        authors = []
        for author_el in art.findall(".//Author"):
            last = self._text(author_el.find("LastName"))
            fore = self._text(author_el.find("ForeName"))
            if last:
                authors.append(f"{fore} {last}".strip())

        # Date
        pub_date = self._parse_date(art)

        # Journal
        journal_el = art.find(".//Journal/Title")
        journal = self._text(journal_el)

        # DOI
        doi = ""
        for eid in article_el.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text or ""
                break

        # PMC ID (for PDF download)
        pmcid = ""
        for eid in article_el.findall(".//ArticleId"):
            if eid.get("IdType") == "pmc":
                pmcid = eid.text or ""
                break

        # Keywords / MeSH
        keywords = []
        for kw in medline.findall(".//MeshHeading/DescriptorName"):
            if kw.text:
                keywords.append(kw.text)
        for kw in medline.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)

        paper_id = doi if doi else f"pmid:{pmid}"
        url = PMC_PDF_URL.format(pmcid=pmcid) if pmcid else ""

        return Paper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            doi=doi or None,
            pmid=pmid or None,
            pmcid=pmcid or None,
            source=PaperSource.PUBMED,
            journal=journal,
            keywords=keywords,
            url=url,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _text(el: Optional[ET.Element]) -> str:
        if el is None:
            return ""
        return "".join(el.itertext()).strip()

    @staticmethod
    def _parse_date(art_el: ET.Element) -> Optional[date]:
        for path in [
            ".//ArticleDate",
            ".//PubDate",
            ".//JournalIssue/PubDate",
        ]:
            d = art_el.find(path)
            if d is not None:
                y = d.findtext("Year")
                m = d.findtext("Month") or "1"
                day = d.findtext("Day") or "1"
                if y:
                    try:
                        month = int(m) if m.isdigit() else _MONTH_MAP.get(m[:3].lower(), 1)
                        return date(int(y), month, int(day))
                    except (ValueError, TypeError):
                        pass
        return None


_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
