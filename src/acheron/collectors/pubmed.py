"""Collector for PubMed Central via NCBI E-Utilities."""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_PDF_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"


class PubMedCollector(BaseCollector):
    """Harvest papers from PubMed / PubMed Central."""

    source_name = "pubmed"

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        """Search PubMed and return enriched Paper objects."""
        ids = self._esearch(query, max_results)
        if not ids:
            logger.info("PubMed: no results for '%s'", query)
            return []
        logger.info("PubMed: fetching metadata for %d articles", len(ids))
        return self._efetch(ids)

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
