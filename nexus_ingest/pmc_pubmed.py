"""Verified ingestion from NCBI PubMed and PubMed Central.

Fetches directly from NCBI E-utilities:
  - PubMed esearch + efetch (XML) for metadata and abstracts
  - PMC efetch (NXML) for full text when PMCID available

Respects NCBI rate limits (3/sec without key, 10/sec with key).
Supports NCBI_API_KEY, NCBI_EMAIL, and NCBI_TOOL env vars.

Output includes provenance tracking for verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlencode

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# NCBI E-utilities endpoints
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

# Rate limits: 3/sec without key, 10/sec with key
RATE_LIMIT_NO_KEY = 0.34  # seconds between requests
RATE_LIMIT_WITH_KEY = 0.11


@dataclass
class SourceProvenance:
    """Provenance tracking for verification."""

    provider: str = "NCBI"
    database: str = "pubmed"
    fetched_at_utc: str = ""
    request_query: str = ""
    url_used: str = ""
    api_version: str = "e-utilities"


@dataclass
class PubMedRecord:
    """A record fetched from PubMed/PMC with full metadata."""

    pmid: str = ""
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    journal: str = ""
    year: Optional[int] = None
    pub_date: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    fulltext_nxml: Optional[str] = None
    fulltext_sections: list[dict] = field(default_factory=list)
    provenance: SourceProvenance = field(default_factory=SourceProvenance)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "doi": self.doi,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "pub_date": self.pub_date,
            "authors": self.authors,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "mesh_terms": self.mesh_terms,
            "has_fulltext": self.fulltext_nxml is not None,
            "fulltext_sections": self.fulltext_sections,
            "provenance": {
                "provider": self.provenance.provider,
                "database": self.provenance.database,
                "fetched_at_utc": self.provenance.fetched_at_utc,
                "request_query": self.provenance.request_query,
                "url_used": self.provenance.url_used,
                "api_version": self.provenance.api_version,
            },
        }


class PMCPubMedFetcher:
    """Fetch records from PubMed and PMC full text from PubMed Central.

    Usage:
        fetcher = PMCPubMedFetcher()
        records = fetcher.search("bioelectricity planarian", retmax=25)
        for record in records:
            print(record.title, record.pmcid)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        tool: str = "AcheronNexus",
    ):
        self.api_key = api_key or os.environ.get("NCBI_API_KEY", "")
        self.email = email or os.environ.get("NCBI_EMAIL", "")
        self.tool = tool or os.environ.get("NCBI_TOOL", "AcheronNexus")
        self.rate_limit = RATE_LIMIT_WITH_KEY if self.api_key else RATE_LIMIT_NO_KEY
        self._last_request_time = 0.0
        self.client = httpx.Client(timeout=60.0, follow_redirects=True)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        retmax: int = 25,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
        sort: str = "relevance",
    ) -> list[PubMedRecord]:
        """Search PubMed and return enriched records with PMC full text when available.

        Args:
            query: PubMed search query
            retmax: Maximum results to return
            mindate: Minimum date (YYYY or YYYY/MM or YYYY/MM/DD)
            maxdate: Maximum date
            sort: Sort order (relevance, pub_date, first_author, journal)

        Returns:
            List of PubMedRecord with metadata, abstract, and optionally full text.
        """
        logger.info("PubMed search: %r (retmax=%d)", query, retmax)

        # Step 1: esearch to get PMIDs
        pmids = self._esearch(query, retmax, mindate, maxdate, sort)
        if not pmids:
            logger.info("No results found for query: %s", query)
            return []

        logger.info("Found %d PMIDs, fetching metadata...", len(pmids))

        # Step 2: efetch metadata for all PMIDs
        records = self._efetch_pubmed(pmids, query)

        # Step 3: For records with PMCID, attempt to fetch full text
        pmc_count = 0
        for record in records:
            if record.pmcid:
                nxml = self._fetch_pmc_nxml(record.pmcid)
                if nxml:
                    record.fulltext_nxml = nxml
                    record.fulltext_sections = self._parse_nxml_sections(nxml)
                    pmc_count += 1

        logger.info(
            "Fetched %d records, %d with PMC full text", len(records), pmc_count
        )
        return records

    def search_iter(
        self,
        query: str,
        retmax: int = 25,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
    ) -> Iterator[PubMedRecord]:
        """Iterator version of search for memory efficiency."""
        pmids = self._esearch(query, retmax, mindate, maxdate)
        for pmid in pmids:
            records = self._efetch_pubmed([pmid], query)
            for record in records:
                if record.pmcid:
                    nxml = self._fetch_pmc_nxml(record.pmcid)
                    if nxml:
                        record.fulltext_nxml = nxml
                        record.fulltext_sections = self._parse_nxml_sections(nxml)
                yield record

    # ------------------------------------------------------------------
    # Internal: E-utilities calls
    # ------------------------------------------------------------------
    def _esearch(
        self,
        query: str,
        retmax: int,
        mindate: Optional[str] = None,
        maxdate: Optional[str] = None,
        sort: str = "relevance",
    ) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "xml",
            "sort": sort,
        }
        if mindate:
            params["mindate"] = mindate
            params["datetype"] = "pdat"
        if maxdate:
            params["maxdate"] = maxdate
        self._add_ncbi_params(params)

        resp = self._get(ESEARCH_URL, params)
        root = ET.fromstring(resp.text)

        # Check for errors
        error = root.find(".//ErrorList/PhraseNotFound")
        if error is not None:
            logger.warning("PubMed phrase not found: %s", error.text)

        pmids = [el.text for el in root.findall(".//Id") if el.text]
        return pmids

    def _efetch_pubmed(self, pmids: list[str], query: str) -> list[PubMedRecord]:
        """Fetch full metadata from PubMed for given PMIDs."""
        records = []
        batch_size = 100

        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "full",
            }
            self._add_ncbi_params(params)

            url = f"{EFETCH_URL}?{urlencode(params)}"
            resp = self._get(EFETCH_URL, params)

            try:
                root = ET.fromstring(resp.text)
            except ET.ParseError as e:
                logger.error("Failed to parse PubMed XML: %s", e)
                continue

            for article_el in root.findall(".//PubmedArticle"):
                try:
                    record = self._parse_pubmed_article(article_el, query, url)
                    if record:
                        records.append(record)
                except Exception as e:
                    logger.warning("Failed to parse article: %s", e)

        return records

    def _parse_pubmed_article(
        self, article_el: ET.Element, query: str, url: str
    ) -> Optional[PubMedRecord]:
        """Parse a PubmedArticle XML element into a PubMedRecord."""
        medline = article_el.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        if not pmid:
            return None

        art = medline.find("Article")
        if art is None:
            return None

        # Title
        title_el = art.find("ArticleTitle")
        title = self._get_text(title_el)

        # Abstract (handle structured abstracts)
        abstract_parts = []
        abs_el = art.find("Abstract")
        if abs_el is not None:
            for abs_text in abs_el.findall("AbstractText"):
                label = abs_text.get("Label", "")
                text = self._get_text(abs_text)
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # Authors
        authors = []
        for author_el in art.findall(".//Author"):
            last = self._get_text(author_el.find("LastName"))
            fore = self._get_text(author_el.find("ForeName"))
            if last:
                authors.append(f"{fore} {last}".strip())

        # Publication date
        year = None
        pub_date = None
        for path in [".//ArticleDate", ".//PubDate", ".//JournalIssue/PubDate"]:
            date_el = art.find(path)
            if date_el is not None:
                y = date_el.findtext("Year")
                m = date_el.findtext("Month") or "01"
                d = date_el.findtext("Day") or "01"
                if y:
                    year = int(y)
                    # Normalize month
                    if not m.isdigit():
                        m = str(_MONTH_MAP.get(m[:3].lower(), 1)).zfill(2)
                    pub_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                    break

        # Journal
        journal_el = art.find(".//Journal/Title")
        journal = self._get_text(journal_el)

        # DOI
        doi = None
        for eid in article_el.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text
                break

        # PMCID
        pmcid = None
        for eid in article_el.findall(".//ArticleId"):
            if eid.get("IdType") == "pmc":
                pmcid = eid.text
                break

        # MeSH terms
        mesh_terms = []
        for mesh in medline.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        # Keywords
        keywords = []
        for kw in medline.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)

        # Provenance
        provenance = SourceProvenance(
            provider="NCBI",
            database="pubmed",
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            request_query=query,
            url_used=url,
        )

        return PubMedRecord(
            pmid=pmid,
            pmcid=pmcid,
            doi=doi,
            title=title,
            journal=journal,
            year=year,
            pub_date=pub_date,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            mesh_terms=mesh_terms,
            provenance=provenance,
        )

    def _fetch_pmc_nxml(self, pmcid: str) -> Optional[str]:
        """Fetch full text NXML from PubMed Central."""
        # Normalize PMCID format
        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        params = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "full",
            "retmode": "xml",
        }
        self._add_ncbi_params(params)

        try:
            resp = self._get(EFETCH_URL, params)
            # Check if we got valid XML with article content
            if "<article" in resp.text or "<pmc-articleset" in resp.text:
                return resp.text
            logger.debug("No full text available for %s", pmcid)
            return None
        except Exception as e:
            logger.debug("Failed to fetch PMC %s: %s", pmcid, e)
            return None

    def _parse_nxml_sections(self, nxml: str) -> list[dict]:
        """Parse NXML into sections with xpath-like location tracking."""
        sections = []
        try:
            root = ET.fromstring(nxml)
        except ET.ParseError:
            return sections

        # Find body sections
        ns_body = ".//{http://www.ncbi.nlm.nih.gov/pmc}body"
        for body in root.findall(ns_body) or root.findall(".//body"):
            for sec in body.findall(".//sec") or body.findall(".//{http://www.ncbi.nlm.nih.gov/pmc}sec"):
                title_el = sec.find("title")
                if title_el is None:
                    title_el = sec.find("{http://www.ncbi.nlm.nih.gov/pmc}title")
                heading = self._get_text(title_el) if title_el is not None else ""

                # Collect all paragraph text in this section
                paragraphs = []
                for p in sec.findall(".//p") or sec.findall(".//{http://www.ncbi.nlm.nih.gov/pmc}p"):
                    text = self._get_text(p)
                    if text:
                        paragraphs.append(text)

                if paragraphs:
                    sections.append({
                        "heading": heading,
                        "text": "\n\n".join(paragraphs),
                        "xpath": f"body/sec[@title='{heading}']" if heading else "body/sec",
                    })

        # Also try to get abstract from front matter
        for abstract in root.findall(".//abstract") or root.findall(".//{http://www.ncbi.nlm.nih.gov/pmc}abstract"):
            text = self._get_text(abstract)
            if text:
                sections.insert(0, {
                    "heading": "Abstract",
                    "text": text,
                    "xpath": "front/article-meta/abstract",
                })

        return sections

    # ------------------------------------------------------------------
    # HTTP helpers with rate limiting and retry
    # ------------------------------------------------------------------
    def _add_ncbi_params(self, params: dict):
        """Add NCBI authentication params."""
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        if self.tool:
            params["tool"] = self.tool

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    def _get(self, url: str, params: dict) -> httpx.Response:
        """HTTP GET with rate limiting and retry."""
        # Enforce rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        resp = self.client.get(url, params=params)
        self._last_request_time = time.time()
        resp.raise_for_status()
        return resp

    @staticmethod
    def _get_text(el: Optional[ET.Element]) -> str:
        """Extract text content from an XML element."""
        if el is None:
            return ""
        return "".join(el.itertext()).strip()


_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ======================================================================
# Library saving functions
# ======================================================================
def save_record_to_library(
    record: PubMedRecord,
    library_dir: Path,
) -> tuple[Path, Optional[Path]]:
    """Save a PubMedRecord to the Library directory.

    Returns:
        Tuple of (json_path, nxml_path or None)
    """
    library_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic, collision-safe filename
    safe_id = f"pubmed_{record.pmid}"
    json_path = library_dir / f"{safe_id}.json"

    # Save metadata JSON
    json_path.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save NXML if available
    nxml_path = None
    if record.fulltext_nxml and record.pmcid:
        pmcid = record.pmcid.upper()
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        nxml_path = library_dir / f"pmc_{pmcid}.nxml"
        nxml_path.write_text(record.fulltext_nxml, encoding="utf-8")

    return json_path, nxml_path


# ======================================================================
# CLI entrypoint for manual testing
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fetch records from PubMed/PMC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q", "--query",
        required=True,
        help="PubMed search query",
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=25,
        help="Maximum results (default: 25)",
    )
    parser.add_argument(
        "--mindate",
        help="Minimum publication date (YYYY or YYYY/MM)",
    )
    parser.add_argument(
        "--maxdate",
        help="Maximum publication date",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Save records to this directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with PMCPubMedFetcher() as fetcher:
        records = fetcher.search(
            args.query,
            retmax=args.retmax,
            mindate=args.mindate,
            maxdate=args.maxdate,
        )

    print(f"\nFetched {len(records)} records\n")

    for i, rec in enumerate(records, 1):
        fulltext_status = "YES" if rec.fulltext_nxml else "no"
        print(f"[{i}] PMID:{rec.pmid} PMCID:{rec.pmcid or '-'} DOI:{rec.doi or '-'}")
        print(f"    {rec.title[:80]}...")
        authors_n = len(rec.authors)
        print(
            f"    {rec.journal} ({rec.year}) | Authors: {authors_n}"
            f" | Fulltext: {fulltext_status}"
        )
        if rec.fulltext_sections:
            print(f"    Sections: {[s['heading'] for s in rec.fulltext_sections[:5]]}")
        print()

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for rec in records:
            json_path, nxml_path = save_record_to_library(rec, args.output_dir)
            print(f"Saved: {json_path.name}", end="")
            if nxml_path:
                print(f" + {nxml_path.name}")
            else:
                print()
        print(f"\nSaved {len(records)} records to {args.output_dir}")


if __name__ == "__main__":
    main()
