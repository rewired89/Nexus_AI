"""Collector for bioRxiv and medRxiv preprints via their public API."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource

logger = logging.getLogger(__name__)

# bioRxiv content-detail API
BIORXIV_API = "https://api.biorxiv.org/details/biorxiv/{start}/{end}/{cursor}"
BIORXIV_SEARCH = "https://api.biorxiv.org/search/{query}/0/{max_results}"


class BiorxivCollector(BaseCollector):
    """Harvest preprints from bioRxiv."""

    source_name = "biorxiv"

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        """Search bioRxiv by keyword.

        The bioRxiv API doesn't have a great full-text search endpoint,
        so we use their search endpoint and also fall back to date-range
        scanning with keyword filtering in abstracts.
        """
        papers = self._api_search(query, max_results)
        if not papers:
            # Fallback: scan recent months and filter locally
            papers = self._date_range_search(query, max_results)
        logger.info("bioRxiv: found %d papers for '%s'", len(papers), query)
        return papers

    def _api_search(self, query: str, max_results: int) -> list[Paper]:
        """Try the bioRxiv search-style endpoint (best-effort)."""
        # The bioRxiv API has limited search; we try a direct content API
        # with date range and filter client-side.
        return self._date_range_search(query, max_results)

    def _date_range_search(self, query: str, max_results: int) -> list[Paper]:
        """Scan recent date ranges and filter by keyword match."""
        papers: list[Paper] = []
        keywords = [kw.strip().lower() for kw in query.split(",") if kw.strip()]
        if not keywords:
            keywords = [query.lower()]

        end = date.today()
        start = end - timedelta(days=365 * 2)  # Last 2 years as starting window

        cursor = 0
        page_size = 100
        max_pages = 20  # Safety limit

        for _ in range(max_pages):
            if len(papers) >= max_results:
                break

            url = BIORXIV_API.format(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                cursor=cursor,
            )

            try:
                resp = self._get(url)
                data = resp.json()
            except Exception:
                logger.warning("bioRxiv API request failed at cursor %d", cursor)
                break

            collection = data.get("collection", [])
            if not collection:
                break

            for item in collection:
                if len(papers) >= max_results:
                    break
                paper = self._parse_item(item)
                if paper and self._matches_keywords(paper, keywords):
                    papers.append(paper)

            cursor += page_size
            time.sleep(1.0)  # Rate-limit respect

        return papers

    def _parse_item(self, item: dict) -> Optional[Paper]:
        doi = item.get("doi", "")
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        authors_str = item.get("authors", "")
        authors = [a.strip() for a in authors_str.split(";") if a.strip()]
        pub_date_str = item.get("date", "")
        category = item.get("category", "")

        pub_date = None
        if pub_date_str:
            try:
                parts = pub_date_str.split("-")
                pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                pass

        pdf_url = f"https://www.biorxiv.org/content/{doi}v1.full.pdf" if doi else ""

        return Paper(
            paper_id=doi if doi else f"biorxiv:{title[:50]}",
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            doi=doi or None,
            source=PaperSource.BIORXIV,
            journal="bioRxiv",
            keywords=[category] if category else [],
            url=pdf_url,
        )

    @staticmethod
    def _matches_keywords(paper: Paper, keywords: list[str]) -> bool:
        text = f"{paper.title} {paper.abstract}".lower()
        return any(kw in text for kw in keywords)
