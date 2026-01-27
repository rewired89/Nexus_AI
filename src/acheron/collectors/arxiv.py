"""Collector for arXiv papers via the arXiv API (Atom feed)."""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from datetime import date
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"

# arXiv namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivCollector(BaseCollector):
    """Harvest papers from arXiv (primarily q-bio section)."""

    source_name = "arxiv"

    def search(self, query: str, max_results: int = 50) -> list[Paper]:
        """Search arXiv and return Paper objects.

        Automatically scopes to q-bio categories unless the query already
        contains a cat: prefix.
        """
        papers: list[Paper] = []
        batch_size = min(max_results, 100)

        # Build the search query — scope to q-bio if not already scoped
        if "cat:" not in query.lower():
            search_query = (
                f"(cat:q-bio* OR cat:physics.bio-ph) AND "
                f"(all:{query})"
            )
        else:
            search_query = query

        start = 0
        while len(papers) < max_results:
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": batch_size,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            try:
                resp = self._get(ARXIV_API, params=params)
            except Exception:
                logger.warning("arXiv API request failed at offset %d", start)
                break

            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", NS)
            if not entries:
                break

            for entry in entries:
                if len(papers) >= max_results:
                    break
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)

            start += batch_size
            time.sleep(3.0)  # arXiv asks for 3-second delay between requests

        logger.info("arXiv: found %d papers for '%s'", len(papers), query)
        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Paper]:
        # ID & arXiv ID
        id_url = self._text(entry.find("atom:id", NS))
        arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url

        title = " ".join(self._text(entry.find("atom:title", NS)).split())
        abstract = " ".join(self._text(entry.find("atom:summary", NS)).split())

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", NS):
            name = self._text(author_el.find("atom:name", NS))
            if name:
                authors.append(name)

        # Date
        published = self._text(entry.find("atom:published", NS))
        pub_date = None
        if published:
            try:
                pub_date = date.fromisoformat(published[:10])
            except ValueError:
                pass

        # DOI (may or may not exist)
        doi_el = entry.find("arxiv:doi", NS)
        doi = self._text(doi_el) if doi_el is not None else None

        # Categories
        categories = []
        for cat in entry.findall("atom:category", NS):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        # PDF link
        pdf_url = ""
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        return Paper(
            paper_id=f"arxiv:{arxiv_id}",
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            doi=doi,
            arxiv_id=arxiv_id,
            source=PaperSource.ARXIV,
            journal="arXiv",
            keywords=categories,
            url=pdf_url,
        )

    @staticmethod
    def _text(el: Optional[ET.Element]) -> str:
        if el is None:
            return ""
        return (el.text or "").strip()
