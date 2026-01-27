"""Collector for PhysioNet datasets relevant to bioelectricity / EEG."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Optional

from acheron.collectors.base import BaseCollector
from acheron.models import Paper, PaperSource

logger = logging.getLogger(__name__)

PHYSIONET_SEARCH = "https://physionet.org/api/v1/published/"


class PhysioNetCollector(BaseCollector):
    """Discover datasets on PhysioNet relevant to bioelectricity research."""

    source_name = "physionet"

    def search(self, query: str, max_results: int = 30) -> list[Paper]:
        """Search PhysioNet for published datasets matching keywords."""
        keywords = [kw.strip().lower() for kw in query.split(",") if kw.strip()]
        if not keywords:
            keywords = [query.lower()]

        try:
            resp = self._get(PHYSIONET_SEARCH)
            data = resp.json()
        except Exception:
            logger.warning("PhysioNet API request failed")
            return []

        papers: list[Paper] = []
        for item in data:
            if len(papers) >= max_results:
                break
            paper = self._parse_item(item)
            if paper and self._matches(paper, keywords):
                papers.append(paper)

        logger.info("PhysioNet: found %d datasets for '%s'", len(papers), query)
        return papers

    def _parse_item(self, item: dict) -> Optional[Paper]:
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        slug = item.get("slug", "")
        version = item.get("version", "")
        doi = item.get("doi", "")

        # Authors
        authors = []
        for a in item.get("authors", []):
            name = a if isinstance(a, str) else a.get("name", "")
            if name:
                authors.append(name)

        # Date
        pub_date = None
        date_str = item.get("publish_date", "")
        if date_str:
            try:
                pub_date = date.fromisoformat(date_str[:10])
            except ValueError:
                pass

        url = f"https://physionet.org/content/{slug}/{version}/" if slug else ""

        return Paper(
            paper_id=doi if doi else f"physionet:{slug}",
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            doi=doi or None,
            source=PaperSource.PHYSIONET,
            journal="PhysioNet",
            keywords=[],
            url=url,
        )

    @staticmethod
    def _matches(paper: Paper, keywords: list[str]) -> bool:
        text = f"{paper.title} {paper.abstract}".lower()
        return any(kw in text for kw in keywords)
