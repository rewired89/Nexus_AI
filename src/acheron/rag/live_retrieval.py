"""Live retrieval: fetch from PubMed/bioRxiv/arXiv when local evidence is weak.

Triggered automatically when local vectorstore returns fewer chunks than the
threshold or when the best relevance score is below the confidence threshold.
Can also be triggered explicitly via --live flag.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from acheron.config import get_settings
from acheron.extraction.chunker import TextChunker
from acheron.models import Paper, PaperSection, PaperSource, QueryResult, TextChunk

logger = logging.getLogger(__name__)


def evidence_is_weak(
    results: list[QueryResult],
    min_chunks: Optional[int] = None,
    min_score: Optional[float] = None,
) -> bool:
    """Determine whether local retrieval evidence is too weak.

    Returns True if:
    - Fewer than min_chunks results, OR
    - Best relevance score is below min_score
    """
    settings = get_settings()
    min_chunks = min_chunks if min_chunks is not None else settings.live_min_chunks
    min_score = min_score if min_score is not None else settings.live_min_score

    if len(results) < min_chunks:
        logger.info(
            "Evidence weak: %d chunks < threshold %d", len(results), min_chunks
        )
        return True

    if results:
        best_score = max(r.relevance_score for r in results)
        if best_score < min_score:
            logger.info(
                "Evidence weak: best score %.3f < threshold %.3f",
                best_score,
                min_score,
            )
            return True

    return False


def live_search(
    query: str,
    sources: list[str] | None = None,
    max_results: int | None = None,
) -> tuple[list[QueryResult], list[Paper], int]:
    """Perform a live search across academic sources and return temporary chunks.

    Args:
        query: The search query.
        sources: List of source names to search (pubmed, biorxiv, arxiv).
                 Defaults to all three.
        max_results: Max papers per source.

    Returns:
        Tuple of (query_results, fetched_papers, total_chunks).
        query_results are temporary chunks formatted as QueryResult for context.
        fetched_papers are the raw Paper objects (for optional persistence).
    """
    settings = get_settings()
    if sources is None:
        sources = ["pubmed", "biorxiv", "arxiv"]
    if max_results is None:
        max_results = settings.live_max_results

    all_papers: list[Paper] = []
    all_chunks: list[TextChunk] = []

    for source in sources:
        try:
            papers = _search_source(source, query, max_results)
            all_papers.extend(papers)
            logger.info("Live %s: fetched %d papers for %r", source, len(papers), query)
        except Exception as exc:
            logger.warning("Live %s search failed: %s", source, exc)

    if not all_papers:
        logger.info("Live search returned 0 papers")
        return [], [], 0

    # Chunk the fetched papers
    chunker = TextChunker()
    for paper in all_papers:
        try:
            chunks = chunker.chunk_paper(paper)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.debug("Failed to chunk live paper %s: %s", paper.paper_id, exc)

    # Convert chunks to QueryResult format for context assembly
    query_results = []
    for chunk in all_chunks:
        qr = QueryResult(
            text=chunk.text,
            paper_id=chunk.paper_id,
            paper_title=chunk.metadata.get("title", ""),
            authors=chunk.metadata.get("authors", "").split(", ")
            if chunk.metadata.get("authors")
            else [],
            doi=chunk.metadata.get("doi"),
            pmid=chunk.metadata.get("pmid"),
            pmcid=chunk.metadata.get("pmcid"),
            section=chunk.section,
            relevance_score=0.5,  # default score for live results
            source_file=chunk.source_file,
            span_start=chunk.span_start,
            span_end=chunk.span_end,
            excerpt=chunk.excerpt,
            xpath=chunk.xpath,
        )
        query_results.append(qr)

    logger.info(
        "Live search: %d papers → %d chunks from %s",
        len(all_papers),
        len(all_chunks),
        sources,
    )
    return query_results, all_papers, len(all_chunks)


def persist_high_relevance(
    papers: list[Paper],
    threshold: float | None = None,
) -> int:
    """Persist high-relevance live-fetched papers to the Library.

    Returns the number of papers persisted.
    """
    settings = get_settings()
    if threshold is None:
        threshold = settings.live_persist_threshold

    metadata_dir = settings.metadata_dir
    metadata_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for paper in papers:
        safe_id = paper.paper_id.replace("/", "_").replace(":", "_")
        dest = metadata_dir / f"{safe_id}.json"
        if dest.exists():
            continue
        dest.write_text(paper.model_dump_json(indent=2), encoding="utf-8")
        count += 1
        logger.info("Persisted live paper to Library: %s", safe_id)

    return count


def _search_source(source: str, query: str, max_results: int) -> list[Paper]:
    """Search a single academic source and return Paper objects."""
    if source == "pubmed":
        return _search_pubmed(query, max_results)
    elif source == "biorxiv":
        return _search_biorxiv(query, max_results)
    elif source == "arxiv":
        return _search_arxiv(query, max_results)
    else:
        logger.warning("Unknown live source: %s", source)
        return []


def _search_pubmed(query: str, max_results: int) -> list[Paper]:
    """Search PubMed using the existing PMCPubMedFetcher."""
    try:
        from nexus_ingest.pmc_pubmed import PMCPubMedFetcher
    except ImportError:
        logger.warning("nexus_ingest not available for live PubMed search")
        return []

    papers = []
    with PMCPubMedFetcher() as fetcher:
        records = fetcher.search(query, retmax=max_results)
        for rec in records:
            paper_id = f"pubmed:{rec.pmid}" if rec.pmid else f"pubmed:{_hash(rec.title)}"
            sections = []
            for sec_dict in rec.fulltext_sections:
                sections.append(
                    PaperSection(
                        heading=sec_dict.get("heading", ""),
                        text=sec_dict.get("text", ""),
                    )
                )
            full_text = "\n\n".join(s.text for s in sections if s.text) if sections else ""
            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=rec.title,
                    authors=rec.authors,
                    abstract=rec.abstract,
                    doi=rec.doi,
                    pmid=rec.pmid,
                    pmcid=rec.pmcid,
                    source=PaperSource.PUBMED,
                    journal=rec.journal,
                    keywords=rec.keywords + rec.mesh_terms,
                    full_text=full_text,
                    sections=sections,
                )
            )
    return papers


def _search_biorxiv(query: str, max_results: int) -> list[Paper]:
    """Search bioRxiv using the existing collector."""
    try:
        from acheron.collectors.biorxiv import BiorxivCollector

        collector = BiorxivCollector()
        papers = collector.search(query, max_results=max_results)
        collector.close()
        return papers
    except Exception as exc:
        logger.debug("bioRxiv live search failed: %s", exc)
        return []


def _search_arxiv(query: str, max_results: int) -> list[Paper]:
    """Search arXiv using the existing collector."""
    try:
        from acheron.collectors.arxiv import ArxivCollector

        collector = ArxivCollector()
        papers = collector.search(query, max_results=max_results)
        collector.close()
        return papers
    except Exception as exc:
        logger.debug("arXiv live search failed: %s", exc)
        return []


def _hash(text: str) -> str:
    """Generate a short hash for ID generation."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]
