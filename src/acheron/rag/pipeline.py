"""Core RAG pipeline: retrieve → rerank → generate with citations."""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI

from acheron.config import get_settings
from acheron.models import QueryResult, RAGResponse
from acheron.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Acheron Nexus, a specialist research assistant for bioelectricity, \
biomedical engineering, EEG analysis, regenerative biology, ion channel dynamics, \
and bioelectric morphogenesis.

Your role:
1. Answer questions using ONLY the provided source passages.
2. Cite every claim using [1], [2], etc. matching the source numbers.
3. If the sources don't contain enough information, say so explicitly.
4. Prefer precise technical language appropriate for a researcher.
5. When discussing experimental results, note the methodology and any limitations.
6. For conflicting findings across sources, present both perspectives with citations.

Do NOT:
- Make up information not in the sources.
- Cite sources that don't support your statement.
- Speculate beyond what the literature states.
"""

QUERY_TEMPLATE = """\
Based on the following source passages from the bioelectricity and biomedical \
research literature, answer the user's question.

=== SOURCE PASSAGES ===
{context}
========================

User question: {query}

Provide a detailed, well-cited answer. Use [1], [2], etc. to cite sources inline."""


class RAGPipeline:
    """End-to-end RAG: retrieval → context assembly → LLM generation."""

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        n_retrieve: int = 12,
        n_context: int = 8,
    ) -> None:
        self.settings = get_settings()
        self.store = store or VectorStore()
        self.n_retrieve = n_retrieve
        self.n_context = n_context
        self._llm_client: Optional[OpenAI] = None

    @property
    def llm(self) -> OpenAI:
        if self._llm_client is None:
            self._llm_client = OpenAI(
                api_key=self.settings.llm_api_key or "not-set",
                base_url=self.settings.llm_base_url,
            )
        return self._llm_client

    # ------------------------------------------------------------------
    # Main query flow
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        filter_source: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a question."""
        n = n_results or self.n_retrieve

        # 1. Retrieve
        results = self.store.search(
            query=question, n_results=n, filter_source=filter_source
        )
        logger.info("Retrieved %d chunks for query", len(results))

        if not results:
            return RAGResponse(
                query=question,
                answer="No relevant sources found in the database. "
                "Try adding more papers or rephrasing your query.",
                sources=[],
                model_used=self.settings.llm_model,
            )

        # 2. Rerank / select top context passages
        top_results = self._select_context(results)

        # 3. Build prompt
        context_str = self._format_context(top_results)
        user_prompt = QUERY_TEMPLATE.format(context=context_str, query=question)

        # 4. Generate
        answer = self._generate(user_prompt)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=top_results,
            model_used=self.settings.llm_model,
            total_chunks_searched=len(results),
        )

    def retrieve_only(
        self,
        question: str,
        n_results: int = 10,
        filter_source: Optional[str] = None,
    ) -> list[QueryResult]:
        """Retrieve relevant passages without generating an answer."""
        return self.store.search(
            query=question, n_results=n_results, filter_source=filter_source
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _select_context(self, results: list[QueryResult]) -> list[QueryResult]:
        """Select the best passages for context, deduplicating by paper."""
        # Sort by relevance, then pick top N, preferring diversity of sources
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)

        selected: list[QueryResult] = []
        papers_included: dict[str, int] = {}
        max_per_paper = 3

        for r in sorted_results:
            if len(selected) >= self.n_context:
                break
            count = papers_included.get(r.paper_id, 0)
            if count >= max_per_paper:
                continue
            selected.append(r)
            papers_included[r.paper_id] = count + 1

        return selected

    def _format_context(self, results: list[QueryResult]) -> str:
        """Format retrieved passages into a numbered context block."""
        parts = []
        for i, r in enumerate(results, 1):
            author_str = ", ".join(r.authors[:3])
            if len(r.authors) > 3:
                author_str += " et al."
            doi_str = f" DOI: {r.doi}" if r.doi else ""
            header = f"[{i}] {author_str}. \"{r.paper_title}\".{doi_str}"
            section_note = f" (Section: {r.section})" if r.section else ""
            parts.append(f"{header}{section_note}\n{r.text}")
        return "\n\n".join(parts)

    def _generate(self, user_prompt: str) -> str:
        """Call the LLM to generate a cited answer."""
        try:
            response = self.llm.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("LLM generation failed")
            return (
                "Error: Could not generate an answer. "
                "Check your LLM API key and configuration."
            )
