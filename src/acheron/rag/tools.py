"""Nexus Tool Definitions — tools the ReAct agent can invoke.

Each tool wraps an existing Nexus capability (vectorstore search,
live retrieval, computation engine) behind a clean interface that
the LLM can call via function-calling / tool-use.

Design rule: NO business logic lives here — only bridging code
that translates tool inputs to existing pipeline method calls and
formats the output for the LLM.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from acheron.models import QueryResult

logger = logging.getLogger(__name__)


# ======================================================================
# Tool result container
# ======================================================================

@dataclass
class ToolResult:
    """Result of executing a tool."""

    text: str  # Formatted text for the LLM to read
    results: list[QueryResult]  # Structured results (for context assembly)
    computation_context: str = ""  # Computation output to inject


# ======================================================================
# Tool definitions — Anthropic format
# ======================================================================

TOOL_DEFINITIONS_ANTHROPIC = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the local research paper library (vectorstore) for "
            "passages relevant to a query.  Returns text chunks from "
            "ingested papers with relevance scores.  Use this first to "
            "check what evidence is already available locally."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — use specific scientific terms",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_pubmed",
        "description": (
            "Search PubMed for peer-reviewed biomedical research papers.  "
            "Use when local knowledge base lacks sufficient evidence or "
            "you need the latest published findings on a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "PubMed search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max papers to fetch (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_biorxiv",
        "description": (
            "Search bioRxiv for biology preprints.  Use when you need "
            "cutting-edge, not-yet-peer-reviewed findings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "bioRxiv search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max papers to fetch (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for preprints in physics, computational biology, "
            "and information theory.  Use for theoretical or computational "
            "aspects of bioelectricity research."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "arXiv search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max papers to fetch (default 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_computation",
        "description": (
            "Run a deterministic reasoning engine computation.  Available "
            "modules: 'ribozyme' (QT45/Bio-RAID redundancy), 'freeze_thaw' "
            "(kinetics), 'mosaic' (topology comparison), 'spectral_rewiring' "
            "(spectral gap/energy optimization), 'state_space' (eigenvalue/"
            "attractor analysis), 'denram' (delay encoding), 'heterogeneity' "
            "(substrate variability), 'homeostatic' (BESS/scaling), "
            "'model_validation' (sensitivity/falsification).  Use when the "
            "query requires quantitative calculations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "module": {
                    "type": "string",
                    "enum": [
                        "ribozyme", "freeze_thaw", "mosaic",
                        "spectral_rewiring", "state_space", "denram",
                        "heterogeneity", "homeostatic", "model_validation",
                    ],
                    "description": "Which computation module to run",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "The original query (used to extract numeric "
                        "parameters like fidelity, temperature, node count)"
                    ),
                },
            },
            "required": ["module", "query"],
        },
    },
]


def get_tool_definitions_openai() -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    openai_tools = []
    for tool in TOOL_DEFINITIONS_ANTHROPIC:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        })
    return openai_tools


# ======================================================================
# Tool execution
# ======================================================================

def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    pipeline: Any,
) -> ToolResult:
    """Execute a tool and return the result.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute.
    tool_input : dict
        Tool input parameters (from the LLM).
    pipeline : RAGPipeline
        The pipeline instance (provides vectorstore, live retrieval, etc.).

    Returns
    -------
    ToolResult
        Formatted text for the LLM plus structured results.
    """
    try:
        if tool_name == "search_knowledge_base":
            return _exec_search_knowledge_base(tool_input, pipeline)
        elif tool_name == "search_pubmed":
            return _exec_search_live(tool_input, pipeline, source="pubmed")
        elif tool_name == "search_biorxiv":
            return _exec_search_live(tool_input, pipeline, source="biorxiv")
        elif tool_name == "search_arxiv":
            return _exec_search_live(tool_input, pipeline, source="arxiv")
        elif tool_name == "run_computation":
            return _exec_computation(tool_input)
        else:
            return ToolResult(
                text=f"Unknown tool: {tool_name}",
                results=[],
            )
    except Exception as exc:
        logger.exception("Tool execution failed: %s", tool_name)
        return ToolResult(
            text=f"Tool '{tool_name}' failed: {exc}",
            results=[],
        )


# ------------------------------------------------------------------
# Individual tool executors
# ------------------------------------------------------------------

def _exec_search_knowledge_base(
    inputs: dict, pipeline: Any,
) -> ToolResult:
    """Search the local vectorstore."""
    query = inputs.get("query", "")
    n_results = inputs.get("n_results", 10)

    results = pipeline.store.search(query=query, n_results=n_results)
    text = _format_results_for_llm(results, label="Local Knowledge Base")
    return ToolResult(text=text, results=results)


def _exec_search_live(
    inputs: dict, pipeline: Any, source: str,
) -> ToolResult:
    """Search a live academic source (PubMed, bioRxiv, arXiv)."""
    from acheron.rag.live_retrieval import live_search, persist_high_relevance

    query = inputs.get("query", "")
    max_results = inputs.get("max_results", 5)

    query_results, papers, chunk_count = live_search(
        query, sources=[source], max_results=max_results,
    )

    # Persist high-relevance papers to the library.
    if papers:
        try:
            persist_high_relevance(papers)
        except Exception:
            pass  # Non-critical.

    label = {"pubmed": "PubMed", "biorxiv": "bioRxiv", "arxiv": "arXiv"}.get(
        source, source
    )
    text = _format_results_for_llm(query_results, label=label)
    return ToolResult(text=text, results=query_results)


def _exec_computation(inputs: dict) -> ToolResult:
    """Run a reasoning engine computation module."""
    from acheron.rag.hypothesis_engine import run_computation

    module = inputs.get("module", "")
    query = inputs.get("query", "")

    context = run_computation([module], query)
    return ToolResult(
        text=context or "No computation output.",
        results=[],
        computation_context=context,
    )


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------

def _format_results_for_llm(
    results: list[QueryResult], label: str = "Search",
) -> str:
    """Format QueryResult list as readable text for the LLM."""
    if not results:
        return f"[{label}] No results found."

    lines = [f"[{label}] Found {len(results)} passages:\n"]
    for i, r in enumerate(results, 1):
        author_str = ", ".join(r.authors[:3]) if r.authors else "Unknown"
        if len(r.authors) > 3:
            author_str += " et al."
        doi_str = f" DOI:{r.doi}" if r.doi else ""
        pmid_str = f" PMID:{r.pmid}" if r.pmid else ""
        score_str = f" (score: {r.relevance_score:.3f})" if r.relevance_score else ""

        lines.append(
            f"[{i}] {r.paper_title} — {author_str}{doi_str}{pmid_str}{score_str}"
        )
        # Show section and excerpt.
        if r.section:
            lines.append(f"    Section: {r.section}")
        excerpt = r.excerpt or r.text[:400]
        lines.append(f"    \"{excerpt}\"")
        lines.append("")

    return "\n".join(lines)
