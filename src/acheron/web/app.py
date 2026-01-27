"""FastAPI web application for Acheron Nexus — bioelectric research intelligence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from acheron.config import get_settings
from acheron.extraction.chunker import TextChunker
from acheron.extraction.pdf_parser import PDFParser
from acheron.models import PaperSource
from acheron.rag.pipeline import RAGPipeline
from acheron.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Acheron Nexus", version="0.2.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Lazy singletons
_store: VectorStore | None = None
_pipeline: RAGPipeline | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(store=get_store())
    return _pipeline


# ======================================================================
# API models
# ======================================================================
class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None
    n_results: int = 10
    retrieve_only: bool = False
    discover_mode: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model_used: str
    total_chunks_searched: int
    evidence: list[str] = []
    inference: list[str] = []
    speculation: list[str] = []


class DiscoverResponse(BaseModel):
    evidence: list[str]
    inference: list[str]
    speculation: list[str]
    variables: list[dict]
    hypotheses: list[dict]
    uncertainty: list[str]
    sources: list[dict]
    model_used: str
    total_chunks_searched: int


class StatsResponse(BaseModel):
    total_papers: int
    total_chunks: int
    papers_by_source: dict[str, int]
    ledger_entries: int = 0


# ======================================================================
# Web UI
# ======================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    store = get_store()
    stats = {
        "total_papers": len(store.list_papers()),
        "total_chunks": store.count(),
    }
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats})


# ======================================================================
# API endpoints
# ======================================================================
@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    pipeline = get_pipeline()

    if req.retrieve_only:
        results = pipeline.retrieve_only(
            req.question, n_results=req.n_results, filter_source=req.source_filter
        )
        return QueryResponse(
            answer="",
            sources=[
                {
                    "text": r.text,
                    "paper_title": r.paper_title,
                    "authors": r.authors,
                    "doi": r.doi,
                    "section": r.section,
                    "score": r.relevance_score,
                }
                for r in results
            ],
            model_used="",
            total_chunks_searched=len(results),
        )

    response = pipeline.query(
        req.question, filter_source=req.source_filter, n_results=req.n_results
    )
    return QueryResponse(
        answer=response.answer,
        sources=[
            {
                "text": s.text[:500],
                "paper_title": s.paper_title,
                "authors": s.authors,
                "doi": s.doi,
                "section": s.section,
                "score": s.relevance_score,
            }
            for s in response.sources
        ],
        model_used=response.model_used,
        total_chunks_searched=response.total_chunks_searched,
        evidence=response.evidence_statements,
        inference=response.inference_statements,
        speculation=response.speculation_statements,
    )


@app.post("/api/discover", response_model=DiscoverResponse)
async def api_discover(req: QueryRequest):
    """Execute the discovery loop and return structured results."""
    from acheron.rag.ledger import ExperimentLedger

    pipeline = get_pipeline()
    result = pipeline.discover(
        req.question, filter_source=req.source_filter, n_results=req.n_results
    )

    # Auto-log to ledger
    ledger = ExperimentLedger()
    ledger.record(result)

    return DiscoverResponse(
        evidence=result.evidence,
        inference=result.inference,
        speculation=result.speculation,
        variables=[
            {"name": v.name, "value": v.value, "unit": v.unit, "source_ref": v.source_ref}
            for v in result.variables
        ],
        hypotheses=[
            {
                "statement": h.statement,
                "confidence": h.confidence,
                "refs": h.supporting_refs,
                "validation": h.validation_strategy,
            }
            for h in result.hypotheses
        ],
        uncertainty=result.uncertainty_notes,
        sources=[
            {
                "text": s.text[:500],
                "paper_title": s.paper_title,
                "authors": s.authors,
                "doi": s.doi,
                "section": s.section,
                "score": s.relevance_score,
            }
            for s in result.sources
        ],
        model_used=result.model_used,
        total_chunks_searched=result.total_chunks_searched,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def api_stats():
    from acheron.rag.ledger import ExperimentLedger

    store = get_store()
    papers = store.list_papers()
    by_source: dict[str, int] = {}
    for p in papers:
        s = p.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1

    ledger = ExperimentLedger()

    return StatsResponse(
        total_papers=len(papers),
        total_chunks=store.count(),
        papers_by_source=by_source,
        ledger_entries=ledger.count(),
    )


@app.get("/api/papers")
async def api_papers():
    store = get_store()
    return store.list_papers()


@app.get("/api/ledger")
async def api_ledger():
    """Return all experiment ledger entries."""
    from acheron.rag.ledger import ExperimentLedger

    ledger = ExperimentLedger()
    entries = ledger.list_entries()
    return [e.model_dump(mode="json") for e in entries]


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    doi: str = Form(""),
    authors: str = Form(""),
):
    """Upload and index a PDF into the Library."""
    settings = get_settings()
    pdf_dir = settings.pdf_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)

    dest = pdf_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    parser = PDFParser()
    chunker = TextChunker()
    store = get_store()

    paper = parser.parse(str(dest))
    if not paper:
        return JSONResponse({"error": "Failed to parse PDF"}, status_code=400)

    if title:
        paper.title = title
    if doi:
        paper.doi = doi
        paper.paper_id = doi
    if authors:
        paper.authors = [a.strip() for a in authors.split(",")]
    paper.source = PaperSource.MANUAL

    chunks = chunker.chunk_paper(paper)
    added = store.add_chunks(chunks)

    return {"message": f"Indexed {added} chunks", "title": paper.title, "total": store.count()}
