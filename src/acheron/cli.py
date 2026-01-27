"""Acheron Nexus CLI — collect papers, index, and query."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from acheron.config import get_settings

console = Console()

# Default search topics for bioelectricity research
DEFAULT_TOPICS = {
    "planarian_bioelectricity": (
        "planarian bioelectricity membrane voltage regeneration"
    ),
    "eeg_cognitive": (
        "EEG cognitive patterns brain oscillations neural dynamics"
    ),
    "ion_channels": (
        "ion channel dynamics voltage-gated signaling electrophysiology"
    ),
    "bioelectric_morphogenesis": (
        "bioelectric morphogenesis pattern formation developmental bioelectricity"
    ),
    "regeneration_memory": (
        "memory mechanisms regenerating tissue planarian decapitation"
    ),
    "bioelectric_computing": (
        "bioelectric computing biological computation cellular signaling networks"
    ),
}


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """Acheron Nexus — Bioelectricity research RAG assistant."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# COLLECT — harvest papers from sources
# ======================================================================
@main.command()
@click.option(
    "--source",
    type=click.Choice(["pubmed", "biorxiv", "arxiv", "physionet", "all"]),
    default="all",
    help="Which source to collect from",
)
@click.option("--topic", "-t", multiple=True, help="Custom search queries")
@click.option("--max-results", "-n", default=50, help="Max papers per query per source")
@click.option("--download-pdfs", is_flag=True, help="Also download PDFs")
def collect(source: str, topic: tuple, max_results: int, download_pdfs: bool) -> None:
    """Collect papers from academic sources."""
    from acheron.collectors.arxiv import ArxivCollector
    from acheron.collectors.biorxiv import BiorxivCollector
    from acheron.collectors.physionet import PhysioNetCollector
    from acheron.collectors.pubmed import PubMedCollector

    topics = list(topic) if topic else list(DEFAULT_TOPICS.values())

    collectors = []
    if source in ("pubmed", "all"):
        collectors.append(("PubMed", PubMedCollector()))
    if source in ("biorxiv", "all"):
        collectors.append(("bioRxiv", BiorxivCollector()))
    if source in ("arxiv", "all"):
        collectors.append(("arXiv", ArxivCollector()))
    if source in ("physionet", "all"):
        collectors.append(("PhysioNet", PhysioNetCollector()))

    total_papers = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for coll_name, collector in collectors:
            for query in topics:
                task = progress.add_task(
                    f"[cyan]{coll_name}[/]: {query[:50]}...", total=None
                )
                try:
                    papers = collector.search(query, max_results=max_results)
                    for paper in papers:
                        collector.save_metadata(paper)
                        if download_pdfs and paper.url:
                            collector.download_pdf(paper)
                    total_papers += len(papers)
                    progress.update(task, description=f"[green]{coll_name}[/]: {len(papers)} papers")
                except Exception as e:
                    progress.update(task, description=f"[red]{coll_name}[/]: error — {e}")
                finally:
                    progress.update(task, completed=True)
            collector.close()

    console.print(f"\n[bold green]Collected {total_papers} papers total.[/]")


# ======================================================================
# INDEX — build vector store from collected papers
# ======================================================================
@main.command()
@click.option("--reindex", is_flag=True, help="Force re-index all papers")
def index(reindex: bool) -> None:
    """Index collected papers into the vector store."""
    from acheron.extraction.chunker import TextChunker
    from acheron.extraction.pdf_parser import PDFParser
    from acheron.models import Paper
    from acheron.vectorstore.store import VectorStore

    settings = get_settings()
    store = VectorStore()
    parser = PDFParser()
    chunker = TextChunker()

    metadata_dir = settings.metadata_dir
    if not metadata_dir.exists():
        console.print("[red]No metadata found. Run 'acheron collect' first.[/]")
        return

    metadata_files = list(metadata_dir.glob("*.json"))
    console.print(f"Found {len(metadata_files)} paper metadata files")

    total_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing papers...", total=len(metadata_files))

        for meta_file in metadata_files:
            try:
                paper_data = json.loads(meta_file.read_text(encoding="utf-8"))
                paper = Paper(**paper_data)

                # If we have a PDF, parse it for full text
                if paper.pdf_path and Path(paper.pdf_path).exists():
                    parsed = parser.parse(paper.pdf_path)
                    if parsed:
                        paper.full_text = parsed.full_text
                        paper.sections = parsed.sections
                        paper.tables = parsed.tables

                # Chunk and index
                chunks = chunker.chunk_paper(paper)
                if chunks:
                    added = store.add_chunks(chunks)
                    total_chunks += added

            except Exception as e:
                logging.getLogger(__name__).debug("Error indexing %s: %s", meta_file.name, e)

            progress.advance(task)

    console.print(
        f"\n[bold green]Indexed {total_chunks} new chunks. "
        f"Total in store: {store.count()}[/]"
    )


# ======================================================================
# QUERY — ask questions
# ======================================================================
@main.command()
@click.argument("question", required=False)
@click.option("--source-filter", "-s", default=None, help="Filter by source")
@click.option("--n-results", "-n", default=10, help="Number of passages to retrieve")
@click.option("--retrieve-only", "-r", is_flag=True, help="Only retrieve, don't generate")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def query(
    question: str | None,
    source_filter: str | None,
    n_results: int,
    retrieve_only: bool,
    interactive: bool,
) -> None:
    """Query the bioelectricity knowledge base."""
    from acheron.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    if interactive or question is None:
        _interactive_loop(pipeline, source_filter, n_results, retrieve_only)
        return

    _run_query(pipeline, question, source_filter, n_results, retrieve_only)


def _interactive_loop(pipeline, source_filter, n_results, retrieve_only):
    """Interactive query loop."""
    console.print(
        Panel(
            "[bold cyan]Acheron Nexus[/] — Bioelectricity Research Assistant\n"
            "Type your question and press Enter. Type 'quit' to exit.",
            title="Interactive Mode",
        )
    )
    while True:
        try:
            q = console.input("\n[bold cyan]Question>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        _run_query(pipeline, q, source_filter, n_results, retrieve_only)


def _run_query(pipeline, question, source_filter, n_results, retrieve_only):
    """Execute a single query and display results."""
    with console.status("[bold cyan]Searching knowledge base..."):
        if retrieve_only:
            results = pipeline.retrieve_only(
                question, n_results=n_results, filter_source=source_filter
            )
            _display_retrieval_results(results)
        else:
            response = pipeline.query(
                question, filter_source=source_filter, n_results=n_results
            )
            _display_rag_response(response)


def _display_retrieval_results(results):
    """Display raw retrieval results."""
    if not results:
        console.print("[yellow]No results found.[/]")
        return

    table = Table(title="Retrieved Passages", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Score", width=6)
    table.add_column("Paper", max_width=50)
    table.add_column("Section", max_width=15)
    table.add_column("Text", max_width=80)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{r.relevance_score:.3f}",
            r.paper_title[:50],
            r.section[:15] if r.section else "",
            r.text[:200] + "..." if len(r.text) > 200 else r.text,
        )
    console.print(table)


def _display_rag_response(response):
    """Display a full RAG response with citations."""
    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title="[bold cyan]Answer[/]",
        border_style="cyan",
    ))

    # Source list
    console.print("\n[bold]Sources:[/]")
    seen = set()
    for i, src in enumerate(response.sources, 1):
        if src.paper_id in seen:
            continue
        seen.add(src.paper_id)
        doi_str = f" DOI: {src.doi}" if src.doi else ""
        author_str = ", ".join(src.authors[:3])
        if len(src.authors) > 3:
            author_str += " et al."
        console.print(
            f"  [{i}] {author_str}. [italic]\"{src.paper_title}\"[/].{doi_str}"
        )
    console.print(
        f"\n[dim]Model: {response.model_used} | "
        f"Chunks searched: {response.total_chunks_searched}[/]"
    )


# ======================================================================
# ADD — manually add a paper
# ======================================================================
@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", default=None, help="Paper title")
@click.option("--doi", default=None, help="Paper DOI")
@click.option("--authors", "-a", default=None, help="Comma-separated authors")
def add(path: str, title: str | None, doi: str | None, authors: str | None) -> None:
    """Add a paper manually from a local PDF file."""
    from acheron.extraction.chunker import TextChunker
    from acheron.extraction.pdf_parser import PDFParser
    from acheron.models import PaperSource
    from acheron.vectorstore.store import VectorStore

    parser = PDFParser()
    chunker = TextChunker()
    store = VectorStore()

    with console.status("[bold cyan]Parsing PDF..."):
        paper = parser.parse(path)

    if not paper:
        console.print("[red]Failed to parse PDF.[/]")
        return

    # Override metadata if provided
    if title:
        paper.title = title
    if doi:
        paper.doi = doi
        paper.paper_id = doi
    if authors:
        paper.authors = [a.strip() for a in authors.split(",")]
    paper.source = PaperSource.MANUAL

    with console.status("[bold cyan]Chunking and indexing..."):
        chunks = chunker.chunk_paper(paper)
        added = store.add_chunks(chunks)

    console.print(
        f"[bold green]Added '{paper.title}'[/] — {added} chunks indexed "
        f"(total: {store.count()})"
    )


# ======================================================================
# STATS — show collection statistics
# ======================================================================
@main.command()
def stats() -> None:
    """Show statistics about the indexed collection."""
    from acheron.vectorstore.store import VectorStore

    settings = get_settings()
    store = VectorStore()

    papers = store.list_papers()
    total_chunks = store.count()

    console.print(Panel(
        f"[bold]Papers:[/] {len(papers)}\n"
        f"[bold]Chunks:[/] {total_chunks}\n"
        f"[bold]Data dir:[/] {settings.data_dir}\n"
        f"[bold]Vector store:[/] {settings.vectorstore_dir}\n"
        f"[bold]Embedding model:[/] {settings.embedding_model}\n"
        f"[bold]LLM model:[/] {settings.llm_model}",
        title="[bold cyan]Acheron Nexus Stats[/]",
    ))

    if papers:
        # Source breakdown
        by_source: dict[str, int] = {}
        for p in papers:
            s = p.get("source", "unknown")
            by_source[s] = by_source.get(s, 0) + 1

        table = Table(title="Papers by Source")
        table.add_column("Source")
        table.add_column("Count", justify="right")
        for src, cnt in sorted(by_source.items()):
            table.add_row(src, str(cnt))
        console.print(table)


# ======================================================================
# SERVE — web UI
# ======================================================================
@main.command()
@click.option("--host", default=None, help="Override host")
@click.option("--port", "-p", default=None, type=int, help="Override port")
def serve(host: str | None, port: int | None) -> None:
    """Start the web interface."""
    import uvicorn

    from acheron.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "acheron.web.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
