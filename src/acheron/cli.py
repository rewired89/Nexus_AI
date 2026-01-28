"""Acheron Nexus CLI — System-2 bioelectric research engine.

Three-layer architecture:
  Layer 1 (Knowledge)  — immutable source corpus
  Layer 2 (Synthesis)  — semantic retrieval, bioelectric variable extraction
  Layer 3 (Discovery)  — analysis, pattern detection, hypothesis generation,
                          bioelectric schematic construction, validation paths
"""

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
    """Nexus — Bioelectric research intelligence (Acheron project)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ======================================================================
# COLLECT — harvest papers from sources (Library layer)
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
    """Collect papers into the Library from academic sources."""
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

    console.print(f"\n[bold green]Library updated: {total_papers} papers collected.[/]")


# ======================================================================
# INDEX — build vector store from collected papers (Index layer)
# ======================================================================
@main.command()
@click.option("--reindex", is_flag=True, help="Force re-index all papers")
def index(reindex: bool) -> None:
    """Build the Index layer from Library material."""
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
        console.print("[red]Library empty. Run 'acheron collect' first.[/]")
        return

    metadata_files = list(metadata_dir.glob("*.json"))
    console.print(f"Library contains {len(metadata_files)} paper records")

    total_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building Index...", total=len(metadata_files))

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
        f"\n[bold green]Index built: {total_chunks} new chunks. "
        f"Total in store: {store.count()}[/]"
    )


# ======================================================================
# QUERY — ask questions (Layer 2+3)
# ======================================================================
@main.command()
@click.argument("question", required=False)
@click.option("--source-filter", "-s", default=None, help="Filter by source")
@click.option("--n-results", "-n", default=10, help="Number of passages to retrieve")
@click.option("--retrieve-only", "-r", is_flag=True, help="Layer 2 only — no Compute")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def query(
    question: str | None,
    source_filter: str | None,
    n_results: int,
    retrieve_only: bool,
    interactive: bool,
) -> None:
    """Query the research corpus (Index + Compute layers)."""
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
            "[bold cyan]Nexus[/] — System-2 Bioelectric Research Engine\n"
            "Three-layer architecture: Knowledge | Synthesis | Discovery\n\n"
            "Commands:\n"
            "  [dim]Type a question to query (Synthesis + Discovery)[/]\n"
            "  [dim]Prefix with[/] /discover [dim]to run the full discovery loop[/]\n"
            "  [dim]Type[/] quit [dim]to exit[/]",
            title="Acheron Nexus",
            border_style="cyan",
        )
    )
    while True:
        try:
            q = console.input("\n[bold cyan]nexus>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        if q.startswith("/discover "):
            _run_discover(pipeline, q[10:].strip(), source_filter, n_results)
        else:
            _run_query(pipeline, q, source_filter, n_results, retrieve_only)


def _run_query(pipeline, question, source_filter, n_results, retrieve_only):
    """Execute a single query and display results."""
    with console.status("[bold cyan]Retrieving from Index..."):
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


def _run_discover(pipeline, question, source_filter, n_results):
    """Execute the discovery loop and display structured results."""
    from acheron.rag.ledger import ExperimentLedger

    with console.status("[bold cyan]Running discovery loop (Compute layer)..."):
        result = pipeline.discover(
            question, filter_source=source_filter, n_results=n_results
        )

    _display_discovery_result(result)

    # Auto-log to ledger
    ledger = ExperimentLedger()
    entry = ledger.record(result)
    console.print(f"\n[dim]Logged to experiment ledger: {entry.entry_id}[/]")


def _display_retrieval_results(results):
    """Display raw retrieval results (Layer 2 only)."""
    if not results:
        console.print("[yellow]No results in Index.[/]")
        return

    table = Table(title="Retrieved Passages (Layer 2)", show_lines=True)
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
    """Display a full RAG response with epistemic structure."""
    console.print()

    # Show structured sections if parsed
    has_structured = (
        response.evidence_statements
        or response.inference_statements
        or response.speculation_statements
    )
    if has_structured:
        if response.evidence_statements:
            console.print(Panel(
                "\n".join(f"  - {s}" for s in response.evidence_statements),
                title="[bold green]EVIDENCE[/]",
                border_style="green",
            ))
        if response.inference_statements:
            console.print(Panel(
                "\n".join(f"  - {s}" for s in response.inference_statements),
                title="[bold yellow]INFERENCE[/]",
                border_style="yellow",
            ))
        if response.speculation_statements:
            console.print(Panel(
                "\n".join(f"  - {s}" for s in response.speculation_statements),
                title="[bold red]SPECULATION[/]",
                border_style="red",
            ))
        if response.bioelectric_schematic:
            console.print(Panel(
                response.bioelectric_schematic,
                title="[bold magenta]BIOELECTRIC SCHEMATIC[/]",
                border_style="magenta",
            ))
    else:
        # Fallback: display raw answer
        console.print(Panel(
            Markdown(response.answer),
            title="[bold cyan]Response[/]",
            border_style="cyan",
        ))

    # Source list
    console.print("\n[bold]Sources (Knowledge):[/]")
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
        f"\n[dim]Discovery: {response.model_used} | "
        f"Synthesis: {response.total_chunks_searched} chunks searched[/]"
    )


def _display_discovery_result(result):
    """Display a full discovery loop result with all structured fields."""
    console.print()

    # Evidence
    if result.evidence:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.evidence),
            title="[bold green]1. EVIDENCE EXTRACTION[/]",
            border_style="green",
        ))

    # Variables
    if result.variables:
        var_table = Table(title="2. BIOELECTRIC VARIABLES", show_lines=True)
        var_table.add_column("Name", style="cyan")
        var_table.add_column("Value")
        var_table.add_column("Unit", style="dim")
        var_table.add_column("Type", style="dim")
        var_table.add_column("Source", style="dim")
        for v in result.variables:
            var_table.add_row(
                v.name, v.value, v.unit, v.variable_type, v.source_ref
            )
        console.print(var_table)

    # Inference / Patterns
    if result.inference:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.inference),
            title="[bold yellow]3. PATTERN COMPARISON (Cross-source)[/]",
            border_style="yellow",
        ))

    # Hypotheses
    if result.hypotheses:
        hyp_table = Table(title="4. HYPOTHESES", show_lines=True)
        hyp_table.add_column("Hypothesis", max_width=60)
        hyp_table.add_column("Prior\nConf.", width=8)
        hyp_table.add_column("Predicted Impact", max_width=40)
        hyp_table.add_column("Refs", width=10)
        hyp_table.add_column("Validation", max_width=40)
        for h in result.hypotheses:
            conf_style = {"high": "green", "medium": "yellow", "low": "red"}.get(
                h.confidence, "dim"
            )
            hyp_table.add_row(
                h.statement[:60],
                f"[{conf_style}]{h.confidence}[/{conf_style}]",
                h.predicted_impact[:40] if h.predicted_impact else "",
                ", ".join(h.supporting_refs[:3]),
                h.validation_strategy[:40] if h.validation_strategy else "",
            )
        console.print(hyp_table)

    # Bioelectric Schematic
    if result.bioelectric_schematic:
        console.print(Panel(
            result.bioelectric_schematic,
            title="[bold magenta]5. BIOELECTRIC SCHEMATIC[/]",
            border_style="magenta",
        ))

    # Validation Path
    if result.validation_path:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.validation_path),
            title="[bold blue]6. VALIDATION PATH[/]",
            border_style="blue",
        ))

    # Cross-species Notes
    if result.cross_species_notes:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.cross_species_notes),
            title="[bold cyan]7. CROSS-SPECIES NOTES[/]",
            border_style="cyan",
        ))

    # Speculation
    if result.speculation:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.speculation),
            title="[bold red]SPECULATION[/]",
            border_style="red",
        ))

    # Uncertainty
    if result.uncertainty_notes:
        console.print(Panel(
            "\n".join(f"  - {s}" for s in result.uncertainty_notes),
            title="[bold dim]8. UNCERTAINTY[/]",
            border_style="dim",
        ))

    # Sources
    console.print("\n[bold]Sources (Knowledge):[/]")
    seen = set()
    for i, src in enumerate(result.sources, 1):
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
        f"\n[dim]Discovery: {result.model_used} | "
        f"Synthesis: {result.total_chunks_searched} chunks searched[/]"
    )


# ======================================================================
# DISCOVER — run the full discovery loop (dedicated command)
# ======================================================================
@main.command()
@click.argument("question")
@click.option("--source-filter", "-s", default=None, help="Filter by source")
@click.option("--n-results", "-n", default=12, help="Number of passages to retrieve")
@click.option("--tag", "-t", multiple=True, help="Tags for the ledger entry")
@click.option("--notes", default="", help="Notes to attach to the ledger entry")
def discover(
    question: str,
    source_filter: str | None,
    n_results: int,
    tag: tuple,
    notes: str,
) -> None:
    """Run the discovery loop: retrieve, extract, compare, hypothesize, log."""
    from acheron.rag.ledger import ExperimentLedger
    from acheron.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    ledger = ExperimentLedger()

    with console.status("[bold cyan]Executing discovery loop (Compute layer)..."):
        result = pipeline.discover(
            question, filter_source=source_filter, n_results=n_results
        )

    _display_discovery_result(result)

    # Log to ledger
    entry = ledger.record(result, notes=notes, tags=list(tag))
    console.print(f"\n[bold green]Recorded in experiment ledger:[/] {entry.entry_id}")


# ======================================================================
# LEDGER — view experiment ledger
# ======================================================================
@main.command()
@click.option("--tag", "-t", default=None, help="Filter entries by tag")
@click.option("--search", "-s", default=None, help="Search entries by keyword")
@click.option("--export", "-e", default=None, type=click.Path(), help="Export to JSON file")
@click.option("--entry-id", default=None, help="View a specific entry")
def ledger(
    tag: str | None,
    search: str | None,
    export: str | None,
    entry_id: str | None,
) -> None:
    """View and search the experiment ledger."""
    from acheron.rag.ledger import ExperimentLedger

    ledger_store = ExperimentLedger()

    if export:
        count = ledger_store.export_all(Path(export))
        console.print(f"[green]Exported {count} entries to {export}[/]")
        return

    if entry_id:
        entry = ledger_store.get_entry(entry_id)
        if not entry:
            console.print(f"[red]Entry not found: {entry_id}[/]")
            return
        _display_ledger_entry(entry)
        return

    if search:
        entries = ledger_store.search_entries(search)
    else:
        entries = ledger_store.list_entries(tag=tag)

    if not entries:
        console.print("[yellow]No ledger entries found.[/]")
        return

    table = Table(title=f"Experiment Ledger ({len(entries)} entries)", show_lines=True)
    table.add_column("ID", max_width=35)
    table.add_column("Timestamp", width=19)
    table.add_column("Query", max_width=50)
    table.add_column("Hypotheses", width=5, justify="right")
    table.add_column("Tags")

    for e in entries:
        table.add_row(
            e.entry_id,
            e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            e.query[:50],
            str(len(e.hypotheses)),
            ", ".join(e.tags) if e.tags else "",
        )
    console.print(table)


def _display_ledger_entry(entry):
    """Display a single ledger entry in detail."""
    console.print(Panel(
        f"[bold]ID:[/] {entry.entry_id}\n"
        f"[bold]Time:[/] {entry.timestamp}\n"
        f"[bold]Query:[/] {entry.query}\n"
        f"[bold]Tags:[/] {', '.join(entry.tags) if entry.tags else '(none)'}\n"
        f"[bold]Notes:[/] {entry.notes or '(none)'}",
        title="[bold cyan]Ledger Entry[/]",
    ))
    if entry.evidence_summary:
        console.print(Panel(
            entry.evidence_summary,
            title="[green]Evidence Summary[/]",
            border_style="green",
        ))
    if entry.hypotheses:
        for i, h in enumerate(entry.hypotheses, 1):
            console.print(f"  [bold]H{i}.[/] [{h.confidence}] {h.statement}")
            if h.validation_strategy:
                console.print(f"      [dim]Validation: {h.validation_strategy}[/]")
    if entry.variables:
        for v in entry.variables:
            console.print(f"  [cyan]{v.name}[/] = {v.value} ({v.unit}) {v.source_ref}")
    console.print(f"\n[dim]Source papers: {len(entry.source_ids)}[/]")


# ======================================================================
# ADD — manually add a paper (Library layer)
# ======================================================================
@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", default=None, help="Paper title")
@click.option("--doi", default=None, help="Paper DOI")
@click.option("--authors", "-a", default=None, help="Comma-separated authors")
def add(path: str, title: str | None, doi: str | None, authors: str | None) -> None:
    """Add a paper to the Library from a local PDF file."""
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

    if title:
        paper.title = title
    if doi:
        paper.doi = doi
        paper.paper_id = doi
    if authors:
        paper.authors = [a.strip() for a in authors.split(",")]
    paper.source = PaperSource.MANUAL

    with console.status("[bold cyan]Indexing into vector store..."):
        chunks = chunker.chunk_paper(paper)
        added = store.add_chunks(chunks)

    console.print(
        f"[bold green]Library updated:[/] '{paper.title}' — {added} chunks indexed "
        f"(total: {store.count()})"
    )


# ======================================================================
# STATS — show collection statistics
# ======================================================================
@main.command()
def stats() -> None:
    """Show status of all three layers."""
    from acheron.rag.ledger import ExperimentLedger
    from acheron.vectorstore.store import VectorStore

    settings = get_settings()
    store = VectorStore()
    ledger_store = ExperimentLedger()

    papers = store.list_papers()
    total_chunks = store.count()
    ledger_count = ledger_store.count()

    # Count metadata files in Library
    metadata_dir = settings.metadata_dir
    lib_count = len(list(metadata_dir.glob("*.json"))) if metadata_dir.exists() else 0

    console.print(Panel(
        f"[bold]Layer 1 — Knowledge:[/]\n"
        f"  Paper records:    {lib_count}\n"
        f"  Data dir:         {settings.data_dir}\n\n"
        f"[bold]Layer 2 — Synthesis:[/]\n"
        f"  Papers indexed:   {len(papers)}\n"
        f"  Chunks indexed:   {total_chunks}\n"
        f"  Vector store:     {settings.vectorstore_dir}\n"
        f"  Embedding model:  {settings.embedding_model}\n\n"
        f"[bold]Layer 3 — Discovery:[/]\n"
        f"  LLM model:        {settings.llm_model}\n"
        f"  Ledger entries:   {ledger_count}",
        title="[bold cyan]Nexus Status[/]",
        border_style="cyan",
    ))

    if papers:
        by_source: dict[str, int] = {}
        for p in papers:
            s = p.get("source", "unknown")
            by_source[s] = by_source.get(s, 0) + 1

        table = Table(title="Library Sources")
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
