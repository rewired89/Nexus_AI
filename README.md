# Acheron Nexus

Domain-specific RAG assistant for **bioelectricity and biomedical research**. Collects papers from PubMed, bioRxiv, arXiv, and PhysioNet, indexes them into a vector store, and answers natural-language queries with grounded, cited responses.

## Focus Areas

- Planarian bioelectricity and regeneration
- EEG cognitive patterns and brain oscillations
- Ion channel dynamics and electrophysiology
- Bioelectric morphogenesis and pattern formation
- Memory mechanisms in regenerating tissue
- Bioelectric computing

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Collect papers from all sources
acheron collect

# 4. Index into vector store
acheron index

# 5. Query (interactive mode)
acheron query -i

# 6. Or start the web UI
acheron serve
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `acheron collect` | Harvest papers from PubMed, bioRxiv, arXiv, PhysioNet |
| `acheron index` | Build/update the vector store from collected papers |
| `acheron query "question"` | Ask a question (single shot) |
| `acheron query -i` | Interactive query mode |
| `acheron query -r "question"` | Retrieve passages without LLM generation |
| `acheron add paper.pdf` | Add a local PDF manually |
| `acheron stats` | Show collection statistics |
| `acheron serve` | Start the web interface |

## Collection Options

```bash
# Collect from specific source
acheron collect --source pubmed

# Custom search queries
acheron collect -t "voltage-gated potassium channels" -t "gap junction signaling"

# More results per query
acheron collect -n 100

# Also download PDFs
acheron collect --download-pdfs
```

## Architecture

```
src/acheron/
├── cli.py                 # Click CLI interface
├── config.py              # Pydantic settings from .env
├── models.py              # Domain models (Paper, TextChunk, etc.)
├── collectors/
│   ├── base.py            # Base collector with HTTP + retry
│   ├── pubmed.py          # PubMed/PMC via NCBI E-Utilities
│   ├── biorxiv.py         # bioRxiv content API
│   ├── arxiv.py           # arXiv Atom feed API
│   └── physionet.py       # PhysioNet dataset API
├── extraction/
│   ├── pdf_parser.py      # PDF → structured text (PyMuPDF + pdfplumber)
│   └── chunker.py         # Text → overlapping chunks
├── vectorstore/
│   └── store.py           # ChromaDB vector store
├── rag/
│   └── pipeline.py        # Retrieve → rerank → generate with citations
└── web/
    ├── app.py             # FastAPI application
    └── templates/
        └── index.html     # Web UI
```

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACHERON_LLM_API_KEY` | (required) | OpenAI-compatible API key |
| `ACHERON_LLM_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint |
| `ACHERON_LLM_MODEL` | `gpt-4o` | Model for generation |
| `ACHERON_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer model |
| `ACHERON_EMBEDDING_API_KEY` | (empty) | If set, uses OpenAI embeddings instead |
| `NCBI_API_KEY` | (empty) | Optional; increases PubMed rate limits |
| `ACHERON_DATA_DIR` | `./data` | Storage root |
| `ACHERON_HOST` | `127.0.0.1` | Web server host |
| `ACHERON_PORT` | `8000` | Web server port |

## Embedding Options

**Local (default):** Uses `sentence-transformers/all-MiniLM-L6-v2` — no API key needed, runs on CPU.

**API-based:** Set `ACHERON_EMBEDDING_API_KEY` and `ACHERON_EMBEDDING_MODEL` to use OpenAI's `text-embedding-3-small` or similar.

## LLM Options

Supports any OpenAI-compatible endpoint:

- **OpenAI**: Set API key, use `gpt-4o` or `gpt-4o-mini`
- **Ollama**: Set `ACHERON_LLM_BASE_URL=http://localhost:11434/v1`, model to `llama3.1` or similar
- **vLLM / TGI**: Point to your local endpoint

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

## License

MIT
