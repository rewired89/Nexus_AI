"""RAG pipeline — three-layer research architecture with discovery loop."""

# Lazy imports to avoid pulling in heavy dependencies (chromadb, openai)
# at module scan time. Use: from acheron.rag.pipeline import RAGPipeline
# or: from acheron.rag.ledger import ExperimentLedger

__all__ = ["RAGPipeline", "ExperimentLedger"]


def __getattr__(name: str):
    if name == "RAGPipeline":
        from acheron.rag.pipeline import RAGPipeline
        return RAGPipeline
    if name == "ExperimentLedger":
        from acheron.rag.ledger import ExperimentLedger
        return ExperimentLedger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
