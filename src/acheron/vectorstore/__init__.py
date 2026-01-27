"""Vector storage and retrieval."""

__all__ = ["VectorStore"]


def __getattr__(name: str):
    if name == "VectorStore":
        from acheron.vectorstore.store import VectorStore
        return VectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
