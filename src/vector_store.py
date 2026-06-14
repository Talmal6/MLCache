"""Compatibility wrapper for the old flat vector_store module."""

from mlcache.retrieval import (
    FaissOutboxIndexer,
    FaissVectorStore,
    FileVectorStore,
    InMemoryVectorStore,
    VectorSearchResult,
    VectorStore,
)

__all__ = [
    "FaissOutboxIndexer",
    "FaissVectorStore",
    "FileVectorStore",
    "InMemoryVectorStore",
    "VectorSearchResult",
    "VectorStore",
]
