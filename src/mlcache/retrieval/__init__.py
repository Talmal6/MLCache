"""Semantic candidate retrieval contracts."""

from mlcache.retrieval.faiss_store import FaissOutboxIndexer, FaissVectorStore
from mlcache.retrieval.file_store import FileVectorStore
from mlcache.retrieval.in_memory import InMemoryVectorStore
from mlcache.retrieval.vector_store import VectorSearchResult, VectorStore

__all__ = [
    "FaissOutboxIndexer",
    "FaissVectorStore",
    "FileVectorStore",
    "InMemoryVectorStore",
    "VectorSearchResult",
    "VectorStore",
]
