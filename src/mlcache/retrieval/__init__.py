"""Semantic candidate retrieval contracts."""

from mlcache.retrieval.file_store import FileVectorStore
from mlcache.retrieval.in_memory import InMemoryVectorStore
from mlcache.retrieval.vector_store import VectorSearchResult, VectorStore

__all__ = ["FileVectorStore", "InMemoryVectorStore", "VectorSearchResult", "VectorStore"]
