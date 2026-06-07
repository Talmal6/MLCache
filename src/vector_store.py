"""Compatibility wrapper for the old flat vector_store module."""

from mlcache.retrieval import FileVectorStore, InMemoryVectorStore, VectorSearchResult, VectorStore

__all__ = ["FileVectorStore", "InMemoryVectorStore", "VectorSearchResult", "VectorStore"]
