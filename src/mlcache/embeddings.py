"""Embedding provider abstractions."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Return semantic embeddings for raw text."""

    @abstractmethod
    def embed(self, text: str) -> tuple[float, ...]:
        raise NotImplementedError


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Sentence-Transformers backed embedding provider."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.normalize = bool(normalize)
        self.local_files_only = bool(local_files_only)
        self._model = self._load_model()

    def embed(self, text: str) -> tuple[float, ...]:
        vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
        if self.normalize:
            norm = float(np.linalg.norm(array))
            if norm > 0.0:
                array = array / norm
        return tuple(float(value) for value in array.tolist())

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing optional dependency sentence-transformers. "
                "Install embedding dependencies with: pip install -e '.[embeddings,dev]'"
            ) from exc

        try:
            return SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}' (local_files_only={self.local_files_only}): {exc}"
            ) from exc


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedding provider using the hashing trick.

    Maps each whitespace token to a coordinate via a SHA-256 digest, so the
    same text always yields the same vector with no model download or network
    access. Useful for tests, smoke runs, and mock-LLM flows where a real
    embedding model isn't available or desired.
    """

    def __init__(self, *, dimensions: int = 64) -> None:
        self.dimensions = int(dimensions)

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return tuple(vector)


__all__ = ["EmbeddingProvider", "HashingEmbeddingProvider", "SentenceTransformersEmbeddingProvider"]