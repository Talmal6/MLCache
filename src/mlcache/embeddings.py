"""Embedding provider abstractions."""

from __future__ import annotations

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
                'Missing optional dependency sentence-transformers. Install with: python -m pip install -e ".[embeddings]"'
            ) from exc

        try:
            return SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}' (local_files_only={self.local_files_only}): {exc}"
            ) from exc


__all__ = ["EmbeddingProvider", "SentenceTransformersEmbeddingProvider"]