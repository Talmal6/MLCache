"""Normalized Hadamard pair feature builder."""

from __future__ import annotations

from math import sqrt

from mlcache.features.base import PairFeatureBuilder, PairFeatureKind, PairFeatures
from mlcache.semantic_types import Embedding, Score


class NormalizedHadamardFeatureBuilder(PairFeatureBuilder):
    """Build elementwise products of L2-normalized embeddings."""

    def build(self, query_embedding: Embedding, candidate_embedding: Embedding) -> PairFeatures:
        query = self._normalize(query_embedding, name="query_embedding")
        candidate = self._normalize(candidate_embedding, name="candidate_embedding")
        if len(query) != len(candidate):
            raise ValueError(
                "query_embedding and candidate_embedding must have the same length, "
                f"got {len(query)} and {len(candidate)}"
            )

        hadamard = tuple(float(q * c) for q, c in zip(query, candidate))
        cosine = Score(float(sum(hadamard)))
        return PairFeatures(
            cosine=cosine,
            hadamard=hadamard,
            values={
                "feature_type": "normalized_hadamard",
                "embedding_dim": len(hadamard),
                "dtype": "float64",
            },
        )

    def default_kind(self) -> PairFeatureKind:
        return PairFeatureKind.HADAMARD

    @staticmethod
    def _normalize(embedding: Embedding, *, name: str) -> tuple[float, ...]:
        values = tuple(float(value) for value in embedding)
        norm_sq = sum(value * value for value in values)
        if norm_sq <= 0.0:
            raise ValueError(f"{name} must not be a zero vector")
        norm = sqrt(norm_sq)
        return tuple(float(value / norm) for value in values)
