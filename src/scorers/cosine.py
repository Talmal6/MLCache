"""Cosine scorer."""

from __future__ import annotations

from features import PairFeatures
from scorers.base import BaseScorer
from scorers.utils import feature_vector, np_module
from semantic_types import LabeledPairBatch, ScorerName, Score


class CosineScorer(BaseScorer):
    """Cosine baseline; with Hadamard features this is sum(U * V)."""

    _name = ScorerName("Cosine")

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del batch, kwargs

    def score(self, features: PairFeatures) -> Score:
        np = np_module()
        if features.cosine is not None:
            return Score(float(features.cosine))
        if features.hadamard:
            return Score(float(np.sum(np.asarray(features.hadamard, dtype=np.float32))))
        return Score(float(np.sum(feature_vector(features))))
