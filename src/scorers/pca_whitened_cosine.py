"""PCA-whitened cosine scorer."""

from __future__ import annotations

from semantic_desider.features import PairFeatures
from semantic_desider.scorers.base import BaseScorer
from semantic_desider.scorers.utils import as_matrix, feature_vector, np_module
from semantic_desider.types import LabeledPairBatch, ScorerName, Score


class PCAWhitenedCosineScorer(BaseScorer):
    """PCA-truncated whitened cosine scorer."""

    _name = ScorerName("PCAWhitenedCosine")

    def __init__(self, *, n_components: int = 128, eps: float = 1e-6) -> None:
        self.n_components = int(n_components)
        self.eps = float(eps)
        self._mean = None
        self._transform = None
        self._prototype = None

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        np = np_module()
        del kwargs
        h0 = as_matrix(batch.h0, name="h0")
        h1 = as_matrix(batch.h1, name="h1")
        x = np.concatenate([h0, h1], axis=0).astype(np.float64, copy=False)
        if x.shape[0] == 0:
            raise ValueError("PCAWhitenedCosineScorer.fit received no rows")

        self._mean = x.mean(axis=0)
        centered = x - self._mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        k = max(1, min(self.n_components, vt.shape[0], x.shape[0] - 1, x.shape[1]))
        components = vt[:k].T
        variances = (singular_values[:k] ** 2) / max(1, x.shape[0] - 1)
        self._transform = components / np.sqrt(np.maximum(variances, self.eps))[None, :]

        z1 = (h1.astype(np.float64, copy=False) - self._mean) @ self._transform
        proto = z1.mean(axis=0) if z1.shape[0] else np.zeros(k, dtype=np.float64)
        norm = float(np.linalg.norm(proto))
        self._prototype = proto / norm if norm > self.eps else None

    def score(self, features: PairFeatures) -> Score:
        np = np_module()
        if self._mean is None or self._transform is None:
            raise ValueError("fit() must be called before score()")
        x = feature_vector(features).astype(np.float64, copy=False)
        z = (x - self._mean) @ self._transform
        if self._prototype is None:
            return Score(0.0)
        denom = max(float(np.linalg.norm(z)), self.eps)
        return Score(float(np.clip(np.dot(z, self._prototype) / denom, -1.0, 1.0)))
