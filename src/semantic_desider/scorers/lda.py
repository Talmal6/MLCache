"""LDA scorer."""

from __future__ import annotations

from semantic_desider.features import PairFeatures
from semantic_desider.scorers.base import BaseScorer
from semantic_desider.scorers.utils import as_matrix, feature_vector, labels, np_module
from semantic_desider.types import LabeledPairBatch, ScorerName, Score


class LDAScorer(BaseScorer):
    _name = ScorerName("LDA")

    def __init__(self) -> None:
        self._clf = None
        self._w = None
        self._b = 0.0

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        np = np_module()
        del kwargs
        h0 = as_matrix(batch.h0, name="h0")
        h1 = as_matrix(batch.h1, name="h1")
        x = np.concatenate([h0, h1], axis=0)
        y = labels(h0, h1)
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(x, y)
            self._clf = clf
            self._w = None
            return
        except Exception:
            self._clf = None

        m0 = h0.mean(axis=0)
        m1 = h1.mean(axis=0)
        z0 = h0 - m0
        z1 = h1 - m1
        sw = (z0.T @ z0 + z1.T @ z1) / max(1, h0.shape[0] + h1.shape[0] - 2)
        sw = sw + 1e-3 * np.eye(sw.shape[0], dtype=np.float32)
        self._w = np.linalg.solve(sw, (m1 - m0).astype(np.float32))
        self._b = -0.5 * float(np.dot(m0 + m1, self._w))

    def score(self, features: PairFeatures) -> Score:
        x = feature_vector(features).reshape(1, -1)
        if self._clf is not None:
            return Score(float(self._clf.decision_function(x)[0]))
        if self._w is None:
            raise ValueError("fit() must be called before score()")
        return Score(float(x.reshape(-1) @ self._w + self._b))
