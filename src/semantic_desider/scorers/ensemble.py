"""Weighted ensemble scorer."""

from __future__ import annotations

from typing import Iterable

from semantic_desider.features import PairFeatures
from semantic_desider.scorers.base import BaseScorer, SemanticScorer
from semantic_desider.scorers.cosine import CosineScorer
from semantic_desider.scorers.lda import LDAScorer
from semantic_desider.scorers.pca_whitened_cosine import PCAWhitenedCosineScorer
from semantic_desider.scorers.tiny_mlp import TinyMLPScorer
from semantic_desider.scorers.utils import as_matrix, np_module, scores_to_threshold
from semantic_desider.thresholds import ThresholdCalibrationRequest
from semantic_desider.types import InputSpace, LabeledPairBatch, ScorerName, Score


class EnsembleScorer(BaseScorer):
    """Non-negative weighted ensemble over fitted scorer judges."""

    _name = ScorerName("Ensemble")
    _input_space = InputSpace.MIXED

    def __init__(self, judges: Iterable[SemanticScorer] | None = None) -> None:
        self._judges = list(judges) if judges is not None else [
            CosineScorer(),
            PCAWhitenedCosineScorer(),
            LDAScorer(),
            TinyMLPScorer(),
        ]
        if not self._judges:
            raise ValueError("EnsembleScorer requires at least one judge")
        self._weights = tuple(1.0 / len(self._judges) for _ in self._judges)

    def members(self) -> Iterable[SemanticScorer]:
        return tuple(self._judges)

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        h0 = as_matrix(batch.h0, name="h0")
        h1 = as_matrix(batch.h1, name="h1")
        for judge in self._judges:
            judge.fit(batch, **kwargs)

        z0 = self._score_matrix(h0)
        z1 = self._score_matrix(h1)
        self._weights = self._fit_np_weights(z0, z1, alpha=float(kwargs.get("alpha", 0.05)))

    def _score_matrix(self, x):
        from semantic_desider.features import PairFeatures

        np = np_module()
        rows = []
        for row in x:
            features = PairFeatures(hadamard=tuple(float(v) for v in row))
            rows.append([float(judge.score(features)) for judge in self._judges])
        return np.asarray(rows, dtype=np.float64)

    def _fit_np_weights(self, z0, z1, *, alpha: float):
        np = np_module()
        m = z0.shape[1]
        candidates = []
        for idx in range(m):
            w = np.zeros(m, dtype=np.float64)
            w[idx] = 1.0
            candidates.append(w)
        candidates.append(np.ones(m, dtype=np.float64) / m)

        rng = np.random.default_rng(42)
        for _ in range(64):
            candidates.append(rng.dirichlet(np.ones(m, dtype=np.float64)))

        best_w = candidates[0]
        best_tpr = -1.0
        for w in candidates:
            s0 = z0 @ w
            tau = scores_to_threshold(
                ThresholdCalibrationRequest(
                    h0_scores=[Score(float(s)) for s in s0],
                    target_false_accept_rate=alpha,
                )
            )
            if not np.isfinite(float(tau)):
                continue
            s1 = z1 @ w
            tpr = float(np.mean(s1 >= float(tau)))
            if tpr > best_tpr:
                best_tpr = tpr
                best_w = w
        return best_w / max(float(best_w.sum()), 1e-12)

    def score(self, features: PairFeatures) -> Score:
        np = np_module()
        scores = np.asarray([float(judge.score(features)) for judge in self._judges], dtype=np.float64)
        return Score(float(scores @ self._weights))
