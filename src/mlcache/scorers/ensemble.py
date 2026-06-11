"""Weighted ensemble scorer."""

from __future__ import annotations

from typing import Iterable

from mlcache.calibration import ThresholdCalibrationRequest
from mlcache.features import PairFeatures
from mlcache.scorers.base import BaseScorer, SemanticScorer
from mlcache.scorers.cosine import CosineScorer
from mlcache.scorers.lda import LDAScorer
from mlcache.scorers.mlp import TinyMLPScorer
from mlcache.scorers.pca_whitened import PCAWhitenedCosineScorer
from mlcache.scorers.utils import as_matrix, np_module, scores_to_threshold
from mlcache.semantic_types import InputSpace, LabeledPairBatch, ScorerName, Score


_WEIGHT_HOLDOUT_FRACTION = 0.3
_MIN_FIT_SAMPLES = 4


def _holdout_split(x):
    """Split rows into (fit_subset, weight_selection_subset).

    Falls back to using all rows for both subsets when there isn't enough
    data to carve out a non-degenerate holdout (e.g. tiny test fixtures);
    in that case weight selection is in-sample, same as before this fix.
    """
    np = np_module()
    n = x.shape[0]
    if n < _MIN_FIT_SAMPLES * 2:
        return x, x
    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    n_holdout = max(1, int(round(n * _WEIGHT_HOLDOUT_FRACTION)))
    n_holdout = min(n_holdout, n - _MIN_FIT_SAMPLES)
    holdout_idx = order[:n_holdout]
    fit_idx = order[n_holdout:]
    return x[fit_idx], x[holdout_idx]


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

    def copy_for_refit(self):
        return type(self)(judges=[judge.copy_for_refit() for judge in self._judges])

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        h0 = as_matrix(batch.h0, name="h0")
        h1 = as_matrix(batch.h1, name="h1")

        # Hold out a slice of each class for weight selection so that flexible,
        # *fitted* judges (lda/pca_whitened/xgboost/mlp) don't get to vote on
        # their own fitting data. Scored in-sample, those judges can partially
        # memorize this batch and look stronger than the unfit `cosine` judge,
        # biasing the weight search away from the component that actually
        # generalizes best. `cosine` is unaffected either way since it never
        # fits anything.
        h0_fit, h0_weight = _holdout_split(h0)
        h1_fit, h1_weight = _holdout_split(h1)

        fit_batch = LabeledPairBatch(h0=h0_fit, h1=h1_fit, weights=None, metadata=batch.metadata)
        for judge in self._judges:
            judge.fit(fit_batch, **kwargs)

        z0 = self._score_matrix(h0_weight)
        z1 = self._score_matrix(h1_weight)
        self._weights = self._fit_np_weights(z0, z1, alpha=float(kwargs.get("alpha", 0.05)))

    def _score_matrix(self, x):
        from mlcache.features import PairFeatures

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
