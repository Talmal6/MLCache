from __future__ import annotations

import numpy as np

from mlcache.features import PairFeatures
from mlcache.scorers.base import BaseScorer
from mlcache.scorers.ensemble import EnsembleScorer
from mlcache.semantic_types import InputSpace, LabeledPairBatch, ScorerName, Score


class _ColumnScorer(BaseScorer):
    _input_space = InputSpace.MIXED

    def __init__(self, column: int, *, scale: float = 1.0, offset: float = 0.0) -> None:
        self._column = column
        self._scale = scale
        self._offset = offset
        self._name = ScorerName(f"column_{column}")

    def copy_for_refit(self) -> "_ColumnScorer":
        return type(self)(self._column, scale=self._scale, offset=self._offset)

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        del batch, kwargs

    def score(self, features: PairFeatures) -> Score:
        value = features.hadamard[self._column]
        return Score(float(value * self._scale + self._offset))


def test_ensemble_weight_fit_is_invariant_to_member_score_scale() -> None:
    h0 = [(float(i), float((i * 7) % 41)) for i in range(40)]
    h1 = [(float(30 + i), float((i * 11) % 41)) for i in range(20)]
    batch = LabeledPairBatch(h0=h0, h1=h1)

    baseline = EnsembleScorer(judges=[_ColumnScorer(0), _ColumnScorer(1)])
    rescaled = EnsembleScorer(
        judges=[
            _ColumnScorer(0, scale=1000.0, offset=-700.0),
            _ColumnScorer(1, scale=0.001, offset=5.0),
        ]
    )

    baseline.fit(batch, alpha=0.2)
    rescaled.fit(batch, alpha=0.2)

    assert np.allclose(baseline.weights, rescaled.weights)
    features = PairFeatures(hadamard=(35.0, 17.0))
    assert float(baseline.score(features)) == float(rescaled.score(features))


def test_equal_tpr_weight_solution_uses_more_of_the_fpr_budget() -> None:
    ensemble = EnsembleScorer(judges=[_ColumnScorer(0), _ColumnScorer(1)])
    z0 = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=np.float64,
    )
    z1 = np.asarray([[2.0, 4.0], [3.0, 5.0]], dtype=np.float64)

    weights = ensemble._fit_np_weights(z0, z1, alpha=0.5)

    assert np.allclose(weights, [0.0, 1.0])
