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
    # z-score normalization is affine-invariant for positive member rescaling,
    # but the two arithmetic paths differ in float rounding, so compare with a
    # tolerance rather than bit-exact equality.
    assert np.isclose(float(baseline.score(features)), float(rescaled.score(features)))


def test_equal_tpr_weight_solution_prefers_blended_anticollapse() -> None:
    # Both single-judge solutions ([1,0] and [0,1]) reach TPR=1.0 at alpha=0.5,
    # as does the uniform blend [0.5,0.5]. The weight search must NOT collapse
    # onto one judge: among equal-TPR feasible solutions it prefers the more
    # spread (higher-entropy) vector, which keeps a complementary judge in the
    # mix to break score ties. Collapsing is exactly what produced an
    # uncalibratable H0/H1 score atom in the online ensemble.
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

    assert np.allclose(weights, [0.5, 0.5])
