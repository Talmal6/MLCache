"""Tiny MLP scorer."""

from __future__ import annotations

from semantic_desider.features import PairFeatures
from semantic_desider.scorers.base import BaseScorer
from semantic_desider.scorers.utils import as_matrix, feature_vector, labels, np_module
from semantic_desider.types import LabeledPairBatch, ScorerName, Score


class TinyMLPScorer(BaseScorer):
    _name = ScorerName("Tiny MLP")

    def __init__(self, *, hidden_layer_sizes: tuple[int, ...] = (16,), max_iter: int = 800) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = int(max_iter)
        self._clf = None

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        np = np_module()
        h0 = as_matrix(batch.h0, name="h0")
        h1 = as_matrix(batch.h1, name="h1")
        x = np.concatenate([h0, h1], axis=0)
        y = labels(h0, h1)
        try:
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.neural_network import MLPClassifier
        except Exception as exc:
            raise ImportError("TinyMLPScorer requires scikit-learn") from exc

        import warnings

        seed = kwargs.get("seed", None)
        clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            alpha=0.001,
            random_state=int(seed) if seed is not None else 42,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(x, y)
        self._clf = clf

    def score(self, features: PairFeatures) -> Score:
        if self._clf is None:
            raise ValueError("fit() must be called before score()")
        x = feature_vector(features).reshape(1, -1)
        return Score(float(self._clf.predict_proba(x)[0, 1]))
