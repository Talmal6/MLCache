"""XGBoost scorer."""

from __future__ import annotations

from semantic_desider.features import PairFeatures
from semantic_desider.scorers.base import BaseScorer
from semantic_desider.scorers.utils import as_matrix, feature_vector, labels, np_module
from semantic_desider.types import LabeledPairBatch, ScorerName, Score


class XGBoostScorer(BaseScorer):
    _name = ScorerName("XGBoost")

    def __init__(self, **model_kwargs: object) -> None:
        self._model_kwargs = dict(model_kwargs)
        self._clf = None

    def fit(self, batch: LabeledPairBatch, **kwargs: object) -> None:
        np = np_module()
        h0 = as_matrix(batch.h0, name="h0")
        h1 = as_matrix(batch.h1, name="h1")
        x = np.concatenate([h0, h1], axis=0)
        y = labels(h0, h1)
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ImportError("XGBoostScorer requires the xgboost package") from exc

        seed = kwargs.get("seed", None)
        params = {
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "verbosity": 0,
            "eval_metric": "logloss",
            "random_state": int(seed) if seed is not None else 42,
        }
        params.update(self._model_kwargs)
        clf = XGBClassifier(**params)
        clf.fit(x, y)
        self._clf = clf

    def score(self, features: PairFeatures) -> Score:
        if self._clf is None:
            raise ValueError("fit() must be called before score()")
        x = feature_vector(features).reshape(1, -1)
        return Score(float(self._clf.predict_proba(x)[0, 1]))
