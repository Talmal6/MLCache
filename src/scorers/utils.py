"""Compatibility wrapper for scorers.utils."""

from mlcache.scorers.utils import (
    as_matrix,
    feature_vector,
    labels,
    np_module,
    score_rows_with_scorer,
    scores_to_threshold,
)

__all__ = [
    "as_matrix",
    "feature_vector",
    "labels",
    "np_module",
    "score_rows_with_scorer",
    "scores_to_threshold",
]

