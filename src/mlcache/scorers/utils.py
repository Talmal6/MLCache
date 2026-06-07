"""Shared utilities for scorer implementations."""

from __future__ import annotations

from typing import Sequence

from mlcache.calibration import ThresholdCalibrationRequest
from mlcache.features import PairFeatures
from mlcache.semantic_types import Score, Threshold, TieMode


def np_module():
    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("semantic_desider scorers require numpy") from exc
    return np


def as_matrix(rows: Sequence[Sequence[float]], *, name: str):
    np = np_module()
    x = np.asarray(rows, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape={x.shape}")
    return x


def labels(h0, h1):
    np = np_module()
    return np.concatenate(
        [np.zeros(h0.shape[0], dtype=np.int32), np.ones(h1.shape[0], dtype=np.int32)],
        axis=0,
    )


def feature_vector(features: PairFeatures):
    np = np_module()
    if features.concat:
        return np.asarray(features.concat, dtype=np.float32)
    if features.hadamard:
        return np.asarray(features.hadamard, dtype=np.float32)
    if features.abs_diff:
        return np.asarray(features.abs_diff, dtype=np.float32)
    if features.cosine is not None:
        return np.asarray([float(features.cosine)], dtype=np.float32)
    main = features.values.get("main")
    if main is not None:
        return np.asarray(main, dtype=np.float32).reshape(-1)
    raise ValueError("PairFeatures contains no usable scorer input")


def scores_to_threshold(request: ThresholdCalibrationRequest) -> Threshold:
    np = np_module()
    scores = np.asarray(list(request.h0_scores), dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return Threshold(float("inf"))

    alpha = float(request.target_false_accept_rate)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"target_false_accept_rate must be in (0,1), got {alpha}")

    uniq, counts = np.unique(scores, return_counts=True)
    n = int(scores.size)
    cumsum = np.cumsum(counts)
    for idx, tau in enumerate(uniq):
        if request.tie_mode == TieMode.GT:
            false_accepts = int(n - cumsum[idx])
        else:
            false_accepts = int(n - (cumsum[idx - 1] if idx > 0 else 0))
        if false_accepts / max(1, n) <= alpha:
            return Threshold(float(tau))
    return Threshold(float("inf"))


def score_rows_with_scorer(rows, scorer):
    from mlcache.features import PairFeatures

    np = np_module()
    out = []
    for row in rows:
        features = PairFeatures(hadamard=tuple(float(v) for v in row))
        out.append(float(scorer.score(features)))
    return np.asarray(out, dtype=np.float64)
