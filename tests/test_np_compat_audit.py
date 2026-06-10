"""Tests for the NP-compatibility audit (experiments/np_compat_audit.py).

These verify the two parity claims the audit rests on, plus the structural
invariants of np_compat_pair_eval, without needing the external NeighborCache
repo:

  1. MLCache `scores_to_threshold` (the np_compat threshold) is byte-identical to
     the NeighborCache `_select_tau` rule with guardrail='none' (the old NP
     experiment's tau selection), for both tie modes, on random score sets.
  2. The cosine scorer on a hadamard row equals sum(hadamard) == cosine-to-anchor
     (so cosine in np_compat matches NP PrecomputedCosine).
  3. split_global produces disjoint, correctly-labelled train/calib/eval.
  4. A tiny deterministic end-to-end np_compat run keeps threshold/scorer frozen
     during eval and reports denominators that match the raw counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
for _p in (str(ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import np_compat_audit as nca  # noqa: E402
from mlcache.builder import build_scorer  # noqa: E402
from mlcache.features import PairFeatures  # noqa: E402


def _np_select_tau_reference(scores: np.ndarray, *, alpha: float, tie_mode: str) -> float:
    """Verbatim port of NeighborCache evaluation._select_tau guardrail='none'."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return float("inf")
    uniq, counts = np.unique(s, return_counts=True)
    n = int(s.size)
    cumsum = np.cumsum(counts)
    for i, tau in enumerate(uniq):
        if tie_mode == "gt":
            k = int(n - cumsum[i])
        else:
            k = int(n - (cumsum[i - 1] if i > 0 else 0))
        if (k / max(1, n)) <= alpha:
            return float(tau)
    return float("inf")


@pytest.mark.parametrize("tie_mode", ["ge", "gt"])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
def test_threshold_rule_matches_np_select_tau(tie_mode, alpha):
    from mlcache.calibration import ThresholdCalibrationRequest
    from mlcache.scorers.utils import scores_to_threshold
    from mlcache.semantic_types import Score, TieMode

    rng = np.random.default_rng(0)
    for _ in range(50):
        # mix of unique and heavily-tied scores to stress tie handling
        scores = np.round(rng.normal(size=int(rng.integers(5, 200))), 2)
        ref = _np_select_tau_reference(scores, alpha=alpha, tie_mode=tie_mode)
        got = float(
            scores_to_threshold(
                ThresholdCalibrationRequest(
                    h0_scores=[Score(float(s)) for s in scores],
                    target_false_accept_rate=alpha,
                    tie_mode=TieMode.GT if tie_mode == "gt" else TieMode.GE,
                )
            )
        )
        assert (got == ref) or (np.isinf(got) and np.isinf(ref)), (
            f"tau mismatch tie={tie_mode} alpha={alpha}: got={got} ref={ref}"
        )


def test_cosine_scorer_on_hadamard_equals_sum():
    scorer = build_scorer("cosine")
    rng = np.random.default_rng(1)
    for _ in range(20):
        row = rng.normal(size=16).astype(np.float32)
        score = float(scorer.score(PairFeatures(hadamard=tuple(float(v) for v in row))))
        assert score == pytest.approx(float(np.sum(row)), rel=1e-5, abs=1e-5)


def test_hadamard_cos_equals_cosine_to_anchor():
    # cos column produced by build_hadamard_features must equal cosine(row, anchor)
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(60, 8)).astype(np.float32)
    region = np.array([i % 5 for i in range(60)], dtype=np.int64)
    had, cos, anchors = nca.build_hadamard_features(emb, region, strategy="centroid_nearest", seed=7)
    Xn = nca.l2_normalize_rows(emb)
    for i in range(emb.shape[0]):
        a = anchors[int(region[i])]
        assert float(cos[i, 0]) == pytest.approx(float(np.dot(Xn[i], a)), rel=1e-5, abs=1e-5)


def test_split_global_disjoint_and_labelled():
    rng = np.random.default_rng(3)
    y = np.where(rng.random(400) < 0.3, 1, 0).astype(np.int32)
    # inject some uncertain rows that must never appear in any split
    y[:20] = -1
    split = nca.split_global(y, n_train=20, n_calib=20, n_eval=20, seed=42)
    pooled = np.concatenate(list(split.values()))
    assert pooled.size == len(set(pooled.tolist())), "splits overlap"
    assert not np.any(y[pooled] == -1), "uncertain rows leaked into a split"
    for k in ("H0_train", "H0_calib", "H0_eval"):
        assert np.all(y[split[k]] == 0)
    for k in ("H1_train", "H1_calib", "H1_eval"):
        assert np.all(y[split[k]] == 1)


def test_forced_gold_anchor_matches_np_compat():
    """The online wrapper (feature_builder + scorer + _predict_score) fed the gold
    anchor must reproduce np_compat scores/accepts within tolerance, and keep the
    scorer + threshold frozen during eval."""
    rng = np.random.default_rng(5)
    h1 = rng.normal(0.6, 1.0, size=(160, 12)).astype(np.float32)
    h0 = rng.normal(0.0, 1.0, size=(320, 12)).astype(np.float32)
    emb = np.vstack([h0, h1])
    label = np.array([0] * 320 + [1] * 160, dtype=np.int32)
    region = (rng.random(emb.shape[0]) * 6).astype(np.int64)

    had, _cos, _a = nca.build_hadamard_features(emb, region, strategy="centroid_nearest", seed=42)
    anchor_idx = nca.select_region_anchor_indices(emb, region, strategy="centroid_nearest", seed=42)
    split = nca.split_global(label, n_train=60, n_calib=60, n_eval=60, seed=42)

    for name in ("cosine", "lda", "pca_whitened_cosine"):
        r = nca.evaluate_scorer_forced_gold_anchor(
            name, emb, had, split, anchor_idx, region,
            target_fpr=0.1, seed=42, match_tol=1e-4,
        )
        assert r["scorer_version_frozen"] is True, name
        assert r["threshold_frozen"] is True, name
        assert r["matches_np_compat"] is True, (
            f"{name}: max_score_diff={r['max_score_diff_vs_np_compat']:.3e} "
            f"accept_mismatches={r['accept_mismatches_vs_np_compat']} sample={r['diff_sample'][:3]}"
        )
        # forced-online raw counts and np_compat raw counts must agree exactly.
        ref = nca.evaluate_scorer_np_compat(name, had, split, target_fpr=0.1, seed=42)
        assert r["accepted_h0_FP"] == ref["accepted_h0_FP"], name
        assert r["accepted_h1_TP"] == ref["accepted_h1_TP"], name
        assert r["threshold"] == pytest.approx(ref["threshold"], rel=1e-9, abs=1e-9), name


def test_end_to_end_tiny_run_frozen_and_consistent():
    rng = np.random.default_rng(4)
    # separable-ish two-class embeddings so calibration is meaningful
    h1 = rng.normal(0.6, 1.0, size=(150, 12)).astype(np.float32)
    h0 = rng.normal(0.0, 1.0, size=(300, 12)).astype(np.float32)
    emb = np.vstack([h0, h1])
    label = np.array([0] * 300 + [1] * 150, dtype=np.int32)
    region = (rng.random(emb.shape[0]) * 6).astype(np.int64)
    had, _cos, _a = nca.build_hadamard_features(emb, region, strategy="centroid_nearest", seed=42)
    split = nca.split_global(label, n_train=60, n_calib=60, n_eval=60, seed=42)
    r = nca.evaluate_scorer_np_compat("cosine", had, split, target_fpr=0.1, seed=42)

    assert r["threshold_frozen_during_eval"] is True
    # denominators must equal the raw counts the report claims
    assert r["accepted_h0_FP"] + r["rejected_h0_TN"] == r["eval_h0"]
    assert r["accepted_h1_TP"] + r["rejected_h1_FN"] == r["eval_h1"]
    if r["shadow_pair_fpr"] is not None:
        assert r["shadow_pair_fpr"] == pytest.approx(r["accepted_h0_FP"] / r["eval_h0"])
    if r["pair_precision"] is not None and (r["accepted_h1_TP"] + r["accepted_h0_FP"]) > 0:
        assert r["pair_precision"] == pytest.approx(
            r["accepted_h1_TP"] / (r["accepted_h1_TP"] + r["accepted_h0_FP"])
        )
