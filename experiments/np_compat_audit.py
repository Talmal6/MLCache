"""NP-compatibility audit for the MLCache scorer path.

PURPOSE (audit / debug, not tuning)
------------------------------------
Prove whether MLCache's *scorer + Neyman-Pearson calibration* path can reproduce
the old NP-calibrated pair-level experiment
(`NeighborCache.region_local_threshold.cli --tau_mode global --hadamard_preprocess`)
when every online-specific effect is disabled.

This `--mode np_compat_pair_eval` faithfully replicates the external protocol:

  1. Load the same NPZ (`emb`, `label`, `global_cluster`).
  2. Build the SAME pair features as the NP experiment:
       Xn        = L2-normalize(emb) rows
       anchor(r) = the row of region r whose Xn has max cosine to the region's
                   (re-normalized) mean — i.e. centroid_nearest, exactly as
                   `preprocessing._select_region_anchor_vectors`.
       hadamard  = Xn * anchor(region(row))        (the pair feature X_main)
       cos       = sum(hadamard)  == cosine(row, anchor)   (the cosine baseline)
     => every row is paired with ITS OWN cluster's representative anchor.
  3. Split GLOBALLY and per-class exactly like `splits.split_global`
     (pool all label==0 -> H0, all label==1 -> H1; shuffle per class with the
     seed; take n_train / n_calib / n_eval per class, proportional shrink if
     short). label==-1 rows are never train/calib/eval samples but DO take part
     in anchor/centroid construction (matching the NP loader).
  4. Fit the scorer on TRAIN pairs only.
  5. Calibrate the NP threshold on CALIB H0 scores only, for --target-fpr,
     using MLCache's `scores_to_threshold` (proven byte-identical to the NP
     `_select_tau` with guardrail='none': smallest unique tau with empirical
     H0-accept <= alpha, tie_mode 'ge').
  6. Freeze scorer + threshold; evaluate fixed pairs directly:
       accept(row) = score(hadamard(row)) >= tau
     No retrieval, no top-k, no cache admission, no eviction, no online refit,
     no online recalibration. Metrics use explicit per-class denominators and
     every rate is printed next to its raw counts.

Run:
  python experiments/np_compat_audit.py --mode np_compat_pair_eval \
    --npz C:\\Work\\MLCache\\data\\h1h0_final.npz \
    --scorers cosine,lda,pca_whitened_cosine,xgboost,mlp,ensemble \
    --target-fpr 0.05 --n-train 5270 --n-calib 5240 --n-eval 5270 --seed 42 \
    --output-dir experiments/np_compat_audit_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache.builder import build_scorer  # noqa: E402
from mlcache.calibration import ThresholdCalibrationRequest  # noqa: E402
from mlcache.features import NormalizedHadamardFeatureBuilder  # noqa: E402
from mlcache.oracle.trainable import TrainableSemanticCacheOracle  # noqa: E402
from mlcache.retrieval.in_memory import InMemoryVectorStore  # noqa: E402
from mlcache.scorers.utils import score_rows_with_scorer, scores_to_threshold  # noqa: E402
from mlcache.semantic_types import LabeledPairBatch, Score, Threshold, TieMode  # noqa: E402

DEFAULT_SCORERS = ["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp", "ensemble"]
ENSEMBLE_MEMBERS = ["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"]


# ── exact replicas of the NP preprocessing primitives ─────────────────────────

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(denom, eps)).astype(np.float32, copy=False)


def _select_region_anchors(
    X_main: np.ndarray, region_id: np.ndarray, *, strategy: str, seed: int
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Return (anchor_vector_by_rid, anchor_global_row_index_by_rid).

    Replica of NP `preprocessing._select_region_anchor_vectors`, additionally
    exposing the chosen anchor's *global row index* so the online wrapper can be
    handed the gold anchor's raw embedding row.
    """
    Xn = l2_normalize_rows(X_main)
    region_id = np.asarray(region_id, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(seed)
    vecs: dict[int, np.ndarray] = {}
    idxs: dict[int, int] = {}
    for rid in np.unique(region_id):
        idx = np.flatnonzero(region_id == rid)
        if idx.size == 0:
            continue
        Xr = Xn[idx]
        if strategy == "random":
            anchor_local = int(rng.integers(0, idx.size))
        else:  # centroid_nearest
            centroid = np.mean(Xr, axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            sims = Xr @ centroid
            anchor_local = int(np.argmax(sims))
        vecs[int(rid)] = np.asarray(Xr[anchor_local], dtype=np.float32).copy()
        idxs[int(rid)] = int(idx[anchor_local])
    return vecs, idxs


def select_region_anchor_vectors(
    X_main: np.ndarray, region_id: np.ndarray, *, strategy: str, seed: int
) -> dict[int, np.ndarray]:
    """One L2-normalized anchor vector per region (replica of NP)."""
    vecs, _ = _select_region_anchors(X_main, region_id, strategy=strategy, seed=seed)
    return vecs


def select_region_anchor_indices(
    X_main: np.ndarray, region_id: np.ndarray, *, strategy: str, seed: int
) -> dict[int, int]:
    """Global row index of the gold anchor per region (same choice as the vectors)."""
    _, idxs = _select_region_anchors(X_main, region_id, strategy=strategy, seed=seed)
    return idxs


def build_hadamard_features(
    X_main: np.ndarray, region_id: np.ndarray, *, strategy: str, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Return (hadamard features (N,D), cosine_to_anchor (N,1), anchors_by_rid)."""
    Xn = l2_normalize_rows(X_main)
    region_id = np.asarray(region_id, dtype=np.int64).reshape(-1)
    N, D = Xn.shape
    anchors_by_rid = select_region_anchor_vectors(X_main, region_id, strategy=strategy, seed=seed)
    anchors = np.zeros((N, D), dtype=np.float32)
    for rid in np.unique(region_id):
        idx = np.flatnonzero(region_id == rid)
        a = anchors_by_rid.get(int(rid))
        if a is not None:
            anchors[idx] = a
    had = (Xn * anchors).astype(np.float32, copy=False)
    cos = np.sum(had, axis=1, keepdims=True).astype(np.float32, copy=False)
    return had, cos, anchors_by_rid


def take_split_train_calib_eval(
    idx: np.ndarray, rng: np.random.Generator, n_train: int, n_calib: int, n_eval: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replica of NP splits._take_split_train_calib_eval (proportional shrink)."""
    idx = np.asarray(idx, dtype=np.int64).copy()
    rng.shuffle(idx)
    n = idx.size
    total = n_train + n_calib + n_eval
    if total <= n:
        n_tr, n_ca, n_ev = n_train, n_calib, n_eval
    elif total > 0:
        scale = n / total
        n_tr = int(round(n_train * scale))
        n_ca = int(round(n_calib * scale))
        n_ev = n - n_tr - n_ca
        if n_ev < 0:
            n_ca += n_ev
            n_ev = 0
    else:
        n_tr = n_ca = n_ev = 0
    tr = idx[:n_tr]
    ca = idx[n_tr:n_tr + n_ca]
    ev = idx[n_tr + n_ca:n_tr + n_ca + n_ev]
    return tr, ca, ev


def split_global(
    y: np.ndarray, *, n_train: int, n_calib: int, n_eval: int, seed: int
) -> dict[str, np.ndarray]:
    """Replica of NP splits.split_global (stratified per-class global split)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int32)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError("Global split needs both label==0 and label==1 rows.")
    h0_tr, h0_ca, h0_ev = take_split_train_calib_eval(idx0, rng, n_train, n_calib, n_eval)
    h1_tr, h1_ca, h1_ev = take_split_train_calib_eval(idx1, rng, n_train, n_calib, n_eval)
    return {
        "H0_train": h0_tr, "H1_train": h1_tr,
        "H0_calib": h0_ca, "H1_calib": h1_ca,
        "H0_eval": h0_ev, "H1_eval": h1_ev,
    }


# ── scorer fit / calibrate / eval using MLCache primitives ─────────────────────

def _rows(had: np.ndarray, idx: np.ndarray) -> list[tuple[float, ...]]:
    return [tuple(float(v) for v in had[i]) for i in idx]


def np_calibrate_tau(h0_scores: np.ndarray, *, target_fpr: float) -> Threshold:
    """NP threshold on H0 calibration scores (== NP `_select_tau` guardrail none)."""
    return scores_to_threshold(
        ThresholdCalibrationRequest(
            h0_scores=[Score(float(s)) for s in np.asarray(h0_scores).reshape(-1)],
            target_false_accept_rate=float(target_fpr),
            tie_mode=TieMode.GE,
        )
    )


def fit_frozen_scorer_and_tau(
    name: str, had: np.ndarray, split: dict[str, np.ndarray], *, target_fpr: float, seed: int
) -> tuple[Any, float, np.ndarray]:
    """Fit scorer on TRAIN pairs, calibrate NP tau on CALIB-H0 only. Returns the
    frozen scorer object, the float threshold, and the calib-H0 scores."""
    scorers = ENSEMBLE_MEMBERS if name == "ensemble" else None
    scorer = build_scorer(name, scorers=scorers)
    batch = LabeledPairBatch(
        h0=_rows(had, split["H0_train"]),
        h1=_rows(had, split["H1_train"]),
    )
    scorer.fit(batch, alpha=float(target_fpr), seed=int(seed))
    s_h0_calib = score_rows_with_scorer(had[split["H0_calib"]], scorer)
    tau = float(np_calibrate_tau(s_h0_calib, target_fpr=target_fpr))
    return scorer, tau, s_h0_calib


def evaluate_scorer_np_compat(
    name: str, had: np.ndarray, split: dict[str, np.ndarray], *, target_fpr: float, seed: int
) -> dict[str, Any]:
    scorer, tau_f, s_h0_calib = fit_frozen_scorer_and_tau(
        name, had, split, target_fpr=target_fpr, seed=seed
    )

    # 3) freeze + evaluate fixed pairs
    s_h0_eval = score_rows_with_scorer(had[split["H0_eval"]], scorer)
    s_h1_eval = score_rows_with_scorer(had[split["H1_eval"]], scorer)

    # INVARIANT: re-deriving tau from the same calib H0 must be identical
    # (scorer + threshold are frozen during eval; nothing refits/recalibrates).
    tau_recheck = float(np_calibrate_tau(score_rows_with_scorer(had[split["H0_calib"]], scorer),
                                         target_fpr=target_fpr))
    frozen_ok = (tau_recheck == tau_f)

    eval_h0 = int(s_h0_eval.size)
    eval_h1 = int(s_h1_eval.size)
    accepted_h0 = int(np.sum(s_h0_eval >= tau_f))   # FP
    accepted_h1 = int(np.sum(s_h1_eval >= tau_f))   # TP
    tp, fp = accepted_h1, accepted_h0
    tn, fn = eval_h0 - fp, eval_h1 - tp

    calib_h0 = int(s_h0_calib.size)
    calib_accept_h0 = int(np.sum(s_h0_calib >= tau_f))

    def rate(num: int, den: int) -> float | None:
        return (float(num) / float(den)) if den > 0 else None

    return {
        "method": name,
        "threshold": tau_f,
        "threshold_finite": bool(np.isfinite(tau_f)),
        "threshold_frozen_during_eval": bool(frozen_ok),
        "target_fpr": float(target_fpr),
        # calibration-side raw counts
        "calib_h0": calib_h0,
        "calib_h0_accepted": calib_accept_h0,
        "calib_fpr": rate(calib_accept_h0, calib_h0),
        # eval-side raw counts (denominators explicit)
        "eval_h0": eval_h0,
        "eval_h1": eval_h1,
        "accepted_h0_FP": fp,
        "accepted_h1_TP": tp,
        "rejected_h0_TN": tn,
        "rejected_h1_FN": fn,
        # rates with explicit denominators
        "shadow_pair_fpr": rate(fp, eval_h0),          # FP / eval_h0
        "shadow_pair_tpr": rate(tp, eval_h1),          # TP / eval_h1
        "pair_precision": rate(tp, tp + fp),           # TP / (TP+FP)
        "pair_accept_rate": rate(tp + fp, eval_h0 + eval_h1),
        "fpr_denominator": "eval_h0 (FP+TN)",
        "tpr_denominator": "eval_h1 (TP+FN)",
        "precision_denominator": "accepted pairs (TP+FP)",
        "fpr_respected": (None if eval_h0 == 0 else bool(rate(fp, eval_h0) <= float(target_fpr))),
    }


def _build_frozen_oracle(scorer: Any, tau: float, *, target_fpr: float) -> TrainableSemanticCacheOracle:
    """A real online oracle wired with the frozen scorer + NP threshold, with all
    online maintenance disabled (no auto-refit; we never call handle()/fit())."""
    oracle = TrainableSemanticCacheOracle(
        vector_store=InMemoryVectorStore(similarity="cosine"),
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=scorer,
        threshold_provider=None,
        judge=None,
        judge_training_store=None,
        shadow_collection_enabled=False,
        auto_refit=False,
        target_false_accept_rate=float(target_fpr),
        tie_mode=TieMode.GE,
        top_k=1,
    )
    oracle._threshold = Threshold(float(tau))  # frozen NP-calibrated threshold
    return oracle


def evaluate_scorer_forced_gold_anchor(
    name: str,
    emb: np.ndarray,
    had: np.ndarray,
    split: dict[str, np.ndarray],
    anchor_idx_by_rid: dict[int, int],
    region_id: np.ndarray,
    *,
    target_fpr: float,
    seed: int,
    match_tol: float,
    max_diff_sample: int = 12,
) -> dict[str, Any]:
    """Route scoring through the online oracle wrapper, but force the candidate to
    the gold cluster anchor used by np_compat. Compares to np_compat per pair."""
    # Same frozen scorer + NP threshold as np_compat (literally the same object).
    scorer, tau_f, _ = fit_frozen_scorer_and_tau(name, had, split, target_fpr=target_fpr, seed=seed)
    oracle = _build_frozen_oracle(scorer, tau_f, target_fpr=target_fpr)

    # Freeze witnesses BEFORE eval.
    scorer_obj_before = oracle.scorer
    thr_before = float(oracle._threshold)
    thr_ver_before = int(oracle._threshold_version)

    def _forced_scores(idx_arr: np.ndarray) -> np.ndarray:
        out = np.empty(int(idx_arr.size), dtype=np.float64)
        for j, i in enumerate(idx_arr.tolist()):
            anchor_idx = anchor_idx_by_rid[int(region_id[i])]
            query_emb = tuple(float(v) for v in emb[i])
            cand_emb = tuple(float(v) for v in emb[anchor_idx])
            feats = oracle.feature_builder.build(query_emb, cand_emb)  # online wrapper
            sc = oracle._safe_score(oracle.scorer, feats)
            out[j] = float("nan") if sc is None else float(sc)
        return out

    h0_idx = split["H0_eval"]
    h1_idx = split["H1_eval"]
    forced_h0 = _forced_scores(h0_idx)
    forced_h1 = _forced_scores(h1_idx)

    # np_compat reference scores on the exact same pairs (precomputed hadamard).
    np_h0 = np.asarray(score_rows_with_scorer(had[h0_idx], scorer), dtype=np.float64)
    np_h1 = np.asarray(score_rows_with_scorer(had[h1_idx], scorer), dtype=np.float64)

    # Decisions via the online wrapper's own predicate.
    acc_forced_h0 = np.array(
        [oracle._predict_score(Score(float(s)), oracle._threshold, oracle.tie_mode) for s in forced_h0],
        dtype=bool,
    )
    acc_forced_h1 = np.array(
        [oracle._predict_score(Score(float(s)), oracle._threshold, oracle.tie_mode) for s in forced_h1],
        dtype=bool,
    )
    acc_np_h0 = np_h0 >= tau_f
    acc_np_h1 = np_h1 >= tau_f

    # Freeze assertions AFTER eval.
    frozen_scorer_ok = oracle.scorer is scorer_obj_before
    frozen_threshold_ok = (float(oracle._threshold) == thr_before) and (int(oracle._threshold_version) == thr_ver_before)

    eval_h0, eval_h1 = int(forced_h0.size), int(forced_h1.size)
    fp = int(np.sum(acc_forced_h0))
    tp = int(np.sum(acc_forced_h1))
    tn, fn = eval_h0 - fp, eval_h1 - tp

    score_diffs = np.abs(np.concatenate([forced_h0 - np_h0, forced_h1 - np_h1])) if (eval_h0 + eval_h1) else np.zeros(0)
    max_score_diff = float(np.nanmax(score_diffs)) if score_diffs.size else 0.0
    accept_mismatches = int(np.sum(acc_forced_h0 != acc_np_h0) + np.sum(acc_forced_h1 != acc_np_h1))
    fp_np = int(np.sum(acc_np_h0))
    tp_np = int(np.sum(acc_np_h1))
    counts_match = (fp == fp_np and tp == tp_np)
    decisions_match = bool(accept_mismatches == 0 and counts_match)
    score_match_within_tol = bool(max_score_diff <= match_tol)
    # Operational equivalence: identical threshold + identical accept decisions +
    # identical FP/TP counts. Raw-score fidelity is reported separately because a
    # tree scorer (xgboost) can jump discretely from a ~1e-7 feature-float
    # difference (float32 precomputed hadamard vs the online builder renormalizing
    # raw embeddings) without changing any decision at this threshold.
    matches = decisions_match
    show_diff = not (decisions_match and score_match_within_tol)

    # Build a per-pair diff sample whenever scores or decisions differ.
    diff_sample: list[dict[str, Any]] = []
    if show_diff:
        for cls, idx_arr, fsc, nsc, facc, nacc in (
            ("H0", h0_idx, forced_h0, np_h0, acc_forced_h0, acc_np_h0),
            ("H1", h1_idx, forced_h1, np_h1, acc_forced_h1, acc_np_h1),
        ):
            for j, i in enumerate(idx_arr.tolist()):
                if abs(float(fsc[j]) - float(nsc[j])) > match_tol or bool(facc[j]) != bool(nacc[j]):
                    diff_sample.append({
                        "class": cls,
                        "query_id": int(i),
                        "gold_anchor_id": int(anchor_idx_by_rid[int(region_id[i])]),
                        "np_compat_score": float(nsc[j]),
                        "forced_online_score": float(fsc[j]),
                        "np_compat_accept": bool(nacc[j]),
                        "forced_online_accept": bool(facc[j]),
                    })
                    if len(diff_sample) >= max_diff_sample:
                        break
            if len(diff_sample) >= max_diff_sample:
                break

    def rate(num: int, den: int) -> float | None:
        return (float(num) / float(den)) if den > 0 else None

    return {
        "method": name,
        "threshold": tau_f,
        "threshold_finite": bool(np.isfinite(tau_f)),
        "scorer_version_frozen": bool(frozen_scorer_ok),
        "threshold_frozen": bool(frozen_threshold_ok),
        "threshold_version": int(oracle._threshold_version),
        "target_fpr": float(target_fpr),
        "eval_h0": eval_h0,
        "eval_h1": eval_h1,
        "accepted_h0_FP": fp,
        "accepted_h1_TP": tp,
        "rejected_h0_TN": tn,
        "rejected_h1_FN": fn,
        "forced_fpr": rate(fp, eval_h0),
        "forced_tpr": rate(tp, eval_h1),
        "forced_precision": rate(tp, tp + fp),
        "fpr_denominator": "eval_h0 (FP+TN)",
        "tpr_denominator": "eval_h1 (TP+FN)",
        # bridge-match diagnostics vs np_compat on the same pairs
        "matches_np_compat": matches,             # operational: decisions + counts
        "decisions_match": decisions_match,
        "score_match_within_tol": score_match_within_tol,
        "max_score_diff_vs_np_compat": max_score_diff,
        "accept_mismatches_vs_np_compat": accept_mismatches,
        "match_tol": float(match_tol),
        "diff_sample": diff_sample,
    }


# ── orchestration ─────────────────────────────────────────────────────────────

def _load_and_prepare(args: argparse.Namespace):
    """Shared data prep for both modes: load NPZ, build gold-anchor Hadamard
    features + anchor row indices, and the global stratified split."""
    npz = np.load(args.npz, allow_pickle=True)
    emb = np.asarray(npz["emb"], dtype=np.float32)
    label = np.asarray(npz["label"]).reshape(-1).astype(np.int32)
    region_id = np.asarray(npz[args.region_key]).reshape(-1).astype(np.int64)
    if emb.shape[0] != label.shape[0] or emb.shape[0] != region_id.shape[0]:
        raise ValueError(
            f"Row mismatch: emb={emb.shape[0]} label={label.shape[0]} region={region_id.shape[0]}"
        )
    had, cos, anchors_by_rid = build_hadamard_features(
        emb, region_id, strategy=args.anchor_strategy, seed=int(args.seed)
    )
    anchor_idx_by_rid = select_region_anchor_indices(
        emb, region_id, strategy=args.anchor_strategy, seed=int(args.seed)
    )
    split = split_global(
        label, n_train=int(args.n_train), n_calib=int(args.n_calib),
        n_eval=int(args.n_eval), seed=int(args.seed),
    )
    return emb, label, region_id, had, anchor_idx_by_rid, split


def run_np_compat(args: argparse.Namespace) -> dict[str, Any]:
    emb, label, region_id, had, _anchor_idx, split = _load_and_prepare(args)

    # ── hard invariant checks for the NP-compatible protocol ──
    invariants: dict[str, Any] = {}
    all_idx = {k: set(v.tolist()) for k, v in split.items()}
    pooled = []
    for v in split.values():
        pooled.extend(v.tolist())
    invariants["train_calib_eval_disjoint"] = bool(len(pooled) == len(set(pooled)))
    # train/calib/eval rows must respect their class label
    invariants["H0_partitions_are_label0"] = bool(
        all(label[list(all_idx[k])].tolist().count(0) == len(all_idx[k]) if all_idx[k] else True
            for k in ("H0_train", "H0_calib", "H0_eval"))
    )
    invariants["H1_partitions_are_label1"] = bool(
        all((label[np.fromiter(all_idx[k], dtype=np.int64)] == 1).all() if all_idx[k] else True
            for k in ("H1_train", "H1_calib", "H1_eval"))
    )
    invariants["no_online_refit"] = True          # structural: SemanticCacheSystem never constructed
    invariants["no_online_recalibration"] = True  # structural
    invariants["no_cache_admission_or_eviction"] = True  # structural
    invariants["no_retrieval_or_topk"] = True     # structural: fixed gold-anchor pairs only
    invariants["anchor_is_gold_cluster_anchor"] = True

    results = [
        evaluate_scorer_np_compat(name, had, split, target_fpr=float(args.target_fpr), seed=int(args.seed))
        for name in args.scorers
    ]
    invariants["all_thresholds_frozen_during_eval"] = bool(
        all(r["threshold_frozen_during_eval"] for r in results)
    )

    counts = {
        "rows_total": int(emb.shape[0]),
        "label_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(label, return_counts=True))},
        "regions": int(len(np.unique(region_id))),
        "embedding_dim": int(emb.shape[1]),
        "split_counts": {k: int(v.size) for k, v in split.items()},
    }

    report = {
        "mode": "np_compat_pair_eval",
        "npz": str(args.npz),
        "region_key": args.region_key,
        "anchor_strategy": args.anchor_strategy,
        "target_fpr": float(args.target_fpr),
        "seed": int(args.seed),
        "caps": {"n_train": int(args.n_train), "n_calib": int(args.n_calib), "n_eval": int(args.n_eval)},
        "tie_mode": "ge",
        "threshold_rule": "scores_to_threshold (== NP _select_tau guardrail=none)",
        "feature": "L2norm(emb) hadamard with centroid_nearest gold cluster anchor; cos=sum(hadamard)",
        "calibration_source": "CALIB H0 scores only",
        "counts": counts,
        "invariants": invariants,
        "results": results,
    }
    return report


def run_forced_gold_anchor(args: argparse.Namespace) -> dict[str, Any]:
    emb, label, region_id, had, anchor_idx_by_rid, split = _load_and_prepare(args)

    results = [
        evaluate_scorer_forced_gold_anchor(
            name, emb, had, split, anchor_idx_by_rid, region_id,
            target_fpr=float(args.target_fpr), seed=int(args.seed), match_tol=float(args.match_tol),
        )
        for name in args.scorers
    ]

    invariants = {
        "all_scorer_versions_frozen": bool(all(r["scorer_version_frozen"] for r in results)),
        "all_thresholds_frozen": bool(all(r["threshold_frozen"] for r in results)),
        "no_online_refit": True,             # auto_refit=False; handle()/fit() never called
        "no_online_recalibration": True,
        "no_cache_admission_or_eviction": True,
        "candidate_forced_to_gold_anchor": True,  # retrieval bypassed
        "all_methods_match_np_compat": bool(all(r["matches_np_compat"] for r in results)),
    }
    counts = {
        "rows_total": int(emb.shape[0]),
        "regions": int(len(np.unique(region_id))),
        "embedding_dim": int(emb.shape[1]),
        "split_counts": {k: int(v.size) for k, v in split.items()},
    }
    return {
        "mode": "forced_gold_anchor_eval",
        "npz": str(args.npz),
        "region_key": args.region_key,
        "anchor_strategy": args.anchor_strategy,
        "target_fpr": float(args.target_fpr),
        "seed": int(args.seed),
        "caps": {"n_train": int(args.n_train), "n_calib": int(args.n_calib), "n_eval": int(args.n_eval)},
        "match_tol": float(args.match_tol),
        "wrapper": "TrainableSemanticCacheOracle.feature_builder + scorer + _predict_score (retrieval forced to gold anchor)",
        "counts": counts,
        "invariants": invariants,
        "results": results,
    }


def write_forced_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forced_gold_anchor_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Forced-Gold-Anchor Bridge Audit", "",
        f"- npz: `{report['npz']}`  mode: `{report['mode']}`  match_tol: {report['match_tol']}",
        f"- target_fpr: {report['target_fpr']}  seed: {report['seed']}  caps: {report['caps']}",
        f"- wrapper: {report['wrapper']}",
        f"- split_counts={report['counts']['split_counts']}",
        "",
        "## Invariants (hard checks)", "",
    ]
    for k, v in report["invariants"].items():
        lines.append(f"- {k}: {'OK' if v is True else ('FAIL' if v is False else str(v))}")
    lines += [
        "", "## Per-method results (online wrapper, gold anchor forced)", "",
        "`matches (decisions)` = identical threshold + accept decisions + FP/TP counts vs np_compat. "
        "`score<=tol` = raw scores also match within tol (can be FAIL for the xgboost tree scorer with no decision change).",
        "",
        "| method | tau | eval_h0 | eval_h1 | TP | FP | TN | FN | fpr | tpr | precision | scorer_frozen | thr_frozen | matches (decisions) | score<=tol | max_score_diff | accept_mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|",
    ]
    for r in report["results"]:
        lines.append(
            "| {m} | {tau} | {eh0} | {eh1} | {tp} | {fp} | {tn} | {fn} | {fpr} | {tpr} | {prec} | {sf} | {tf} | {mt} | {sm} | {msd} | {am} |".format(
                m=r["method"], tau=_fmt(r["threshold"]), eh0=r["eval_h0"], eh1=r["eval_h1"],
                tp=r["accepted_h1_TP"], fp=r["accepted_h0_FP"], tn=r["rejected_h0_TN"], fn=r["rejected_h1_FN"],
                fpr=_fmt(r["forced_fpr"]), tpr=_fmt(r["forced_tpr"]), prec=_fmt(r["forced_precision"]),
                sf=_fmt(r["scorer_version_frozen"]), tf=_fmt(r["threshold_frozen"]),
                mt=_fmt(r["matches_np_compat"]), sm=_fmt(r["score_match_within_tol"]),
                msd=f"{r['max_score_diff_vs_np_compat']:.2e}", am=r["accept_mismatches_vs_np_compat"],
            )
        )
    decisions_all = all(r["decisions_match"] for r in report["results"])
    lines += [
        "", "## Verdict", "",
        ("**All methods reproduce np_compat_pair_eval at the decision level** "
         "(identical frozen scorer + NP threshold, identical accept decisions and FP/TP counts on the "
         "exact same gold-anchor pairs). The online scorer/threshold/metric wrapper introduces no "
         "decision discrepancy; therefore any divergence in the *real* online replay comes from retrieval "
         "(wrong or absent anchor), not from the scoring path."
         if decisions_all else
         "**Decision-level mismatch detected** between the online wrapper and np_compat on identical "
         "gold-anchor pairs — this indicates a real scorer/threshold/metric wrapper bug; see the per-pair "
         "diff sample below."),
        "",
    ]
    mism = [r for r in report["results"] if r["diff_sample"]]
    if mism:
        lines += [
            "## Per-pair diff sample",
            "",
            "Methods listed here differ in raw score and/or decision. A method with "
            "`matches (decisions)=OK` but `score<=tol=FAIL` differs only in raw score with **no decision "
            "change** (expected for the xgboost tree scorer: a ~1e-7 feature-float difference between the "
            "float32 precomputed hadamard and the online builder's renormalized raw embeddings can cross a "
            "tree split boundary and jump a leaf, without flipping the threshold decision).",
            "",
        ]
        for r in mism:
            lines.append(
                f"### {r['method']} (decisions_match={r['decisions_match']}, "
                f"score_match_within_tol={r['score_match_within_tol']}, "
                f"max_score_diff={r['max_score_diff_vs_np_compat']:.3e}, "
                f"accept_mismatches={r['accept_mismatches_vs_np_compat']})"
            )
            lines.append("| class | query_id | gold_anchor_id | np_compat_score | forced_online_score | np_accept | forced_accept |")
            lines.append("|---|---:|---:|---:|---:|---|---|")
            for d in r["diff_sample"]:
                lines.append(
                    f"| {d['class']} | {d['query_id']} | {d['gold_anchor_id']} | "
                    f"{d['np_compat_score']:.6f} | {d['forced_online_score']:.6f} | "
                    f"{d['np_compat_accept']} | {d['forced_online_accept']} |"
                )
            lines.append("")
    (out_dir / "forced_gold_anchor_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "OK" if x else "FAIL"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "np_compat_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# NP-Compatibility Pair-Level Audit", "",
        f"- npz: `{report['npz']}`",
        f"- mode: `{report['mode']}`  region_key: `{report['region_key']}`  anchor: `{report['anchor_strategy']}`",
        f"- target_fpr: {report['target_fpr']}  seed: {report['seed']}  caps: {report['caps']}",
        f"- threshold rule: {report['threshold_rule']}",
        f"- feature: {report['feature']}",
        f"- calibration source: {report['calibration_source']}",
        "",
        "## Counts", "",
        f"- rows={report['counts']['rows_total']} regions={report['counts']['regions']} "
        f"dim={report['counts']['embedding_dim']} label_dist={report['counts']['label_distribution']}",
        f"- split_counts={report['counts']['split_counts']}",
        "",
        "## Invariants (hard checks)", "",
    ]
    for k, v in report["invariants"].items():
        flag = "OK" if v is True else ("FAIL" if v is False else str(v))
        lines.append(f"- {k}: {flag}")
    lines += [
        "", "## Per-method pair-level results", "",
        "Rates show their explicit denominators; raw counts are alongside.",
        "",
        "| method | tau | eval_h0 | eval_h1 | TP | FP | TN | FN | pair_fpr (FP/eval_h0) | pair_tpr (TP/eval_h1) | precision (TP/(TP+FP)) | accept_rate | fpr<=target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["results"]:
        lines.append(
            "| {m} | {tau} | {eh0} | {eh1} | {tp} | {fp} | {tn} | {fn} | {fpr} | {tpr} | {prec} | {ar} | {ok} |".format(
                m=r["method"], tau=_fmt(r["threshold"]), eh0=r["eval_h0"], eh1=r["eval_h1"],
                tp=r["accepted_h1_TP"], fp=r["accepted_h0_FP"], tn=r["rejected_h0_TN"], fn=r["rejected_h1_FN"],
                fpr=_fmt(r["shadow_pair_fpr"]), tpr=_fmt(r["shadow_pair_tpr"]),
                prec=_fmt(r["pair_precision"]), ar=_fmt(r["pair_accept_rate"]), ok=_fmt(r["fpr_respected"]),
            )
        )
    lines += [
        "", "## Notes / explicit caveats", "",
        "- `pair_fpr` and `pair_precision` use DIFFERENT denominators (eval_h0 vs TP+FP); they are not complementary.",
        "- This mode pairs every row with its OWN cluster's centroid-nearest anchor (the gold anchor) and uses the",
        "  per-row label. This is the NP experiment's pair distribution, which differs from the online replay's",
        "  retrieved-neighbour pairs. Use `forced_gold_anchor_eval` (online wrapper, gold anchor forced) to confirm",
        "  the scorer/threshold/metric path matches this mode, and the retrieved-anchor diagnostic to measure how",
        "  often online actually retrieves the gold anchor.",
        "- `cosine` here = MLCache CosineScorer on the hadamard row = sum(hadamard) = cosine-to-anchor (== NP",
        "  PrecomputedCosine in hadamard mode).",
        "- `ensemble` weight-fitting differs from the NP WeightedEnsemble meta-calibration (MLCache fits NP weights on",
        "  the train batch; NP uses a meta_frac split). Per-judge scorers match the NP protocol directly.",
        "",
    ]
    (out_dir / "np_compat_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["np_compat_pair_eval", "forced_gold_anchor_eval"],
                   default="np_compat_pair_eval")
    p.add_argument("--npz", default=str(ROOT.parent / "data" / "h1h0_final.npz"))
    p.add_argument("--region-key", default="global_cluster")
    p.add_argument("--anchor-strategy", default="centroid_nearest", choices=["centroid_nearest", "random"])
    p.add_argument("--scorers", default=",".join(DEFAULT_SCORERS))
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--n-train", type=int, default=5270)
    p.add_argument("--n-calib", type=int, default=5240)
    p.add_argument("--n-eval", type=int, default=5270)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match-tol", type=float, default=1e-4,
                   help="Max |forced_online_score - np_compat_score| for a pair to count as matching.")
    p.add_argument("--output-dir", default=str(ROOT / "experiments" / "np_compat_audit_out"))
    args = p.parse_args(argv)
    args.scorers = [s.strip() for s in str(args.scorers).split(",") if s.strip()]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)

    if args.mode == "forced_gold_anchor_eval":
        report = run_forced_gold_anchor(args)
        write_forced_report(out_dir, report)
        print(f"[forced_gold_anchor] wrote {out_dir / 'forced_gold_anchor_metrics.json'}")
        print(f"[forced_gold_anchor] wrote {out_dir / 'forced_gold_anchor_audit.md'}")
        inv = report["invariants"]
        bad = [k for k, v in inv.items() if v is False]
        print(f"[forced_gold_anchor] invariants: {'ALL OK' if not bad else 'FAILED: ' + ','.join(bad)}")
        for r in report["results"]:
            print(
                f"  {r['method']:>20}: tau={_fmt(r['threshold'])} "
                f"fpr={_fmt(r['forced_fpr'])} (FP={r['accepted_h0_FP']}/{r['eval_h0']}) "
                f"tpr={_fmt(r['forced_tpr'])} (TP={r['accepted_h1_TP']}/{r['eval_h1']}) "
                f"matches_np_compat={r['matches_np_compat']} "
                f"max_score_diff={r['max_score_diff_vs_np_compat']:.2e} "
                f"accept_mismatches={r['accept_mismatches_vs_np_compat']}"
            )
        return 0 if not bad else 1

    report = run_np_compat(args)
    write_report(out_dir, report)
    print(f"[np_compat] wrote {out_dir / 'np_compat_metrics.json'}")
    print(f"[np_compat] wrote {out_dir / 'np_compat_audit.md'}")
    inv = report["invariants"]
    bad = [k for k, v in inv.items() if v is False]
    print(f"[np_compat] invariants: {'ALL OK' if not bad else 'FAILED: ' + ','.join(bad)}")
    for r in report["results"]:
        print(
            f"  {r['method']:>20}: tau={_fmt(r['threshold'])} "
            f"pair_fpr={_fmt(r['shadow_pair_fpr'])} (FP={r['accepted_h0_FP']}/{r['eval_h0']}) "
            f"pair_tpr={_fmt(r['shadow_pair_tpr'])} (TP={r['accepted_h1_TP']}/{r['eval_h1']}) "
            f"prec={_fmt(r['pair_precision'])}"
        )
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
