"""Online cosine-vs-full-ensemble replay on data/h1h0_final.npz.

This is a REAL online experiment built on `SemanticCacheSystem`:

- dataset embeddings come straight from the NPZ `emb` field (no model loads,
  no sentence-transformer encoding, no random vectors)
- reuse labels come from the NPZ `global_cluster` + `label` fields (no synthetic
  TopicJudge, no #topic prompts, no LLM judge)
- no offline `run_cache.py` / `prefit_and_calibrate` path is used; both systems
  learn online through the runtime shadow-collection + refit lifecycle

Two systems replay the *identical* request stream and differ ONLY in the reuse
scorer:

    A. cosine baseline  -> scorer="cosine"
    B. full ensemble    -> scorer="ensemble", scorers=cosine,lda,
                           pca_whitened_cosine,xgboost,mlp

The ensemble follows the online lifecycle: shadow-judge top-k candidates,
accumulate H0/H1 pairs, train + calibrate once the explicit train/calibration
gates are met, atomically promote the learned scorer, and keep serving.

The headline question this script is built to answer: previous diagnostics
showed a LOW calibration FPR but a HIGH online active-hit error. So every
ensemble active HIT is audited against the dataset oracle label, and the
final report contrasts the calibration-H0 score distribution with the online
active-hit score distribution to explain any disagreement.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for _p in (str(SRC), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlcache import MockLLM, SemanticCacheSystem  # noqa: E402
from mlcache.policies import ConservativeRefitConfig, ConservativeRefitPolicy  # noqa: E402
from mlcache.policies.refit import RefitAction  # noqa: E402
from mlcache.scorers.utils import score_rows_with_scorer  # noqa: E402

# Proven, already-tested dataset components are reused rather than reimplemented.
from compare_cosine_vs_ensemble import (  # noqa: E402
    DatasetEmbeddingProvider,
    DatasetH1H0Judge,
    DatasetRow,
    FULL_ENSEMBLE,
    MetricsRecorder,
    append_jsonl,
    load_rows,
    safe_rate,
    setup_logging,
    snapshot_policy,
    snapshot_training_counts,
    store_breakdown,
    write_csv_rows,
    write_json,
)

WILSON_Z = 1.96


def preview_text(text: str, max_chars: int = 120) -> str:
    """Create a short, single-line preview of text."""
    if max_chars <= 0:
        return ""
    s = str(text).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    return s


# ── reuse oracle (same rule as DatasetH1H0Judge, exposed for the audit) ─────

def reuse_label(query: DatasetRow, anchor: DatasetRow) -> str:
    """Return 'H1' (reusable), 'H0' (not reusable), or 'UNCERTAIN'.

    Mirrors `DatasetH1H0Judge`: H1 iff both rows are label==1 and share a
    global_cluster; UNCERTAIN if either is label==-1; H0 otherwise.
    """
    if query.label == -1 or anchor.label == -1:
        return "UNCERTAIN"
    if query.label == 1 and anchor.label == 1 and query.group == anchor.group:
        return "H1"
    return "H0"


# ── explicit gate wiring with Wilson-safe calibration floor ────────────────

@dataclass(frozen=True)
class GateDiagnostics:
    requested_min_train_h0: int
    requested_min_train_h1: int
    requested_min_calib_h0: int
    requested_min_calib_h1: int
    wilson_min_calib_h0: int
    effective_min_calib_h0: int
    wilson_floor_applied: bool
    target_fpr: float
    fpr_wilson_margin: float
    allowed_fpr_bound: float
    config: dict[str, Any]


def build_refit_policy(
    *,
    min_train_h0: int,
    min_train_h1: int,
    min_calib_h0: int,
    min_calib_h1: int,
    target_fpr: float,
) -> tuple[ConservativeRefitPolicy, GateDiagnostics]:
    """Build a ConservativeRefitPolicy from explicit gates.

    The activation gate uses the Wilson upper bound on calibration FPR; with
    k=0 false accepts that bound is z^2/(n+z^2), so the gate can only ever pass
    when n_calib_h0 >= z^2*(1-allowed)/allowed. If the requested --min-calib-h0
    is below that floor the gate is mathematically unsatisfiable, so we raise it
    to the floor and record the adjustment (per the user's "auto-raise + warn").
    """
    fpr_wilson_margin = max(0.03, float(target_fpr))
    allowed = float(target_fpr) + fpr_wilson_margin
    wilson_min = max(1, ceil(WILSON_Z * WILSON_Z * (1.0 - allowed) / allowed))
    eff_calib_h0 = max(int(min_calib_h0), wilson_min)
    floor_applied = eff_calib_h0 != int(min_calib_h0)

    cfg = ConservativeRefitConfig(
        # Trigger thresholds (total store counts) before the first fit fires.
        min_h0_for_fit=int(min_train_h0) + eff_calib_h0,
        min_h1_for_fit=int(min_train_h1) + int(min_calib_h1),
        min_h0_for_calibration=int(min_train_h0) + eff_calib_h0,
        # Activation-gate per-bucket requirements.
        min_train_total=max(2, int(min_train_h0) + int(min_train_h1)),
        min_train_h0=max(1, int(min_train_h0)),
        min_train_h1=max(1, int(min_train_h1)),
        min_calibration_h0=eff_calib_h0,
        min_calibration_h1=max(1, int(min_calib_h1)),
        # New-data cadence: tie recalibration/refit to the gate sizes so the
        # threshold does not churn on every request (see the runaway-recalib
        # diagnosis). One recalibration per ~min_calib_h0 fresh H0 pairs.
        min_new_h0_for_calibration=max(1, eff_calib_h0),
        min_new_h0_for_refit=max(1, int(min_train_h0)),
        min_new_h1_for_refit=max(1, int(min_train_h1)),
        fpr_wilson_margin=fpr_wilson_margin,
        deactivate_on_failed_refit=False,
    )
    diag = GateDiagnostics(
        requested_min_train_h0=int(min_train_h0),
        requested_min_train_h1=int(min_train_h1),
        requested_min_calib_h0=int(min_calib_h0),
        requested_min_calib_h1=int(min_calib_h1),
        wilson_min_calib_h0=int(wilson_min),
        effective_min_calib_h0=int(eff_calib_h0),
        wilson_floor_applied=bool(floor_applied),
        target_fpr=float(target_fpr),
        fpr_wilson_margin=float(fpr_wilson_margin),
        allowed_fpr_bound=float(allowed),
        config={k: getattr(cfg, k) for k in (
            "min_h0_for_fit", "min_h1_for_fit", "min_h0_for_calibration",
            "min_train_total", "min_train_h0", "min_train_h1",
            "min_calibration_h0", "min_calibration_h1",
            "min_new_h0_for_calibration", "min_new_h0_for_refit",
            "min_new_h1_for_refit", "fpr_wilson_margin",
        )},
    )
    return ConservativeRefitPolicy(config=cfg), diag


# ── stream construction (bounded cache universe + identical order) ─────────

def build_stream_uniform(rows: list[DatasetRow], *, max_requests: int, seed: int) -> list[DatasetRow]:
    """Build the shared request stream of length `max_requests`.

    `rows` is the seeded cache universe (at most --cache-size distinct rows, so
    the vector store never exceeds that capacity). If max_requests exceeds the
    universe, requests repeat in seeded epochs; otherwise it is a prefix. Both
    systems consume this exact list, so order and capacity are identical.
    """
    if max_requests <= len(rows):
        return list(rows[:max_requests])
    rng = np.random.default_rng(int(seed))
    stream: list[DatasetRow] = list(rows)
    while len(stream) < max_requests:
        order = rng.permutation(len(rows))
        stream.extend(rows[int(i)] for i in order)
    return stream[:max_requests]


def _cluster_stats(rows: list[DatasetRow]) -> dict[str, dict[str, Any]]:
    """Group rows by `group` (anchor/global_cluster), counting label==1 members."""
    stats: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = stats.setdefault(r.group, {"rows": [], "h1_count": 0, "total": 0})
        s["rows"].append(r)
        s["total"] += 1
        if r.label == 1:
            s["h1_count"] += 1
    return stats


def select_reuse_clusters(
    rows: list[DatasetRow], *, top_k: int, min_cluster_size: int,
) -> list[tuple[str, list[DatasetRow]]]:
    """Eligible clusters have >= `min_cluster_size` label==1 members.

    The top `top_k` eligible clusters are kept, ranked by total cluster size
    (then h1 count, then cluster id for a fully deterministic tie-break).
    """
    stats = _cluster_stats(rows)
    eligible = [(g, s) for g, s in stats.items() if s["h1_count"] >= int(min_cluster_size)]
    eligible.sort(key=lambda kv: (kv[1]["total"], kv[1]["h1_count"], kv[0]), reverse=True)
    return [(g, s["rows"]) for g, s in eligible[: int(top_k)]]


def build_stream_cluster_reuse(
    rows: list[DatasetRow], *, max_requests: int, seed: int, top_k: int, min_cluster_size: int,
) -> tuple[list[DatasetRow], list[dict[str, Any]]]:
    """Build a stream drawn only from the top eligible reuse clusters.

    Each cluster's rows are deterministically shuffled (via --seed) and then
    round-robin interleaved, so consecutive requests cycle across clusters
    while still revisiting the same cluster's rows often enough to create
    real same-cluster H1 (cache hit) opportunities. Raises if no cluster
    meets `min_cluster_size`; cluster_reuse is opt-in, so we never silently
    fall back to uniform.
    """
    clusters = select_reuse_clusters(rows, top_k=top_k, min_cluster_size=min_cluster_size)
    if not clusters:
        raise ValueError(
            "--selection cluster_reuse: no cluster in the cache universe has >= "
            f"{int(min_cluster_size)} label==1 members (--reuse-min-cluster-size="
            f"{int(min_cluster_size)}). Lower --reuse-min-cluster-size, raise "
            "--cache-size, or use --selection uniform."
        )
    rng = np.random.default_rng(int(seed))
    shuffled: list[list[DatasetRow]] = []
    for _, crows in clusters:
        order = rng.permutation(len(crows))
        shuffled.append([crows[int(i)] for i in order])

    stream: list[DatasetRow] = []
    cursors = [0] * len(shuffled)
    i = 0
    while len(stream) < max_requests:
        c = i % len(shuffled)
        lst = shuffled[c]
        stream.append(lst[cursors[c] % len(lst)])
        cursors[c] += 1
        i += 1

    cluster_info = [
        {"cluster": g, "total_rows": len(crows), "h1_rows": sum(1 for r in crows if r.label == 1)}
        for g, crows in clusters
    ]
    return stream[:max_requests], cluster_info


def build_stream_mixed(
    rows: list[DatasetRow], *, max_requests: int, seed: int, top_k: int, min_cluster_size: int,
    mixed_reuse_fraction: float,
) -> tuple[list[DatasetRow], list[dict[str, Any]], int]:
    """Mostly-uniform stream with a `mixed_reuse_fraction` of positions replaced
    by cluster_reuse traffic (same eligible-cluster selection as cluster_reuse).
    Deterministic under --seed; raises if no cluster is eligible.
    """
    uniform_stream = build_stream_uniform(rows, max_requests=max_requests, seed=seed)
    reuse_stream, cluster_info = build_stream_cluster_reuse(
        rows, max_requests=max_requests, seed=int(seed) + 1,
        top_k=top_k, min_cluster_size=min_cluster_size,
    )
    rng = np.random.default_rng(int(seed) + 2)
    mask = rng.random(max_requests) < float(mixed_reuse_fraction)
    stream = [reuse_stream[i] if mask[i] else uniform_stream[i] for i in range(max_requests)]
    return stream, cluster_info, int(mask.sum())


def build_stream(
    rows: list[DatasetRow], *, max_requests: int, seed: int, selection: str,
    reuse_clusters_top_k: int, reuse_min_cluster_size: int, mixed_reuse_fraction: float,
) -> tuple[list[DatasetRow], dict[str, Any]]:
    """Dispatch on --selection. Returns (stream, selection_diagnostics)."""
    if selection == "uniform":
        stream = build_stream_uniform(rows, max_requests=max_requests, seed=seed)
        return stream, {"mode": "uniform"}
    if selection == "cluster_reuse":
        stream, cluster_info = build_stream_cluster_reuse(
            rows, max_requests=max_requests, seed=seed,
            top_k=reuse_clusters_top_k, min_cluster_size=reuse_min_cluster_size,
        )
        return stream, {"mode": "cluster_reuse", "selected_clusters": cluster_info}
    if selection == "mixed":
        stream, cluster_info, injected = build_stream_mixed(
            rows, max_requests=max_requests, seed=seed,
            top_k=reuse_clusters_top_k, min_cluster_size=reuse_min_cluster_size,
            mixed_reuse_fraction=mixed_reuse_fraction,
        )
        return stream, {
            "mode": "mixed",
            "selected_clusters": cluster_info,
            "injected_reuse_requests": injected,
        }
    raise ValueError(f"Unknown --selection mode: {selection!r}")


def scorer_state(policy: dict[str, Any], *, method: str) -> str:
    """Map lifecycle flags to untrained / global / learned."""
    if not policy["calibrated"]:
        return "untrained"
    if method == "ensemble":
        return "learned" if policy["trained"] else "global"
    return "global"  # cosine baseline serves on the global/cold scorer


# ── one system replay with full active-decision audit ──────────────────────

def run_one_system(
    *,
    method: str,
    scorer: str,
    scorers: list[str] | None,
    stream: list[DatasetRow],
    rows_by_id: dict[int, DatasetRow],
    target_fpr: float,
    top_k: int,
    batch_size: int,
    gate_min_train_h0: int,
    gate_min_train_h1: int,
    gate_min_calib_h0: int,
    gate_min_calib_h1: int,
    parallelism: int,
    persistence: bool,
    state_dir: Path,
    logger,
    progress_every: int,
    log_judge_details: bool = False,
    warmup_stream: list[DatasetRow] | None = None,
) -> dict[str, Any]:
    # Fresh run: clear any prior persisted state. With --persist the cache still
    # persists *within* this run; we only drop leftovers from earlier runs.
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
        logger.info("[%s] cleared prior persisted state at %s", method, state_dir)

    recorder = MetricsRecorder(method=method, target_fpr=float(target_fpr), logger=logger)

    # min_h0/min_h1 seed the constructor (and the stopping monitor); the refit
    # policy is then overridden with the explicit gates for auditable control.
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None,
        scorer=scorer,
        scorers=scorers if scorer == "ensemble" else None,
                judge=DatasetH1H0Judge(rows_by_id, recorder=recorder, logger=logger if log_judge_details else None),
        embedding_provider=DatasetEmbeddingProvider(rows_by_id),
        target_fpr=float(target_fpr),
        top_k=int(top_k),
        root_dir=state_dir,
        batch_size=int(batch_size),
        min_h0=int(gate_min_train_h0) + int(gate_min_calib_h0),
        min_h1=int(gate_min_train_h1) + int(gate_min_calib_h1),
        persistence=bool(persistence),
        parallelism=int(parallelism),
        namespace=f"h1h0-online-{method}",
    )

    policy_obj, gate_diag = build_refit_policy(
        min_train_h0=gate_min_train_h0,
        min_train_h1=gate_min_train_h1,
        min_calib_h0=gate_min_calib_h0,
        min_calib_h1=gate_min_calib_h1,
        target_fpr=target_fpr,
    )
    system.cache.runtime.oracle.refit_policy = policy_obj
    if gate_diag.wilson_floor_applied:
        logger.warning(
            "[%s] raised min_calibration_h0 %d -> %d (Wilson-safe floor for "
            "target_fpr=%.3f); the activation gate would be unsatisfiable below this.",
            method, gate_diag.requested_min_calib_h0, gate_diag.effective_min_calib_h0, target_fpr,
        )

    # Per-request bookkeeping.
    key_to_row: dict[str, int] = {}       # exact anchor recovery: cache_key -> row_id

    if warmup_stream:
        logger.info("[%s] warm-up: replaying %d requests (scorer=%s)", method, len(warmup_stream),
                    ",".join(scorers) if scorers else scorer)
        for idx, row in enumerate(warmup_stream, start=1):
            recorder.set_request_index(idx)
            response = system.handle(row.prompt)
            if response.source == "llm":
                if response.cache_key is not None:
                    key_to_row[response.cache_key] = row.row_id
            recorder.resolve_request(
                idx, source=response.source,
                accepted_key=response.cache_key if response.source == "cache" else None,
            )
            if idx % max(1, batch_size) == 0:
                system._maybe_check_stopping()
            if progress_every > 0 and idx % progress_every == 0:
                wp = snapshot_policy(system)
                wc = snapshot_training_counts(system)
                logger.info(
                    "[%s][warm-up %d/%d] trained=%s calibrated=%s thr=%s "
                    "train_h0=%d train_h1=%d calib_h0=%d calib_h1=%d",
                    method, idx, len(warmup_stream),
                    wp["trained"], wp["calibrated"], _fmt(wp["threshold"]),
                    wc["h0_train"], wc["h1_train"], wc["h0_calibration"], wc["h1_calibration"],
                )

        # Let any in-flight fit (triggered during warm-up replay) finish, then
        # re-check the refit policy with the now-larger train/calibration counts:
        # the fit that was *triggered* mid-warm-up may have been built from a
        # snapshot too small to pass the activation gate, even though enough
        # data has accumulated by the time the warm-up stream is exhausted.
        oracle = system.cache.runtime.oracle
        for attempt in range(5):
            try:
                oracle.wait_for_fit(timeout=120)
            except Exception as exc:
                logger.warning("[%s] warm-up: wait_for_fit raised: %s", method, exc)
                break
            if snapshot_policy(system)["trained"]:
                break
            decision = oracle._maybe_auto_refit(current_threshold=oracle._threshold)
            if decision is None or decision.action != RefitAction.REFIT_SCORER:
                break
            logger.info("[%s] warm-up: retry refit attempt %d (%s)", method, attempt + 1, decision.reason)

        wp = snapshot_policy(system)
        wc = snapshot_training_counts(system)
        logger.info(
            "[%s] warm-up complete: trained=%s calibrated=%s thr=%s scorer_v=%d "
            "train_h0=%d train_h1=%d calib_h0=%d calib_h1=%d",
            method, wp["trained"], wp["calibrated"], _fmt(wp["threshold"]), wp["scorer_version"],
            wc["h0_train"], wc["h1_train"], wc["h0_calibration"], wc["h1_calibration"],
        )
        recorder.reset()

    minimal_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    activation_index: int | None = None
    first_hit_at: int | None = None
    first_calibrated_at: int | None = None

    started = time.time()
    logger.info("[%s] replaying %d requests (scorer=%s)", method, len(stream),
                ",".join(scorers) if scorers else scorer)

    for idx, row in enumerate(stream, start=1):
        recorder.set_request_index(idx)
        response = system.handle(row.prompt)
        policy = snapshot_policy(system)

        if response.source == "llm":
            # New cache entry written through on a MISS.
            if response.cache_key is not None:
                key_to_row[response.cache_key] = row.row_id
        else:  # cache HIT -> exact accept/reject join
            if first_hit_at is None:
                first_hit_at = idx
        recorder.resolve_request(
            idx, source=response.source,
            accepted_key=response.cache_key if response.source == "cache" else None,
        )

        if first_calibrated_at is None and policy["calibrated"]:
            first_calibrated_at = idx
        if activation_index is None and method == "ensemble" and policy["trained"]:
            activation_index = idx
            logger.info("[%s] ensemble ACTIVATED (learned scorer promoted) at request %d "
                        "threshold=%s scorer_v=%d", method, idx,
                        _fmt(policy["threshold"]), policy["scorer_version"])

        # Minimal raw decision row for both systems.
        minimal_rows.append({
            "request_index": idx,
            "query_row_id": row.row_id,
            "source": response.source,
            "score": response.score,
            "threshold": response.threshold,
            "trained": bool(policy["trained"]),
            "calibrated": bool(policy["calibrated"]),
            "scorer_version": int(policy["scorer_version"]),
            "threshold_version": int(policy["threshold_version"]),
        })

        # Active-decision audit: every cache HIT (the active accept decisions).
        if response.source == "cache":
            anchor_row_id = key_to_row.get(response.cache_key)
            anchor = rows_by_id.get(anchor_row_id) if anchor_row_id is not None else None
            oracle = reuse_label(row, anchor) if anchor is not None else "UNKNOWN"
            is_fp = oracle == "H0"
            audit_rows.append({
                "request_index": idx,
                "query_row_id": row.row_id,
                "query_text": row.text[:160].replace("\n", " "),
                "anchor_row_id": anchor_row_id if anchor_row_id is not None else "",
                "anchor_text": (anchor.text[:160].replace("\n", " ") if anchor is not None else ""),
                "query_cluster": row.group,
                "anchor_cluster": (anchor.group if anchor is not None else ""),
                "score": response.score,
                "threshold": response.threshold,
                "predicted_decision": "HIT",
                "oracle_label": oracle,
                "is_false_positive": bool(is_fp),
                "scorer_state": scorer_state(policy, method=method),
                "cache_size": len(key_to_row),
                "candidate_rank": 1,  # served candidate is the top accepted; deeper rank not exposed
                "post_activation": bool(activation_index is not None and idx >= activation_index),
            })

        if idx % max(1, batch_size) == 0:
            system._maybe_check_stopping()
        if progress_every > 0 and idx % progress_every == 0:
            m = recorder.metrics()
            logger.info(
                "[%s][%d/%d] hits=%d miss=%d fpr=%s tpr=%s prec=%s trained=%s thr=%s judged=%d",
                method, idx, len(stream),
                sum(1 for r in minimal_rows if r["source"] == "cache"),
                sum(1 for r in minimal_rows if r["source"] == "llm"),
                _fmt(m["empirical_fpr"], 3), _fmt(m["tpr"], 3), _fmt(m["precision"], 3),
                policy["trained"], _fmt(policy["threshold"]),
                snapshot_judged := store_breakdown(system)["total"],
            )

    runtime = time.time() - started

    # Final lifecycle snapshot + shadow-view metrics.
    final_policy = snapshot_policy(system)
    final_counts = snapshot_training_counts(system)
    shadow_metrics = recorder.metrics()
    hits = sum(1 for r in minimal_rows if r["source"] == "cache")
    misses = sum(1 for r in minimal_rows if r["source"] == "llm")

    # Active-hit (online decision) view of TP/FP.
    active = [a for a in audit_rows if a["oracle_label"] in ("H0", "H1")]
    active_tp = sum(1 for a in active if a["oracle_label"] == "H1")
    active_fp = sum(1 for a in active if a["oracle_label"] == "H0")
    post = [a for a in active if a["post_activation"]]
    post_tp = sum(1 for a in post if a["oracle_label"] == "H1")
    post_fp = sum(1 for a in post if a["oracle_label"] == "H0")

    # Calibration vs online active-hit score distribution under the active scorer.
    calib_diag = _calibration_vs_active(system, audit_rows, target_fpr=target_fpr)

    summary = {
        "method": method,
        "scorer": scorer,
        "scorers": scorers,
        "runtime_seconds": runtime,
        "warmup_requests": len(warmup_stream) if warmup_stream else 0,
        "total_requests": len(stream),
        "hits": hits,
        "misses": misses,
        "hit_rate": safe_rate(hits, len(stream)),
        "first_hit_at_request": first_hit_at,
        "first_calibrated_at_request": first_calibrated_at,
        "activation_request_index": activation_index,
        "threshold": final_policy["threshold"],
        "finite_threshold": final_policy["finite_threshold"],
        "scorer_version": final_policy["scorer_version"],
        "threshold_version": final_policy["threshold_version"],
        "trained": final_policy["trained"],
        "calibrated": final_policy["calibrated"],
        "train_h0": final_counts["h0_train"],
        "train_h1": final_counts["h1_train"],
        "calibration_h0": final_counts["h0_calibration"],
        "calibration_h1": final_counts["h1_calibration"],
        "judged_pair_count": final_counts["total"],
        # shadow-view (all retrieved candidates judged) — exact accept/reject:
        "shadow_empirical_fpr": shadow_metrics["empirical_fpr"],
        "shadow_tpr": shadow_metrics["tpr"],
        "shadow_precision": shadow_metrics["precision"],
        "shadow_true_accepts": shadow_metrics["true_accepts"],
        "shadow_false_accepts": shadow_metrics["false_accepts"],
        "shadow_true_rejects": shadow_metrics["true_rejects"],
        "shadow_false_rejects": shadow_metrics["false_rejects"],
        "target_fpr_respected": shadow_metrics["target_fpr_respected"],
        # active-hit (online serving decision) view:
        "active_hits": len(audit_rows),
        "active_decisive_hits": len(active),
        "active_true_positives": active_tp,
        "active_false_positives": active_fp,
        "active_uncertain": len(audit_rows) - len(active),
        "active_fp_rate": safe_rate(active_fp, active_tp + active_fp),
        "post_activation_active_hits": len(post),
        "post_activation_false_positives": post_fp,
        "post_activation_true_positives": post_tp,
        "post_activation_fp_rate": safe_rate(post_fp, post_tp + post_fp),
        "gate_diagnostics": gate_diag.__dict__,
        "calibration_vs_active": calib_diag,
        "activation_status": _activation_status(
            method=method, trained=final_policy["trained"],
            h1_total=final_counts["h1_total"],
            required_h1=int(gate_min_train_h1) + int(gate_min_calib_h1),
        ),
    }
    system._executor.shutdown(wait=False)
    return {"summary": summary, "minimal_rows": minimal_rows, "audit_rows": audit_rows,
            "gate_diag": gate_diag, "calib_diag": calib_diag}


def _activation_status(*, method: str, trained: bool, h1_total: int, required_h1: int) -> str:
    """Classify why the ensemble did/did not activate (no-op for cosine).

    Lack of activation is not itself a failure; if the judged H1 pair count
    never reached the configured train+calibration H1 gates, this is reported
    as starvation rather than an unexplained "never trained" result.
    """
    if method != "ensemble":
        return "not_applicable_baseline_scorer"
    if trained:
        return "activated"
    if int(h1_total) < int(required_h1):
        return "no_activation_due_to_h1_starvation"
    return "no_activation_other"


def _calibration_vs_active(system: SemanticCacheSystem, audit_rows: list[dict[str, Any]], *, target_fpr: float) -> dict[str, Any]:
    """Compare calibration-H0 scores (under the active scorer) with the online
    active-hit score distribution. A big gap is the signature of calibration
    FPR / online FPR disagreement.
    """
    store = system.cache.runtime.judge_training_store
    scorer = system.cache.runtime.oracle.scorer
    out: dict[str, Any] = {"calibration_h0": None, "active_hits": None, "active_false_positive_hits": None}
    try:
        h0_calib = store.h0_calibration() if store is not None else ()
        if h0_calib:
            calib_scores = np.asarray(
                score_rows_with_scorer([ex.features for ex in h0_calib], scorer), dtype=np.float64
            )
            out["calibration_h0"] = _dist(calib_scores)
    except Exception as exc:  # pragma: no cover - diagnostics must never crash the run
        out["calibration_h0_error"] = str(exc)

    active_scores = np.asarray([a["score"] for a in audit_rows if a["score"] is not None], dtype=np.float64)
    if active_scores.size:
        out["active_hits"] = _dist(active_scores)
    fp_scores = np.asarray(
        [a["score"] for a in audit_rows if a["is_false_positive"] and a["score"] is not None], dtype=np.float64
    )
    if fp_scores.size:
        out["active_false_positive_hits"] = _dist(fp_scores)
    return out


def _dist(arr: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


# ── FP diagnosis report (Step 7) ───────────────────────────────────────────

def build_fp_diagnosis(ensemble: dict[str, Any]) -> dict[str, Any]:
    summary = ensemble["summary"]
    audit = ensemble["audit_rows"]
    activation = summary["activation_request_index"]
    fps = [a for a in audit if a["is_false_positive"]]

    # 1) timing: are FPs concentrated right after activation or throughout?
    timing: dict[str, Any] = {"activation_request_index": activation, "total_false_positives": len(fps)}
    if activation is not None and fps:
        post = [a for a in fps if a["request_index"] >= activation]
        span = max((a["request_index"] for a in audit), default=activation) - activation + 1
        early_cut = activation + max(1, int(0.2 * span))
        early = sum(1 for a in post if a["request_index"] < early_cut)
        timing.update({
            "post_activation_false_positives": len(post),
            "first_20pct_window_end": early_cut,
            "false_positives_in_first_20pct": early,
            "fraction_in_first_20pct": safe_rate(early, len(post)),
            "verdict": ("front-loaded right after activation"
                        if (safe_rate(early, len(post)) or 0) > 0.4
                        else "spread throughout the stream"),
        })

    # 2) concentration by anchor / cluster.
    by_anchor: dict[Any, int] = {}
    by_cluster: dict[Any, int] = {}
    for a in fps:
        by_anchor[a["anchor_row_id"]] = by_anchor.get(a["anchor_row_id"], 0) + 1
        by_cluster[a["anchor_cluster"]] = by_cluster.get(a["anchor_cluster"], 0) + 1
    top_anchors = sorted(by_anchor.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_clusters = sorted(by_cluster.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top5_share = safe_rate(sum(c for _, c in top_anchors[:5]), len(fps))
    concentration = {
        "unique_fp_anchors": len(by_anchor),
        "unique_fp_clusters": len(by_cluster),
        "top_anchors": [{"anchor_row_id": k, "false_positives": v} for k, v in top_anchors],
        "top_clusters": [{"cluster": k, "false_positives": v} for k, v in top_clusters],
        "top5_anchor_share_of_fps": top5_share,
        "verdict": ("concentrated in a few anchors" if (top5_share or 0) > 0.5
                    else "spread across many anchors"),
    }

    # 3) threshold applied correctly: every active HIT score must be >= threshold.
    thr = summary["threshold"]
    scored = [a["score"] for a in audit if a["score"] is not None]
    below = [a for a in audit if a["score"] is not None and thr is not None and a["score"] < thr]
    threshold_check = {
        "final_threshold": thr,
        "active_hit_min_score": (min(scored) if scored else None),
        "active_hits_below_threshold": len(below),
        "verdict": ("threshold applied correctly" if not below
                    else "WARNING: active hits scored below the final threshold "
                         "(threshold moved after those hits, or mis-application)"),
    }

    # 4) is the active scorer actually the trained ensemble?
    active_scorer = {
        "trained": summary["trained"],
        "scorer_version": summary["scorer_version"],
        "activation_request_index": activation,
        "verdict": ("active scorer is the trained/promoted ensemble"
                    if summary["trained"] and activation is not None
                    else "active scorer NEVER became the learned ensemble (served on cosine/global fallback)"),
    }

    # 5) calibration vs online active distribution.
    cva = summary["calibration_vs_active"]
    calib = cva.get("calibration_h0")
    activ = cva.get("active_hits")
    dist_verdict = "insufficient data"
    gap = None
    if calib and activ:
        gap = activ["mean"] - calib["mean"]
        dist_verdict = (
            "online active-hit scores sit ABOVE the calibration-H0 mean, i.e. "
            "calibration H0 under-represents the hard near-threshold H0 pairs seen "
            "online -> low calibration FPR but high online FPR"
            if gap > 0 else
            "online active-hit and calibration-H0 score distributions are aligned"
        )
    distribution = {
        "calibration_h0": calib,
        "active_hits": activ,
        "active_minus_calib_mean_gap": gap,
        "verdict": dist_verdict,
    }

    return {
        "shadow_empirical_fpr": summary["shadow_empirical_fpr"],
        "active_fp_rate": summary["active_fp_rate"],
        "post_activation_fp_rate": summary["post_activation_fp_rate"],
        "fpr_disagreement": _fpr_disagreement(summary),
        "timing": timing,
        "concentration": concentration,
        "threshold_application": threshold_check,
        "active_scorer": active_scorer,
        "distribution": distribution,
    }


def _fpr_disagreement(summary: dict[str, Any]) -> dict[str, Any]:
    shadow = summary["shadow_empirical_fpr"]
    active = summary["active_fp_rate"]
    diff = (active - shadow) if (shadow is not None and active is not None) else None
    return {
        "shadow_empirical_fpr": shadow,
        "active_hit_fp_rate": active,
        "difference": diff,
        "note": ("active-hit FP rate exceeds the shadow/calibration FPR -> the "
                 "serving decision is accepting hard H0 pairs the calibration set "
                 "did not penalize" if (diff is not None and diff > 0.02)
                 else "shadow and active FP rates are consistent"),
    }


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


# ── artifact writers ───────────────────────────────────────────────────────

_MINIMAL_FIELDS = [
    "method", "request_index", "query_row_id", "source", "score", "threshold",
    "trained", "calibrated", "scorer_version", "threshold_version",
]
_AUDIT_FIELDS = [
    "method", "request_index", "query_row_id", "query_text", "anchor_row_id",
    "anchor_text", "query_cluster", "anchor_cluster", "score", "threshold",
    "predicted_decision", "oracle_label", "is_false_positive", "scorer_state",
    "cache_size", "candidate_rank", "post_activation",
]
_SUMMARY_CSV_FIELDS = [
    "method", "total_requests", "hits", "misses", "hit_rate",
    "shadow_empirical_fpr", "shadow_tpr", "shadow_precision",
    "active_hits", "active_true_positives", "active_false_positives",
    "active_fp_rate", "post_activation_fp_rate",
    "threshold", "activation_request_index", "trained", "calibrated",
    "train_h0", "train_h1", "calibration_h0", "calibration_h1",
]


def write_artifacts(out_dir: Path, cosine: dict[str, Any], ensemble: dict[str, Any], config: dict[str, Any]) -> None:
    cs, es = cosine["summary"], ensemble["summary"]

    summary_metrics = {
        "experiment": "online_replay_h1h0_final",
        "offline_prefit_used": False,
        "config": config,
        "cosine": cs,
        "ensemble": es,
        "comparison": {
            "delta_hit_rate": _delta(es["hit_rate"], cs["hit_rate"]),
            "delta_shadow_fpr": _delta(es["shadow_empirical_fpr"], cs["shadow_empirical_fpr"]),
            "delta_shadow_tpr": _delta(es["shadow_tpr"], cs["shadow_tpr"]),
            "ensemble_improves_over_cosine": _improvement_verdict(cs, es),
        },
    }
    write_json(out_dir / "summary_metrics.json", summary_metrics)

    write_csv_rows(
        out_dir / "summary_metrics.csv",
        [_summary_csv_row("cosine", cs), _summary_csv_row("ensemble", es)],
        _SUMMARY_CSV_FIELDS,
    )

    minimal = ([{**r, "method": "cosine"} for r in cosine["minimal_rows"]]
               + [{**r, "method": "ensemble"} for r in ensemble["minimal_rows"]])
    write_csv_rows(out_dir / "raw_decisions_minimal.csv", minimal, _MINIMAL_FIELDS)

    audit = ([{**r, "method": "cosine"} for r in cosine["audit_rows"]]
             + [{**r, "method": "ensemble"} for r in ensemble["audit_rows"]])
    write_csv_rows(out_dir / "active_decision_audit.csv", audit, _AUDIT_FIELDS)

    write_json(out_dir / "activation_diagnostics.json", {
        "selection": config["selection"],
        "cosine": {
            "activation_request_index": cs["activation_request_index"],
            "first_calibrated_at_request": cs["first_calibrated_at_request"],
            "trained": cs["trained"], "calibrated": cs["calibrated"],
            "gate_diagnostics": cs["gate_diagnostics"],
            "activation_status": cs["activation_status"],
        },
        "ensemble": {
            "activation_request_index": es["activation_request_index"],
            "first_calibrated_at_request": es["first_calibrated_at_request"],
            "trained": es["trained"], "calibrated": es["calibrated"],
            "scorer_version": es["scorer_version"], "threshold_version": es["threshold_version"],
            "gate_diagnostics": es["gate_diagnostics"],
            "activation_status": es["activation_status"],
        },
    })

    write_json(out_dir / "calibration_diagnostics.json", {
        "cosine": cs["calibration_vs_active"],
        "ensemble": es["calibration_vs_active"],
        "ensemble_fp_diagnosis": build_fp_diagnosis(ensemble),
    })


def _summary_csv_row(method: str, s: dict[str, Any]) -> dict[str, Any]:
    return {k: (method if k == "method" else s.get(k)) for k in _SUMMARY_CSV_FIELDS}


def _delta(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _improvement_verdict(cs: dict[str, Any], es: dict[str, Any]) -> str:
    if not es["trained"] or es["activation_request_index"] is None:
        return "inconclusive: ensemble never activated its learned scorer"
    e_fpr, c_fpr = es["shadow_empirical_fpr"], cs["shadow_empirical_fpr"]
    e_tpr, c_tpr = es["shadow_tpr"], cs["shadow_tpr"]
    if None in (e_fpr, c_fpr, e_tpr, c_tpr):
        return "inconclusive: FPR/TPR not computable for both systems"
    if e_fpr <= c_fpr + 1e-9 and e_tpr >= c_tpr - 1e-9 and (e_tpr > c_tpr or e_fpr < c_fpr):
        return "yes: ensemble dominates (>= TPR at <= FPR)"
    if e_tpr > c_tpr and e_fpr > c_fpr:
        return "mixed: ensemble higher TPR but also higher FPR"
    return "no: ensemble does not improve over cosine on this stream"


# ── orchestration ──────────────────────────────────────────────────────────

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT.parent / "experiments" / f"online_replay_h1h0_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir)

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    logger.info("=== ONLINE REPLAY CONFIG ===")
    logger.info("mode: online SemanticCacheSystem (offline_prefit_used=false)")
    logger.info("npz: %s", args.npz)
    logger.info("output_dir: %s", str(out_dir))
    logger.info("ensemble_scorers: %s", ",".join(scorers))
    for k in ("selection", "reuse_clusters_top_k", "reuse_min_cluster_size",
              "mixed_reuse_fraction", "target_fpr", "min_train_h0", "min_train_h1",
              "min_calib_h0", "min_calib_h1", "seed", "max_requests", "cache_size",
              "top_k", "batch_size", "parallelism", "persist"):
        logger.info("%s: %s", k, getattr(args, k))

    # Step 1 guard: fail loudly if required fields are missing (no synthetic fallback).
    rows, schema = load_rows(
        args.npz,
        query_field=args.query_field,
        label_field=args.label_field,
        anchor_field=args.anchor_field,
        embedding_field=args.embedding_field,
        max_rows=int(args.cache_size),
        seed=int(args.seed),
        include_uncertain=False,
        selection="random",
        allow_pickle=True,
    )
    dist = schema["label_distribution_selected"]
    if int(dist.get("0", 0)) <= 0 or int(dist.get("1", 0)) <= 0:
        raise ValueError(
            f"Selected rows need both label=0 and label=1 for online H0/H1 learning; got {dist}. "
            "Increase --cache-size or change --seed."
        )
    logger.info("=== DATASET SUMMARY ===")
    logger.info("rows(universe)=%d label_dist=%s unique_clusters=%d emb_dim=%d",
                len(rows), dist, schema["unique_groups_selected"], schema["embedding_dim"])

    rows_by_id = {r.row_id: r for r in rows}
    stream, selection_diag = build_stream(
        rows, max_requests=int(args.max_requests), seed=int(args.seed),
        selection=str(args.selection),
        reuse_clusters_top_k=int(args.reuse_clusters_top_k),
        reuse_min_cluster_size=int(args.reuse_min_cluster_size),
        mixed_reuse_fraction=float(args.mixed_reuse_fraction),
    )
    logger.info("shared stream length=%d (cache universe=%d) selection=%s",
                len(stream), len(rows), selection_diag)

    warmup_stream: list[DatasetRow] | None = None
    warmup_selection_diag: dict[str, Any] | None = None
    if int(args.warmup_requests) > 0:
        warmup_stream, warmup_selection_diag = build_stream(
            rows, max_requests=int(args.warmup_requests), seed=int(args.seed) + 1_000_000,
            selection=str(args.selection),
            reuse_clusters_top_k=int(args.reuse_clusters_top_k),
            reuse_min_cluster_size=int(args.reuse_min_cluster_size),
            mixed_reuse_fraction=float(args.mixed_reuse_fraction),
        )
        logger.info("warm-up stream length=%d (cache universe=%d) selection=%s",
                    len(warmup_stream), len(rows), warmup_selection_diag)

    selection = {
        "mode": args.selection,
        "seed": args.seed,
        "reuse_clusters_top_k": args.reuse_clusters_top_k,
        "reuse_min_cluster_size": args.reuse_min_cluster_size,
        "mixed_reuse_fraction": args.mixed_reuse_fraction,
        **{k: v for k, v in selection_diag.items() if k != "mode"},
    }

    config = {
        "npz": str(args.npz), "output_dir": str(out_dir), "offline_prefit_used": False,
        "scorers": scorers, "target_fpr": args.target_fpr,
        "min_train_h0": args.min_train_h0, "min_train_h1": args.min_train_h1,
        "min_calib_h0": args.min_calib_h0, "min_calib_h1": args.min_calib_h1,
        "seed": args.seed, "max_requests": args.max_requests, "cache_size": args.cache_size,
        "top_k": args.top_k, "batch_size": args.batch_size, "parallelism": args.parallelism,
        "persistence": bool(args.persist), "stream_length": len(stream),
        "cache_universe": len(rows), "embedding_field": args.embedding_field,
        "label_field": args.label_field, "anchor_field": args.anchor_field,
        "dataset_label_distribution": dist,
        "selection": selection,
        "warmup_requests": int(args.warmup_requests),
        "warmup_stream_length": len(warmup_stream) if warmup_stream is not None else 0,
        "warmup_selection": warmup_selection_diag,
    }
    write_json(out_dir / "schema_report.json", schema)

    results: dict[str, dict[str, Any]] = {}
    for method, scorer, scs in (("ensemble", "ensemble", scorers), ("cosine", "cosine", None)):
        results[method] = run_one_system(
            method=method, scorer=scorer, scorers=scs,
            stream=stream, rows_by_id=rows_by_id,
            target_fpr=float(args.target_fpr), top_k=int(args.top_k),
            batch_size=int(args.batch_size),
            gate_min_train_h0=int(args.min_train_h0), gate_min_train_h1=int(args.min_train_h1),
            gate_min_calib_h0=int(args.min_calib_h0), gate_min_calib_h1=int(args.min_calib_h1),
            parallelism=int(args.parallelism), persistence=bool(args.persist),
            state_dir=out_dir / method / "state", logger=logger,
            progress_every=int(args.progress_every),
            log_judge_details=bool(args.log_judge_details),
            warmup_stream=warmup_stream,
        )

    write_artifacts(out_dir, results["cosine"], results["ensemble"], config)

    cs, es = results["cosine"]["summary"], results["ensemble"]["summary"]
    logger.info("=== RESULT ===")
    logger.info("cosine  : hit_rate=%s shadow_fpr=%s shadow_tpr=%s active_fp_rate=%s",
                _fmt(cs["hit_rate"]), _fmt(cs["shadow_empirical_fpr"]),
                _fmt(cs["shadow_tpr"]), _fmt(cs["active_fp_rate"]))
    logger.info("ensemble: hit_rate=%s shadow_fpr=%s shadow_tpr=%s active_fp_rate=%s activated@%s",
                _fmt(es["hit_rate"]), _fmt(es["shadow_empirical_fpr"]),
                _fmt(es["shadow_tpr"]), _fmt(es["active_fp_rate"]),
                es["activation_request_index"])
    logger.info("ensemble_improves_over_cosine: %s", _improvement_verdict(cs, es))
    logger.info("artifacts -> %s", str(out_dir.resolve()))
    return {"output_dir": str(out_dir), "cosine": cs, "ensemble": es}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default=str(ROOT.parent / "data" / "h1h0_final.npz"))
    p.add_argument("--output-dir", default=None, help="Default: timestamped experiments/ dir.")
    p.add_argument("--query-field", default="text")
    p.add_argument("--label-field", default="label")
    p.add_argument("--anchor-field", default="global_cluster")
    p.add_argument("--embedding-field", default="emb")
    p.add_argument("--selection", choices=["uniform", "cluster_reuse", "mixed"], default="uniform",
                   help="Request stream construction mode. uniform (default): sample/shuffle "
                        "rows normally. cluster_reuse: opt-in, draw only from clusters with "
                        ">= --reuse-min-cluster-size label==1 members. mixed: opt-in, mostly "
                        "uniform with --mixed-reuse-fraction of requests injected from reuse "
                        "clusters.")
    p.add_argument("--reuse-clusters-top-k", type=int, default=3,
                   help="cluster_reuse/mixed: keep the top-k largest eligible clusters.")
    p.add_argument("--reuse-min-cluster-size", type=int, default=5,
                   help="cluster_reuse/mixed: minimum label==1 members for a cluster to be eligible.")
    p.add_argument("--mixed-reuse-fraction", type=float, default=0.3,
                   help="mixed: fraction of stream positions replaced with reuse-cluster traffic.")
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--min-train-h0", type=int, default=40)
    p.add_argument("--min-train-h1", type=int, default=10)
    p.add_argument("--min-calib-h0", type=int, default=20)
    p.add_argument("--min-calib-h1", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-requests", type=int, default=0,
                   help="Replay this many extra requests (same selection, different seed) "
                        "before the measured stream, so train/calibration gates and the "
                        "ensemble's first refit/activation can be reached before timing "
                        "starts. Warm-up traffic is excluded from reported metrics. "
                        "Default: 0 (disabled).")
    p.add_argument("--max-requests", type=int, default=2000)
    p.add_argument("--cache-size", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--parallelism", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--scorers", default=",".join(FULL_ENSEMBLE))
    p.add_argument("--persist", action="store_true",
                   help="Persist within the run (cleared at run start). Default: in-memory.")
    p.add_argument("--log-judge-details", action="store_true",
                   help="Log per-call DatasetH1H0Judge start/finish lines (verbose). Default: off.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_experiment(args)
    except Exception as exc:
        print(f"online_replay_h1h0_final.py failed: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
