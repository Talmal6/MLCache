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
EXP = ROOT / "experiments"
for _p in (str(SRC), str(SCRIPTS), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlcache import MockLLM, SemanticCacheSystem  # noqa: E402
from mlcache.policies import ConservativeRefitConfig, ConservativeRefitPolicy  # noqa: E402
from mlcache.policies.refit import RefitAction  # noqa: E402
from mlcache.scorers.utils import score_rows_with_scorer  # noqa: E402
from mlcache.semantic_types import CacheEntry, CacheMetadata, Query, Response  # noqa: E402

# Gold-anchor machinery: the retrieved-anchor diagnostic compares against the
# SAME centroid-nearest cluster anchor np_compat_pair_eval uses, so we import
# its exact anchor-selection primitive rather than re-deriving it here.
import np_compat_audit as nca  # noqa: E402

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
    use_wilson: bool = True,
) -> tuple[ConservativeRefitPolicy, GateDiagnostics]:
    """Build a ConservativeRefitPolicy from explicit gates.

    The activation gate normally uses the Wilson upper bound on calibration FPR;
    with k=0 false accepts that bound is z^2/(n+z^2), so the gate can only ever
    pass when n_calib_h0 >= z^2*(1-allowed)/allowed. If the requested
    --min-calib-h0 is below that floor the gate is mathematically unsatisfiable,
    so we raise it to the floor and record the adjustment.

    With `use_wilson=False` (the --no-wilson switch) the Wilson upper-bound gate
    and its calib-H0 floor are disabled: a large margin makes the allowed FPR
    bound >= 1.0 so the gate's `wilson_upper <= allowed` check always passes.
    Activation then depends only on the count gates plus a FINITE NP threshold
    (the NP threshold already pins empirical calib FPR to ~target on the calib
    set by construction, so FPR is still controlled -- only the extra Wilson
    safety margin is dropped). NOTE: --no-wilson does NOT relax
    `require_finite_threshold`; if the scorer cannot place a finite NP cut at the
    target FPR (e.g. a tree scorer with >target-fraction of H0 tied at the top
    score), activation still fails with reason `threshold_not_finite`.
    """
    if use_wilson:
        fpr_wilson_margin = max(0.03, float(target_fpr))
        allowed = float(target_fpr) + fpr_wilson_margin
        wilson_min = max(1, ceil(WILSON_Z * WILSON_Z * (1.0 - allowed) / allowed))
        eff_calib_h0 = max(int(min_calib_h0), wilson_min)
    else:
        fpr_wilson_margin = 1.0  # allowed bound >= 1.0 -> Wilson gate never binds
        allowed = float(target_fpr) + fpr_wilson_margin
        wilson_min = 1
        eff_calib_h0 = int(min_calib_h0)
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


# ── shared-cache write-through (fair A/B: identical retrieval pool) ─────────

def _write_through_row(system: SemanticCacheSystem, row: DatasetRow) -> str:
    """Force `row` into the cache regardless of HIT/MISS, returning its cache key.

    Retrieval ranks candidates by cosine over whatever is IN the cache, and the
    cache is normally populated only by MISS write-throughs -- so a more/less
    permissive scorer builds a different cache and both systems end up retrieving
    *different* nearest neighbours despite using the same cosine metric. Writing
    through on EVERY request makes both systems' caches hold the identical set of
    rows at every step, so cosine-NN returns the same candidates and the two
    scorers are judged on exactly the same retrieved pairs (only the accept/reject
    decision differs). Called AFTER `handle()` so the row never appears in its own
    retrieval (no self-hit); the stream is distinct rows, so each is written once.
    """
    cache_key = system._cache_key_for(row.prompt)
    embedding = system._embeddings.embed(row.prompt)
    system.cache.put(
        CacheEntry(
            cache_key=cache_key,
            query=Query(row.prompt),
            response=Response("shared-cache-writethrough"),
            embedding=embedding,
            metadata=CacheMetadata(namespace=system.namespace),
        )
    )
    return str(cache_key)


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
    use_wilson: bool = True,
    shared_cache: bool = False,
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
        use_wilson=use_wilson,
    )
    system.cache.runtime.oracle.refit_policy = policy_obj
    logger.info("[%s] activation gate: use_wilson=%s allowed_fpr_bound=%.3f min_calibration_h0=%d",
                method, use_wilson, gate_diag.allowed_fpr_bound, gate_diag.effective_min_calib_h0)
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
            elif shared_cache:  # HIT: force the row in so caches stay identical
                key_to_row[_write_through_row(system, row)] = row.row_id
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
            # Fair A/B: write the row through even on a HIT so both systems'
            # caches hold the identical set of rows and cosine-NN retrieves the
            # same candidates for both. (No-op for the default policy-built cache.)
            if shared_cache:
                key_to_row[_write_through_row(system, row)] = row.row_id
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


# ── retrieved-anchor diagnostic (audit: retrieval vs gold NP anchor) ────────
#
# np_compat_pair_eval (pair-level) and forced_gold_anchor_eval (online wrapper,
# gold anchor forced) already PASS: the scorer + NP threshold + online wrapper
# reproduce the calibrated pair experiment. So any residual *real online* active
# false positives can only come from the one thing those two modes bypass:
# retrieval / cache state choosing the anchor. This mode runs the REAL online
# replay path (real SemanticCacheSystem, real retrieval, real serving) and, per
# request, contrasts the anchor the online cache actually served against the gold
# NP anchor for the query's cluster, so the FP can be attributed to a concrete
# bucket (gold absent / not retrieved / not selected / wrong anchor accepted /
# same- vs different-cluster confusion / self-pair).
#
# "gold anchor" here is np_compat's centroid_nearest cluster representative
# (nca.select_region_anchor_indices) evaluated on EXACTLY the rows the online
# system loaded (the cache universe) -- i.e. the achievable same-cluster anchor
# the online cache could serve. When --cache-size covers the whole NPZ this is
# byte-for-byte the np_compat_pair_eval gold anchor; on a subsample it is that
# same rule restricted to retrievable rows (so "gold absent" measures cache
# state, not a subsampling artifact).


@dataclass(frozen=True)
class AnchorDiagRecord:
    """One real-online request's gold-vs-retrieved anchor comparison.

    Stores only primitives observed at decision time; all diagnostic buckets are
    derived from these by `aggregate_anchor_diagnostics`, so the counters are
    unit-testable from hand-built records without running the online system.
    """

    request_index: int
    query_id: int
    query_label: int
    query_cluster: str
    gold_anchor_id: int | None
    gold_anchor_label: int | None
    gold_cluster: str | None
    gold_anchor_available_in_cache: bool
    top1_retrieved_id: int | None
    retrieved_anchor_id: int | None
    retrieved_anchor_label: int | None
    retrieved_cluster: str | None
    topk_contains_gold_anchor: bool
    gold_pair_label: str           # H1 / H0 / UNCERTAIN / UNKNOWN
    retrieved_pair_label: str      # H1 / H0 / UNCERTAIN / UNKNOWN / NONE
    score_gold_anchor: float | None
    score_retrieved_anchor: float | None
    threshold: float | None
    accepted_gold_anchor: bool     # FORCED: would the scorer accept the gold pair
    served_hit: bool               # the real online serving decision (HIT)


def _diag_top1_is_gold(r: AnchorDiagRecord) -> bool:
    return r.top1_retrieved_id is not None and r.gold_anchor_id is not None and r.top1_retrieved_id == r.gold_anchor_id


def _diag_retr_same_cluster(r: AnchorDiagRecord) -> bool:
    return r.retrieved_cluster is not None and r.gold_cluster is not None and r.retrieved_cluster == r.gold_cluster


def _diag_retr_diff_cluster(r: AnchorDiagRecord) -> bool:
    return r.retrieved_cluster is not None and r.gold_cluster is not None and r.retrieved_cluster != r.gold_cluster


def _diag_is_self_pair(r: AnchorDiagRecord) -> bool:
    return r.served_hit and r.retrieved_anchor_id is not None and r.retrieved_anchor_id == r.query_id


def _diag_served_class(r: AnchorDiagRecord) -> tuple[str, str | None]:
    """Classify a request's SERVED decision: ('miss'|'tp'|'fp'|'unjudged', reason)."""
    if not r.served_hit:
        return ("miss", None)
    if r.retrieved_anchor_id is None:
        return ("unjudged", "unknown_anchor")
    if r.retrieved_anchor_id == r.query_id:
        return ("unjudged", "self_pair")
    if r.retrieved_pair_label == "H1":
        return ("tp", None)
    if r.retrieved_pair_label == "H0":
        return ("fp", None)
    if r.retrieved_pair_label == "UNCERTAIN":
        return ("unjudged", "uncertain_label")
    return ("unjudged", "unknown_label")


def _diag_accepted_wrong_anchor(r: AnchorDiagRecord) -> bool:
    """System served/reused a non-gold, non-self anchor (retrieved != gold)."""
    return (
        r.served_hit
        and r.retrieved_anchor_id is not None
        and not _diag_is_self_pair(r)
        and (r.gold_anchor_id is None or r.retrieved_anchor_id != r.gold_anchor_id)
    )


def _diag_rejected_wrong_anchor(r: AnchorDiagRecord) -> bool:
    """A wrong anchor was the top-1 retrieved candidate but the request MISSED."""
    return (
        not r.served_hit
        and r.top1_retrieved_id is not None
        and (r.gold_anchor_id is None or r.top1_retrieved_id != r.gold_anchor_id)
    )


def _rate(num: int, den: int) -> dict[str, Any]:
    """A rate plus its explicit numerator/denominator (never a bare percentage)."""
    return {
        "value": (float(num) / float(den)) if den > 0 else None,
        "numerator": int(num),
        "denominator": int(den),
    }


def aggregate_anchor_diagnostics(records: list[AnchorDiagRecord]) -> dict[str, Any]:
    """Pure aggregation of per-request records into raw counts + derived rates.

    Every rate carries its numerator and denominator. `active_fp_rate` and
    `active_precision` are deliberately built from the SAME judged-served-hit
    population (active_tp + active_fp == active_judged_hits) so they cannot
    silently disagree.
    """
    total = len(records)

    gold_available = sum(1 for r in records if r.gold_anchor_available_in_cache)
    gold_missing = total - gold_available

    top1_gold = sum(1 for r in records if _diag_top1_is_gold(r))
    top1_wrong = sum(1 for r in records if r.top1_retrieved_id is not None and not _diag_top1_is_gold(r))
    topk_gold = sum(1 for r in records if r.topk_contains_gold_anchor)
    topk_miss = total - topk_gold

    # Forced gold-anchor scorer view, restricted to gold pairs that SHOULD be
    # accepted (gold_pair_label == H1) and where a forced decision is defined
    # (scorer calibrated -> finite threshold + a score). This answers "if the
    # gold anchor were forced, would the scorer accept it?" as a TPR.
    gold_h1 = [
        r for r in records
        if r.gold_pair_label == "H1" and r.threshold is not None and r.score_gold_anchor is not None
    ]
    accepted_gold = sum(1 for r in gold_h1 if r.accepted_gold_anchor)
    rejected_gold = len(gold_h1) - accepted_gold

    # Wrong-anchor serving view (the REAL serving decision on a non-gold anchor).
    accepted_wrong = sum(1 for r in records if _diag_accepted_wrong_anchor(r))
    rejected_wrong = sum(1 for r in records if _diag_rejected_wrong_anchor(r))
    same_cluster_wrong = sum(1 for r in records if _diag_accepted_wrong_anchor(r) and _diag_retr_same_cluster(r))
    diff_cluster_wrong = sum(1 for r in records if _diag_accepted_wrong_anchor(r) and _diag_retr_diff_cluster(r))

    # Active (served-hit) outcome view.
    served_hits = sum(1 for r in records if r.served_hit)
    misses = total - served_hits
    classes = [_diag_served_class(r) for r in records]
    active_tp = sum(1 for c, _ in classes if c == "tp")
    active_fp = sum(1 for c, _ in classes if c == "fp")
    active_unjudged = sum(1 for c, _ in classes if c == "unjudged")
    active_judged_hits = active_tp + active_fp  # == judged served hits, by construction

    raw_counts = {
        "total_requests": total,
        "gold_anchor_available_count": gold_available,
        "gold_anchor_missing_count": gold_missing,
        "top1_is_gold_anchor_count": top1_gold,
        "top1_wrong_anchor_count": top1_wrong,
        "topk_contains_gold_anchor_count": topk_gold,
        "topk_misses_gold_anchor_count": topk_miss,
        "accepted_gold_anchor": accepted_gold,
        "rejected_gold_anchor": rejected_gold,
        "accepted_wrong_anchor": accepted_wrong,
        "rejected_wrong_anchor": rejected_wrong,
        "same_cluster_wrong_anchor": same_cluster_wrong,
        "different_cluster_wrong_anchor": diff_cluster_wrong,
        "active_judged_hits": active_judged_hits,
        "active_unjudged_hits": active_unjudged,
        "active_tp": active_tp,
        "active_fp": active_fp,
        "served_hits": served_hits,
        "misses": misses,
    }

    rates = {
        "gold_anchor_cache_availability_rate": _rate(gold_available, total),
        "top1_gold_match_rate": _rate(top1_gold, total),
        "topk_gold_recall": _rate(topk_gold, total),
        "scorer_tpr_given_gold_anchor": _rate(accepted_gold, accepted_gold + rejected_gold),
        "wrong_anchor_accept_rate": _rate(accepted_wrong, accepted_wrong + rejected_wrong),
        "active_precision": _rate(active_tp, active_tp + active_fp),
        "active_fp_rate": _rate(active_fp, active_judged_hits),
        "active_hit_rate": _rate(served_hits, total),
    }

    # Attribute each ACTIVE FALSE POSITIVE to a single bucket so the dominant
    # failure mode is unambiguous.
    fp_records = [r for r, (c, _) in zip(records, classes) if c == "fp"]
    attribution = {
        "gold_anchor_absent": 0,
        "gold_present_not_in_topk": 0,
        "gold_in_topk_not_selected": 0,
        "other": 0,
        "same_cluster_confusion": 0,
        "different_cluster_confusion": 0,
    }
    for r in fp_records:
        if not r.gold_anchor_available_in_cache:
            attribution["gold_anchor_absent"] += 1
        elif not r.topk_contains_gold_anchor:
            attribution["gold_present_not_in_topk"] += 1
        elif r.gold_anchor_id is None or r.retrieved_anchor_id != r.gold_anchor_id:
            attribution["gold_in_topk_not_selected"] += 1
        else:
            attribution["other"] += 1
        if _diag_retr_same_cluster(r):
            attribution["same_cluster_confusion"] += 1
        elif _diag_retr_diff_cluster(r):
            attribution["different_cluster_confusion"] += 1

    primary_buckets = {
        k: attribution[k]
        for k in ("gold_anchor_absent", "gold_present_not_in_topk", "gold_in_topk_not_selected", "other")
    }
    dominant = max(primary_buckets, key=primary_buckets.get) if active_fp > 0 else None

    return {
        "raw_counts": raw_counts,
        "rates": rates,
        "active_fp_attribution": attribution,
        "dominant_active_fp_bucket": dominant,
    }


_ANCHOR_DIAG_FIELDS = [
    "request_index", "query_id", "query_label", "query_cluster",
    "gold_anchor_id", "gold_anchor_label", "gold_cluster", "gold_anchor_available_in_cache",
    "top1_retrieved_anchor_id", "retrieved_anchor_id", "retrieved_anchor_label", "retrieved_cluster",
    "top1_is_gold_anchor", "topk_contains_gold_anchor",
    "retrieved_is_same_cluster_as_gold", "retrieved_is_different_cluster_from_gold",
    "gold_pair_label", "retrieved_pair_label",
    "score_gold_anchor", "score_retrieved_anchor", "threshold",
    "accepted_gold_anchor", "accepted_retrieved_anchor",
    "served_hit", "served_is_tp", "served_is_fp", "served_unjudged", "reason_if_unjudged",
]


def anchor_record_to_csv_row(r: AnchorDiagRecord) -> dict[str, Any]:
    cls, reason = _diag_served_class(r)
    return {
        "request_index": r.request_index,
        "query_id": r.query_id,
        "query_label": r.query_label,
        "query_cluster": r.query_cluster,
        "gold_anchor_id": "" if r.gold_anchor_id is None else r.gold_anchor_id,
        "gold_anchor_label": "" if r.gold_anchor_label is None else r.gold_anchor_label,
        "gold_cluster": "" if r.gold_cluster is None else r.gold_cluster,
        "gold_anchor_available_in_cache": bool(r.gold_anchor_available_in_cache),
        "top1_retrieved_anchor_id": "" if r.top1_retrieved_id is None else r.top1_retrieved_id,
        "retrieved_anchor_id": "" if r.retrieved_anchor_id is None else r.retrieved_anchor_id,
        "retrieved_anchor_label": "" if r.retrieved_anchor_label is None else r.retrieved_anchor_label,
        "retrieved_cluster": "" if r.retrieved_cluster is None else r.retrieved_cluster,
        "top1_is_gold_anchor": _diag_top1_is_gold(r),
        "topk_contains_gold_anchor": bool(r.topk_contains_gold_anchor),
        "retrieved_is_same_cluster_as_gold": _diag_retr_same_cluster(r),
        "retrieved_is_different_cluster_from_gold": _diag_retr_diff_cluster(r),
        "gold_pair_label": r.gold_pair_label,
        "retrieved_pair_label": r.retrieved_pair_label,
        "score_gold_anchor": "" if r.score_gold_anchor is None else r.score_gold_anchor,
        "score_retrieved_anchor": "" if r.score_retrieved_anchor is None else r.score_retrieved_anchor,
        "threshold": "" if r.threshold is None else r.threshold,
        "accepted_gold_anchor": bool(r.accepted_gold_anchor),
        "accepted_retrieved_anchor": bool(r.served_hit),
        "served_hit": bool(r.served_hit),
        "served_is_tp": cls == "tp",
        "served_is_fp": cls == "fp",
        "served_unjudged": cls == "unjudged",
        "reason_if_unjudged": reason or "",
    }


def compute_gold_anchor_map(
    rows: list[DatasetRow], *, seed: int, strategy: str = "centroid_nearest",
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-cluster gold anchor (np_compat centroid_nearest) over the online universe.

    Returns (gold_row_id_by_cluster, gold_label_by_cluster). Cluster ids are the
    DatasetRow.group strings; they are factorised to dense int codes purely so
    `nca.select_region_anchor_indices` can group them -- the anchor chosen depends
    only on the grouping + embeddings + seed, exactly as np_compat_pair_eval.
    """
    if not rows:
        return {}, {}
    emb = np.stack([np.asarray(r.embedding, dtype=np.float32).reshape(-1) for r in rows])
    clusters = [r.group for r in rows]
    code_by_group = {g: i for i, g in enumerate(sorted(set(clusters)))}
    region_id = np.array([code_by_group[g] for g in clusters], dtype=np.int64)
    idx_by_code = nca.select_region_anchor_indices(emb, region_id, strategy=strategy, seed=int(seed))
    gold_rid_by_cluster: dict[str, int] = {}
    gold_label_by_cluster: dict[str, int] = {}
    for group, code in code_by_group.items():
        local_idx = idx_by_code.get(int(code))
        if local_idx is None:
            continue
        anchor_row = rows[int(local_idx)]
        gold_rid_by_cluster[group] = int(anchor_row.row_id)
        gold_label_by_cluster[group] = int(anchor_row.label)
    return gold_rid_by_cluster, gold_label_by_cluster


def _emb_tuple(row: DatasetRow) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(row.embedding, dtype=np.float64).reshape(-1))


def replay_with_anchor_capture(
    *,
    method: str,
    scorer: str,
    scorers: list[str] | None,
    stream: list[DatasetRow],
    rows_by_id: dict[int, DatasetRow],
    gold_rid_by_cluster: dict[str, int],
    gold_label_by_cluster: dict[str, int],
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
    use_wilson: bool = True,
    warmup_stream: list[DatasetRow] | None = None,
    require_activation: bool = False,
) -> tuple[list[AnchorDiagRecord], dict[str, Any]]:
    """Replay the REAL online path, capturing the decision-time top-k and the
    served anchor per request so each can be compared against the gold NP anchor.

    Retrieval is probed via the oracle's OWN vector store BEFORE serving (so the
    candidate set is exactly what the oracle saw, prior to this request's
    write-through), then `system.handle` performs the real serving + learning.
    Nothing about the scorer, calibration, or policy is altered.

    When `warmup_stream` is given, it is replayed first (with label collection
    on) and the refit is force/retried exactly like the old online comparison,
    so a trainable scorer (the ensemble) activates BEFORE the measured phase.
    Warm-up requests are NOT recorded; the measured diagnostic records start at
    request 1 of `stream`. With `require_activation=True` the run fails fast if
    the scorer is still uncalibrated after warm-up (rather than emitting a
    diagnostic with threshold=None that would mislabel every request a cold MISS).
    """
    if state_dir.exists():
        shutil.rmtree(state_dir, ignore_errors=True)
        logger.info("[%s] cleared prior persisted state at %s", method, state_dir)

    recorder = MetricsRecorder(method=method, target_fpr=float(target_fpr), logger=logger)
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None,
        scorer=scorer,
        scorers=scorers if scorer == "ensemble" else None,
        judge=DatasetH1H0Judge(rows_by_id, recorder=recorder, logger=None),
        embedding_provider=DatasetEmbeddingProvider(rows_by_id),
        target_fpr=float(target_fpr),
        top_k=int(top_k),
        root_dir=state_dir,
        batch_size=int(batch_size),
        min_h0=int(gate_min_train_h0) + int(gate_min_calib_h0),
        min_h1=int(gate_min_train_h1) + int(gate_min_calib_h1),
        persistence=bool(persistence),
        parallelism=int(parallelism),
        namespace=f"h1h0-anchor-diag-{method}",
    )
    policy_obj, gate_diag = build_refit_policy(
        min_train_h0=gate_min_train_h0, min_train_h1=gate_min_train_h1,
        min_calib_h0=gate_min_calib_h0, min_calib_h1=gate_min_calib_h1,
        target_fpr=target_fpr, use_wilson=use_wilson,
    )
    system.cache.runtime.oracle.refit_policy = policy_obj
    logger.info("[%s] activation gate: use_wilson=%s fpr_wilson_margin=%.3f allowed_fpr_bound=%.3f "
                "min_calibration_h0=%d", method, use_wilson, gate_diag.fpr_wilson_margin,
                gate_diag.allowed_fpr_bound, gate_diag.effective_min_calib_h0)
    if gate_diag.wilson_floor_applied:
        logger.warning(
            "[%s] raised min_calibration_h0 %d -> %d (Wilson-safe floor for target_fpr=%.3f)",
            method, gate_diag.requested_min_calib_h0, gate_diag.effective_min_calib_h0, target_fpr,
        )

    oracle = system.cache.runtime.oracle
    vstore = oracle.vector_store
    key_to_row: dict[str, int] = {}
    records: list[AnchorDiagRecord] = []

    warmup_diag: dict[str, Any] = {
        "warmup_requests": 0,
        "warmup_train_h0": 0, "warmup_train_h1": 0,
        "warmup_calib_h0": 0, "warmup_calib_h1": 0,
        "threshold_after_warmup": None, "scorer_version_after_warmup": 0,
        "activated_before_measurement": False,
    }

    # ---- warm-up / activation phase (excluded from measured records) ----
    if warmup_stream:
        logger.info("[%s] warm-up: replaying %d requests before measurement (label collection on)",
                    method, len(warmup_stream))
        for widx, row in enumerate(warmup_stream, start=1):
            recorder.set_request_index(widx)
            response = system.handle(row.prompt)
            if response.source == "llm" and response.cache_key is not None:
                key_to_row[response.cache_key] = row.row_id
            recorder.resolve_request(
                widx, source=response.source,
                accepted_key=response.cache_key if response.source == "cache" else None,
            )
            if widx % max(1, batch_size) == 0:
                system._maybe_check_stopping()
            if progress_every > 0 and widx % progress_every == 0:
                wp = snapshot_policy(system)
                logger.info("[%s][warm-up %d/%d] trained=%s calibrated=%s thr=%s",
                            method, widx, len(warmup_stream), wp["trained"], wp["calibrated"],
                            _fmt(wp["threshold"]))

        # Force/retry the refit exactly like the old online comparison: a fit
        # triggered mid-warm-up may have been built from a snapshot too small to
        # pass the activation gate even though enough data has since accumulated.
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
        activation_status = dict(oracle.activation_status)
        gate_reason = activation_status.get("activation_gate_reason")
        gate_passed = activation_status.get("activation_gate_passed")
        activated = bool(wp["trained"] and wp["calibrated"] and wp["threshold"] is not None)
        warmup_diag = {
            "warmup_requests": len(warmup_stream),
            "warmup_train_h0": wc["h0_train"], "warmup_train_h1": wc["h1_train"],
            "warmup_calib_h0": wc["h0_calibration"], "warmup_calib_h1": wc["h1_calibration"],
            "threshold_after_warmup": wp["threshold"],
            "scorer_version_after_warmup": wp["scorer_version"],
            "activated_before_measurement": activated,
            "use_wilson": bool(use_wilson),
            "activation_gate_reason": gate_reason,
            "activation_gate_passed": gate_passed,
            "candidate_threshold": activation_status.get("candidate_threshold"),
            "empirical_calibration_fpr": activation_status.get("empirical_calibration_fpr"),
            "wilson_upper_fpr": activation_status.get("wilson_upper_fpr"),
            "allowed_fpr_bound": activation_status.get("allowed_fpr_bound"),
            "threshold_is_finite": activation_status.get("threshold_is_finite"),
        }
        logger.info(
            "[%s] warm-up complete: trained=%s calibrated=%s threshold_after_warmup=%s "
            "scorer_version_after_warmup=%d warmup_train_h0=%d warmup_train_h1=%d "
            "warmup_calib_h0=%d warmup_calib_h1=%d activated_before_measurement=%s "
            "gate_reason=%s candidate_threshold=%s empirical_calib_fpr=%s wilson_upper_fpr=%s "
            "allowed_fpr_bound=%s threshold_is_finite=%s",
            method, wp["trained"], wp["calibrated"], _fmt(wp["threshold"]), wp["scorer_version"],
            wc["h0_train"], wc["h1_train"], wc["h0_calibration"], wc["h1_calibration"], activated,
            gate_reason, _fmt(warmup_diag["candidate_threshold"]),
            _fmt(warmup_diag["empirical_calibration_fpr"]), _fmt(warmup_diag["wilson_upper_fpr"]),
            _fmt(warmup_diag["allowed_fpr_bound"]), warmup_diag["threshold_is_finite"],
        )
        recorder.reset()

        if require_activation and not activated:
            system._executor.shutdown(wait=False)
            raise RuntimeError(
                f"[{method}] diag-scorer requires activation but the scorer did not train/calibrate after "
                f"warm-up (trained={wp['trained']} calibrated={wp['calibrated']} threshold={wp['threshold']}); "
                f"activation_gate_reason={gate_reason!r} candidate_threshold={warmup_diag['candidate_threshold']} "
                f"threshold_is_finite={warmup_diag['threshold_is_finite']} use_wilson={use_wilson} "
                f"warmup_train_h0={wc['h0_train']} warmup_train_h1={wc['h1_train']} "
                f"warmup_calib_h0={wc['h0_calibration']} warmup_calib_h1={wc['h1_calibration']}. "
                "If reason is 'threshold_not_finite', --no-wilson will NOT help: the scorer cannot place a finite "
                "NP cut at this target FPR (raise --target-fpr or address score ties). If reason is a Wilson/FPR "
                "bound, pass --no-wilson or raise --target-fpr."
            )
    elif require_activation:
        # No warm-up but activation required: still verify before measuring.
        wp = snapshot_policy(system)
        if not (wp["trained"] and wp["calibrated"] and wp["threshold"] is not None):
            system._executor.shutdown(wait=False)
            raise RuntimeError(
                f"[{method}] diag-scorer requires activation but no warm-up was provided and the scorer is "
                "cold (threshold=None). Provide --warmup-requests so the scorer activates before measurement."
            )

    logger.info("[%s] anchor-diagnostic replay of %d MEASURED requests (scorer=%s, top_k=%d)",
                method, len(stream), ",".join(scorers) if scorers else scorer, top_k)
    started = time.time()

    for idx, row in enumerate(stream, start=1):
        recorder.set_request_index(idx)
        prompt = row.prompt
        embedding = system._embeddings.embed(prompt)
        query_emb_t = tuple(float(v) for v in embedding)

        # ---- decision-time retrieval probe (BEFORE this request's write) ----
        candidates = vstore.search(embedding, namespace=system.namespace, top_k=system.top_k)
        cand_keys = [str(c.cache_key) for c in candidates]
        top1_rid = key_to_row.get(cand_keys[0]) if cand_keys else None

        gold_rid = gold_rid_by_cluster.get(row.group)
        gold_label = gold_label_by_cluster.get(row.group)
        gold_row = rows_by_id.get(gold_rid) if gold_rid is not None else None
        gold_key = system._cache_key_for(gold_row.prompt) if gold_row is not None else None
        gold_available = bool(gold_key is not None and vstore.get(gold_key) is not None)
        topk_contains_gold = bool(gold_key is not None and str(gold_key) in cand_keys)

        # ---- forced gold-anchor score under the CURRENT serving policy ----
        pre_scorer = oracle.scorer
        pre_threshold = system.cache.threshold
        score_gold: float | None = None
        if gold_row is not None:
            feats_g = oracle.feature_builder.build(query_emb_t, _emb_tuple(gold_row))
            sg = oracle._safe_score(pre_scorer, feats_g)
            score_gold = None if sg is None else float(sg)

        # ---- serve through the REAL online path (retrieval + scoring + learn) ----
        response = system.handle(prompt)
        if response.source == "llm" and response.cache_key is not None:
            key_to_row[response.cache_key] = row.row_id
        recorder.resolve_request(
            idx, source=response.source,
            accepted_key=response.cache_key if response.source == "cache" else None,
        )

        served_hit = response.source == "cache"
        served_rid = key_to_row.get(response.cache_key) if served_hit else None
        retrieved_rid = served_rid if served_hit else top1_rid
        threshold = (
            float(response.threshold) if response.threshold is not None
            else (float(pre_threshold) if pre_threshold is not None else None)
        )

        if served_hit:
            score_retr = float(response.score) if response.score is not None else None
        elif top1_rid is not None and top1_rid in rows_by_id:
            feats_r = oracle.feature_builder.build(query_emb_t, _emb_tuple(rows_by_id[top1_rid]))
            sr = oracle._safe_score(pre_scorer, feats_r)
            score_retr = None if sr is None else float(sr)
        else:
            score_retr = None

        accepted_gold_forced = bool(
            threshold is not None and score_gold is not None and score_gold >= threshold
        )

        retrieved_row = rows_by_id.get(retrieved_rid) if retrieved_rid is not None else None
        gold_pair_label = reuse_label(row, gold_row) if gold_row is not None else "UNKNOWN"
        if retrieved_row is not None:
            retrieved_pair_label = reuse_label(row, retrieved_row)
        elif served_hit or top1_rid is not None:
            retrieved_pair_label = "UNKNOWN"
        else:
            retrieved_pair_label = "NONE"

        records.append(AnchorDiagRecord(
            request_index=idx,
            query_id=row.row_id, query_label=row.label, query_cluster=row.group,
            gold_anchor_id=gold_rid, gold_anchor_label=gold_label,
            gold_cluster=(gold_row.group if gold_row is not None else None),
            gold_anchor_available_in_cache=gold_available,
            top1_retrieved_id=top1_rid,
            retrieved_anchor_id=retrieved_rid,
            retrieved_anchor_label=(retrieved_row.label if retrieved_row is not None else None),
            retrieved_cluster=(retrieved_row.group if retrieved_row is not None else None),
            topk_contains_gold_anchor=topk_contains_gold,
            gold_pair_label=gold_pair_label,
            retrieved_pair_label=retrieved_pair_label,
            score_gold_anchor=score_gold,
            score_retrieved_anchor=score_retr,
            threshold=threshold,
            accepted_gold_anchor=accepted_gold_forced,
            served_hit=served_hit,
        ))

        if idx % max(1, batch_size) == 0:
            system._maybe_check_stopping()
        if progress_every > 0 and idx % progress_every == 0:
            pol = snapshot_policy(system)
            served = sum(1 for r in records if r.served_hit)
            logger.info("[%s][%d/%d] hits=%d trained=%s calibrated=%s thr=%s",
                        method, idx, len(stream), served, pol["trained"], pol["calibrated"],
                        _fmt(pol["threshold"]))

    final_policy = snapshot_policy(system)
    final_policy["runtime_seconds"] = time.time() - started
    final_policy["gate_diagnostics"] = gate_diag.__dict__
    final_policy["warmup"] = warmup_diag
    system._executor.shutdown(wait=False)
    return records, final_policy


def write_anchor_diagnostics(
    out_dir: Path,
    records: list[AnchorDiagRecord],
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        out_dir / "anchor_retrieval_diagnostics.csv",
        [anchor_record_to_csv_row(r) for r in records],
        _ANCHOR_DIAG_FIELDS,
    )
    write_json(out_dir / "retrieved_anchor_metrics.json", {"config": config, **metrics})
    (out_dir / "retrieved_anchor_diagnostics.md").write_text(
        _render_anchor_markdown(metrics, config), encoding="utf-8"
    )


def _rate_str(rate: dict[str, Any]) -> str:
    v = rate["value"]
    vs = "n/a" if v is None else f"{v:.4f}"
    return f"{vs} ({rate['numerator']}/{rate['denominator']})"


def _render_anchor_markdown(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    rc = metrics["raw_counts"]
    rt = metrics["rates"]
    attr = metrics["active_fp_attribution"]
    dom = metrics["dominant_active_fp_bucket"]

    lines = [
        "# Retrieved-Anchor Diagnostics (real online replay vs gold NP anchor)",
        "",
        f"- npz: `{config.get('npz')}`  scorer: `{config.get('diag_scorer')}`  seed: {config.get('seed')}",
        f"- cache_universe: {config.get('cache_universe')}  stream_length: {config.get('stream_length')}  "
        f"top_k: {config.get('top_k')}  target_fpr: {config.get('target_fpr')}  selection: {config.get('selection')}",
        f"- gold anchor rule: np_compat centroid_nearest over the cache universe (== np_compat_pair_eval anchor "
        f"restricted to retrievable rows)",
        "",
        "## Raw counts (every rate below prints numerator/denominator)",
        "",
        "| count | value |",
        "|---|---:|",
    ]
    for k, v in rc.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Derived rates",
        "",
        "| rate | value (num/den) |",
        "|---|---|",
    ]
    for k, rate in rt.items():
        lines.append(f"| {k} | {_rate_str(rate)} |")

    lines += [
        "",
        "## Active false-positive attribution",
        "",
        f"- gold_anchor_absent: {attr['gold_anchor_absent']}",
        f"- gold_present_not_in_topk: {attr['gold_present_not_in_topk']}",
        f"- gold_in_topk_not_selected: {attr['gold_in_topk_not_selected']}",
        f"- other: {attr['other']}",
        f"- same_cluster_confusion: {attr['same_cluster_confusion']}",
        f"- different_cluster_confusion: {attr['different_cluster_confusion']}",
        f"- **dominant active-FP bucket: {dom}**",
        "",
        "## Direct answers",
        "",
        f"1. Gold anchor available in cache: {_rate_str(rt['gold_anchor_cache_availability_rate'])}.",
        f"2. Top-1 retrieval returns the gold anchor: {_rate_str(rt['top1_gold_match_rate'])}.",
        f"3. Gold anchor present in top-k: {_rate_str(rt['topk_gold_recall'])}.",
        f"4. When online serves a false positive ({rc['active_fp']} FPs), the dominant cause is "
        f"`{dom}` (absent={attr['gold_anchor_absent']}, present-but-not-in-topk="
        f"{attr['gold_present_not_in_topk']}, in-topk-not-selected={attr['gold_in_topk_not_selected']}).",
        f"5. Wrong accepted anchors by cluster: same-cluster={attr['same_cluster_confusion']}, "
        f"different-cluster={attr['different_cluster_confusion']}.",
        f"6. If the gold anchor were forced, the scorer accepts the H1 gold pair at "
        f"{_rate_str(rt['scorer_tpr_given_gold_anchor'])} (forced gold-anchor acceptance among judged H1 gold "
        f"pairs).",
        "7. " + _anchor_verdict(metrics),
        "",
        "## Interpretation rule",
        "",
        "np_compat_pair_eval and forced_gold_anchor_eval already pass, so the scorer + NP threshold are not the "
        "cause unless `scorer_tpr_given_gold_anchor` is low (the scorer rejects even the correct, forced gold "
        "anchor). If gold-anchor acceptance is high but online active FPs are high, the failure is retrieval / "
        "anchor selection / cache state, per the attribution above -- NOT the scorer or calibration.",
        "",
    ]
    return "\n".join(lines)


def _anchor_verdict(metrics: dict[str, Any]) -> str:
    rc = metrics["raw_counts"]
    rt = metrics["rates"]
    tpr = rt["scorer_tpr_given_gold_anchor"]["value"]
    dom = metrics["dominant_active_fp_bucket"]
    if rc["active_fp"] == 0:
        return ("No active false positives in this run: the served judged anchors were all true positives, so "
                "there is no online FP gap to attribute here.")

    tpr_str = _rate_str(rt["scorer_tpr_given_gold_anchor"])
    retrieval_clause = (
        f"the dominant active-FP bucket is `{dom}` and top-1 retrieval returns the gold anchor only "
        f"{_rate_str(rt['top1_gold_match_rate'])} of the time (gold available "
        f"{_rate_str(rt['gold_anchor_cache_availability_rate'])})"
    )

    if tpr is None:
        return ("Inconclusive on the scorer axis: no judged H1 gold pairs had a calibrated forced decision in "
                f"this run, so gold-anchor acceptance could not be measured. The {rc['active_fp']} active FPs are "
                f"still attributable by retrieval: {retrieval_clause}.")
    if tpr >= 0.7:
        return ("This is a RETRIEVAL / anchor-selection problem, NOT a scorer/calibration problem: the scorer "
                f"accepts the forced gold anchor at a high rate ({tpr_str}) -- consistent with the passing "
                f"np_compat / forced-gold modes -- yet online serves {rc['active_fp']} active FPs because "
                f"{retrieval_clause}.")
    if tpr < 0.5:
        return ("This looks like a SCORER / calibration problem: even the forced gold anchor is accepted at only "
                f"{tpr_str}, so the scorer rejects correct anchors -- inconsistent with the passing "
                "np_compat / forced-gold modes; re-check the online calibration path before blaming retrieval.")
    return ("MIXED: the forced gold anchor is accepted at a middling rate "
            f"({tpr_str}), so the scorer is imperfect on the gold anchor, but retrieval is also implicated -- "
            f"{retrieval_clause}. Both axes contribute; the dominant active-FP bucket is `{dom}`.")


def run_retrieved_anchor_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT.parent / "experiments" / f"retrieved_anchor_diag_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir)

    logger.info("=== RETRIEVED-ANCHOR DIAGNOSTICS ===")
    logger.info("npz: %s  output_dir: %s  diag_scorer: %s", args.npz, str(out_dir), args.diag_scorer)

    rows, schema = load_rows(
        args.npz, query_field=args.query_field, label_field=args.label_field,
        anchor_field=args.anchor_field, embedding_field=args.embedding_field,
        max_rows=int(args.cache_size), seed=int(args.seed),
        include_uncertain=False, selection="random", allow_pickle=True,
    )
    dist = schema["label_distribution_selected"]
    if int(dist.get("0", 0)) <= 0 or int(dist.get("1", 0)) <= 0:
        raise ValueError(f"Selected rows need both label=0 and label=1; got {dist}.")
    rows_by_id = {r.row_id: r for r in rows}
    gold_rid_by_cluster, gold_label_by_cluster = compute_gold_anchor_map(rows, seed=int(args.seed))
    logger.info("cache_universe=%d clusters_with_gold_anchor=%d emb_dim=%d",
                len(rows), len(gold_rid_by_cluster), schema["embedding_dim"])

    stream, selection_diag = build_stream(
        rows, max_requests=int(args.max_requests), seed=int(args.seed),
        selection=str(args.selection), reuse_clusters_top_k=int(args.reuse_clusters_top_k),
        reuse_min_cluster_size=int(args.reuse_min_cluster_size),
        mixed_reuse_fraction=float(args.mixed_reuse_fraction),
    )

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    diag_scorers = scorers if args.diag_scorer == "ensemble" else None

    # The ensemble must be warm-started/activated before measurement (cosine may
    # run cold). If no --warmup-requests is given for the ensemble, default to a
    # warm-up large enough to reach the train/calibration gates.
    require_activation = args.diag_scorer == "ensemble"
    # Wilson is OFF by default in the diagnostic (--use-wilson re-enables it). The
    # NP threshold already pins calib FPR to ~target; we drop the extra Wilson
    # safety margin so activation is not blocked by it.
    use_wilson = bool(getattr(args, "use_wilson", False))
    logger.info("activation gate Wilson upper-bound: %s", "ON (--use-wilson)" if use_wilson else "OFF (default)")
    warmup_requests = int(args.warmup_requests)
    if require_activation and warmup_requests <= 0:
        warmup_requests = max(4000, int(args.max_requests))
        logger.info("ensemble diag: no --warmup-requests given; defaulting warm-up to %d requests",
                    warmup_requests)
    warmup_stream: list[DatasetRow] | None = None
    if warmup_requests > 0:
        warmup_stream, warmup_selection_diag = build_stream(
            rows, max_requests=warmup_requests, seed=int(args.seed) + 1_000_000,
            selection=str(args.selection), reuse_clusters_top_k=int(args.reuse_clusters_top_k),
            reuse_min_cluster_size=int(args.reuse_min_cluster_size),
            mixed_reuse_fraction=float(args.mixed_reuse_fraction),
        )
        logger.info("warm-up stream length=%d (seed offset +1e6) selection=%s",
                    len(warmup_stream), warmup_selection_diag.get("mode"))

    records, final_policy = replay_with_anchor_capture(
        method=args.diag_scorer, scorer=args.diag_scorer, scorers=diag_scorers,
        stream=stream, rows_by_id=rows_by_id,
        gold_rid_by_cluster=gold_rid_by_cluster, gold_label_by_cluster=gold_label_by_cluster,
        target_fpr=float(args.target_fpr), top_k=int(args.top_k), batch_size=int(args.batch_size),
        gate_min_train_h0=int(args.min_train_h0), gate_min_train_h1=int(args.min_train_h1),
        gate_min_calib_h0=int(args.min_calib_h0), gate_min_calib_h1=int(args.min_calib_h1),
        parallelism=int(args.parallelism), persistence=bool(args.persist),
        state_dir=out_dir / "state", logger=logger, progress_every=int(args.progress_every),
        use_wilson=use_wilson, warmup_stream=warmup_stream, require_activation=require_activation,
    )

    metrics = aggregate_anchor_diagnostics(records)
    config = {
        "npz": str(args.npz), "output_dir": str(out_dir), "diag_scorer": args.diag_scorer,
        "scorers": scorers, "seed": args.seed, "target_fpr": args.target_fpr,
        "cache_universe": len(rows), "stream_length": len(stream), "top_k": args.top_k,
        "use_wilson": use_wilson,
        "selection": selection_diag.get("mode"), "max_requests": args.max_requests,
        "min_train_h0": args.min_train_h0, "min_train_h1": args.min_train_h1,
        "min_calib_h0": args.min_calib_h0, "min_calib_h1": args.min_calib_h1,
        "trained": final_policy["trained"], "calibrated": final_policy["calibrated"],
        "final_threshold": final_policy["threshold"],
        "label_distribution_selected": dist,
        "warmup": final_policy.get("warmup"),
    }
    write_anchor_diagnostics(out_dir, records, metrics, config)

    rc, rt = metrics["raw_counts"], metrics["rates"]
    logger.info("=== ANCHOR DIAGNOSTIC RESULT ===")
    wu = final_policy.get("warmup") or {}
    logger.info("warmup: requests=%s activated_before_measurement=%s threshold_after_warmup=%s "
                "scorer_version_after_warmup=%s (train_h0=%s train_h1=%s calib_h0=%s calib_h1=%s)",
                wu.get("warmup_requests"), wu.get("activated_before_measurement"),
                _fmt(wu.get("threshold_after_warmup")), wu.get("scorer_version_after_warmup"),
                wu.get("warmup_train_h0"), wu.get("warmup_train_h1"),
                wu.get("warmup_calib_h0"), wu.get("warmup_calib_h1"))
    logger.info("gold_availability=%s top1_gold=%s topk_recall=%s",
                _rate_str(rt["gold_anchor_cache_availability_rate"]),
                _rate_str(rt["top1_gold_match_rate"]), _rate_str(rt["topk_gold_recall"]))
    logger.info("active_tp=%d active_fp=%d active_precision=%s active_fp_rate=%s",
                rc["active_tp"], rc["active_fp"],
                _rate_str(rt["active_precision"]), _rate_str(rt["active_fp_rate"]))
    logger.info("scorer_tpr_given_gold_anchor=%s  dominant_active_fp_bucket=%s",
                _rate_str(rt["scorer_tpr_given_gold_anchor"]), metrics["dominant_active_fp_bucket"])
    logger.info("verdict: %s", _anchor_verdict(metrics))
    logger.info("artifacts -> %s", str(out_dir.resolve()))
    return {"output_dir": str(out_dir), "metrics": metrics, "config": config}


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

    use_wilson = bool(getattr(args, "use_wilson", False))
    logger.info("activation gate Wilson upper-bound: %s", "ON (--use-wilson)" if use_wilson else "OFF (default)")
    shared_cache = bool(getattr(args, "shared_cache", False))
    logger.info("shared-cache write-through (fair A/B, identical retrieval pool): %s",
                "ON (--shared-cache)" if shared_cache else "OFF (each scorer builds its own cache)")

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
            use_wilson=use_wilson,
            shared_cache=shared_cache,
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
    p.add_argument("--mode", choices=["compare", "retrieved_anchor_diagnostics"], default="compare",
                   help="compare (default): the cosine-vs-ensemble online replay. "
                        "retrieved_anchor_diagnostics: audit how often real online retrieval serves the gold "
                        "NP anchor (writes anchor_retrieval_diagnostics.csv / retrieved_anchor_metrics.json / "
                        "retrieved_anchor_diagnostics.md).")
    p.add_argument("--retrieved-anchor-diagnostics", dest="retrieved_anchor_diagnostics", action="store_true",
                   help="Convenience alias for --mode retrieved_anchor_diagnostics.")
    p.add_argument("--diag-scorer", choices=["cosine", "ensemble"], default="ensemble",
                   help="retrieved_anchor_diagnostics: which single online system to replay/audit.")
    p.add_argument("--use-wilson", dest="use_wilson", action="store_true",
                   help="retrieved_anchor_diagnostics: re-enable the Wilson upper-bound activation gate "
                        "(+ its calib-H0 floor). Default: OFF -- the NP threshold already controls calib FPR, "
                        "so activation gates only on counts + a finite threshold.")
    p.add_argument("--shared-cache", dest="shared_cache", action="store_true",
                   help="compare mode: write every request's row into the cache even on a HIT, so cosine and "
                        "ensemble caches hold the IDENTICAL set of rows and cosine-NN retrieves the same "
                        "candidates for both -- a fair A/B where only the accept/reject decision differs. "
                        "Default: OFF (each scorer's hit/miss decisions shape its own cache).")
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
    if getattr(args, "retrieved_anchor_diagnostics", False):
        args.mode = "retrieved_anchor_diagnostics"
    try:
        if args.mode == "retrieved_anchor_diagnostics":
            run_retrieved_anchor_diagnostics(args)
        else:
            run_experiment(args)
    except Exception as exc:
        print(f"online_replay_h1h0_final.py failed: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
