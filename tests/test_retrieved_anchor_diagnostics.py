"""Tests for the retrieved-anchor diagnostic (online_replay_h1h0_final.py).

These prove the diagnostic COUNTERS behave correctly (the audit's whole value is
that its buckets are trustworthy), plus a tiny end-to-end run on a real-format
NPZ fixture that exercises the real online retrieval/serving path.

The counter tests build `AnchorDiagRecord`s by hand and call the pure aggregator
`aggregate_anchor_diagnostics`, so they do not depend on the online system at all
-- exactly the property required to trust the numbers the audit reports.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import online_replay_h1h0_final as exp  # noqa: E402


def _sklearn_ok() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_sklearn = pytest.mark.skipif(not _sklearn_ok(), reason="sklearn not installed")


def rec(**overrides) -> exp.AnchorDiagRecord:
    """Build an AnchorDiagRecord with neutral defaults; override per test."""
    base = dict(
        request_index=1,
        query_id=1, query_label=1, query_cluster="A",
        gold_anchor_id=5, gold_anchor_label=1, gold_cluster="A",
        gold_anchor_available_in_cache=True,
        top1_retrieved_id=5,
        retrieved_anchor_id=5,
        retrieved_anchor_label=1, retrieved_cluster="A",
        topk_contains_gold_anchor=True,
        gold_pair_label="H1",
        retrieved_pair_label="H1",
        score_gold_anchor=0.9,
        score_retrieved_anchor=0.9,
        threshold=0.5,
        accepted_gold_anchor=True,
        served_hit=True,
        evaluated_h0_candidate_count=0,
        accepted_h0_candidate_count=0,
    )
    base.update(overrides)
    return exp.AnchorDiagRecord(**base)


# ── 1. retrieved anchor == gold anchor, accepted, H1 ────────────────────────

def test_gold_anchor_served_h1_counts_as_tp_not_wrong():
    m = exp.aggregate_anchor_diagnostics([rec()])
    c = m["raw_counts"]
    assert c["top1_is_gold_anchor_count"] == 1
    assert c["active_tp"] == 1
    assert c["accepted_wrong_anchor"] == 0
    assert c["active_fp"] == 0


# ── 2. retrieved anchor != gold, H0, accepted ───────────────────────────────

def test_wrong_anchor_served_h0_counts_as_fp_and_wrong():
    r = rec(
        top1_retrieved_id=9, retrieved_anchor_id=9, retrieved_anchor_label=0,
        retrieved_cluster="B", retrieved_pair_label="H0", topk_contains_gold_anchor=False,
    )
    c = exp.aggregate_anchor_diagnostics([r])["raw_counts"]
    assert c["top1_wrong_anchor_count"] == 1
    assert c["accepted_wrong_anchor"] == 1
    assert c["active_fp"] == 1
    assert c["active_tp"] == 0


# ── 3. gold anchor in top-k but top-1 is a wrong anchor ─────────────────────

def test_gold_in_topk_but_top1_wrong():
    # top-1 retrieval is a wrong anchor; gold still appears somewhere in top-k.
    r = rec(
        top1_retrieved_id=9, retrieved_anchor_id=9, retrieved_anchor_label=0,
        retrieved_cluster="B", retrieved_pair_label="H0",
        topk_contains_gold_anchor=True,
    )
    c = exp.aggregate_anchor_diagnostics([r])["raw_counts"]
    assert c["topk_contains_gold_anchor_count"] == 1
    assert c["top1_is_gold_anchor_count"] == 0


# ── 4. gold anchor not available in cache ───────────────────────────────────

def test_gold_anchor_missing_not_a_retrieval_success():
    r = rec(
        gold_anchor_available_in_cache=False,
        top1_retrieved_id=9, retrieved_anchor_id=9, retrieved_anchor_label=0,
        retrieved_cluster="B", retrieved_pair_label="H0",
        topk_contains_gold_anchor=False,
    )
    c = exp.aggregate_anchor_diagnostics([r])["raw_counts"]
    assert c["gold_anchor_missing_count"] == 1
    assert c["gold_anchor_available_count"] == 0
    # A missing gold anchor cannot be retrieved, so it counts as no retrieval success.
    assert c["top1_is_gold_anchor_count"] == 0
    assert c["topk_contains_gold_anchor_count"] == 0


# ── 5. denominators: active_fdr and active_precision agree ─────────────────

def test_active_rate_denominators_are_consistent():
    records = []
    # 2 true positives (gold served, H1)
    records += [rec(request_index=i) for i in range(2)]
    # 3 false positives (wrong anchor served, H0)
    records += [
        rec(request_index=10 + i, top1_retrieved_id=9, retrieved_anchor_id=9,
            retrieved_anchor_label=0, retrieved_cluster="B", retrieved_pair_label="H0")
        for i in range(3)
    ]
    # 1 self-pair HIT (unjudged) and 1 miss -> must NOT enter the judged denominator
    records.append(rec(request_index=20, retrieved_anchor_id=1, top1_retrieved_id=1,
                       retrieved_pair_label="H1"))  # self pair (retrieved==query_id==1)
    records.append(rec(request_index=21, served_hit=False, top1_retrieved_id=9,
                       retrieved_anchor_id=9, retrieved_pair_label="H0"))

    m = exp.aggregate_anchor_diagnostics(records)
    c, rt = m["raw_counts"], m["rates"]

    assert c["active_tp"] == 2
    assert c["active_fp"] == 3
    assert c["active_judged_hits"] == c["active_tp"] + c["active_fp"] == 5
    assert c["active_unjudged_hits"] == 1  # the self-pair HIT

    # active_fdr = active_fp / active_judged_hits
    assert rt["active_fdr"]["value"] == pytest.approx(3 / 5)
    assert rt["active_fdr"]["denominator"] == 5
    # active_precision = active_tp / (active_tp + active_fp)
    assert rt["active_precision"]["value"] == pytest.approx(2 / 5)
    assert rt["active_precision"]["denominator"] == 5
    # The two rates must share the identical denominator (not silently different).
    assert rt["active_fdr"]["denominator"] == rt["active_precision"]["denominator"]
    assert rt["active_fdr"]["value"] + rt["active_precision"]["value"] == pytest.approx(1.0)


# ── forced gold-anchor TPR uses only judged H1 gold pairs ───────────────────

def test_scorer_tpr_given_gold_anchor_uses_judged_h1_only():
    records = [
        rec(score_gold_anchor=0.9, accepted_gold_anchor=True),    # H1 gold, accepted
        rec(score_gold_anchor=0.1, accepted_gold_anchor=False),   # H1 gold, rejected
        rec(gold_pair_label="H0", score_gold_anchor=0.9, accepted_gold_anchor=True),  # H0 gold -> excluded
        rec(gold_pair_label="H1", threshold=None, score_gold_anchor=None,
            accepted_gold_anchor=False),  # uncalibrated -> excluded
    ]
    m = exp.aggregate_anchor_diagnostics(records)
    c, rt = m["raw_counts"], m["rates"]
    assert c["accepted_gold_anchor"] == 1
    assert c["rejected_gold_anchor"] == 1
    assert rt["scorer_tpr_given_gold_anchor"]["value"] == pytest.approx(0.5)
    assert rt["scorer_tpr_given_gold_anchor"]["denominator"] == 2


# ── same vs different cluster wrong-anchor confusion ────────────────────────

def test_wrong_anchor_cluster_confusion_split():
    records = [
        rec(top1_retrieved_id=7, retrieved_anchor_id=7, retrieved_cluster="A",
            retrieved_anchor_label=0, retrieved_pair_label="H0"),   # same cluster as gold (A)
        rec(top1_retrieved_id=8, retrieved_anchor_id=8, retrieved_cluster="C",
            retrieved_anchor_label=0, retrieved_pair_label="H0"),   # different cluster
    ]
    c = exp.aggregate_anchor_diagnostics(records)["raw_counts"]
    assert c["accepted_wrong_anchor"] == 2
    assert c["same_cluster_wrong_anchor"] == 1
    assert c["different_cluster_wrong_anchor"] == 1


# ── gold-anchor map matches np_compat's anchor selection ────────────────────

def test_gold_anchor_map_matches_np_compat_indices():
    import np_compat_audit as nca
    from compare_cosine_vs_ensemble import DatasetRow

    rng = np.random.default_rng(3)
    rows = []
    clusters = ["0", "0", "0", "1", "1", "1"]
    embs = rng.normal(size=(6, 8)).astype(np.float32)
    for i, (g, e) in enumerate(zip(clusters, embs)):
        rows.append(DatasetRow(row_id=100 + i, text=f"t{i}", label=i % 2,
                               group=g, embedding=e, prompt=f"row:{100 + i}\tt{i}"))

    gold_rid, gold_label = exp.compute_gold_anchor_map(rows, seed=42)
    # Reference: np_compat anchor indices on the same matrix/grouping.
    region = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    idx_by_code = nca.select_region_anchor_indices(embs, region, strategy="centroid_nearest", seed=42)
    assert gold_rid["0"] == rows[int(idx_by_code[0])].row_id
    assert gold_rid["1"] == rows[int(idx_by_code[1])].row_id
    assert gold_label["0"] == rows[int(idx_by_code[0])].label


# ── tiny real-format NPZ fixture for the end-to-end path ────────────────────

def _make_npz(path: Path, *, clusters=5, per_cluster_h1=7, per_cluster_h0=4, dim=16, seed=7) -> Path:
    rng = np.random.default_rng(seed)
    texts, labels, embs, gclusters = [], [], [], []
    for c in range(clusters):
        centroid = rng.normal(size=dim); centroid /= np.linalg.norm(centroid)
        for j in range(per_cluster_h1 + per_cluster_h0):
            label = 1 if j < per_cluster_h1 else 0
            v = centroid + 0.02 * rng.normal(size=dim); v /= np.linalg.norm(v)
            texts.append(f"cluster {c} row {j} label {label}")
            labels.append(label); gclusters.append(c); embs.append(v.astype(np.float32))
    for _ in range(3):
        v = rng.normal(size=dim); v /= np.linalg.norm(v)
        texts.append("noise"); labels.append(-1); gclusters.append(-1); embs.append(v.astype(np.float32))
    np.savez(path, text=np.array(texts, dtype=object), label=np.array(labels, dtype=np.int32),
             emb=np.stack(embs).astype(np.float32), global_cluster=np.array(gclusters, dtype=np.int32))
    return path


@skip_no_sklearn
def test_end_to_end_diagnostic_writes_consistent_artifacts(tmp_path):
    npz = _make_npz(tmp_path / "tiny.npz")
    out = tmp_path / "diag"
    args = exp.parse_args([
        "--mode", "retrieved_anchor_diagnostics", "--diag-scorer", "cosine",
        "--npz", str(npz), "--output-dir", str(out),
        "--max-requests", "200", "--cache-size", "55", "--top-k", "5",
        "--min-train-h0", "4", "--min-train-h1", "2", "--min-calib-h0", "2",
        "--min-calib-h1", "1", "--target-fpr", "0.30", "--batch-size", "20",
        "--progress-every", "1000", "--seed", "42",
    ])
    result = exp.run_retrieved_anchor_diagnostics(args)

    for name in ("anchor_retrieval_diagnostics.csv", "retrieved_anchor_metrics.json",
                 "retrieved_anchor_diagnostics.md"):
        assert (out / name).exists(), f"missing artifact: {name}"

    metrics = json.loads((out / "retrieved_anchor_metrics.json").read_text(encoding="utf-8"))
    c = metrics["raw_counts"]
    # Structural invariants of the aggregation on a real run.
    assert c["total_requests"] == 200
    assert c["served_hits"] + c["misses"] == c["total_requests"]
    assert c["active_judged_hits"] == c["active_tp"] + c["active_fp"]
    assert c["gold_anchor_available_count"] + c["gold_anchor_missing_count"] == c["total_requests"]

    with (out / "anchor_retrieval_diagnostics.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 200
    # Every required column is present.
    for col in exp._ANCHOR_DIAG_FIELDS:
        assert col in csv_rows[0], f"missing csv column: {col}"
    # CSV served_is_tp / served_is_fp counts match the aggregated metrics.
    assert sum(1 for r in csv_rows if r["served_is_tp"] == "True") == c["active_tp"]
    assert sum(1 for r in csv_rows if r["served_is_fp"] == "True") == c["active_fp"]

    assert result["metrics"]["raw_counts"]["total_requests"] == 200


@skip_no_sklearn
def test_ensemble_warm_starts_and_is_active_before_measurement(tmp_path):
    npz = _make_npz(tmp_path / "tiny.npz")
    out = tmp_path / "diag_ens"
    args = exp.parse_args([
        "--mode", "retrieved_anchor_diagnostics", "--diag-scorer", "ensemble",
        "--scorers", "cosine,lda", "--npz", str(npz), "--output-dir", str(out),
        "--warmup-requests", "400", "--max-requests", "120", "--cache-size", "55", "--top-k", "5",
        "--min-train-h0", "4", "--min-train-h1", "2", "--min-calib-h0", "2",
        "--min-calib-h1", "1", "--target-fpr", "0.30", "--batch-size", "20",
        "--progress-every", "1000", "--seed", "42",
    ])
    exp.run_retrieved_anchor_diagnostics(args)

    metrics = json.loads((out / "retrieved_anchor_metrics.json").read_text(encoding="utf-8"))
    wu = metrics["config"]["warmup"]
    # The ensemble must be active BEFORE the measured phase (item 4).
    assert wu["activated_before_measurement"] is True
    assert wu["threshold_after_warmup"] is not None
    assert wu["scorer_version_after_warmup"] >= 1
    assert wu["warmup_requests"] == 400
    assert metrics["config"]["calibrated"] is True
    assert metrics["config"]["final_threshold"] is not None

    # Warm-up requests must NOT be in the measured records (item 6).
    with (out / "anchor_retrieval_diagnostics.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 120
    assert metrics["raw_counts"]["total_requests"] == 120
    # Measured request indices start at 1 (item 5).
    assert min(int(r["request_index"]) for r in csv_rows) == 1


@skip_no_sklearn
def test_ensemble_fails_fast_when_it_cannot_activate(tmp_path):
    npz = _make_npz(tmp_path / "tiny.npz")
    out = tmp_path / "diag_fail"
    # Unreachable H1 train gate -> scorer can never train/calibrate.
    args = exp.parse_args([
        "--mode", "retrieved_anchor_diagnostics", "--diag-scorer", "ensemble",
        "--scorers", "cosine,lda", "--npz", str(npz), "--output-dir", str(out),
        "--warmup-requests", "120", "--max-requests", "60", "--cache-size", "55", "--top-k", "5",
        "--min-train-h0", "4", "--min-train-h1", "100000", "--min-calib-h0", "2",
        "--min-calib-h1", "1", "--target-fpr", "0.30", "--batch-size", "20",
        "--progress-every", "1000", "--seed", "42",
    ])
    with pytest.raises(RuntimeError, match="requires activation"):
        exp.run_retrieved_anchor_diagnostics(args)
    # Must fail BEFORE writing a misleading threshold=None diagnostic.
    assert not (out / "retrieved_anchor_metrics.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
