"""Tests for the dataset-backed online replay experiment.

These use a TINY but REAL-format NPZ fixture (the actual field names and dtypes
of `h1h0_final.npz`: `text`, `label`, `emb`, `global_cluster`) so the real
loader / embedding provider / judge / online lifecycle are exercised without
the 791MB file and without any synthetic TopicJudge or #topic prompts.

Coverage (Step 8):
1. real NPZ loader does not silently fall back to synthetic data
2. cosine and ensemble replays use an identical request order
3. dataset-backed embedding provider returns exact NPZ embeddings
4. dataset-backed judge is deterministic
5. ensemble activation only happens after train/calibration gates are satisfied
6. the active-decision audit is written and contains FP rows when FPs occur
7. the script fails loudly when required NPZ fields are missing
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
import logging # Moved to top

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _sklearn_ok() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_sklearn = pytest.mark.skipif(not _sklearn_ok(), reason="sklearn not installed")

import online_replay_h1h0_final as exp  # noqa: E402
from compare_cosine_vs_ensemble import (  # noqa: E402
    DatasetEmbeddingProvider,
    DatasetH1H0Judge,
    encode_prompt,
    load_rows,
)
from mlcache import JudgeLabel, JudgeRequest, MockLLM, SemanticCacheSystem  # noqa: E402
from mlcache.semantic_types import CacheKey, Query  # noqa: E402


# ── tiny real-format NPZ fixture ───────────────────────────────────────────

def _make_npz(path: Path, *, clusters: int = 5, per_cluster_h1: int = 7,
              per_cluster_h0: int = 4, dim: int = 16, seed: int = 7,
              include_fields: tuple[str, ...] = ("text", "label", "emb", "global_cluster")) -> Path:
    """Build a small NPZ with the real schema.

    Same-cluster rows share a tight embedding centroid (high cosine) so retrieval
    keeps neighbours within a cluster; cross-cluster rows are well separated.
    Each cluster mixes label==1 (reusable) and label==0 (in-cluster non-reuse)
    rows, so same-cluster retrieval yields both H1 (1&1) and H0 (involving a 0)
    pairs -- and same-cluster label-0 queries hitting a label-1 anchor are the
    realistic false positives.
    """
    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels: list[int] = []
    embs: list[np.ndarray] = []
    gclusters: list[int] = []

    for c in range(clusters):
        centroid = rng.normal(size=dim)
        centroid /= np.linalg.norm(centroid)
        for j in range(per_cluster_h1 + per_cluster_h0):
            label = 1 if j < per_cluster_h1 else 0
            v = centroid + 0.02 * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            texts.append(f"cluster {c} row {j} label {label} :: some real request text")
            labels.append(label)
            gclusters.append(c)
            embs.append(v.astype(np.float32))

    # A few noise rows (label=-1, cluster=-1) that the loader must drop.
    for _ in range(3):
        v = rng.normal(size=dim); v /= np.linalg.norm(v)
        texts.append("noise request"); labels.append(-1); gclusters.append(-1)
        embs.append(v.astype(np.float32))

    arrays = {
        "text": np.array(texts, dtype=object),
        "label": np.array(labels, dtype=np.int32),
        "emb": np.stack(embs).astype(np.float32),
        "global_cluster": np.array(gclusters, dtype=np.int32),
    }
    np.savez(path, **{k: arrays[k] for k in include_fields})
    return path


@pytest.fixture()
def tiny_npz(tmp_path):
    return _make_npz(tmp_path / "tiny_h1h0.npz")


# ── 1. loader does not silently fall back to synthetic data ────────────────

def test_loader_raises_on_missing_file_no_synthetic_fallback(tmp_path):
    missing = tmp_path / "does_not_exist.npz"
    with pytest.raises(Exception):
        load_rows(missing, query_field="text", label_field="label",
                  anchor_field="global_cluster", embedding_field="emb",
                  max_rows=10, seed=1, include_uncertain=False,
                  selection="random", allow_pickle=True)


def test_loader_returns_real_npz_rows(tiny_npz):
    rows, schema = load_rows(tiny_npz, query_field="text", label_field="label",
                             anchor_field="global_cluster", embedding_field="emb",
                             max_rows=1000, seed=1, include_uncertain=False,
                             selection="random", allow_pickle=True)
    # Noise (-1) rows are dropped; only 0/1 remain.
    assert all(r.label in (0, 1) for r in rows)
    assert schema["embedding_dim"] == 16
    assert int(schema["label_distribution_selected"].get("1", 0)) > 0
    assert int(schema["label_distribution_selected"].get("0", 0)) > 0


# ── 7. fail loudly if required NPZ fields are missing ──────────────────────

def test_missing_embedding_field_fails_loudly(tmp_path):
    path = _make_npz(tmp_path / "no_emb.npz", include_fields=("text", "label", "global_cluster"))
    with pytest.raises(ValueError) as ei:
        load_rows(path, query_field="text", label_field="label",
                  anchor_field="global_cluster", embedding_field="emb",
                  max_rows=10, seed=1, include_uncertain=False,
                  selection="random", allow_pickle=True)
    assert "emb" in str(ei.value)


def test_missing_anchor_field_fails_loudly(tmp_path):
    path = _make_npz(tmp_path / "no_cluster.npz", include_fields=("text", "label", "emb"))
    with pytest.raises(ValueError) as ei:
        load_rows(path, query_field="text", label_field="label",
                  anchor_field="global_cluster", embedding_field="emb",
                  max_rows=10, seed=1, include_uncertain=False,
                  selection="random", allow_pickle=True)
    assert "global_cluster" in str(ei.value)


# ── 3. embedding provider returns exact NPZ embeddings ─────────────────────

def test_embedding_provider_returns_exact_npz_embeddings(tiny_npz):
    rows, _ = load_rows(tiny_npz, query_field="text", label_field="label",
                        anchor_field="global_cluster", embedding_field="emb",
                        max_rows=1000, seed=1, include_uncertain=False,
                        selection="random", allow_pickle=True)
    rows_by_id = {r.row_id: r for r in rows}
    provider = DatasetEmbeddingProvider(rows_by_id)

    npz = np.load(tiny_npz, allow_pickle=True)
    emb = npz["emb"]
    for r in rows[:5]:
        got = np.asarray(provider.embed(encode_prompt(r.row_id, r.text)), dtype=np.float64)
        expected = np.asarray(emb[r.row_id], dtype=np.float64)
        assert np.allclose(got, expected, atol=1e-6)


# ── 4. judge is deterministic and follows the dataset rule ─────────────────

def test_dataset_judge_is_deterministic_and_rule_correct(tiny_npz):
    rows, _ = load_rows(tiny_npz, query_field="text", label_field="label",
                        anchor_field="global_cluster", embedding_field="emb",
                        max_rows=1000, seed=1, include_uncertain=False,
                        selection="random", allow_pickle=True)
    rows_by_id = {r.row_id: r for r in rows}
    judge = DatasetH1H0Judge(rows_by_id, logger=logging.getLogger("test_judge"))
    # Same-cluster label-1/label-1 pair -> REUSABLE; repeated calls agree.
    h1_rows = [r for r in rows if r.label == 1]
    by_cluster: dict[int, list] = {}
    for r in h1_rows:
        by_cluster.setdefault(r.group, []).append(r)
    pair = next(v for v in by_cluster.values() if len(v) >= 2)
    a, b = pair[0], pair[1]
    req = JudgeRequest(query=Query(encode_prompt(a.row_id, a.text)),
                       candidate_query=Query(encode_prompt(b.row_id, b.text)),
                       candidate_key=CacheKey("k"))
    first = judge.judge(req).decision.label
    second = judge.judge(req).decision.label
    assert first == second == JudgeLabel.REUSABLE
    assert exp.reuse_label(a, b) == "H1"

    # A label-0 row in the same cluster -> H0 against a label-1 anchor.
    zero = next(r for r in rows if r.label == 0 and r.group == a.group)
    assert exp.reuse_label(zero, a) == "H0"
    req0 = JudgeRequest(query=Query(encode_prompt(zero.row_id, zero.text)),
                        candidate_query=Query(encode_prompt(a.row_id, a.text)),
                        candidate_key=CacheKey("k"))
    assert judge.judge(req0).decision.label == JudgeLabel.NOT_REUSABLE


# ── 2. stream construction is deterministic / identical across systems ─────

def test_build_stream_is_deterministic_and_shared(tiny_npz):
    rows, _ = load_rows(tiny_npz, query_field="text", label_field="label",
                        anchor_field="global_cluster", embedding_field="emb",
                        max_rows=1000, seed=1, include_uncertain=False,
                        selection="random", allow_pickle=True)
    s1, _ = exp.build_stream(rows, max_requests=120, seed=42, selection="uniform",
                              reuse_clusters_top_k=3, reuse_min_cluster_size=5,
                              mixed_reuse_fraction=0.3)
    s2, _ = exp.build_stream(rows, max_requests=120, seed=42, selection="uniform",
                              reuse_clusters_top_k=3, reuse_min_cluster_size=5,
                              mixed_reuse_fraction=0.3)
    assert [r.row_id for r in s1] == [r.row_id for r in s2]
    assert len(s1) == 120


@skip_no_sklearn
def test_cosine_and_ensemble_share_identical_request_order(tiny_npz, tmp_path):
    out = tmp_path / "run"
    args = exp.parse_args([
        "--npz", str(tiny_npz), "--output-dir", str(out),
        "--scorers", "cosine,lda", "--max-requests", "120", "--cache-size", "55",
        "--min-train-h0", "4", "--min-train-h1", "2", "--min-calib-h0", "2",
        "--min-calib-h1", "1", "--target-fpr", "0.25", "--batch-size", "20",
        "--progress-every", "1000", "--seed", "1",
    ])
    exp.run_experiment(args)

    with (out / "raw_decisions_minimal.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cos = [r["query_row_id"] for r in rows if r["method"] == "cosine"]
    ens = [r["query_row_id"] for r in rows if r["method"] == "ensemble"]
    assert cos and ens
    assert cos == ens, "both systems must replay the identical request order"


@skip_no_sklearn
def test_shared_cache_makes_both_systems_judge_identical_pairs(tiny_npz, tmp_path):
    """With --shared-cache, every request writes through even on a HIT, so both
    caches hold the identical rows -> cosine-NN retrieves the same candidates for
    both -> both judge the IDENTICAL set of pairs. The judged H1/H0 composition
    (and thus FP+TN, TP+FN totals) must therefore match exactly across scorers,
    even though their accept/reject decisions differ."""
    import json
    out = tmp_path / "run"
    args = exp.parse_args([
        "--npz", str(tiny_npz), "--output-dir", str(out), "--shared-cache",
        "--scorers", "cosine,lda", "--max-requests", "150", "--cache-size", "55",
        "--min-train-h0", "4", "--min-train-h1", "2", "--min-calib-h0", "2",
        "--min-calib-h1", "1", "--target-fpr", "0.25", "--batch-size", "20",
        "--progress-every", "1000", "--seed", "1",
    ])
    exp.run_experiment(args)
    d = json.loads((out / "summary_metrics.json").read_text(encoding="utf-8"))

    def judged(s):
        h1 = s["shadow_true_accepts"] + s["shadow_false_rejects"]   # TP + FN
        h0 = s["shadow_false_accepts"] + s["shadow_true_rejects"]   # FP + TN
        return h1, h0

    c_h1, c_h0 = judged(d["cosine"])
    e_h1, e_h0 = judged(d["ensemble"])
    assert (c_h1, c_h0) == (e_h1, e_h0), (
        f"shared cache must judge identical pairs: cosine (H1={c_h1},H0={c_h0}) "
        f"vs ensemble (H1={e_h1},H0={e_h0})"
    )
    assert c_h1 + c_h0 > 0


# ── 5. activation only after gates satisfied ───────────────────────────────

@skip_no_sklearn
def test_ensemble_does_not_activate_with_unreachable_gates(tiny_npz, tmp_path):
    rows, _ = load_rows(tiny_npz, query_field="text", label_field="label",
                        anchor_field="global_cluster", embedding_field="emb",
                        max_rows=55, seed=1, include_uncertain=False,
                        selection="random", allow_pickle=True)
    rows_by_id = {r.row_id: r for r in rows}
    stream, _ = exp.build_stream(rows, max_requests=150, seed=1, selection="uniform",
                                  reuse_clusters_top_k=3, reuse_min_cluster_size=5,
                                  mixed_reuse_fraction=0.3)
    import logging
    result = exp.run_one_system(
        method="ensemble", scorer="ensemble", scorers=["cosine", "lda"],
        stream=stream, rows_by_id=rows_by_id, target_fpr=0.25, top_k=5, batch_size=20,
        gate_min_train_h0=4, gate_min_train_h1=100_000,  # unreachable H1
        gate_min_calib_h0=2, gate_min_calib_h1=2,
        parallelism=2, persistence=False, state_dir=tmp_path / "s", logger=logging.getLogger("t"),
        progress_every=1000,
    )
    s = result["summary"]
    assert s["activation_request_index"] is None
    assert s["trained"] is False


@skip_no_sklearn
def test_ensemble_activates_only_after_gates_are_met(tiny_npz, tmp_path):
    rows, _ = load_rows(tiny_npz, query_field="text", label_field="label",
                        anchor_field="global_cluster", embedding_field="emb",
                        max_rows=55, seed=1, include_uncertain=False,
                        selection="random", allow_pickle=True)
    rows_by_id = {r.row_id: r for r in rows}
    stream, _ = exp.build_stream(rows, max_requests=200, seed=1, selection="uniform",
                                  reuse_clusters_top_k=3, reuse_min_cluster_size=5,
                                  mixed_reuse_fraction=0.3)
    import logging
    result = exp.run_one_system(
        method="ensemble", scorer="ensemble", scorers=["cosine", "lda"],
        stream=stream, rows_by_id=rows_by_id, target_fpr=0.25, top_k=5, batch_size=20,
        gate_min_train_h0=4, gate_min_train_h1=2, gate_min_calib_h0=2, gate_min_calib_h1=1,
        parallelism=2, persistence=False, state_dir=tmp_path / "s", logger=logging.getLogger("t"),
        progress_every=1000,
    )
    s = result["summary"]
    if s["activation_request_index"] is not None:
        # If it activated, the gates must have been satisfiable: by the end the
        # train/calibration buckets meet the configured (Wilson-adjusted) gates.
        gate = s["gate_diagnostics"]
        assert s["train_h0"] >= gate["config"]["min_train_h0"]
        assert s["train_h1"] >= gate["config"]["min_train_h1"]
        assert s["calibration_h0"] >= gate["config"]["min_calibration_h0"]
        assert s["trained"] is True
    else:
        # Otherwise it must genuinely lack data (never claims trained).
        assert s["trained"] is False


# ── 6. active-decision audit written and contains FP rows ──────────────────

@skip_no_sklearn
def test_active_decision_audit_written_with_false_positive_rows(tiny_npz, tmp_path):
    out = tmp_path / "run"
    args = exp.parse_args([
        "--npz", str(tiny_npz), "--output-dir", str(out),
        "--scorers", "cosine,lda", "--max-requests", "200", "--cache-size", "55",
        "--min-train-h0", "4", "--min-train-h1", "2", "--min-calib-h0", "2",
        "--min-calib-h1", "1", "--target-fpr", "0.30", "--batch-size", "20",
        "--progress-every", "1000", "--seed", "1",
    ])
    exp.run_experiment(args)

    for name in ("summary_metrics.json", "summary_metrics.csv", "raw_decisions_minimal.csv",
                 "activation_diagnostics.json", "calibration_diagnostics.json",
                 "active_decision_audit.csv"):
        assert (out / name).exists(), f"missing artifact: {name}"

    with (out / "active_decision_audit.csv").open(newline="", encoding="utf-8") as f:
        audit = list(csv.DictReader(f))
    assert audit, "audit must have rows when cache hits occur"
    # Same-cluster label-0 queries hitting label-1 anchors are guaranteed FPs.
    fps = [r for r in audit if r["is_false_positive"] == "True"]
    assert fps, "audit must record false-positive rows when FPs occur"
    # Every audited HIT must carry an oracle label and a decision.
    for r in audit:
        assert r["oracle_label"] in ("H0", "H1", "UNCERTAIN", "UNKNOWN")
        assert r["predicted_decision"] == "HIT"


# ── 8. warm-up freeze guard ─────────────────────────────────────────────────

def test_assert_not_frozen_before_measurement_raises_when_frozen(tmp_path):
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None, scorer="cosine", root_dir=tmp_path / "frozen", persistence=False,
    )
    system.freeze(reason="convergence_during_warmup")

    with pytest.raises(RuntimeError, match="frozen.*convergence_during_warmup"):
        exp._assert_not_frozen_before_measurement(system, "cosine")


def test_assert_not_frozen_before_measurement_passes_when_not_frozen(tmp_path):
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None, scorer="cosine", root_dir=tmp_path / "not_frozen", persistence=False,
    )
    exp._assert_not_frozen_before_measurement(system, "cosine")  # must not raise


# ── 9. anchor recovery guard against stale key_to_row entries ──────────────

def test_recover_audit_anchor_succeeds_for_fresh_entry(tmp_path):
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None, scorer="cosine", root_dir=tmp_path / "fresh", persistence=False,
    )
    response = system.handle("hello world")
    assert response.source == "llm"
    cache_key = response.cache_key
    key_to_row = {cache_key: 7}
    key_to_prompt = {cache_key: "hello world"}

    anchor_row_id, failed = exp._recover_audit_anchor(system, cache_key, key_to_row, key_to_prompt)
    assert anchor_row_id == 7
    assert failed is False


def test_recover_audit_anchor_fails_after_eviction(tmp_path):
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None, scorer="cosine", root_dir=tmp_path / "evicted", persistence=False,
    )
    response = system.handle("hello world")
    cache_key = response.cache_key
    key_to_row = {cache_key: 7}
    key_to_prompt = {cache_key: "hello world"}

    # Simulate the production eviction path (runtime.py invalidates a stale
    # query-level candidate via oracle.remove).
    system.cache.runtime.oracle.remove(cache_key)

    anchor_row_id, failed = exp._recover_audit_anchor(system, cache_key, key_to_row, key_to_prompt)
    assert anchor_row_id is None
    assert failed is True


def test_recover_audit_anchor_fails_on_prompt_mismatch(tmp_path):
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None, scorer="cosine", root_dir=tmp_path / "mismatch", persistence=False,
    )
    response = system.handle("hello world")
    cache_key = response.cache_key
    key_to_row = {cache_key: 7}
    # Recorded witness no longer matches what's actually stored under cache_key.
    key_to_prompt = {cache_key: "a different prompt"}

    anchor_row_id, failed = exp._recover_audit_anchor(system, cache_key, key_to_row, key_to_prompt)
    assert anchor_row_id is None
    assert failed is True


def test_recover_audit_anchor_fails_when_key_unknown(tmp_path):
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        stream=None, scorer="cosine", root_dir=tmp_path / "unknown", persistence=False,
    )
    anchor_row_id, failed = exp._recover_audit_anchor(system, "nonexistent-key", {}, {})
    assert anchor_row_id is None
    assert failed is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
