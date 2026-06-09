"""Tests for the cosine-vs-ensemble comparison harness.

Unit tests (topic extraction, judge, embedding, winner logic) run in
milliseconds.  The one integration test runs a 40-request comparison with a
5-second fit-wait ceiling; it verifies output-file structure but does not
require calibration to have occurred.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for _p in (str(SRC), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _sklearn_ok() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


try:
    from compare_cosine_vs_ensemble import (  # type: ignore[import]
        DEBUG_ENSEMBLE,
        FULL_ENSEMBLE,
        TopicEmbeddingProvider,
        TopicJudge,
        build_delta,
        build_validity,
        determine_winner,
        run_comparison,
        topic_of,
        topic_prompts,
    )
    _OK = True
except ImportError:
    _OK = False

skip_no_harness = pytest.mark.skipif(not _OK, reason="compare harness not importable")
skip_no_sklearn = pytest.mark.skipif(not _sklearn_ok(), reason="sklearn not installed")


# ── topic helpers ──────────────────────────────────────────────────────────

@skip_no_harness
def test_topic_of_extracts_number():
    assert topic_of("What is the capital of country #3?") == "3"
    assert topic_of("no number here") is None


@skip_no_harness
def test_topic_prompts_cycles():
    prompts = topic_prompts(8, topics=4)
    topics = [topic_of(p) for p in prompts]
    assert topics == ["0", "1", "2", "3", "0", "1", "2", "3"]


# ── deterministic judge ────────────────────────────────────────────────────

@skip_no_harness
def test_judge_same_topic_reusable():
    from mlcache import JudgeLabel, JudgeRequest
    from mlcache.semantic_types import CacheKey, Query
    judge = TopicJudge()
    req = JudgeRequest(
        query=Query("What is the capital of country #2?"),
        candidate_query=Query("Tell me the capital city of nation #2."),
        candidate_response=None, candidate_key=CacheKey("k"),
    )
    assert judge.judge(req).decision.label == JudgeLabel.REUSABLE


@skip_no_harness
def test_judge_cross_topic_not_reusable():
    from mlcache import JudgeLabel, JudgeRequest
    from mlcache.semantic_types import CacheKey, Query
    judge = TopicJudge()
    req = JudgeRequest(
        query=Query("What is the capital of country #1?"),
        candidate_query=Query("What is the capital of country #5?"),
        candidate_response=None, candidate_key=CacheKey("k"),
    )
    assert judge.judge(req).decision.label == JudgeLabel.NOT_REUSABLE


# ── embedding provider ─────────────────────────────────────────────────────

@skip_no_harness
def test_embed_is_unit_norm():
    import numpy as np
    p = TopicEmbeddingProvider(dim=16, jitter=0.05)
    v = np.array(p.embed("What is the capital of country #3?"))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


@skip_no_harness
def test_embed_is_deterministic():
    p = TopicEmbeddingProvider(dim=16, jitter=0.05)
    t = "What is the capital of country #0?"
    assert p.embed(t) == p.embed(t)


@skip_no_harness
def test_same_topic_higher_sim_than_cross():
    import numpy as np
    p = TopicEmbeddingProvider(dim=32, jitter=0.05)
    a = np.array(p.embed("What is the capital of country #1?"))
    b = np.array(p.embed("Tell me the capital city of nation #1."))
    c = np.array(p.embed("What is the capital of country #7?"))
    assert float(np.dot(a, b)) > float(np.dot(a, c))


# ── winner / delta logic ───────────────────────────────────────────────────

@skip_no_harness
def test_winner_ensemble_higher_hit():
    w = determine_winner({"calibrated": True, "hit_rate": 0.5},
                         {"calibrated": True, "hit_rate": 0.8})
    assert w["by_hit_rate"] == "ensemble"


@skip_no_harness
def test_winner_cosine_higher_hit():
    w = determine_winner({"calibrated": True, "hit_rate": 0.9},
                         {"calibrated": True, "hit_rate": 0.6})
    assert w["by_hit_rate"] == "cosine"


@skip_no_harness
def test_winner_only_cosine_calibrated():
    w = determine_winner({"calibrated": True, "hit_rate": 0.3},
                         {"calibrated": False, "hit_rate": 0.0})
    assert w["by_hit_rate"] == "cosine"


@skip_no_harness
def test_winner_neither_calibrated():
    w = determine_winner({"calibrated": False, "hit_rate": 0.0},
                         {"calibrated": False, "hit_rate": 0.0})
    assert w["by_hit_rate"] == "undetermined"


@skip_no_harness
def test_winner_tie():
    w = determine_winner({"calibrated": True, "hit_rate": 0.601},
                         {"calibrated": True, "hit_rate": 0.602})
    assert w["by_hit_rate"] == "tie"


@skip_no_harness
def test_delta_computes_difference():
    d = build_delta({"hit_rate": 0.4, "llm_calls": 80, "threshold": 0.7},
                    {"hit_rate": 0.6, "llm_calls": 60, "threshold": 0.5})
    assert abs(d["hit_rate"] - 0.2) < 1e-5
    assert d["llm_calls"] == -20


# ── ensemble constants ─────────────────────────────────────────────────────

@skip_no_harness
def test_default_ensemble_is_full_five_member():
    assert FULL_ENSEMBLE == ["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"]


@skip_no_harness
def test_debug_ensemble_constant():
    assert DEBUG_ENSEMBLE == ["cosine", "lda"]


# ── comparison_valid / build_validity ──────────────────────────────────────

@skip_no_harness
def test_comparison_valid_true_when_all_pass():
    cosine  = {"calibrated": True, "finite_threshold": True}
    ensemble = {"calibrated": True, "finite_threshold": True, "trained": True}
    valid, reason = build_validity(cosine, ensemble)
    assert valid is True
    assert "passed" in reason


@skip_no_harness
def test_comparison_valid_false_when_ensemble_uncalibrated():
    cosine  = {"calibrated": True, "finite_threshold": True}
    ensemble = {"calibrated": False, "finite_threshold": True, "trained": True}
    valid, reason = build_validity(cosine, ensemble)
    assert valid is False
    assert "ensemble.calibrated" in reason


@skip_no_harness
def test_comparison_valid_false_when_ensemble_not_trained():
    cosine  = {"calibrated": True, "finite_threshold": True}
    ensemble = {"calibrated": True, "finite_threshold": True, "trained": False}
    valid, reason = build_validity(cosine, ensemble)
    assert valid is False
    assert "ensemble.trained" in reason


# ── integration: one tiny end-to-end run ──────────────────────────────────

@skip_no_harness
@skip_no_sklearn
@pytest.mark.integration
def test_tiny_comparison_writes_artifacts(tmp_path):
    summary = run_comparison(
        n_requests=40, topics=3, top_k=3, batch_size=10,
        min_h0=6, min_h1=3, target_fpr=0.25,
        ensemble_scorers=["cosine", "lda"],
        fit_wait_secs=5.0,          # short ceiling; calibration not required
        output_dir=str(tmp_path),
        dim=16, jitter=0.05,
    )

    # all four required artifacts exist
    for name in ("comparison_summary.json", "comparison_report.md",
                 "cosine_trace.csv", "ensemble_trace.csv"):
        assert (tmp_path / name).exists(), f"missing: {name}"

    # JSON has required keys
    data = json.loads((tmp_path / "comparison_summary.json").read_text())
    for key in ("config", "cosine", "ensemble", "delta", "winner",
                "comparison_valid", "comparison_valid_reason"):
        assert key in data

    # comparison_valid is a bool
    assert isinstance(data["comparison_valid"], bool)

    # cosine section has required fields
    cosine = data["cosine"]
    for f in ("n_requests", "cache_hits", "llm_calls", "hit_rate",
              "trained", "calibrated", "threshold", "finite_threshold"):
        assert f in cosine, f"cosine missing field: {f}"

    # winner value is valid
    assert data["winner"]["by_hit_rate"] in {"cosine", "ensemble", "tie", "undetermined"}

    # traces have rows
    rows = list(csv.DictReader((tmp_path / "cosine_trace.csv").open(
        newline="", encoding="utf-8")))
    assert len(rows) >= 40

    # same topic sequence for both systems
    def topics_from(name: str) -> list[str]:
        with (tmp_path / name).open(newline="", encoding="utf-8") as f:
            return [r["topic"] for r in csv.DictReader(f)][:40]
    assert topics_from("cosine_trace.csv") == topics_from("ensemble_trace.csv")

    # if ensemble calibrated it must also be trained
    if summary["ensemble"]["calibrated"]:
        assert summary["ensemble"]["trained"] is True
