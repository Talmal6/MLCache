"""Tiny tests for the dataset-backed online comparison harness instrumentation.

These cover only the logging/metrics plumbing (no dataset, no model fits):
- setup_logging creates run.log
- progress CSV fields include FPR/TPR/precision
- lifecycle JSONL rows are valid JSON
- the comparison summary exposes empirical_fpr / tpr / precision
- log text is ASCII-only
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for _p in (str(SRC), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mlcache.feedback import JudgeLabel

try:
    from compare_cosine_vs_ensemble import (  # type: ignore[import]
        PROGRESS_CSV_FIELDS,
        FULL_ENSEMBLE,
        MetricsRecorder,
        append_jsonl,
        build_comparison_summary,
        determine_winner,
        safe_rate,
        setup_logging,
    )
    _OK = True
except ImportError:
    _OK = False

skip_no_harness = pytest.mark.skipif(not _OK, reason="compare harness not importable")


@skip_no_harness
def test_setup_logging_creates_run_log(tmp_path):
    logger = setup_logging(tmp_path)
    logger.info("hello from test")
    for handler in logger.handlers:
        handler.flush()
    log_path = tmp_path / "run.log"
    assert log_path.exists()
    assert "hello from test" in log_path.read_text(encoding="utf-8")


@skip_no_harness
def test_progress_fields_include_fpr_tpr_precision():
    for field in ("empirical_fpr", "tpr", "precision", "true_accepts", "false_accepts",
                  "true_rejects", "false_rejects"):
        assert field in PROGRESS_CSV_FIELDS


@skip_no_harness
def test_lifecycle_jsonl_row_is_valid_json(tmp_path):
    path = tmp_path / "lifecycle_events.jsonl"
    append_jsonl(path, {
        "timestamp": "2026-01-01T00:00:00Z",
        "method": "ensemble",
        "request_index": 400,
        "event": "calibrated_changed",
        "old_value": False,
        "new_value": True,
        "details": {},
    })
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == "calibrated_changed"
    assert parsed["new_value"] is True


@skip_no_harness
def test_comparison_summary_contains_fpr_tpr_precision():
    cosine = {
        "name": "cosine", "hit_rate": 0.3, "hit_count": 30, "miss_count": 70,
        "judged_pair_count": 100, "calibrated": True, "trained": True,
        "empirical_fpr": 0.05, "tpr": 0.6, "precision": 0.9, "target_fpr_respected": True,
        "first_calibrated_at_request": 50, "first_hit_at_request": 60,
        "threshold": 0.8,
    }
    ensemble = dict(cosine, name="ensemble", hit_rate=0.4, empirical_fpr=0.04, tpr=0.7, precision=0.95)
    summary = build_comparison_summary(config={}, dataset={}, cosine=cosine, ensemble=ensemble)
    for key in ("empirical_fpr", "tpr", "precision"):
        assert key in summary
        assert "cosine" in summary[key] and "ensemble" in summary[key]
    # Both respect target_fpr; higher TPR (ensemble) should win.
    assert summary["winner"]["winner"] == "ensemble"


@skip_no_harness
def test_metrics_recorder_metrics_has_required_keys():
    rec = MetricsRecorder(method="ensemble", target_fpr=0.10, logger=logging.getLogger("t"))
    m = rec.metrics()
    for key in ("empirical_fpr", "tpr", "precision", "target_fpr_respected"):
        assert key in m
    # With no data, rates are None (never faked).
    assert m["empirical_fpr"] is None
    assert m["tpr"] is None


@skip_no_harness
def test_metrics_since_excludes_cold_start_true_negatives():
    rec = MetricsRecorder(method="cosine", target_fpr=0.10, logger=logging.getLogger("t"))

    rec.set_request_index(1)
    rec.record_judge(candidate_key="cold", judge_label=JudgeLabel.NOT_REUSABLE)
    rec.resolve_request(1, source="llm", accepted_key=None)
    baseline = rec.count_snapshot()

    rec.set_request_index(2)
    rec.record_judge(candidate_key="fp", judge_label=JudgeLabel.NOT_REUSABLE)
    rec.resolve_request(2, source="cache", accepted_key="fp")

    rec.set_request_index(3)
    rec.record_judge(candidate_key="tn", judge_label=JudgeLabel.NOT_REUSABLE)
    rec.resolve_request(3, source="llm", accepted_key=None)

    active = rec.metrics_since(baseline)
    assert active["false_accepts"] == 1
    assert active["true_rejects"] == 1
    assert active["empirical_fpr"] == pytest.approx(0.5)
    assert rec.metrics()["empirical_fpr"] == pytest.approx(1 / 3)


@skip_no_harness
def test_safe_rate_handles_zero_denominator():
    assert safe_rate(1, 0) is None
    assert safe_rate(None, 5) is None
    assert safe_rate(1, 4) == 0.25


@skip_no_harness
def test_log_text_is_ascii_only(tmp_path):
    logger = setup_logging(tmp_path)
    logger.info("ascii only metrics fpr=0.031 tpr=0.284 precision=0.917")
    for handler in logger.handlers:
        handler.flush()
    raw = (tmp_path / "run.log").read_text(encoding="utf-8")
    # Must be representable as ASCII (no emojis / unicode arrows).
    raw.encode("ascii")
