"""Unit tests for admission metrics (ground-truth + audited estimates)."""

from __future__ import annotations

import pytest

from mlcache.adaptive.metrics import AdmissionEvent, AdmissionMetrics


def make_event(**overrides):
    base = dict(
        stream_index=0,
        query_id="q",
        best_score=0.5,
        tau=0.4,
        alpha_t=0.05,
        accepted=True,
        label=0,
        judged=False,
        control_audit=False,
        diagnostic_audit=False,
        p_control=0.04,
        p_diagnostic=0.0,
        zone="safe_accept",
        threshold_source="adaptive",
        region_id="r",
    )
    base.update(overrides)
    return AdmissionEvent(**base)


def test_fpr_only_over_h0_events():
    m = AdmissionMetrics(alpha_target=0.05)
    # 10 H0, 3 accepted -> fpr 0.3; H1 accepts must NOT affect fpr.
    for i in range(10):
        m.record(make_event(stream_index=i, label=0, accepted=(i < 3)))
    for i in range(5):
        m.record(make_event(stream_index=100 + i, label=1, accepted=True))
    s = m.summary()
    assert s["gt_n_h0"] == 10
    assert s["achieved_fpr"] == pytest.approx(0.3)
    assert s["false_hits"] == 3


def test_tpr_only_over_h1_events():
    m = AdmissionMetrics(alpha_target=0.05)
    for i in range(8):
        m.record(make_event(stream_index=i, label=1, accepted=(i < 6)))
    for i in range(4):
        m.record(make_event(stream_index=50 + i, label=0, accepted=False))
    s = m.summary()
    assert s["gt_n_h1"] == 8
    assert s["tpr"] == pytest.approx(0.75)


def test_unknown_labels_excluded_from_fpr_tpr():
    m = AdmissionMetrics(alpha_target=0.05)
    for i in range(5):
        m.record(make_event(stream_index=i, label=None, accepted=True))
    s = m.summary()
    assert s["gt_n_h0"] == 0
    assert s["gt_n_h1"] == 0
    assert s["achieved_fpr"] is None
    assert s["hit_rate"] == pytest.approx(1.0)  # still served


def test_no_judge_hit_rate_and_judged_fraction():
    m = AdmissionMetrics(alpha_target=0.05)
    # 4 accepted, of which 1 judged; 1 rejected judged. total 5.
    m.record(make_event(accepted=True, judged=False))
    m.record(make_event(accepted=True, judged=False))
    m.record(make_event(accepted=True, judged=True, control_audit=True))
    m.record(make_event(accepted=True, judged=False))
    m.record(make_event(accepted=False, judged=True, control_audit=True))
    s = m.summary()
    assert s["hit_rate"] == pytest.approx(4 / 5)
    assert s["no_judge_hit_rate"] == pytest.approx(3 / 5)  # 3 accepted & unjudged
    assert s["judged_fraction"] == pytest.approx(2 / 5)
    assert s["provider_call_rate"] == pytest.approx(1 / 5)


def test_control_audit_fpr_separate_from_raw():
    m = AdmissionMetrics(alpha_target=0.05)
    # control-audited H0: 2 total, 1 accepted -> control fpr 0.5
    m.record(make_event(label=0, accepted=True, judged=True, control_audit=True))
    m.record(make_event(label=0, accepted=False, judged=True, control_audit=True))
    # diagnostic-only H0 accepted -> raises raw fpr but not control fpr
    m.record(make_event(label=0, accepted=True, judged=True, diagnostic_audit=True, p_diagnostic=0.2))
    s = m.summary()
    assert s["control_audit_fpr"] == pytest.approx(0.5)
    assert s["control_audit_n_h0"] == 2
    # raw over all judged H0: 3 total, 2 accepted
    assert s["raw_audit_fpr"] == pytest.approx(2 / 3)


def test_importance_weighted_fpr_and_kish():
    m = AdmissionMetrics(alpha_target=0.05)
    # Two diagnostic H0 audits with different audit probabilities.
    #   safe-accept false hit at p=0.01 -> weight 100 (accepted)
    #   borderline reject at p=0.20 -> weight 5 (not accepted)
    m.record(make_event(label=0, accepted=True, judged=True, diagnostic_audit=True, p_diagnostic=0.01))
    m.record(make_event(label=0, accepted=False, judged=True, diagnostic_audit=True, p_diagnostic=0.20))
    s = m.summary()
    # HT FPR = (100*1) / (100 + 5) = 100/105
    assert s["iw_audit_fpr"] == pytest.approx(100 / 105)
    # Kish ESS = (sum w)^2 / sum w^2 = 105^2 / (100^2 + 5^2)
    assert s["iw_audit_kish_ess_h0"] == pytest.approx(105 ** 2 / (100 ** 2 + 5 ** 2))


def test_np_score_zero_when_budget_blown():
    m = AdmissionMetrics(alpha_target=0.05)
    for i in range(10):  # fpr 0.5 >> 0.05
        m.record(make_event(stream_index=i, label=0, accepted=(i < 5)))
    for i in range(10):
        m.record(make_event(stream_index=50 + i, label=1, accepted=True))
    s = m.summary()
    assert s["fpr_ok"] is False
    assert s["np_score"] == 0.0


def test_cost_adjusted_utility():
    m = AdmissionMetrics(alpha_target=0.05, judge_cost_ratio=0.1)
    # 3 true hits (accepted H1), 2 judged events overall.
    m.record(make_event(label=1, accepted=True, judged=False))
    m.record(make_event(label=1, accepted=True, judged=True, control_audit=True))
    m.record(make_event(label=1, accepted=True, judged=True, diagnostic_audit=True, p_diagnostic=0.2))
    m.record(make_event(label=0, accepted=False, judged=False))
    s = m.summary()
    assert s["true_hits"] == 3
    # cost_adjusted_utility = true_hits - ratio*judged = 3 - 0.1*2 = 2.8
    assert s["cost_adjusted_utility"] == pytest.approx(2.8)
    assert s["cost_adjusted_utility_per_request"] == pytest.approx(2.8 / 4)


def test_np_score_equals_tpr_when_budget_met():
    m = AdmissionMetrics(alpha_target=0.5)
    for i in range(10):  # fpr 0.3 <= 0.5
        m.record(make_event(stream_index=i, label=0, accepted=(i < 3)))
    for i in range(10):
        m.record(make_event(stream_index=50 + i, label=1, accepted=(i < 8)))
    s = m.summary()
    assert s["fpr_ok"] is True
    assert s["np_score"] == pytest.approx(0.8)


def test_snapshots_include_cumulative_and_window_rates():
    m = AdmissionMetrics(alpha_target=0.05, window=3, snapshot_every=2)

    m.record(make_event(stream_index=0, label=0, accepted=True))
    m.record(make_event(stream_index=1, label=1, accepted=True))
    m.record(make_event(stream_index=2, label=0, accepted=False))
    m.record(make_event(stream_index=3, label=1, accepted=False))

    assert len(m.snapshots) == 2

    first = m.snapshots[0]
    assert first["total_seen"] == 2
    assert first["cumulative_fpr"] == pytest.approx(1.0)
    assert first["cumulative_tpr"] == pytest.approx(1.0)
    assert first["window_fpr"] == pytest.approx(1.0)
    assert first["window_tpr"] == pytest.approx(1.0)

    second = m.snapshots[1]
    assert second["total_seen"] == 4
    assert second["cumulative_fpr"] == pytest.approx(0.5)
    assert second["cumulative_tpr"] == pytest.approx(0.5)
    # Last 3 events are: accepted H1, rejected H0, rejected H1.
    assert second["window_fpr"] == pytest.approx(0.0)
    assert second["window_tpr"] == pytest.approx(0.5)
    assert second["window_n_h0"] == 1
    assert second["window_n_h1"] == 2
