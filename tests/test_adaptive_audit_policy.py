"""Unit tests for the sparse margin-band audit policy."""

from __future__ import annotations

import random

import pytest

from mlcache.adaptive.audit_policy import (
    AuditZone,
    MarginBandAuditConfig,
    MarginBandAuditPolicy,
)


def _rates(policy, best_score, tau, n=20000):
    control = 0
    diagnostic = 0
    for _ in range(n):
        d = policy.decide(best_score, tau)
        control += int(d.control_audit)
        diagnostic += int(d.diagnostic_audit)
    return control / n, diagnostic / n


def test_zone_assignment():
    policy = MarginBandAuditPolicy(MarginBandAuditConfig(accept_margin=0.1, reject_margin=0.1))
    assert policy.zone(1.0, 0.5) == AuditZone.SAFE_ACCEPT
    assert policy.zone(0.55, 0.5) == AuditZone.BORDERLINE_ACCEPT
    assert policy.zone(0.45, 0.5) == AuditZone.BORDERLINE_REJECT
    assert policy.zone(0.2, 0.5) == AuditZone.SAFE_REJECT


def test_safe_accept_audited_with_low_probability():
    cfg = MarginBandAuditConfig(accept_margin=0.1, reject_margin=0.1, p_audit_safe_accept=0.01, p_control=0.0)
    policy = MarginBandAuditPolicy(cfg, rng=random.Random(1))
    _, diag = _rates(policy, best_score=1.0, tau=0.5)
    assert diag == pytest.approx(0.01, abs=0.005)


def test_borderline_accept_audited_with_high_probability():
    cfg = MarginBandAuditConfig(accept_margin=0.1, reject_margin=0.1, p_audit_borderline_accept=0.20, p_control=0.0)
    policy = MarginBandAuditPolicy(cfg, rng=random.Random(2))
    _, diag = _rates(policy, best_score=0.55, tau=0.5)
    assert diag == pytest.approx(0.20, abs=0.02)


def test_borderline_reject_audited_with_high_probability():
    cfg = MarginBandAuditConfig(accept_margin=0.1, reject_margin=0.1, p_audit_borderline_reject=0.20, p_control=0.0)
    policy = MarginBandAuditPolicy(cfg, rng=random.Random(3))
    _, diag = _rates(policy, best_score=0.45, tau=0.5)
    assert diag == pytest.approx(0.20, abs=0.02)


def test_safe_reject_audited_with_low_probability():
    cfg = MarginBandAuditConfig(accept_margin=0.1, reject_margin=0.1, p_audit_safe_reject=0.01, p_control=0.0)
    policy = MarginBandAuditPolicy(cfg, rng=random.Random(4))
    _, diag = _rates(policy, best_score=0.1, tau=0.5)
    assert diag == pytest.approx(0.01, abs=0.005)


def test_control_audit_is_margin_independent():
    # Control rate must be the SAME across every zone (this is what keeps ACI
    # unbiased). Diagnostic rate varies by zone; control does not.
    cfg = MarginBandAuditConfig(
        accept_margin=0.1,
        reject_margin=0.1,
        p_control=0.05,
        p_audit_safe_accept=0.01,
        p_audit_borderline_accept=0.20,
        p_audit_borderline_reject=0.20,
        p_audit_safe_reject=0.01,
    )
    policy = MarginBandAuditPolicy(cfg, rng=random.Random(5))
    control_rates = []
    for best, tau in ((1.0, 0.5), (0.55, 0.5), (0.45, 0.5), (0.1, 0.5)):
        ctrl, _ = _rates(policy, best, tau)
        control_rates.append(ctrl)
    for rate in control_rates:
        assert rate == pytest.approx(0.05, abs=0.006)


def test_audit_probability_logged():
    cfg = MarginBandAuditConfig(accept_margin=0.1, reject_margin=0.1, p_control=0.04, p_audit_borderline_accept=0.2)
    policy = MarginBandAuditPolicy(cfg, rng=random.Random(6))
    d = policy.decide(0.55, 0.5)
    assert d.p_control == pytest.approx(0.04)
    assert d.p_diagnostic == pytest.approx(0.20)
    # combined probability of being judged at all
    assert d.audit_probability == pytest.approx(1 - (1 - 0.04) * (1 - 0.20))
    assert d.audit_channel in {"none", "control", "diagnostic", "both"}


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        MarginBandAuditConfig(p_control=1.5)
    with pytest.raises(ValueError):
        MarginBandAuditConfig(accept_margin=-0.1)
