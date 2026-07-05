"""Unit tests for the Global ACI admission controller."""

from __future__ import annotations

import pytest

from mlcache.adaptive.aci_controller import GlobalACIController, selected_h0_quantile_threshold
from mlcache.semantic_types import TieMode


def make_controller(**overrides):
    kwargs = dict(
        alpha_target=0.05,
        gamma=0.01,
        alpha_min=0.001,
        alpha_max=0.20,
        buffer_size=10000,
        min_buffer_size=100,
        fallback_threshold=0.5,
    )
    kwargs.update(overrides)
    return GlobalACIController(**kwargs)


def _seed_uniform(controller, n=1000):
    # Seed a broad H0 score distribution so the quantile is well-defined.
    scores = [i / n for i in range(n)]
    controller.seed(scores, fallback_threshold=0.5)


def test_alpha_decreases_after_accepted_h0():
    c = make_controller()
    _seed_uniform(c)
    before = c.alpha_t
    c.update(score=0.99, label=0, accepted=True)  # false hit -> err=1
    assert c.alpha_t < before
    assert c.alpha_t == pytest.approx(before + c.gamma * (c.alpha_target - 1.0))


def test_alpha_increases_after_rejected_h0():
    c = make_controller()
    _seed_uniform(c)
    before = c.alpha_t
    c.update(score=0.1, label=0, accepted=False)  # correct reject -> err=0
    assert c.alpha_t > before
    assert c.alpha_t == pytest.approx(before + c.gamma * (c.alpha_target - 0.0))


def test_alpha_unchanged_after_h1():
    c = make_controller()
    _seed_uniform(c)
    before = c.alpha_t
    buf_before = c.buffer_size_current
    c.update(score=0.9, label=1, accepted=True)
    assert c.alpha_t == before
    assert c.buffer_size_current == buf_before  # H1 never enters the H0 buffer


def test_alpha_unchanged_after_unknown_label():
    c = make_controller()
    _seed_uniform(c)
    before = c.alpha_t
    buf_before = c.buffer_size_current
    c.update(score=0.9, label=None, accepted=True)
    assert c.alpha_t == before
    assert c.buffer_size_current == buf_before


def test_alpha_clipped_to_bounds():
    c = make_controller(alpha_target=0.05, gamma=0.5, alpha_min=0.02, alpha_max=0.08)
    _seed_uniform(c)
    for _ in range(50):
        c.update(score=0.99, label=0, accepted=True)  # push down hard
    assert c.alpha_t == pytest.approx(0.02)
    for _ in range(50):
        c.update(score=0.01, label=0, accepted=False)  # push up hard
    assert c.alpha_t == pytest.approx(0.08)


def test_threshold_stricter_when_alpha_smaller():
    # Directly compare the tie-exact quantile at two alpha levels.
    scores = [i / 1000 for i in range(1000)]
    strict = float(selected_h0_quantile_threshold(scores, 0.01))
    loose = float(selected_h0_quantile_threshold(scores, 0.10))
    assert strict > loose  # smaller alpha -> higher quantile -> stricter gate


def test_controller_threshold_moves_with_alpha():
    c = make_controller(gamma=0.02)
    _seed_uniform(c, n=2000)
    tau_start = c.threshold()
    # Drive many accepted H0 (false hits) -> alpha_t down -> threshold up.
    for _ in range(200):
        c.update(score=0.999, label=0, accepted=True)
    assert c.alpha_t < c.alpha_target
    assert c.threshold() >= tau_start


def test_h0_buffer_receives_only_h0_scores():
    c = make_controller(min_buffer_size=1)
    start = c.buffer_size_current
    c.update(score=0.3, label=0, accepted=False)
    c.update(score=0.9, label=1, accepted=True)
    c.update(score=0.9, label=None, accepted=True)
    assert c.buffer_size_current == start + 1


def test_threshold_uses_fallback_when_buffer_small():
    c = make_controller(min_buffer_size=100, fallback_threshold=0.42)
    c.seed([0.1, 0.2, 0.3])  # below min_buffer_size
    assert not c.buffer_ready
    assert c.threshold() == pytest.approx(0.42)
    assert c.threshold_source() == "fallback"


def test_threshold_rejects_by_default_without_fallback():
    c = make_controller(min_buffer_size=100, fallback_threshold=None)
    assert c.threshold() == float("inf")
    assert not c.should_accept(1.0)


def test_threshold_uses_quantile_when_buffer_large():
    c = make_controller(min_buffer_size=100)
    _seed_uniform(c, n=1000)
    assert c.buffer_ready
    assert c.threshold_source() == "adaptive"
    expected = float(selected_h0_quantile_threshold([i / 1000 for i in range(1000)], c.alpha_t))
    assert c.threshold() == pytest.approx(expected)


def test_fifo_buffer_evicts_oldest():
    c = make_controller(buffer_size=5, min_buffer_size=1)
    for i in range(10):
        c.update(score=float(i), label=0, accepted=False)
    assert c.buffer_size_current == 5  # capped, recency window


def test_state_dict_roundtrip():
    c = make_controller(gamma=0.02)
    _seed_uniform(c, n=500)
    for i in range(30):
        c.update(score=0.5 + 0.01 * i, label=0, accepted=(i % 2 == 0))
    state = c.state_dict()
    restored = make_controller()
    restored.load_state_dict(state)
    assert restored.alpha_t == pytest.approx(c.alpha_t)
    assert restored.threshold() == pytest.approx(c.threshold())
    assert restored.buffer_size_current == c.buffer_size_current


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        make_controller(alpha_target=1.5)
    with pytest.raises(ValueError):
        make_controller(alpha_min=0.1, alpha_max=0.05)
    with pytest.raises(ValueError):
        make_controller(alpha_target=0.5, alpha_min=0.001, alpha_max=0.20)  # target outside... 0.5>0.2
    with pytest.raises(ValueError):
        make_controller(gamma=0.0)


def test_update_rejects_bad_label():
    c = make_controller()
    with pytest.raises(ValueError):
        c.update(score=0.5, label=2, accepted=True)


# ---------------------------------------------------------------------------
# Wilson gate (item 6)
# ---------------------------------------------------------------------------


def make_gated(**overrides):
    kwargs = dict(wilson_gate=True, wilson_z=1.96, wilson_window=1000)
    kwargs.update(overrides)
    return make_controller(**kwargs)


def test_wilson_gate_blocks_loosening_when_ucb_above_target():
    # A single accepted H0 in a small window -> phat=0.5, UCB well above 0.05.
    # A subsequent rejected H0 (err=0) *wants* to loosen; the gate must block it.
    c = make_gated(gamma=0.01)
    _seed_uniform(c)
    c.update(score=0.9, label=0, accepted=True)  # tighten + record a false hit
    alpha_after_tighten = c.alpha_t
    assert c.wilson_ucb() > c.alpha_target
    c.update(score=0.1, label=0, accepted=False)  # would loosen, but UCB high
    assert c.alpha_t == pytest.approx(alpha_after_tighten)  # blocked
    assert c.blocked_loosen_count == 1


def test_wilson_gate_still_allows_tightening_when_ucb_above_target():
    c = make_gated(gamma=0.01)
    _seed_uniform(c)
    c.update(score=0.9, label=0, accepted=True)
    before = c.alpha_t
    assert c.wilson_ucb() > c.alpha_target
    c.update(score=0.95, label=0, accepted=True)  # err=1 -> tighten, always allowed
    assert c.alpha_t < before


def test_wilson_gate_opens_once_ucb_drops_below_target():
    # Many clean (rejected) H0 -> empirical FPR 0, UCB shrinks under target ->
    # loosening becomes allowed again.
    # alpha_max high so alpha_t does not saturate before the final probe step.
    c = make_gated(gamma=0.001, alpha_max=0.9, wilson_window=5000)
    _seed_uniform(c)
    for _ in range(2000):
        c.update(score=0.1, label=0, accepted=False)
    assert c.wilson_ucb() <= c.alpha_target
    before = c.alpha_t
    assert before < c.alpha_max  # not clipped
    c.update(score=0.1, label=0, accepted=False)  # loosening now permitted
    assert c.alpha_t > before


def test_wilson_gate_keeps_overshooting_stream_within_budget():
    # A conservatively-calibrated controller fed control-audit H0 whose recent
    # empirical FPR sits *below* target but with too few samples to be sure.
    # Ungated ACI loosens (alpha_t rises); gated ACI must not net-loosen.
    # 40 control-audit H0, one false hit -> phat = 1/40 = 0.025 < 0.05, but
    # UCB(1, 40) >> 0.05, so the gate must withhold loosening.
    def run(gate: bool):
        c = make_controller(gamma=0.01, wilson_gate=gate, wilson_window=1000)
        _seed_uniform(c)
        start = c.alpha_t
        seq = [0] * 40
        seq[20] = 1
        for accepted in seq:
            c.update(score=0.5, label=0, accepted=bool(accepted))
        return start, c.alpha_t

    start, ungated = run(False)
    _, gated = run(True)
    assert ungated > start          # ungated loosened
    assert gated <= start           # gate prevented net loosening
    assert gated < ungated


def test_ungated_controller_ignores_wilson_state():
    c = make_controller(wilson_gate=False, gamma=0.01)
    _seed_uniform(c)
    c.update(score=0.9, label=0, accepted=True)
    before = c.alpha_t
    c.update(score=0.1, label=0, accepted=False)  # loosens freely, no gate
    assert c.alpha_t > before
    assert c.blocked_loosen_count == 0


def test_wilson_state_roundtrips():
    c = make_gated(gamma=0.02)
    _seed_uniform(c)
    for i in range(50):
        c.update(score=0.5, label=0, accepted=(i % 7 == 0))
    state = c.state_dict()
    restored = make_gated()
    restored.load_state_dict(state)
    assert restored.wilson_gate is True
    assert restored.wilson_ucb() == pytest.approx(c.wilson_ucb())
    assert restored.blocked_loosen_count == c.blocked_loosen_count
