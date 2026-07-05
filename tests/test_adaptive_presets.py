"""The recommended default adaptive configuration."""

from __future__ import annotations

from mlcache.adaptive import (
    RECOMMENDED_ADAPTIVE_CONFIG,
    RECOMMENDED_ENSEMBLE_MEMBERS,
    build_recommended_controller,
    recommended_audit_config,
)


def test_recommended_members_are_the_winning_ensemble():
    assert RECOMMENDED_ENSEMBLE_MEMBERS == (
        "cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp",
    )


def test_recommended_controller_is_wilson_gated_and_seeded():
    h0 = tuple(i / 500 for i in range(500))
    c = build_recommended_controller(fallback_threshold=0.9, h0_scores=h0)
    assert c.wilson_gate is True
    assert c.alpha_target == RECOMMENDED_ADAPTIVE_CONFIG.alpha_target == 0.05
    assert c.buffer_size_current == 500          # seeded
    assert c.threshold_source() == "adaptive"    # buffer above min


def test_recommended_controller_holds_budget_on_overshooting_stream():
    # Fed control-audit H0 whose empirical FPR is just under target but with
    # too few samples to be sure, the gated recommended controller must not
    # net-loosen (this is the Wilson gate doing its job by default).
    c = build_recommended_controller(fallback_threshold=0.5,
                                     h0_scores=tuple(i / 500 for i in range(500)))
    start = c.alpha_t
    seq = [0] * 40
    seq[20] = 1  # phat = 1/40 = 0.025 < 0.05, UCB(1,40) >> 0.05
    for accepted in seq:
        c.update(score=0.5, label=0, accepted=bool(accepted))
    assert c.alpha_t <= start


def test_recommended_audit_config_uses_control_slice():
    cfg = recommended_audit_config()
    assert cfg.p_control == RECOMMENDED_ADAPTIVE_CONFIG.p_control == 0.04
