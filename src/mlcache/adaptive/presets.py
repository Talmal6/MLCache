"""Recommended default adaptive-admission configuration.

Encodes the best configuration found in the replay study: the WeightedEnsemble
scorer with a **Wilson-gated** Global ACI controller and sparse two-channel
auditing. The ensemble gives the highest TPR at a fixed FPR budget and is
robust to workload shift; the Wilson gate keeps the online threshold from
over-loosening past the hard budget on sparse control-audit updates.

This is the recommended *system default* for correctness-controlled online
serving. Use :func:`build_recommended_controller` for the controller,
:data:`RECOMMENDED_ENSEMBLE_MEMBERS` for the scorer, and
:func:`recommended_audit_config` for the audit policy.

Note: the Wilson gate defaults to ON here (budget-safe) even though the
low-level :class:`GlobalACIController` keeps it OFF by default -- the primitive
stays neutral; this preset makes the safe choice.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlcache.adaptive.aci_controller import GlobalACIController
from mlcache.adaptive.audit_policy import MarginBandAuditConfig
from mlcache.semantic_types import TieMode

# WeightedEnsemble members (the winning scorer): the search leans on the MLP's
# nonlinear boundary with a large cosine anchor, and stays in the FPR budget
# across random / natural / cluster-block orderings.
RECOMMENDED_ENSEMBLE_MEMBERS: tuple[str, ...] = (
    "cosine",
    "lda",
    "pca_whitened_cosine",
    "xgboost",
    "mlp",
)


@dataclass(frozen=True, slots=True)
class RecommendedAdaptiveConfig:
    """Tuned defaults for the adaptive admission layer."""

    alpha_target: float = 0.05
    gamma: float = 0.01
    alpha_min: float = 0.001
    alpha_max: float = 0.20
    buffer_size: int = 3000
    min_buffer_size: int = 300
    wilson_gate: bool = True
    wilson_z: float = 1.96
    wilson_window: int = 1000
    p_control: float = 0.04
    tie_mode: TieMode = TieMode.GE


RECOMMENDED_ADAPTIVE_CONFIG = RecommendedAdaptiveConfig()


def recommended_audit_config(
    config: RecommendedAdaptiveConfig = RECOMMENDED_ADAPTIVE_CONFIG,
) -> MarginBandAuditConfig:
    """Audit policy config for the recommended setup (uniform control slice)."""

    return MarginBandAuditConfig(p_control=config.p_control)


def build_recommended_controller(
    *,
    fallback_threshold: float | None = None,
    h0_scores: tuple[float, ...] = (),
    config: RecommendedAdaptiveConfig = RECOMMENDED_ADAPTIVE_CONFIG,
) -> GlobalACIController:
    """Build the recommended Wilson-gated Global ACI controller.

    Seed it with the batch-calibration selected-H0 scores and threshold so
    cold-start uses the real NP threshold.
    """

    controller = GlobalACIController(
        alpha_target=config.alpha_target,
        gamma=config.gamma,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
        buffer_size=config.buffer_size,
        min_buffer_size=config.min_buffer_size,
        fallback_threshold=fallback_threshold,
        tie_mode=config.tie_mode,
        wilson_gate=config.wilson_gate,
        wilson_z=config.wilson_z,
        wilson_window=config.wilson_window,
    )
    if h0_scores:
        controller.seed(h0_scores, fallback_threshold=fallback_threshold)
    return controller
