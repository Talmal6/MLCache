"""Adaptive Neyman-Pearson cache admission (Global ACI + sparse auditing).

This subpackage adds an *online admission* layer on top of the existing
scorers/calibration. The scorer is unchanged and still owns low-FPR ranking;
the Global ACI controller only adapts the global admission threshold under
workload shift, and a sparse audit policy decides which few queries to judge.

Nothing here re-implements selected-candidate calibration, Wilson bounds, or
sliding metrics from scratch -- it reuses ``mlcache.calibration.query_level``
for the tie-exact selected-H0 quantile threshold.
"""

from mlcache.adaptive.aci_controller import (
    GlobalACIController,
    selected_h0_quantile_threshold,
)
from mlcache.adaptive.audit_policy import (
    AuditDecision,
    AuditZone,
    MarginBandAuditConfig,
    MarginBandAuditPolicy,
)
from mlcache.adaptive.metrics import AdmissionEvent, AdmissionMetrics
from mlcache.adaptive.presets import (
    RECOMMENDED_ADAPTIVE_CONFIG,
    RECOMMENDED_ENSEMBLE_MEMBERS,
    RecommendedAdaptiveConfig,
    build_recommended_controller,
    recommended_audit_config,
)

__all__ = [
    "GlobalACIController",
    "selected_h0_quantile_threshold",
    "AuditDecision",
    "AuditZone",
    "MarginBandAuditConfig",
    "MarginBandAuditPolicy",
    "AdmissionEvent",
    "AdmissionMetrics",
    "RecommendedAdaptiveConfig",
    "RECOMMENDED_ADAPTIVE_CONFIG",
    "RECOMMENDED_ENSEMBLE_MEMBERS",
    "recommended_audit_config",
    "build_recommended_controller",
]
