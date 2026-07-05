"""Sparse audit policy with two independent channels.

Production does not label every served hit. We audit only a small fraction of
queries, and we deliberately separate two channels:

* **control-audit** -- a uniform, *margin-independent* ``Bernoulli(p_control)``
  draw. Only these events feed the Global ACI update, so the controller sees an
  unbiased sample of served H0 outcomes and truly targets the served FPR (not
  the FPR of the margin-selected subpopulation).
* **diagnostic-audit** -- a *margin-dependent* draw (higher probability near the
  threshold). Used for hard-example mining, region diagnostics, local-separator
  training candidates, and importance-weighted diagnostic metrics. It must NOT
  drive the controller.

The two draws are independent; an event can land in both, one, or neither.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum


class AuditZone(StrEnum):
    SAFE_ACCEPT = "safe_accept"
    BORDERLINE_ACCEPT = "borderline_accept"
    BORDERLINE_REJECT = "borderline_reject"
    SAFE_REJECT = "safe_reject"


@dataclass(frozen=True, slots=True)
class MarginBandAuditConfig:
    accept_margin: float = 0.05
    reject_margin: float = 0.05
    p_control: float = 0.04  # margin-independent slice feeding ACI (~3-5%)
    p_audit_safe_accept: float = 0.01
    p_audit_borderline_accept: float = 0.20
    p_audit_borderline_reject: float = 0.20
    p_audit_safe_reject: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "p_control",
            "p_audit_safe_accept",
            "p_audit_borderline_accept",
            "p_audit_borderline_reject",
            "p_audit_safe_reject",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if float(self.accept_margin) < 0.0 or float(self.reject_margin) < 0.0:
            raise ValueError("accept_margin and reject_margin must be non-negative")


@dataclass(frozen=True, slots=True)
class AuditDecision:
    zone: AuditZone
    control_audit: bool
    diagnostic_audit: bool
    p_control: float
    p_diagnostic: float

    @property
    def judged(self) -> bool:
        """Whether the event is labeled at all (either channel spends a judge)."""

        return self.control_audit or self.diagnostic_audit

    @property
    def audit_channel(self) -> str:
        if self.control_audit and self.diagnostic_audit:
            return "both"
        if self.control_audit:
            return "control"
        if self.diagnostic_audit:
            return "diagnostic"
        return "none"

    @property
    def audit_probability(self) -> float:
        """Probability that this event was selected for auditing at all.

        ``1 - (1 - p_control)(1 - p_diagnostic)`` -- the two channels are
        independent. Logged per-event for downstream inspection.
        """

        return 1.0 - (1.0 - self.p_control) * (1.0 - self.p_diagnostic)


class MarginBandAuditPolicy:
    """Assigns each decision to a margin band and draws both audit channels."""

    def __init__(
        self,
        config: MarginBandAuditConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or MarginBandAuditConfig()
        self._rng = rng or random.Random()

    def zone(self, best_score: float, tau: float) -> AuditZone:
        margin = float(best_score) - float(tau)
        if margin >= self.config.accept_margin:
            return AuditZone.SAFE_ACCEPT
        if margin >= 0.0:
            return AuditZone.BORDERLINE_ACCEPT
        if margin >= -self.config.reject_margin:
            return AuditZone.BORDERLINE_REJECT
        return AuditZone.SAFE_REJECT

    def _p_diagnostic(self, zone: AuditZone) -> float:
        return {
            AuditZone.SAFE_ACCEPT: self.config.p_audit_safe_accept,
            AuditZone.BORDERLINE_ACCEPT: self.config.p_audit_borderline_accept,
            AuditZone.BORDERLINE_REJECT: self.config.p_audit_borderline_reject,
            AuditZone.SAFE_REJECT: self.config.p_audit_safe_reject,
        }[zone]

    def decide(self, best_score: float, tau: float) -> AuditDecision:
        zone = self.zone(best_score, tau)
        p_control = float(self.config.p_control)
        p_diag = float(self._p_diagnostic(zone))
        # Independent draws. Control is margin-INDEPENDENT by construction.
        control = self._rng.random() < p_control
        diagnostic = self._rng.random() < p_diag
        return AuditDecision(
            zone=zone,
            control_audit=control,
            diagnostic_audit=diagnostic,
            p_control=p_control,
            p_diagnostic=p_diag,
        )
