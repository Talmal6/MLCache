"""Refit policy contracts for scorer and threshold maintenance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlcache.online import OnlineMetrics
    from mlcache.semantic_types import Threshold


class RefitAction(StrEnum):
    NOOP = "noop"
    RECALIBRATE_THRESHOLD = "recalibrate_threshold"
    REFIT_SCORER = "refit_scorer"
    DISABLE_SEMANTIC_HITS = "disable_semantic_hits"


@dataclass(frozen=True, slots=True)
class RefitPolicyContext:
    total_h0: int
    total_h1: int
    target_false_accept_rate: float = 0.05
    current_threshold: Threshold | None = None
    metrics: OnlineMetrics | None = None
    new_h0_since_fit: int = 0
    new_h1_since_fit: int = 0
    new_h0_since_calibration: int = 0
    monitor_fpr: float | None = None
    monitor_fpr_upper_bound: float | None = None
    monitor_tpr: float | None = None
    previous_monitor_tpr: float | None = None
    decisions_since_fit: int = 0
    decisions_since_calibration: int = 0
    seconds_since_fit: float | None = None
    seconds_since_calibration: float | None = None
    fit_in_progress: bool = False
    scorer_changed: bool = False
    embedding_model_changed: bool = False
    traffic_distribution_changed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_target_false_accept_rate(self) -> float:
        if self.metrics is not None:
            return float(self.metrics.target_false_accept_rate)
        return float(self.target_false_accept_rate)

    @property
    def effective_monitor_fpr(self) -> float | None:
        if self.monitor_fpr is not None:
            return float(self.monitor_fpr)
        if self.metrics is not None:
            return float(self.metrics.monitor_fpr)
        return None

    @property
    def effective_monitor_tpr(self) -> float | None:
        if self.monitor_tpr is not None:
            return float(self.monitor_tpr)
        if self.metrics is not None:
            return float(self.metrics.monitor_tpr)
        return None


@dataclass(frozen=True, slots=True)
class RefitPolicyDecision:
    action: RefitAction
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def should_recalibrate_threshold(self) -> bool:
        return self.action == RefitAction.RECALIBRATE_THRESHOLD

    @property
    def should_refit_scorer(self) -> bool:
        return self.action == RefitAction.REFIT_SCORER


class RefitPolicy(ABC):
    """Decides when online feedback should update oracle-owned model state."""

    @abstractmethod
    def decide(self, context: RefitPolicyContext) -> RefitPolicyDecision:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ConservativeRefitConfig:
    min_h0_for_fit: int = 100
    min_h1_for_fit: int = 50
    min_h0_for_calibration: int = 100
    min_new_h0_for_calibration: int = 25
    min_new_h0_for_refit: int = 50
    min_new_h1_for_refit: int = 50
    min_new_fraction_for_refit: float = 0.25
    fpr_margin: float = 0.01
    tpr_drop_margin: float = 0.05
    min_decisions_between_calibrations: int = 1
    min_decisions_between_refits: int = 5
    max_decisions_between_calibrations: int | None = None
    max_decisions_between_refits: int | None = None
    min_seconds_between_calibrations: float | None = None
    min_seconds_between_refits: float | None = None
    max_seconds_between_calibrations: float | None = None
    max_seconds_between_refits: float | None = None


class ConservativeRefitPolicy(RefitPolicy):
    """Default policy that separates safety recalibration from scorer refits."""

    def __init__(self, config: ConservativeRefitConfig | None = None) -> None:
        self.config = config or ConservativeRefitConfig()
        self._validate_config()

    def decide(self, context: RefitPolicyContext) -> RefitPolicyDecision:
        self._validate_context(context)
        metadata = self._metadata(context)

        if context.fit_in_progress:
            return self._decision(RefitAction.NOOP, "fit_in_progress", metadata)

        can_fit = self._has_enough_fit_data(context)
        can_calibrate = self._has_enough_calibration_data(context)

        if context.current_threshold is None:
            if can_fit and self._refit_cooldown_elapsed(context):
                return self._decision(RefitAction.REFIT_SCORER, "initial_fit_ready", metadata)
            return self._decision(RefitAction.NOOP, "insufficient_initial_training_data", metadata)

        if self._requires_full_refit(context):
            if can_fit and self._refit_cooldown_elapsed(context):
                return self._decision(RefitAction.REFIT_SCORER, "semantic_distribution_changed", metadata)
            return self._decision(RefitAction.NOOP, "distribution_changed_but_refit_not_ready", metadata)

        if self._fpr_above_limit(context):
            if can_calibrate and self._has_new_calibration_data(context):
                if self._calibration_cooldown_elapsed(context):
                    return self._decision(RefitAction.RECALIBRATE_THRESHOLD, "fpr_above_limit", metadata)
                return self._decision(RefitAction.NOOP, "calibration_cooldown_active", metadata)
            return self._decision(RefitAction.NOOP, "fpr_above_limit_waiting_for_h0", metadata)

        if self._tpr_dropped(context):
            if can_fit and self._has_new_refit_data(context):
                if self._refit_cooldown_elapsed(context):
                    return self._decision(RefitAction.REFIT_SCORER, "tpr_degraded", metadata)
                return self._decision(RefitAction.NOOP, "refit_cooldown_active", metadata)
            return self._decision(RefitAction.NOOP, "tpr_degraded_waiting_for_labels", metadata)

        if can_fit and self._has_new_refit_data(context):
            if self._refit_cooldown_elapsed(context):
                return self._decision(RefitAction.REFIT_SCORER, "enough_new_balanced_examples", metadata)
            return self._decision(RefitAction.NOOP, "refit_cooldown_active", metadata)

        if can_calibrate and self._threshold_is_stale(context):
            if self._calibration_cooldown_elapsed(context):
                return self._decision(RefitAction.RECALIBRATE_THRESHOLD, "threshold_stale", metadata)
            return self._decision(RefitAction.NOOP, "calibration_cooldown_active", metadata)

        return self._decision(RefitAction.NOOP, "no_refit_signal", metadata)

    def _has_enough_fit_data(self, context: RefitPolicyContext) -> bool:
        return (
            int(context.total_h0) >= self.config.min_h0_for_fit
            and int(context.total_h1) >= self.config.min_h1_for_fit
        )

    def _has_enough_calibration_data(self, context: RefitPolicyContext) -> bool:
        return int(context.total_h0) >= self.config.min_h0_for_calibration

    def _has_new_calibration_data(self, context: RefitPolicyContext) -> bool:
        return int(context.new_h0_since_calibration) >= self.config.min_new_h0_for_calibration

    def _has_new_refit_data(self, context: RefitPolicyContext) -> bool:
        new_h0 = int(context.new_h0_since_fit)
        new_h1 = int(context.new_h1_since_fit)
        if new_h0 < self.config.min_new_h0_for_refit:
            return False
        if new_h1 < self.config.min_new_h1_for_refit:
            return False
        return self._new_fraction_since_fit(context) >= self.config.min_new_fraction_for_refit

    def _new_fraction_since_fit(self, context: RefitPolicyContext) -> float:
        new_total = max(0, int(context.new_h0_since_fit)) + max(0, int(context.new_h1_since_fit))
        total = max(0, int(context.total_h0)) + max(0, int(context.total_h1))
        previous_total = max(1, total - new_total)
        return float(new_total) / float(previous_total)

    def _requires_full_refit(self, context: RefitPolicyContext) -> bool:
        return (
            bool(context.scorer_changed)
            or bool(context.embedding_model_changed)
            or bool(context.traffic_distribution_changed)
        )

    def _fpr_above_limit(self, context: RefitPolicyContext) -> bool:
        if context.monitor_fpr_upper_bound is not None:
            return float(context.monitor_fpr_upper_bound) > context.effective_target_false_accept_rate
        monitor_fpr = context.effective_monitor_fpr
        if monitor_fpr is None:
            return False
        limit = context.effective_target_false_accept_rate + self.config.fpr_margin
        return float(monitor_fpr) > float(limit)

    def _tpr_dropped(self, context: RefitPolicyContext) -> bool:
        monitor_tpr = context.effective_monitor_tpr
        previous = context.previous_monitor_tpr
        if monitor_tpr is None or previous is None:
            return False
        return float(previous) - float(monitor_tpr) >= self.config.tpr_drop_margin

    def _threshold_is_stale(self, context: RefitPolicyContext) -> bool:
        if self.config.max_decisions_between_calibrations is not None:
            if int(context.decisions_since_calibration) >= self.config.max_decisions_between_calibrations:
                return True
        if self.config.max_seconds_between_calibrations is not None:
            elapsed = context.seconds_since_calibration
            if elapsed is not None and float(elapsed) >= self.config.max_seconds_between_calibrations:
                return True
        if self.config.max_decisions_between_refits is not None:
            if int(context.decisions_since_fit) >= self.config.max_decisions_between_refits:
                return True
        if self.config.max_seconds_between_refits is not None:
            elapsed = context.seconds_since_fit
            if elapsed is not None and float(elapsed) >= self.config.max_seconds_between_refits:
                return True
        return False

    def _calibration_cooldown_elapsed(self, context: RefitPolicyContext) -> bool:
        if int(context.decisions_since_calibration) < self.config.min_decisions_between_calibrations:
            return False
        min_seconds = self.config.min_seconds_between_calibrations
        elapsed = context.seconds_since_calibration
        if min_seconds is not None and elapsed is not None and float(elapsed) < min_seconds:
            return False
        return True

    def _refit_cooldown_elapsed(self, context: RefitPolicyContext) -> bool:
        if int(context.decisions_since_fit) < self.config.min_decisions_between_refits:
            return False
        min_seconds = self.config.min_seconds_between_refits
        elapsed = context.seconds_since_fit
        if min_seconds is not None and elapsed is not None and float(elapsed) < min_seconds:
            return False
        return True

    def _metadata(self, context: RefitPolicyContext) -> dict[str, Any]:
        monitor_fpr = context.effective_monitor_fpr
        monitor_tpr = context.effective_monitor_tpr
        target = context.effective_target_false_accept_rate
        new_fraction = self._new_fraction_since_fit(context)
        fpr_limit = target if context.monitor_fpr_upper_bound is not None else target + self.config.fpr_margin
        return {
            "total_h0": int(context.total_h0),
            "total_h1": int(context.total_h1),
            "new_h0_since_fit": int(context.new_h0_since_fit),
            "new_h1_since_fit": int(context.new_h1_since_fit),
            "new_h0_since_calibration": int(context.new_h0_since_calibration),
            "new_fraction_since_fit": new_fraction,
            "monitor_fpr": monitor_fpr,
            "monitor_fpr_upper_bound": context.monitor_fpr_upper_bound,
            "monitor_tpr": monitor_tpr,
            "target_false_accept_rate": target,
            "fpr_limit": fpr_limit,
            "current_threshold": None if context.current_threshold is None else float(context.current_threshold),
            "decisions_since_fit": int(context.decisions_since_fit),
            "decisions_since_calibration": int(context.decisions_since_calibration),
            **context.metadata,
        }

    def _validate_config(self) -> None:
        integer_fields = (
            "min_h0_for_fit",
            "min_h1_for_fit",
            "min_h0_for_calibration",
            "min_new_h0_for_calibration",
            "min_new_h0_for_refit",
            "min_new_h1_for_refit",
            "min_decisions_between_calibrations",
            "min_decisions_between_refits",
        )
        for name in integer_fields:
            if int(getattr(self.config, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

        optional_integer_fields = (
            "max_decisions_between_calibrations",
            "max_decisions_between_refits",
        )
        for name in optional_integer_fields:
            value = getattr(self.config, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when set")

        optional_seconds_fields = (
            "min_seconds_between_calibrations",
            "min_seconds_between_refits",
            "max_seconds_between_calibrations",
            "max_seconds_between_refits",
        )
        for name in optional_seconds_fields:
            value = getattr(self.config, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative when set")

        if self.config.min_new_fraction_for_refit < 0.0:
            raise ValueError("min_new_fraction_for_refit must be non-negative")
        if self.config.fpr_margin < 0.0:
            raise ValueError("fpr_margin must be non-negative")
        if self.config.tpr_drop_margin < 0.0:
            raise ValueError("tpr_drop_margin must be non-negative")

    @staticmethod
    def _validate_context(context: RefitPolicyContext) -> None:
        if int(context.total_h0) < 0 or int(context.total_h1) < 0:
            raise ValueError("total_h0 and total_h1 must be non-negative")
        if not 0.0 < context.effective_target_false_accept_rate < 1.0:
            raise ValueError("target_false_accept_rate must be in (0,1)")

    @staticmethod
    def _decision(
        action: RefitAction,
        reason: str,
        metadata: dict[str, Any],
    ) -> RefitPolicyDecision:
        return RefitPolicyDecision(action=action, reason=reason, metadata=metadata)
