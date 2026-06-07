"""Online stopping controller implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from mlcache.online.types import OnlineMetrics, StopStatus


@dataclass(frozen=True, slots=True)
class OnlineStoppingConfig:
    check_every: int = 1
    window: int = 5
    patience: int = 3
    eps_tpr: float = 1e-3
    eps_fpr: float = 1e-3
    eps_tau: float = 1e-3
    fpr_margin: float = 0.0
    min_monitor_h0: int = 1
    min_monitor_h1: int = 1


class OnlineStoppingController(ABC):
    """Controls whether online updates should continue."""

    @abstractmethod
    def should_stop(self) -> StopStatus:
        raise NotImplementedError

    @abstractmethod
    def observe(self, metrics: OnlineMetrics) -> StopStatus:
        raise NotImplementedError


class WindowedOnlineStoppingController(OnlineStoppingController):
    """Stopping controller for online NP calibration convergence."""

    def __init__(self, config: OnlineStoppingConfig | None = None) -> None:
        self.config = config or OnlineStoppingConfig()
        if self.config.check_every <= 0:
            raise ValueError("check_every must be positive")
        if self.config.window <= 1:
            raise ValueError("window must be greater than 1")
        if self.config.patience <= 0:
            raise ValueError("patience must be positive")

        self._history: list[OnlineMetrics] = []
        self._stable_checks = 0
        self._status = StopStatus(stopped=False, reason=None)

    def should_stop(self) -> StopStatus:
        return self._status

    def observe(self, metrics: OnlineMetrics) -> StopStatus:
        if self._status.stopped:
            return self._status

        self._history.append(metrics)
        if not self._is_check_batch(metrics):
            return self._pending_status("waiting_for_check_interval")

        if not self._has_enough_monitor_data(metrics):
            self._stable_checks = 0
            return self._pending_status("insufficient_monitor_data")

        if len(self._history) < self.config.window:
            return self._pending_status("insufficient_window")

        window = self._history[-self.config.window :]
        tpr_delta = self._range([m.monitor_tpr for m in window])
        fpr_delta = self._range([m.monitor_fpr for m in window])
        tau_delta = self._range([float(m.threshold) for m in window])
        fpr_limit = float(metrics.target_false_accept_rate) + self.config.fpr_margin
        fpr_ok = float(metrics.monitor_fpr) <= fpr_limit

        stable = (
            fpr_ok
            and tpr_delta <= self.config.eps_tpr
            and fpr_delta <= self.config.eps_fpr
            and tau_delta <= self.config.eps_tau
        )

        if stable:
            self._stable_checks += 1
        else:
            self._stable_checks = 0

        status_metadata = {
            "stable_checks": self._stable_checks,
            "patience": self.config.patience,
            "window": self.config.window,
            "tpr_delta": tpr_delta,
            "fpr_delta": fpr_delta,
            "tau_delta": tau_delta,
            "fpr_limit": fpr_limit,
            "fpr_ok": fpr_ok,
        }

        if self._stable_checks >= self.config.patience:
            self._status = StopStatus(
                stopped=True,
                reason="online_metrics_converged",
                metadata=status_metadata,
            )
            return self._status

        self._status = StopStatus(
            stopped=False,
            reason="not_converged",
            metadata=status_metadata,
        )
        return self._status

    def reset(self) -> None:
        self._history = []
        self._stable_checks = 0
        self._status = StopStatus(stopped=False, reason=None)

    def history(self) -> tuple[OnlineMetrics, ...]:
        return tuple(self._history)

    def _is_check_batch(self, metrics: OnlineMetrics) -> bool:
        return int(metrics.batch_index) % self.config.check_every == 0

    def _has_enough_monitor_data(self, metrics: OnlineMetrics) -> bool:
        h0_ok = metrics.monitor_h0_count is None or metrics.monitor_h0_count >= self.config.min_monitor_h0
        h1_ok = metrics.monitor_h1_count is None or metrics.monitor_h1_count >= self.config.min_monitor_h1
        return h0_ok and h1_ok

    def _pending_status(self, reason: str) -> StopStatus:
        self._status = StopStatus(
            stopped=False,
            reason=reason,
            metadata={
                "stable_checks": self._stable_checks,
                "history_size": len(self._history),
            },
        )
        return self._status

    @staticmethod
    def _range(values: list[float]) -> float:
        finite = [float(v) for v in values if isfinite(float(v))]
        if not finite:
            return float("inf")
        return max(finite) - min(finite)
