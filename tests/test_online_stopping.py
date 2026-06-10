"""Robust unit tests for the online stopping mechanism.

`WindowedOnlineStoppingController` is the convergence detector that decides
when an online calibration loop has stabilised and should freeze. These tests
exercise it in isolation (no cache, no scorers) so every branch of the
convergence logic is pinned deterministically:

- config validation
- the three "pending" gates (check interval, monitor-data sufficiency, window
  fill) and which of them reset the stable-check counter
- convergence only after `patience` consecutive stable checks
- instability (TPR/FPR/threshold jumps) resetting the counter
- the FPR-limit gate (a low-variance-but-too-high FPR never converges)
- non-finite thresholds never converging
- terminal latching (`stopped` stays stopped) and `reset()`
"""

from __future__ import annotations

import unittest

from mlcache.online import (
    OnlineMetrics,
    OnlineStoppingConfig,
    WindowedOnlineStoppingController,
)
from mlcache.semantic_types import Threshold


def _metrics(
    batch_index: int,
    *,
    tpr: float = 0.9,
    fpr: float = 0.05,
    threshold: float = 0.7,
    target_fpr: float = 0.10,
    h0: int | None = 100,
    h1: int | None = 100,
) -> OnlineMetrics:
    return OnlineMetrics(
        batch_index=batch_index,
        monitor_tpr=tpr,
        monitor_fpr=fpr,
        threshold=Threshold(threshold),
        target_false_accept_rate=target_fpr,
        monitor_h0_count=h0,
        monitor_h1_count=h1,
    )


def _observe_stable(controller: WindowedOnlineStoppingController, n: int, *, start: int = 1):
    """Feed `n` identical, converged observations; return the last status."""
    status = None
    for i in range(start, start + n):
        status = controller.observe(_metrics(i))
    return status


class ConfigValidationTests(unittest.TestCase):
    def test_window_must_exceed_one(self) -> None:
        with self.assertRaises(ValueError):
            WindowedOnlineStoppingController(OnlineStoppingConfig(window=1))

    def test_patience_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            WindowedOnlineStoppingController(OnlineStoppingConfig(patience=0))

    def test_check_every_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            WindowedOnlineStoppingController(OnlineStoppingConfig(check_every=0))

    def test_default_config_constructs(self) -> None:
        controller = WindowedOnlineStoppingController()
        self.assertFalse(controller.should_stop().stopped)


class ConvergenceTests(unittest.TestCase):
    def test_converges_after_patience_stable_checks(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=3, patience=2, min_monitor_h0=1, min_monitor_h1=1)
        )
        # window=3 -> first 2 observations are "insufficient_window".
        # 3rd fills the window and is stable check #1; 4th is stable check #2
        # -> patience reached -> stop.
        status = _observe_stable(controller, 4)
        self.assertTrue(status.stopped)
        self.assertEqual(status.reason, "online_metrics_converged")
        self.assertEqual(status.metadata["stable_checks"], 2)

    def test_does_not_stop_before_window_is_filled(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=5, patience=1, min_monitor_h0=1, min_monitor_h1=1)
        )
        for i in range(1, 5):  # only 4 < window
            status = controller.observe(_metrics(i))
            self.assertFalse(status.stopped)
            self.assertEqual(status.reason, "insufficient_window")

    def test_exact_patience_boundary_does_not_stop_one_check_early(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=3, min_monitor_h0=1, min_monitor_h1=1)
        )
        # window=2: obs1 pending(window). obs2 stable#1, obs3 stable#2 -> not yet.
        controller.observe(_metrics(1))
        s2 = controller.observe(_metrics(2))
        s3 = controller.observe(_metrics(3))
        self.assertFalse(s2.stopped)
        self.assertFalse(s3.stopped)
        self.assertEqual(s3.metadata["stable_checks"], 2)
        s4 = controller.observe(_metrics(4))
        self.assertTrue(s4.stopped)

    def test_fpr_margin_allows_convergence_slightly_above_target(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, fpr_margin=0.05, min_monitor_h0=1, min_monitor_h1=1)
        )
        # monitor_fpr=0.12 > target 0.10 but <= target + margin(0.05)=0.15.
        for i in range(1, 4):
            status = controller.observe(_metrics(i, fpr=0.12, target_fpr=0.10))
        self.assertTrue(status.stopped)


class InstabilityResetsTests(unittest.TestCase):
    def test_tpr_jump_resets_stable_checks(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=2, eps_tpr=1e-3, min_monitor_h0=1, min_monitor_h1=1)
        )
        controller.observe(_metrics(1, tpr=0.90))
        s2 = controller.observe(_metrics(2, tpr=0.90))  # stable #1
        self.assertEqual(s2.metadata["stable_checks"], 1)
        s3 = controller.observe(_metrics(3, tpr=0.50))  # big jump in window -> reset
        self.assertEqual(s3.metadata["stable_checks"], 0)
        self.assertFalse(s3.stopped)

    def test_threshold_jump_resets_stable_checks(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=2, eps_tau=1e-3, min_monitor_h0=1, min_monitor_h1=1)
        )
        controller.observe(_metrics(1, threshold=0.70))
        s2 = controller.observe(_metrics(2, threshold=0.70))
        self.assertEqual(s2.metadata["stable_checks"], 1)
        s3 = controller.observe(_metrics(3, threshold=0.95))
        self.assertEqual(s3.metadata["stable_checks"], 0)

    def test_fpr_above_limit_never_converges_even_if_flat(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, fpr_margin=0.0, min_monitor_h0=1, min_monitor_h1=1)
        )
        # Perfectly flat but fpr=0.30 >> target 0.10 -> fpr_ok False -> never stable.
        for i in range(1, 8):
            status = controller.observe(_metrics(i, fpr=0.30, target_fpr=0.10))
        self.assertFalse(status.stopped)
        self.assertFalse(status.metadata["fpr_ok"])
        self.assertEqual(status.metadata["stable_checks"], 0)

    def test_non_finite_threshold_never_converges(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, min_monitor_h0=1, min_monitor_h1=1)
        )
        # _range() returns inf when all thresholds are non-finite -> tau_delta
        # never within eps -> never stable.
        for i in range(1, 6):
            status = controller.observe(_metrics(i, threshold=float("inf")))
        self.assertFalse(status.stopped)
        self.assertEqual(status.metadata["stable_checks"], 0)


class MonitorDataGateTests(unittest.TestCase):
    def test_insufficient_monitor_h1_blocks_and_resets(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, min_monitor_h0=1, min_monitor_h1=200)
        )
        # h1 count below the gate the whole time -> never evaluates convergence.
        for i in range(1, 10):
            status = controller.observe(_metrics(i, h1=50))
        self.assertFalse(status.stopped)
        self.assertEqual(status.reason, "insufficient_monitor_data")

    def test_recovering_monitor_data_then_converges(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, min_monitor_h0=1, min_monitor_h1=10)
        )
        # Early batches starved of H1, then H1 arrives and convergence proceeds.
        controller.observe(_metrics(1, h1=2))
        controller.observe(_metrics(2, h1=2))
        status = None
        for i in range(3, 6):
            status = controller.observe(_metrics(i, h1=50))
        self.assertTrue(status.stopped)

    def test_none_counts_are_treated_as_sufficient(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, min_monitor_h0=5, min_monitor_h1=5)
        )
        # None means "unknown / not gated" -> must not block.
        for i in range(1, 4):
            status = controller.observe(_metrics(i, h0=None, h1=None))
        self.assertTrue(status.stopped)


class CheckIntervalTests(unittest.TestCase):
    def test_non_check_batches_are_pending_but_recorded(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(check_every=5, window=2, patience=1, min_monitor_h0=1, min_monitor_h1=1)
        )
        s1 = controller.observe(_metrics(1))
        self.assertFalse(s1.stopped)
        self.assertEqual(s1.reason, "waiting_for_check_interval")
        # History still grows on non-check batches.
        self.assertEqual(len(controller.history()), 1)

    def test_convergence_only_evaluated_on_check_batches(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(check_every=5, window=2, patience=1, min_monitor_h0=1, min_monitor_h1=1)
        )
        status = None
        for i in range(1, 11):
            status = controller.observe(_metrics(i))
        # batch 5 and 10 are check batches; by batch 10 the window is filled and
        # stable -> stop.
        self.assertTrue(status.stopped)


class TerminalAndResetTests(unittest.TestCase):
    def test_stopped_status_is_latched(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, min_monitor_h0=1, min_monitor_h1=1)
        )
        _observe_stable(controller, 3)
        self.assertTrue(controller.should_stop().stopped)
        # Even a wildly unstable observation after stopping must not un-stop it.
        after = controller.observe(_metrics(99, tpr=0.0, fpr=0.99, threshold=0.0))
        self.assertTrue(after.stopped)
        self.assertEqual(after.reason, "online_metrics_converged")

    def test_reset_clears_history_and_status(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=2, patience=1, min_monitor_h0=1, min_monitor_h1=1)
        )
        _observe_stable(controller, 3)
        self.assertTrue(controller.should_stop().stopped)
        controller.reset()
        self.assertFalse(controller.should_stop().stopped)
        self.assertEqual(controller.history(), ())
        # Fully usable again after reset.
        status = _observe_stable(controller, 3)
        self.assertTrue(status.stopped)

    def test_history_preserves_observation_order(self) -> None:
        controller = WindowedOnlineStoppingController(
            OnlineStoppingConfig(window=10, patience=1, min_monitor_h0=1, min_monitor_h1=1)
        )
        for i in range(1, 5):
            controller.observe(_metrics(i, tpr=0.5 + i / 100))
        indices = [m.batch_index for m in controller.history()]
        self.assertEqual(indices, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
