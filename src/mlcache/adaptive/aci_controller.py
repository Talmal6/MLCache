"""Global Adaptive Conformal Inference (ACI) admission-threshold controller.

The controller adapts a single global admission threshold so that the long-run
empirical false-positive rate (FPR) over *selected-candidate H0 events* tracks a
target budget ``alpha_target`` under workload shift. It does **not** replace the
scorer: the scorer still produces ``best_score`` for the selected candidate, and
the controller only decides where the accept threshold sits.

Design (see README):

* ``threshold()`` = ``Quantile(recent selected-H0 scores, 1 - alpha_t)``, computed
  with the repo's tie-exact selected-H0 logic
  (:func:`selected_h0_quantile_threshold`), never ``numpy.quantile``.
* ``alpha_t`` is nudged by the ACI update
  ``alpha_t += gamma * (alpha_target - err_t)`` on H0 events, where
  ``err_t = 1`` iff the H0 was accepted (a false hit).
* Updates are fed **only** from a uniform, margin-independent *control-audit*
  slice (enforced by the caller), so the controller is not biased by the
  margin-dependent diagnostic audit. On H1 / unknown labels it does nothing.
* The recent-H0 buffer is a FIFO (recency) window so the quantile estimate
  tracks drift; ACI corrects residual miscalibration.
"""

from __future__ import annotations

from collections import deque
from math import isfinite
from typing import Any

from mlcache.calibration.query_level import threshold_from_selected_h0_scores
from mlcache.calibration.wilson import wilson_upper_bound
from mlcache.semantic_types import Score, Threshold, TieMode


def selected_h0_quantile_threshold(
    scores: list[float] | tuple[float, ...],
    alpha: float,
    *,
    tie_mode: TieMode = TieMode.GE,
) -> Threshold | None:
    """Tie-exact ``Quantile(scores, 1 - alpha)`` for selected-H0 scores.

    Delegates to the repo's calibration logic so the ACI threshold and the
    batch NP threshold are computed identically (crucial for fair fixed-vs-ACI
    replay comparisons). Returns ``None`` for an empty score list.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not scores:
        return None
    typed = [Score(float(value)) for value in scores]
    # Reuse the tie-exact selected-H0 threshold logic; do NOT use numpy.quantile.
    return threshold_from_selected_h0_scores(
        typed, target_false_accept_rate=float(alpha), tie_mode=tie_mode
    )


class GlobalACIController:
    """One global ACI controller over selected-candidate H0 events."""

    def __init__(
        self,
        *,
        alpha_target: float,
        gamma: float,
        alpha_min: float,
        alpha_max: float,
        buffer_size: int,
        min_buffer_size: int,
        fallback_threshold: float | None = None,
        tie_mode: TieMode = TieMode.GE,
        wilson_gate: bool = False,
        wilson_z: float = 1.96,
        wilson_window: int = 1000,
    ) -> None:
        if not 0.0 < float(alpha_target) < 1.0:
            raise ValueError("alpha_target must be in (0, 1)")
        if float(gamma) <= 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 < float(alpha_min) <= float(alpha_max) < 1.0:
            raise ValueError("require 0 < alpha_min <= alpha_max < 1")
        if not float(alpha_min) <= float(alpha_target) <= float(alpha_max):
            raise ValueError("alpha_target must lie within [alpha_min, alpha_max]")
        if int(buffer_size) <= 0:
            raise ValueError("buffer_size must be positive")
        if int(min_buffer_size) <= 0:
            raise ValueError("min_buffer_size must be positive")
        if float(wilson_z) <= 0.0:
            raise ValueError("wilson_z must be positive")
        if int(wilson_window) <= 0:
            raise ValueError("wilson_window must be positive")

        self.alpha_target = float(alpha_target)
        self.gamma = float(gamma)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.buffer_size = int(buffer_size)
        self.min_buffer_size = int(min_buffer_size)
        self.fallback_threshold = None if fallback_threshold is None else float(fallback_threshold)
        self.tie_mode = tie_mode
        self.wilson_gate = bool(wilson_gate)
        self.wilson_z = float(wilson_z)
        self.wilson_window = int(wilson_window)

        self.alpha_t = float(alpha_target)
        # FIFO / recency window of control-audited selected-H0 scores.
        self._buffer: deque[float] = deque(maxlen=self.buffer_size)
        # Parallel origin flags (True = appended online via update, False = seeded)
        # so we can report how much of the live buffer has turned over post-seed.
        self._buffer_origin: deque[bool] = deque(maxlen=self.buffer_size)
        # Recent control-audit H0 outcomes (1 = false hit) for the Wilson gate.
        self._ctrl_outcomes: deque[int] = deque(maxlen=self.wilson_window)
        self._update_count = 0
        self._h0_seen = 0
        self._blocked_loosen_count = 0

    # -- seeding -----------------------------------------------------------

    def seed(
        self,
        h0_scores: list[float] | tuple[float, ...] = (),
        *,
        fallback_threshold: float | None = None,
    ) -> None:
        """Warm-start the buffer and fallback from batch calibration.

        Typically fed the selected-H0 scores and threshold produced by
        ``DefaultQueryLevelCalibrationBuilder.calibrate(...)`` on a calibration
        prefix, so cold-start uses the real NP threshold rather than an
        arbitrary constant.
        """

        for value in h0_scores:
            self._buffer.append(float(value))
            self._buffer_origin.append(False)
        if fallback_threshold is not None:
            self.fallback_threshold = float(fallback_threshold)

    # -- decision ----------------------------------------------------------

    @property
    def buffer_ready(self) -> bool:
        return len(self._buffer) >= self.min_buffer_size

    def threshold_source(self) -> str:
        return "adaptive" if self.buffer_ready else "fallback"

    def threshold(self) -> float:
        """Current admission threshold ``tau_t``.

        Uses the tie-exact selected-H0 quantile once the buffer is large enough;
        otherwise the seeded fallback. With no fallback and too little data it
        rejects by default (``+inf``) rather than trusting an unstable quantile.
        """

        if not self.buffer_ready:
            if self.fallback_threshold is None:
                return float("inf")
            return self.fallback_threshold
        tau = selected_h0_quantile_threshold(
            tuple(self._buffer), self.alpha_t, tie_mode=self.tie_mode
        )
        if tau is None or not isfinite(float(tau)):
            if self.fallback_threshold is not None:
                return self.fallback_threshold
            return float("inf")
        return float(tau)

    def should_accept(self, score: float) -> bool:
        tau = self.threshold()
        if self.tie_mode == TieMode.GT:
            return float(score) > tau
        return float(score) >= tau

    # -- learning ----------------------------------------------------------

    def wilson_ucb(self) -> float | None:
        """Wilson upper bound on control-audit FPR over the recent window.

        ``None`` until at least one control-audit H0 outcome is observed.
        """

        n = len(self._ctrl_outcomes)
        if n == 0:
            return None
        return wilson_upper_bound(sum(self._ctrl_outcomes), n, z=self.wilson_z)

    def update(self, score: float, label: int | None, accepted: bool) -> None:
        """Apply the ACI update from one *control-audited* selected event.

        Callers must only pass control-audited events here (margin-independent
        sample). ``accepted`` must be the **final** admission decision.

        * label ``0`` (H0): buffer the score and nudge ``alpha_t`` by
          ``gamma * (alpha_target - err)`` where ``err = 1`` iff accepted.
        * label ``1`` (H1) or ``None``: no-op (never updates alpha, never
          buffers).

        With ``wilson_gate=True``, an *upward* (loosening) move is applied only
        while the Wilson upper bound on the recent control-audit FPR stays at or
        below ``alpha_target`` -- this keeps the hard budget from being blown by
        ACI over-loosening on sparse updates. *Downward* (tightening) moves are
        always allowed.
        """

        if label is None:
            return
        if int(label) == 1:
            return
        if int(label) != 0:
            raise ValueError(f"label must be 0, 1, or None; got {label!r}")

        self._h0_seen += 1
        self._buffer.append(float(score))
        self._buffer_origin.append(True)
        err = 1.0 if accepted else 0.0
        self._ctrl_outcomes.append(int(accepted))
        proposed = self.alpha_t + self.gamma * (self.alpha_target - err)

        if self.wilson_gate and proposed > self.alpha_t:
            ucb = self.wilson_ucb()
            # Block loosening unless we have evidence FPR is safely under budget.
            if ucb is None or float(ucb) > self.alpha_target:
                proposed = self.alpha_t
                self._blocked_loosen_count += 1

        self.alpha_t = min(self.alpha_max, max(self.alpha_min, proposed))
        self._update_count += 1

    # -- introspection -----------------------------------------------------

    @property
    def buffer_size_current(self) -> int:
        return len(self._buffer)

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def blocked_loosen_count(self) -> int:
        return self._blocked_loosen_count

    @property
    def control_h0_count(self) -> int:
        return len(self._ctrl_outcomes)

    @property
    def post_seed_buffer_fraction(self) -> float | None:
        """Fraction of the live H0 buffer that was appended online (post-seed).

        Near 0 means the quantile is still dominated by the seeded calibration
        scores (little drift tracking via the buffer); near 1 means the buffer
        has turned over to online control-audit scores.
        """

        n = len(self._buffer_origin)
        if n == 0:
            return None
        return sum(1 for flag in self._buffer_origin if flag) / n

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha_target": self.alpha_target,
            "alpha_t": self.alpha_t,
            "gamma": self.gamma,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "buffer_size": self.buffer_size,
            "min_buffer_size": self.min_buffer_size,
            "fallback_threshold": self.fallback_threshold,
            "tie_mode": self.tie_mode.value,
            "wilson_gate": self.wilson_gate,
            "wilson_z": self.wilson_z,
            "wilson_window": self.wilson_window,
            "buffer": list(self._buffer),
            "buffer_origin": [bool(flag) for flag in self._buffer_origin],
            "ctrl_outcomes": list(self._ctrl_outcomes),
            "update_count": self._update_count,
            "h0_seen": self._h0_seen,
            "blocked_loosen_count": self._blocked_loosen_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.alpha_target = float(state["alpha_target"])
        self.alpha_t = float(state["alpha_t"])
        self.gamma = float(state["gamma"])
        self.alpha_min = float(state["alpha_min"])
        self.alpha_max = float(state["alpha_max"])
        self.buffer_size = int(state["buffer_size"])
        self.min_buffer_size = int(state["min_buffer_size"])
        self.fallback_threshold = (
            None if state["fallback_threshold"] is None else float(state["fallback_threshold"])
        )
        self.tie_mode = TieMode(state["tie_mode"])
        self.wilson_gate = bool(state.get("wilson_gate", False))
        self.wilson_z = float(state.get("wilson_z", 1.96))
        self.wilson_window = int(state.get("wilson_window", 1000))
        self._buffer = deque(
            (float(v) for v in state.get("buffer", ())), maxlen=self.buffer_size
        )
        origin = state.get("buffer_origin")
        if origin is None:
            origin = [False] * len(self._buffer)
        self._buffer_origin = deque(
            (bool(v) for v in origin), maxlen=self.buffer_size
        )
        self._ctrl_outcomes = deque(
            (int(v) for v in state.get("ctrl_outcomes", ())), maxlen=self.wilson_window
        )
        self._update_count = int(state.get("update_count", 0))
        self._h0_seen = int(state.get("h0_seen", 0))
        self._blocked_loosen_count = int(state.get("blocked_loosen_count", 0))
