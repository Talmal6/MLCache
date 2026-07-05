"""Admission metrics: ground-truth, audited estimates, and sliding windows.

Two families of numbers are reported, and they answer different questions:

* **Ground-truth (full-label) metrics** -- computed over *every* labeled
  selected-candidate event in an offline replay. This is the true achieved
  FPR/TPR/hit-rate/utility of a method. It is legitimate offline: the label is
  only ever revealed *after* the admission decision (no leakage), and is used
  purely for post-hoc evaluation.

* **Audited estimates** -- what a production deployment could actually measure
  from its sparse judge calls:
    - ``raw``     : over all judged events (biased if audit is margin-dependent);
    - ``control`` : over the uniform control-audit slice (unbiased for FPR/TPR);
    - ``iw``      : Horvitz-Thompson estimate over the *diagnostic* audits,
                    weighting each event by ``1 / p_diagnostic``, reported with
                    a Kish effective sample size so you know when to trust it.

FPR is computed only over H0 (label==0) selected events; TPR only over H1
(label==1). Unknown labels never enter FPR/TPR.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AdmissionEvent:
    """One selected-candidate decision, with its (possibly hidden) label."""

    stream_index: int
    query_id: str
    best_score: float
    tau: float
    alpha_t: float
    accepted: bool  # final admission decision (served from cache)
    label: int | None  # ground-truth for the selected pair: 0=H0, 1=H1, None=unknown
    judged: bool
    control_audit: bool
    diagnostic_audit: bool
    p_control: float
    p_diagnostic: float
    zone: str
    threshold_source: str
    region_id: str | None = None


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


@dataclass
class _Counter:
    """Accumulates FP/TP over a labeled sub-population."""

    n_h0: int = 0
    n_h1: int = 0
    fp: int = 0  # accepted H0
    tp: int = 0  # accepted H1

    def add(self, label: int | None, accepted: bool) -> None:
        if label == 0:
            self.n_h0 += 1
            if accepted:
                self.fp += 1
        elif label == 1:
            self.n_h1 += 1
            if accepted:
                self.tp += 1

    @property
    def fpr(self) -> float | None:
        return _ratio(self.fp, self.n_h0)

    @property
    def tpr(self) -> float | None:
        return _ratio(self.tp, self.n_h1)


@dataclass
class _WeightedCounter:
    """Horvitz-Thompson FP/TP with Kish effective sample size."""

    w_h0: float = 0.0
    w_h1: float = 0.0
    fp_w: float = 0.0
    tp_w: float = 0.0
    w_h0_sq: float = 0.0
    w_h1_sq: float = 0.0
    n_h0: int = 0
    n_h1: int = 0

    def add(self, label: int | None, accepted: bool, weight: float) -> None:
        if weight <= 0:
            return
        if label == 0:
            self.w_h0 += weight
            self.w_h0_sq += weight * weight
            self.n_h0 += 1
            if accepted:
                self.fp_w += weight
        elif label == 1:
            self.w_h1 += weight
            self.w_h1_sq += weight * weight
            self.n_h1 += 1
            if accepted:
                self.tp_w += weight

    @property
    def fpr(self) -> float | None:
        return _ratio(self.fp_w, self.w_h0)

    @property
    def tpr(self) -> float | None:
        return _ratio(self.tp_w, self.w_h1)

    @staticmethod
    def _kish(sum_w: float, sum_w_sq: float) -> float | None:
        if sum_w_sq <= 0:
            return None
        return (sum_w * sum_w) / sum_w_sq

    @property
    def kish_ess_h0(self) -> float | None:
        return self._kish(self.w_h0, self.w_h0_sq)

    @property
    def kish_ess_h1(self) -> float | None:
        return self._kish(self.w_h1, self.w_h1_sq)


class AdmissionMetrics:
    """Streaming accumulator for one admission method over a replay stream."""

    def __init__(
        self,
        *,
        alpha_target: float,
        window: int = 2000,
        snapshot_every: int = 0,
        judge_cost_ratio: float = 0.1,
    ) -> None:
        self.alpha_target = float(alpha_target)
        self.window = int(window)
        self.snapshot_every = int(snapshot_every)
        # Cost of one judge call relative to one saved provider call, used for the
        # cost-adjusted utility. judge >> provider makes sparse auditing net-negative.
        self.judge_cost_ratio = float(judge_cost_ratio)

        self.total = 0
        self.accepted = 0
        self.no_judge_hits = 0
        self.judged = 0
        self.provider_calls = 0

        self._ground_truth = _Counter()
        self._raw = _Counter()
        self._control = _Counter()
        self._iw = _WeightedCounter()

        # Sliding window over the last ``window`` events (all events, labeled or not).
        self._win: deque[AdmissionEvent] = deque(maxlen=self.window)
        self.snapshots: list[dict[str, Any]] = []

    def record(self, event: AdmissionEvent) -> None:
        self.total += 1
        if event.accepted:
            self.accepted += 1
            if not event.judged:
                self.no_judge_hits += 1
        else:
            self.provider_calls += 1
        if event.judged:
            self.judged += 1

        # Ground-truth uses every labeled event (offline, post-decision).
        self._ground_truth.add(event.label, event.accepted)

        if event.judged:
            self._raw.add(event.label, event.accepted)
        if event.control_audit:
            self._control.add(event.label, event.accepted)
        if event.diagnostic_audit and event.p_diagnostic > 0.0:
            self._iw.add(event.label, event.accepted, 1.0 / event.p_diagnostic)

        self._win.append(event)
        if self.snapshot_every and self.total % self.snapshot_every == 0:
            self.snapshots.append(self._window_snapshot())

    # -- sliding window ----------------------------------------------------

    def _window_snapshot(self) -> dict[str, Any]:
        win = self._win
        n = len(win)
        win_c = _Counter()
        accepted = 0
        no_judge_hits = 0
        judged = 0
        for ev in win:
            win_c.add(ev.label, ev.accepted)
            if ev.accepted:
                accepted += 1
                if not ev.judged:
                    no_judge_hits += 1
            if ev.judged:
                judged += 1
        last = win[-1]
        return {
            "stream_index": last.stream_index,
            "total_seen": self.total,
            "alpha_t": last.alpha_t,
            "tau": last.tau,
            "cumulative_fpr": self._ground_truth.fpr,
            "cumulative_tpr": self._ground_truth.tpr,
            "cumulative_hit_rate": _ratio(self.accepted, self.total),
            "cumulative_no_judge_hit_rate": _ratio(self.no_judge_hits, self.total),
            "cumulative_judged_fraction": _ratio(self.judged, self.total),
            "cumulative_n_h0": self._ground_truth.n_h0,
            "cumulative_n_h1": self._ground_truth.n_h1,
            "cumulative_false_hits": self._ground_truth.fp,
            "cumulative_true_hits": self._ground_truth.tp,
            "window_fpr": win_c.fpr,
            "window_tpr": win_c.tpr,
            "window_hit_rate": _ratio(accepted, n),
            "window_no_judge_hit_rate": _ratio(no_judge_hits, n),
            "window_judged_fraction": _ratio(judged, n),
            "window_n_h0": win_c.n_h0,
            "window_n_h1": win_c.n_h1,
            "window_false_hits": win_c.fp,
            "window_true_hits": win_c.tp,
        }

    # -- summary -----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        gt = self._ground_truth
        gt_fpr = gt.fpr
        gt_tpr = gt.tpr
        fpr_ok = gt_fpr is not None and gt_fpr <= self.alpha_target
        # NP score: TPR at FPR <= alpha. If the method blew the budget it earns 0.
        np_score = gt_tpr if (fpr_ok and gt_tpr is not None) else 0.0
        # Cost-adjusted utility: correct reuses saved minus the judge bill.
        # (Absolute counts; judged is every audited request across both channels.)
        cost_adjusted_utility = float(gt.tp) - self.judge_cost_ratio * float(self.judged)
        return {
            "alpha_target": self.alpha_target,
            "total_requests": self.total,
            # served-decision rates
            "hit_rate": _ratio(self.accepted, self.total),
            "no_judge_hit_rate": _ratio(self.no_judge_hits, self.total),
            "judged_fraction": _ratio(self.judged, self.total),
            "provider_call_rate": _ratio(self.provider_calls, self.total),
            # ground-truth (full-label) admission quality
            "gt_n_h0": gt.n_h0,
            "gt_n_h1": gt.n_h1,
            "achieved_fpr": gt_fpr,
            "tpr": gt_tpr,
            "false_hits": gt.fp,
            "true_hits": gt.tp,
            "correct_hit_utility": _ratio(gt.tp, self.total),
            "fpr_ok": fpr_ok,
            "np_score": np_score,
            "judge_cost_ratio": self.judge_cost_ratio,
            "cost_adjusted_utility": cost_adjusted_utility,
            "cost_adjusted_utility_per_request": _ratio(cost_adjusted_utility, self.total),
            # audited estimates (production-realistic)
            "raw_audit_fpr": self._raw.fpr,
            "raw_audit_tpr": self._raw.tpr,
            "raw_audit_n_h0": self._raw.n_h0,
            "raw_audit_n_h1": self._raw.n_h1,
            "control_audit_fpr": self._control.fpr,
            "control_audit_tpr": self._control.tpr,
            "control_audit_n_h0": self._control.n_h0,
            "control_audit_n_h1": self._control.n_h1,
            "iw_audit_fpr": self._iw.fpr,
            "iw_audit_tpr": self._iw.tpr,
            "iw_audit_kish_ess_h0": self._iw.kish_ess_h0,
            "iw_audit_kish_ess_h1": self._iw.kish_ess_h1,
            "iw_audit_n_h0": self._iw.n_h0,
            "iw_audit_n_h1": self._iw.n_h1,
        }
