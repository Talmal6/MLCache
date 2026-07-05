"""Offline replay over a frozen selected-candidate table.

To keep every method (fixed threshold vs Global ACI, cosine vs ensemble)
comparable, we score the stream **once** into a frozen table of selected
candidates and then run each admission method over the *same* table. A method
only ever sees ``best_score`` at decision time; the label is revealed
afterwards (and only "spent" when the audit policy judges the event), so there
is no way for a method to peek at labels before deciding.

Deployment-shaped protocol per query:
    retrieve top-k anchors -> score each -> select argmax -> record (score, label).

The selected pair's label is resolved through the dataset's judge, which only
knows the (query, own-cluster) pair; when argmax lands on a different anchor the
label is unknown (``None``) and the event is excluded from FPR/TPR, exactly as
in production.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from mlcache.adaptive.aci_controller import GlobalACIController, selected_h0_quantile_threshold
from mlcache.adaptive.audit_policy import MarginBandAuditPolicy
from mlcache.adaptive.metrics import AdmissionEvent, AdmissionMetrics
from mlcache.features.base import PairFeatureBuilder
from mlcache.features.hadamard import NormalizedHadamardFeatureBuilder
from mlcache.feedback.h1h0_npz_adapters import H1H0NPZDataset, H1H0NPZJudgeAdapter, H1H0NPZStreamAdapter
from mlcache.feedback.types import JudgeLabel, JudgeRequest
from mlcache.retrieval.in_memory import InMemoryVectorStore
from mlcache.scorers.base import SemanticScorer
from mlcache.semantic_types import TieMode


# --------------------------------------------------------------------------
# Frozen scored table
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoredSelection:
    """One query's selected candidate after retrieve -> score -> argmax."""

    stream_index: int
    query_id: str
    anchor_id: str
    best_score: float
    label: int | None  # 0=H0, 1=H1, None=unknown (ground-truth for selected pair)
    region_id: str | None
    own_cluster: bool  # selected anchor == the query's labeled cluster anchor


def _judge_label_to_int(label: JudgeLabel) -> int | None:
    if label == JudgeLabel.REUSABLE:
        return 1
    if label == JudgeLabel.NOT_REUSABLE:
        return 0
    return None


def build_frozen_table(
    dataset: H1H0NPZDataset,
    scorer: SemanticScorer,
    *,
    feature_builder: PairFeatureBuilder | None = None,
    judge: H1H0NPZJudgeAdapter | None = None,
    top_k: int = 5,
    selection: str = "argmax",
    progress_every: int = 0,
) -> tuple[ScoredSelection, ...]:
    """Score the whole stream once into a frozen selected-candidate table.

    ``selection="argmax"`` follows the deployment protocol (score all top-k,
    pick the best). ``selection="labeled_pair"`` scores the query against its
    own labeled cluster anchor directly (guaranteed label, faster) -- useful as
    a reference / for datasets that only label one pair per query.
    """

    if selection not in {"argmax", "labeled_pair"}:
        raise ValueError("selection must be 'argmax' or 'labeled_pair'")
    feature_builder = feature_builder or NormalizedHadamardFeatureBuilder()
    records = dataset.records()
    judge = judge or H1H0NPZJudgeAdapter(records)
    adapter = H1H0NPZStreamAdapter(dataset)
    anchors = adapter.anchor_entries()
    anchor_embedding = {str(a.cache_key): a.embedding for a in anchors}

    store = InMemoryVectorStore(similarity="cosine")
    for anchor in anchors:
        store.upsert(anchor)

    selections: list[ScoredSelection] = []
    for idx, record in enumerate(records):
        if selection == "labeled_pair":
            selected_key = str(record.anchor_key)
            features = feature_builder.build(record.query_embedding, anchor_embedding[selected_key])
            best_score = float(scorer.score(features))
        else:
            results = store.search(record.query_embedding, top_k=top_k)
            if not results:
                continue
            best_score = float("-inf")
            selected_key = None
            for candidate in results:
                features = feature_builder.build(record.query_embedding, candidate.embedding)
                candidate_score = float(scorer.score(features))
                if candidate_score > best_score:
                    best_score = candidate_score
                    selected_key = str(candidate.cache_key)
            if selected_key is None:
                continue

        own_cluster = selected_key == str(record.anchor_key)
        label = _resolve_label(judge, record, selected_key)
        selections.append(
            ScoredSelection(
                stream_index=idx,
                query_id=record.query_id,
                anchor_id=selected_key,
                best_score=best_score,
                label=label,
                region_id=selected_key,
                own_cluster=own_cluster,
            )
        )
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"  scored {idx + 1}/{len(records)}", flush=True)

    return tuple(selections)


def _resolve_label(judge: H1H0NPZJudgeAdapter, record: Any, selected_key: str) -> int | None:
    request = JudgeRequest(
        query=record.query,
        candidate_key=selected_key,  # type: ignore[arg-type]
        candidate_query=record.anchor_query if selected_key == str(record.anchor_key) else None,
        context={"query_id": record.query_id},
    )
    result = judge.judge(request)
    return _judge_label_to_int(result.decision.label)


# --------------------------------------------------------------------------
# Stream orderings
# --------------------------------------------------------------------------


def order_stream(
    table: Sequence[ScoredSelection],
    ordering: str,
    *,
    seed: int = 0,
) -> list[ScoredSelection]:
    """Return the table in the requested replay order (does not mutate input)."""

    items = list(table)
    if ordering == "natural":
        return sorted(items, key=lambda s: s.stream_index)
    if ordering == "random":
        rng = random.Random(seed)
        rng.shuffle(items)
        return items
    if ordering == "cluster_block":
        # Group by region so a whole cluster arrives together (worst case for a
        # threshold calibrated on an earlier, different mix).
        rng = random.Random(seed)
        blocks: dict[str | None, list[ScoredSelection]] = {}
        for item in sorted(items, key=lambda s: s.stream_index):
            blocks.setdefault(item.region_id, []).append(item)
        block_keys = list(blocks)
        rng.shuffle(block_keys)
        ordered: list[ScoredSelection] = []
        for key in block_keys:
            ordered.extend(blocks[key])
        return ordered
    raise ValueError(f"unknown ordering {ordering!r}")


# --------------------------------------------------------------------------
# Admission methods
# --------------------------------------------------------------------------


class Admission:
    """Admission-method interface: decide accept from ``best_score`` only."""

    name: str

    def threshold(self) -> float:
        raise NotImplementedError

    def alpha_t(self) -> float:
        raise NotImplementedError

    def threshold_source(self) -> str:
        return "fixed"

    def decide(self, best_score: float) -> bool:
        raise NotImplementedError

    def learn(self, best_score: float, label: int | None, accepted: bool, control_audit: bool) -> None:
        """Post-decision update. Must never change the just-made decision."""


class FixedThresholdAdmission(Admission):
    """Static threshold from batch calibration; never adapts."""

    def __init__(self, threshold: float, *, tie_mode: TieMode = TieMode.GE, name: str = "fixed") -> None:
        self._tau = float(threshold)
        self._tie_mode = tie_mode
        self.name = name

    def threshold(self) -> float:
        return self._tau

    def alpha_t(self) -> float:
        return float("nan")

    def decide(self, best_score: float) -> bool:
        if self._tie_mode == TieMode.GT:
            return float(best_score) > self._tau
        return float(best_score) >= self._tau


class GlobalACIAdmission(Admission):
    """Global ACI controller; updates only from control-audited H0 events."""

    def __init__(self, controller: GlobalACIController, *, name: str = "aci") -> None:
        self.controller = controller
        self.name = name

    def threshold(self) -> float:
        return self.controller.threshold()

    def alpha_t(self) -> float:
        return self.controller.alpha_t

    def threshold_source(self) -> str:
        return self.controller.threshold_source()

    def decide(self, best_score: float) -> bool:
        return self.controller.should_accept(best_score)

    def learn(self, best_score: float, label: int | None, accepted: bool, control_audit: bool) -> None:
        # Only the uniform control-audit slice drives the controller.
        if control_audit:
            self.controller.update(best_score, label, accepted)


# --------------------------------------------------------------------------
# Calibration helpers (reuse repo query-level tie-exact threshold)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    threshold: float | None
    h0_scores: tuple[float, ...]
    n_h0: int
    n_h1: int


def calibrate_selected_h0(
    calibration: Sequence[ScoredSelection],
    *,
    alpha_target: float,
    tie_mode: TieMode = TieMode.GE,
) -> CalibrationResult:
    """Batch NP threshold on the calibration split's selected-H0 scores.

    Uses the same tie-exact selected-H0 logic as the ACI controller so the
    fixed baseline and the ACI warm-start are computed identically.
    """

    h0 = tuple(float(s.best_score) for s in calibration if s.label == 0)
    n_h1 = sum(1 for s in calibration if s.label == 1)
    threshold = selected_h0_quantile_threshold(h0, alpha_target, tie_mode=tie_mode) if h0 else None
    return CalibrationResult(
        threshold=None if threshold is None else float(threshold),
        h0_scores=h0,
        n_h0=len(h0),
        n_h1=n_h1,
    )


# --------------------------------------------------------------------------
# Replay runner
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayResult:
    method: str
    summary: dict[str, Any]
    snapshots: tuple[dict[str, Any], ...]
    calibration: dict[str, Any]


class OfflineReplayRunner:
    """Runs one admission method over an ordered frozen stream."""

    def __init__(
        self,
        *,
        alpha_target: float,
        audit_policy: MarginBandAuditPolicy,
        window: int = 2000,
        snapshot_every: int = 0,
        tie_mode: TieMode = TieMode.GE,
        judge_cost_ratio: float = 0.1,
    ) -> None:
        self.alpha_target = float(alpha_target)
        self.audit_policy = audit_policy
        self.window = int(window)
        self.snapshot_every = int(snapshot_every)
        self.tie_mode = tie_mode
        self.judge_cost_ratio = float(judge_cost_ratio)

    def run(
        self,
        admission: Admission,
        serving_stream: Iterable[ScoredSelection],
        *,
        calibration_meta: dict[str, Any] | None = None,
        collect_events: bool = False,
        snapshot_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[ReplayResult, list[AdmissionEvent]]:
        metrics = AdmissionMetrics(
            alpha_target=self.alpha_target,
            window=self.window,
            snapshot_every=self.snapshot_every,
            judge_cost_ratio=self.judge_cost_ratio,
        )
        events: list[AdmissionEvent] = []
        for position, selection in enumerate(serving_stream):
            best_score = float(selection.best_score)

            # --- decision (no label access) ---
            tau = admission.threshold()
            accepted = admission.decide(best_score)

            # --- audit (still no label access) ---
            audit = self.audit_policy.decide(best_score, tau)

            # --- reveal label only now, after deciding ---
            label = selection.label
            admission.learn(best_score, label, accepted, audit.control_audit)

            event = AdmissionEvent(
                stream_index=position,
                query_id=selection.query_id,
                best_score=best_score,
                tau=tau,
                alpha_t=admission.alpha_t(),
                accepted=accepted,
                label=label,
                judged=audit.judged,
                control_audit=audit.control_audit,
                diagnostic_audit=audit.diagnostic_audit,
                p_control=audit.p_control,
                p_diagnostic=audit.p_diagnostic,
                zone=str(audit.zone),
                threshold_source=admission.threshold_source(),
                region_id=selection.region_id,
            )
            n_snapshots_before = len(metrics.snapshots)
            metrics.record(event)
            if snapshot_callback is not None and len(metrics.snapshots) > n_snapshots_before:
                snapshot_callback(metrics.snapshots[-1])
            if collect_events:
                events.append(event)

        result = ReplayResult(
            method=admission.name,
            summary=metrics.summary(),
            snapshots=tuple(metrics.snapshots),
            calibration=dict(calibration_meta or {}),
        )
        return result, events


def table_separation_diagnostics(
    table_a: Sequence[ScoredSelection],
    table_b: Sequence[ScoredSelection],
    *,
    alpha_target: float,
    tie_mode: TieMode = TieMode.GE,
) -> dict[str, Any]:
    """Quantify how differently two frozen tables rank/admit the same queries.

    Aligns the two tables by ``query_id`` (they share the query stream), then
    reports Spearman rank correlation of the best-scores and the fraction of
    queries where the two induce a *different* accept decision at each table's
    own batch threshold. Near-zero disagreement means the two scorers are
    effectively the same admission rule.
    """

    import numpy as np

    by_id_b = {s.query_id: s for s in table_b}
    a_scores: list[float] = []
    b_scores: list[float] = []
    for sel in table_a:
        other = by_id_b.get(sel.query_id)
        if other is None:
            continue
        a_scores.append(float(sel.best_score))
        b_scores.append(float(other.best_score))
    if not a_scores:
        return {"n_aligned": 0}

    a = np.asarray(a_scores)
    b = np.asarray(b_scores)
    # Spearman = Pearson on ranks (avoids a scipy dependency).
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    spearman = float(np.corrcoef(ar, br)[0, 1]) if len(a) > 1 else float("nan")

    tau_a = threshold_from_selected_h0_scores_for(table_a, alpha_target, tie_mode)
    tau_b = threshold_from_selected_h0_scores_for(table_b, alpha_target, tie_mode)
    disagree = None
    if tau_a is not None and tau_b is not None:
        accept_a = a >= float(tau_a)
        accept_b = b >= float(tau_b)
        disagree = float(np.mean(accept_a != accept_b))
    return {
        "n_aligned": int(len(a)),
        "spearman_best_score": spearman,
        "accept_disagreement_at_batch_tau": disagree,
        "tau_a": None if tau_a is None else float(tau_a),
        "tau_b": None if tau_b is None else float(tau_b),
    }


def threshold_from_selected_h0_scores_for(
    table: Sequence[ScoredSelection],
    alpha_target: float,
    tie_mode: TieMode = TieMode.GE,
) -> float | None:
    result = calibrate_selected_h0(table, alpha_target=alpha_target, tie_mode=tie_mode)
    return result.threshold


def score_percentiles(table: Sequence[ScoredSelection]) -> dict[str, Any]:
    """min / percentiles / max of best-scores over the labeled part of a table."""

    import numpy as np

    vals = np.asarray([float(s.best_score) for s in table if s.label in (0, 1)])
    if vals.size == 0:
        return {"n": 0}
    pct = np.percentile(vals, [0, 1, 25, 50, 75, 99, 100])
    return {
        "n": int(vals.size),
        "min": float(pct[0]),
        "p1": float(pct[1]),
        "p25": float(pct[2]),
        "p50": float(pct[3]),
        "p75": float(pct[4]),
        "p99": float(pct[5]),
        "max": float(pct[6]),
    }


def split_calibration_serving(
    stream: Sequence[ScoredSelection],
    *,
    calib_fraction: float,
) -> tuple[list[ScoredSelection], list[ScoredSelection]]:
    """Prefix split: first ``calib_fraction`` calibrates, the rest is served.

    A prefix split (rather than a random one) is intentional -- under a
    shifted ordering it stresses whether a threshold calibrated early survives
    later traffic.
    """

    if not 0.0 < calib_fraction < 1.0:
        raise ValueError("calib_fraction must be in (0, 1)")
    n_calib = max(1, int(round(len(stream) * calib_fraction)))
    n_calib = min(n_calib, len(stream) - 1)
    return list(stream[:n_calib]), list(stream[n_calib:])
