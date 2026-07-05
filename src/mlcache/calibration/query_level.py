"""Query-level calibration data contracts and builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from math import floor, isfinite, nextafter
from typing import Any

from mlcache.calibration.wilson import wilson_upper_bound
from mlcache.semantic_types import CacheKey, OracleDecisionStatus, Query, Score, Threshold, TieMode


def threshold_from_selected_h0_scores(
    scores: Sequence[Score],
    *,
    target_false_accept_rate: float,
    tie_mode: TieMode = TieMode.GE,
) -> Threshold | None:
    """Smallest tie-exact threshold whose selected-H0 accept rate is <= the budget.

    Neyman-Pearson thresholding on selected-candidate H0 scores: returns the
    ``1 - target_false_accept_rate`` quantile computed so that no more than
    ``floor(target * n)`` H0 scores are accepted under ``tie_mode``. Returns
    ``None`` for an empty score list. Shared by batch query-level calibration and
    the online ACI controller so both compute an identical threshold.
    """

    if not scores:
        return None
    values = sorted((float(score) for score in scores), reverse=True)
    allowed_accepts = floor(float(target_false_accept_rate) * len(values))
    allowed_accepts = max(0, min(int(allowed_accepts), len(values)))

    if tie_mode == TieMode.GE:
        if allowed_accepts == 0:
            return Threshold(nextafter(values[0], float("inf")))
        boundary = values[allowed_accepts - 1]
        accepts_at_boundary = sum(1 for value in values if value >= boundary)
        if accepts_at_boundary <= allowed_accepts:
            return Threshold(boundary)
        return Threshold(nextafter(boundary, float("inf")))

    if allowed_accepts == 0:
        return Threshold(values[0])
    boundary = values[allowed_accepts - 1]
    accepts_if_including_boundary = sum(1 for value in values if value >= boundary)
    if accepts_if_including_boundary <= allowed_accepts:
        return Threshold(nextafter(boundary, float("-inf")))
    return Threshold(boundary)


@dataclass(frozen=True, slots=True)
class QueryLevelCalibrationConfig:
    target_false_accept_rate: float = 0.05
    tie_mode: TieMode = TieMode.GE
    wilson_confidence_z: float = 1.96
    fpr_wilson_margin: float = 0.03
    min_selected_h0: int = 500
    require_finite_threshold: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_false_accept_rate) < 1.0:
            raise ValueError("target_false_accept_rate must be in (0, 1)")
        if int(self.min_selected_h0) <= 0:
            raise ValueError("min_selected_h0 must be positive")
        if float(self.wilson_confidence_z) <= 0.0:
            raise ValueError("wilson_confidence_z must be positive")
        if float(self.fpr_wilson_margin) < 0.0:
            raise ValueError("fpr_wilson_margin must be non-negative")


@dataclass(frozen=True, slots=True)
class QueryCalibrationCandidate:
    """One labeled candidate from a query's retrieved top-k set."""

    score: Score
    label: int | None
    candidate_rank: int | None = None
    candidate_key: CacheKey | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryCalibrationRecord:
    """All candidate evidence collected for one query."""

    query_id: str
    candidates: Sequence[QueryCalibrationCandidate]
    query: Query | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True, slots=True)
class QueryCalibrationDecision:
    """The candidate selected for calibration from a single query."""

    query_id: str
    status: OracleDecisionStatus
    selected_candidate: QueryCalibrationCandidate | None = None
    selected_score: Score | None = None
    label: int | None = None
    candidate_rank: int | None = None
    candidate_key: CacheKey | None = None
    query_metadata: dict[str, Any] = field(default_factory=dict)
    candidate_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryCalibrationDataset:
    """Calibration decisions produced from query-level candidate selection."""

    decisions: Sequence[QueryCalibrationDecision]
    h0_scores: Sequence[Score]
    all_pair_h0_scores: Sequence[Score] = ()
    total_queries: int = 0
    skipped_queries: int = 0
    abstained_queries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "h0_scores", tuple(self.h0_scores))
        object.__setattr__(self, "all_pair_h0_scores", tuple(self.all_pair_h0_scores))


@dataclass(frozen=True, slots=True)
class QueryLevelCalibrationResult:
    """Output container for future query-level threshold calibration."""

    dataset: QueryCalibrationDataset
    threshold: Threshold | None = None
    target_false_accept_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


CandidateSelectionFn = Callable[[Sequence[QueryCalibrationCandidate]], QueryCalibrationCandidate | None]


class QueryLevelCalibrator(ABC):
    """Calibrates decisions selected per query rather than independent pair scores."""

    @abstractmethod
    def build_calibration_decisions(
        self,
        records: Iterable[QueryCalibrationRecord],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryCalibrationDataset:
        raise NotImplementedError

    @abstractmethod
    def calibrate(
        self,
        records_or_dataset: Iterable[QueryCalibrationRecord] | QueryCalibrationDataset,
        *,
        target_false_accept_rate: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QueryLevelCalibrationResult:
        raise NotImplementedError


class DefaultQueryLevelCalibrationBuilder(QueryLevelCalibrator):
    """Builds query-level calibration decisions without changing serving behavior."""

    def __init__(
        self,
        *,
        selection_fn: CandidateSelectionFn | None = None,
        empty_query_policy: str = "skip",
        config: QueryLevelCalibrationConfig | None = None,
    ) -> None:
        if empty_query_policy not in {"skip", "abstain"}:
            raise ValueError("empty_query_policy must be 'skip' or 'abstain'")
        self.selection_fn = selection_fn or self.select_max_score
        self.empty_query_policy = empty_query_policy
        self.config = config or QueryLevelCalibrationConfig()

    def build_calibration_decisions(
        self,
        records: Iterable[QueryCalibrationRecord],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> QueryCalibrationDataset:
        decisions: list[QueryCalibrationDecision] = []
        h0_scores: list[Score] = []
        all_pair_h0_scores: list[Score] = []
        skipped_queries = 0
        abstained_queries = 0
        total_queries = 0

        for record in records:
            total_queries += 1
            candidates = tuple(record.candidates)
            all_pair_h0_scores.extend(candidate.score for candidate in candidates if self._is_h0(candidate.label))

            if not candidates:
                if self.empty_query_policy == "abstain":
                    decisions.append(self._abstain_decision(record))
                    abstained_queries += 1
                else:
                    skipped_queries += 1
                continue

            selected = self.selection_fn(candidates)
            if selected is None:
                if self.empty_query_policy == "abstain":
                    decisions.append(self._abstain_decision(record))
                    abstained_queries += 1
                else:
                    skipped_queries += 1
                continue

            decision = self._selected_decision(record, selected)
            decisions.append(decision)
            if self._is_h0(selected.label):
                h0_scores.append(selected.score)

        dataset_metadata = {
            "selection": getattr(self.selection_fn, "__name__", type(self.selection_fn).__name__),
            "empty_query_policy": self.empty_query_policy,
        }
        if metadata:
            dataset_metadata.update(metadata)

        return QueryCalibrationDataset(
            decisions=tuple(decisions),
            h0_scores=tuple(h0_scores),
            all_pair_h0_scores=tuple(all_pair_h0_scores),
            total_queries=total_queries,
            skipped_queries=skipped_queries,
            abstained_queries=abstained_queries,
            metadata=dataset_metadata,
        )

    def calibrate(
        self,
        records_or_dataset: Iterable[QueryCalibrationRecord] | QueryCalibrationDataset,
        *,
        target_false_accept_rate: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QueryLevelCalibrationResult:
        config = self.config
        if target_false_accept_rate is not None:
            config = replace(config, target_false_accept_rate=float(target_false_accept_rate))
        dataset = (
            records_or_dataset
            if isinstance(records_or_dataset, QueryCalibrationDataset)
            else self.build_calibration_decisions(records_or_dataset, metadata=metadata)
        )
        selected_h0_scores = tuple(Score(float(score)) for score in dataset.h0_scores)
        all_pair_h0_scores = tuple(Score(float(score)) for score in dataset.all_pair_h0_scores)
        candidate_threshold = self._threshold_from_selected_h0_scores(selected_h0_scores, config)
        selected_accepts = self._accept_count(selected_h0_scores, candidate_threshold, config.tie_mode)
        all_pair_accepts = self._accept_count(all_pair_h0_scores, candidate_threshold, config.tie_mode)
        selected_h0_count = len(selected_h0_scores)
        all_pair_h0_count = len(all_pair_h0_scores)
        empirical_selected_fpr = None
        if selected_h0_count > 0:
            empirical_selected_fpr = float(selected_accepts) / float(selected_h0_count)
        empirical_all_pair_fpr = None
        if all_pair_h0_count > 0:
            empirical_all_pair_fpr = float(all_pair_accepts) / float(all_pair_h0_count)
        wilson_upper = wilson_upper_bound(
            int(selected_accepts),
            int(selected_h0_count),
            z=config.wilson_confidence_z,
        )
        allowed_fpr_bound = config.target_false_accept_rate + config.fpr_wilson_margin
        threshold_is_finite = candidate_threshold is not None and isfinite(float(candidate_threshold))

        reason: str | None = None
        if selected_h0_count < config.min_selected_h0:
            reason = "min_selected_h0_not_met"
        elif config.require_finite_threshold and not threshold_is_finite:
            reason = "threshold_not_finite"
        elif wilson_upper is None:
            reason = "wilson_upper_selected_fpr_unavailable"
        elif float(wilson_upper) > float(allowed_fpr_bound):
            reason = "wilson_upper_selected_fpr_exceeds_bound"

        passed = reason is None
        result_metadata = {
            "selected_h0_count": selected_h0_count,
            "all_pair_h0_count": all_pair_h0_count,
            "selected_h0_accepts": int(selected_accepts),
            "selected_h0_accepts_at_threshold": int(selected_accepts),
            "all_pair_h0_accepts_at_threshold": int(all_pair_accepts),
            "empirical_selected_fpr": empirical_selected_fpr,
            "empirical_all_pair_fpr_at_query_threshold": empirical_all_pair_fpr,
            "wilson_upper_selected_fpr": wilson_upper,
            "allowed_fpr_bound": allowed_fpr_bound,
            "threshold_is_finite": threshold_is_finite,
            "calibration_gate_passed": passed,
            "calibration_gate_reason": reason,
            "target_false_accept_rate": config.target_false_accept_rate,
            "tie_mode": config.tie_mode.value,
            "candidate_threshold": candidate_threshold,
            "selected_h0_score_min": self._score_min(selected_h0_scores),
            "selected_h0_score_max": self._score_max(selected_h0_scores),
            "selected_h0_score_mean": self._score_mean(selected_h0_scores),
            "all_pair_h0_score_min": self._score_min(all_pair_h0_scores),
            "all_pair_h0_score_max": self._score_max(all_pair_h0_scores),
            "all_pair_h0_score_mean": self._score_mean(all_pair_h0_scores),
            "selected_vs_all_pair_note": "threshold_calibrated_on_selected_h0_scores_only",
        }
        if metadata:
            result_metadata.update(metadata)
        return QueryLevelCalibrationResult(
            dataset=dataset,
            threshold=candidate_threshold if passed else None,
            target_false_accept_rate=config.target_false_accept_rate,
            metadata=result_metadata,
        )

    @staticmethod
    def _threshold_from_selected_h0_scores(
        scores: Sequence[Score],
        config: QueryLevelCalibrationConfig,
    ) -> Threshold | None:
        return threshold_from_selected_h0_scores(
            scores,
            target_false_accept_rate=config.target_false_accept_rate,
            tie_mode=config.tie_mode,
        )

    @staticmethod
    def _accept_count(scores: Sequence[Score], threshold: Threshold | None, tie_mode: TieMode) -> int:
        if threshold is None:
            return 0
        tau = float(threshold)
        if tie_mode == TieMode.GT:
            return sum(1 for score in scores if float(score) > tau)
        return sum(1 for score in scores if float(score) >= tau)

    @staticmethod
    def _score_min(scores: Sequence[Score]) -> float | None:
        if not scores:
            return None
        return min(float(score) for score in scores)

    @staticmethod
    def _score_max(scores: Sequence[Score]) -> float | None:
        if not scores:
            return None
        return max(float(score) for score in scores)

    @staticmethod
    def _score_mean(scores: Sequence[Score]) -> float | None:
        if not scores:
            return None
        return sum(float(score) for score in scores) / float(len(scores))

    @staticmethod
    def select_max_score(candidates: Sequence[QueryCalibrationCandidate]) -> QueryCalibrationCandidate | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                float(candidate.score),
                -(candidate.candidate_rank if candidate.candidate_rank is not None else len(candidates) + 1),
            ),
        )

    @staticmethod
    def _is_h0(label: int | None) -> bool:
        return label == 0

    @staticmethod
    def _selected_decision(
        record: QueryCalibrationRecord,
        selected: QueryCalibrationCandidate,
    ) -> QueryCalibrationDecision:
        return QueryCalibrationDecision(
            query_id=record.query_id,
            status=OracleDecisionStatus.HIT,
            selected_candidate=selected,
            selected_score=selected.score,
            label=selected.label,
            candidate_rank=selected.candidate_rank,
            candidate_key=selected.candidate_key,
            query_metadata=dict(record.metadata),
            candidate_metadata=dict(selected.metadata),
            metadata={"selection_status": "selected"},
        )

    @staticmethod
    def _abstain_decision(record: QueryCalibrationRecord) -> QueryCalibrationDecision:
        return QueryCalibrationDecision(
            query_id=record.query_id,
            status=OracleDecisionStatus.ABSTAIN,
            query_metadata=dict(record.metadata),
            metadata={"selection_status": "abstain", "reason": "empty_top_k"},
        )
