"""Query-level calibration data contracts and builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mlcache.semantic_types import CacheKey, OracleDecisionStatus, Query, Score, Threshold


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
    ) -> None:
        if empty_query_policy not in {"skip", "abstain"}:
            raise ValueError("empty_query_policy must be 'skip' or 'abstain'")
        self.selection_fn = selection_fn or self.select_max_score
        self.empty_query_policy = empty_query_policy

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
        dataset = (
            records_or_dataset
            if isinstance(records_or_dataset, QueryCalibrationDataset)
            else self.build_calibration_decisions(records_or_dataset, metadata=metadata)
        )
        result_metadata = {
            "status": "not_calibrated",
            "reason": "query_level_threshold_calibration_not_implemented",
        }
        if metadata:
            result_metadata.update(metadata)
        return QueryLevelCalibrationResult(
            dataset=dataset,
            threshold=None,
            target_false_accept_rate=target_false_accept_rate,
            metadata=result_metadata,
        )

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
