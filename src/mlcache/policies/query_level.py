"""Query-level learned policy shadow evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mlcache.calibration.query_level import (
    DefaultQueryLevelCalibrationBuilder,
    QueryCalibrationCandidate,
    QueryCalibrationRecord,
)
from mlcache.semantic_types import CacheKey, OracleDecisionStatus, Score, Threshold, TieMode


class QueryLevelPolicyMode(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class QueryLevelPolicyDecision:
    status: OracleDecisionStatus
    accepted: bool
    selected_candidate_key: CacheKey | None
    selected_candidate_rank: int | None
    selected_score: Score | None
    threshold: Threshold | None
    reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryLevelPolicyConfig:
    mode: QueryLevelPolicyMode = QueryLevelPolicyMode.DISABLED
    tie_mode: TieMode = TieMode.GE
    require_threshold: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", QueryLevelPolicyMode(self.mode))
        object.__setattr__(self, "tie_mode", TieMode(self.tie_mode))


CandidateSelectionFn = Callable[[Sequence[QueryCalibrationCandidate]], QueryCalibrationCandidate | None]


class QueryLevelLearnedPolicy:
    """Pure evaluator for query-level learned reuse decisions."""

    def __init__(
        self,
        *,
        threshold: Threshold | None = None,
        config: QueryLevelPolicyConfig | None = None,
        selection_fn: CandidateSelectionFn | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.threshold = threshold
        self.config = config or QueryLevelPolicyConfig()
        self.selection_fn = selection_fn or DefaultQueryLevelCalibrationBuilder.select_max_score
        self.metadata = dict(metadata or {})

    def evaluate(
        self,
        record_or_candidates: QueryCalibrationRecord | Sequence[QueryCalibrationCandidate],
        *,
        threshold: Threshold | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QueryLevelPolicyDecision:
        """Evaluate what the query-level policy would decide for one top-k set."""

        record = record_or_candidates if isinstance(record_or_candidates, QueryCalibrationRecord) else None
        candidates = tuple(record.candidates if record is not None else record_or_candidates)
        effective_threshold = threshold if threshold is not None else self.threshold
        decision_metadata = self._base_metadata(
            record=record,
            candidates=candidates,
            metadata=metadata,
        )

        if not candidates:
            return QueryLevelPolicyDecision(
                status=OracleDecisionStatus.ABSTAIN,
                accepted=False,
                selected_candidate_key=None,
                selected_candidate_rank=None,
                selected_score=None,
                threshold=effective_threshold,
                reason="no_candidates",
                metadata=decision_metadata,
            )

        selected = self.selection_fn(candidates)
        if selected is None:
            return QueryLevelPolicyDecision(
                status=OracleDecisionStatus.ABSTAIN,
                accepted=False,
                selected_candidate_key=None,
                selected_candidate_rank=None,
                selected_score=None,
                threshold=effective_threshold,
                reason="no_candidates",
                metadata=decision_metadata,
            )

        decision_metadata.update(self._selected_metadata(selected))
        if effective_threshold is None:
            return QueryLevelPolicyDecision(
                status=OracleDecisionStatus.ABSTAIN,
                accepted=False,
                selected_candidate_key=selected.candidate_key,
                selected_candidate_rank=selected.candidate_rank,
                selected_score=selected.score,
                threshold=None,
                reason="query_level_threshold_missing",
                metadata=decision_metadata,
            )

        accepted = self._predict_score(selected.score, effective_threshold, self.config.tie_mode)
        return QueryLevelPolicyDecision(
            status=OracleDecisionStatus.HIT if accepted else OracleDecisionStatus.MISS,
            accepted=accepted,
            selected_candidate_key=selected.candidate_key,
            selected_candidate_rank=selected.candidate_rank,
            selected_score=selected.score,
            threshold=effective_threshold,
            reason=None if accepted else "selected_score_below_threshold",
            metadata=decision_metadata,
        )

    def decide(
        self,
        record_or_candidates: QueryCalibrationRecord | Sequence[QueryCalibrationCandidate],
        *,
        threshold: Threshold | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QueryLevelPolicyDecision:
        return self.evaluate(record_or_candidates, threshold=threshold, metadata=metadata)

    def _base_metadata(
        self,
        *,
        record: QueryCalibrationRecord | None,
        candidates: tuple[QueryCalibrationCandidate, ...],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        selection = getattr(self.selection_fn, "__name__", type(self.selection_fn).__name__)
        base = {
            **self.metadata,
            "mode": self.config.mode.value,
            "tie_mode": self.config.tie_mode.value,
            "require_threshold": bool(self.config.require_threshold),
            "candidate_count": len(candidates),
            "selection": selection,
        }
        if record is not None:
            base.update(
                {
                    "query_id": record.query_id,
                    "query_metadata": dict(record.metadata),
                }
            )
        if metadata:
            base.update(metadata)
        return base

    @staticmethod
    def _selected_metadata(selected: QueryCalibrationCandidate) -> dict[str, Any]:
        return {
            "selected_candidate_key": str(selected.candidate_key) if selected.candidate_key is not None else None,
            "selected_candidate_rank": selected.candidate_rank,
            "selected_score": float(selected.score),
            "selected_candidate_label": selected.label,
            "label": selected.label,
            "selected_candidate_metadata": dict(selected.metadata),
        }

    @staticmethod
    def _predict_score(score: Score, threshold: Threshold, tie_mode: TieMode) -> bool:
        if tie_mode == TieMode.GT:
            return float(score) > float(threshold)
        return float(score) >= float(threshold)


__all__ = [
    "QueryLevelLearnedPolicy",
    "QueryLevelPolicyConfig",
    "QueryLevelPolicyDecision",
    "QueryLevelPolicyMode",
]
