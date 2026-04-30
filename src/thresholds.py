"""Threshold calibration and lookup contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from semantic_desider.types import RegionId, ScorerName, Score, Threshold, TieMode


class ThresholdScope(StrEnum):
    GLOBAL = "global"
    LOCAL = "local"
    CLUSTER_LOCAL = "cluster_local"


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    score: Score
    label: int = 0
    region_id: RegionId | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationRequest:
    h0_scores: Iterable[Score]
    target_false_accept_rate: float
    tie_mode: TieMode = TieMode.GE
    context: dict[str, Any] = field(default_factory=dict)


class NPThresholdCalibrator(ABC):
    """Neyman-Pearson threshold calibration contract."""

    @abstractmethod
    def calibrate(
        self,
        request: ThresholdCalibrationRequest,
    ) -> Threshold:
        raise NotImplementedError

    @abstractmethod
    def predict(self, score: Score, threshold: Threshold, *, tie_mode: TieMode = TieMode.GE) -> bool:
        raise NotImplementedError


class ThresholdProvider(ABC):
    """Provides global, local, or cluster-local thresholds."""

    @abstractmethod
    def get_threshold(
        self,
        *,
        scorer: ScorerName,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
        region_id: RegionId | None = None,
        cluster_id: RegionId | None = None,
        context: dict[str, Any] | None = None,
    ) -> Threshold:
        raise NotImplementedError

    @abstractmethod
    def set_threshold(
        self,
        threshold: Threshold,
        *,
        scorer: ScorerName,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
        region_id: RegionId | None = None,
        cluster_id: RegionId | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError
