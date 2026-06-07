"""Threshold calibration and lookup contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mlcache.calibration.types import CalibrationExample, ThresholdCalibrationRequest, ThresholdScope
from mlcache.semantic_types import RegionId, ScorerName, Score, Threshold, TieMode


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
