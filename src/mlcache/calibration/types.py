"""Calibration data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from mlcache.semantic_types import RegionId, Score, TieMode


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

