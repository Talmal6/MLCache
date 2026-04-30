"""Shared online-update data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semantic_desider.types import OracleDecision, Query, Response, Threshold


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    query: Query
    response: Response | None
    decision: OracleDecision
    accepted: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OnlineBatch:
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OnlineMetrics:
    batch_index: int
    monitor_tpr: float
    monitor_fpr: float
    threshold: Threshold
    target_false_accept_rate: float
    monitor_h0_count: int | None = None
    monitor_h1_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StopStatus:
    stopped: bool
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
