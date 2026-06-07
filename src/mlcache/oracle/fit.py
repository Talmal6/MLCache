"""Oracle fitting result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlcache.semantic_types import Threshold


@dataclass(frozen=True, slots=True)
class ActivationGateResult:
    passed: bool
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OracleFitResult:
    scorer: str
    threshold: Threshold
    target_false_accept_rate: float
    n_h0_train: int
    n_h1_train: int
    n_h0_calib: int
    n_h1_calib: int
    metadata: dict[str, Any] = field(default_factory=dict)
