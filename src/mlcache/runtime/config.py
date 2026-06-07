"""Runtime composition configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ShadowRuntimeConfig:
    enabled: bool = False
    top_k: int = 5
    max_pairs_per_request: int | None = None
    calibration_every_n: int = 5
    collect_uncertain: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeRefitConfig:
    auto_refit: bool = True
    target_false_accept_rate: float = 0.05
    top_k: int = 1


@dataclass(frozen=True, slots=True)
class MLCacheRuntimeConfig:
    shadow: ShadowRuntimeConfig = field(default_factory=ShadowRuntimeConfig)
    refit: RuntimeRefitConfig = field(default_factory=RuntimeRefitConfig)
    namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
