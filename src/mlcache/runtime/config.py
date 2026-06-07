"""Runtime composition configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlcache.policies.query_level import QueryLevelPolicyMode
from mlcache.semantic_types import Threshold


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
class QueryLevelRuntimeConfig:
    enabled: bool = False
    mode: QueryLevelPolicyMode = QueryLevelPolicyMode.DISABLED
    threshold: Threshold | None = None
    require_threshold: bool = True
    fallback_to_pair_level_on_abstain: bool = True
    fallback_to_pair_level_on_missing_record: bool = True
    fallback_to_pair_level_on_kv_miss: bool = True
    active_requires_threshold: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", QueryLevelPolicyMode(self.mode))


@dataclass(frozen=True, slots=True)
class MLCacheRuntimeConfig:
    shadow: ShadowRuntimeConfig = field(default_factory=ShadowRuntimeConfig)
    refit: RuntimeRefitConfig = field(default_factory=RuntimeRefitConfig)
    query_level: QueryLevelRuntimeConfig = field(default_factory=QueryLevelRuntimeConfig)
    namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
