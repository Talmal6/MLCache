"""Runtime composition configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

from mlcache.cache.eviction import EvictionPolicyName
from mlcache.policies.query_level import QueryLevelPolicyMode
from mlcache.semantic_types import Threshold


@dataclass(frozen=True, slots=True)
class ShadowRuntimeConfig:
    enabled: bool = False
    top_k: int = 5
    max_pairs_per_request: int | None = None
    calibration_every_n: int = 5
    calibration_fraction: float | None = None
    collect_uncertain: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeRefitConfig:
    auto_refit: bool = True
    target_false_accept_rate: float = 0.05
    top_k: int = 1


@dataclass(frozen=True, slots=True)
class EvictionRuntimeConfig:
    """Configures the cache eviction behavior.

    ``policy`` selects which entry is removed when the cache is full (one of
    ``lru``, ``lfu``, ``fifo``). ``max_entries`` is the capacity bound; when it
    is ``None`` the cache is unbounded and no eviction ever happens (preserving
    the historical default).
    """

    policy: str = "lru"
    max_entries: int | None = None

    def __post_init__(self) -> None:
        # Validate eagerly so misconfiguration fails at construction time.
        object.__setattr__(self, "policy", EvictionPolicyName(str(self.policy).strip().lower()).value)
        if self.max_entries is not None and int(self.max_entries) <= 0:
            raise ValueError("eviction max_entries must be a positive integer or None")

    @classmethod
    def from_env(cls) -> "EvictionRuntimeConfig":
        policy = os.getenv("MLCACHE_EVICTION_POLICY")
        max_entries = os.getenv("MLCACHE_MAX_ENTRIES")
        return cls(
            policy=policy.strip().lower() if policy else "lru",
            max_entries=int(max_entries) if max_entries else None,
        )


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
class StorageRuntimeConfig:
    storage_backend: str | None = None
    vector_backend: str | None = None
    database_url: str | None = None
    mysql_database_url: str | None = None
    sqlite_database_url: str | None = None
    faiss_index_path: str | None = None
    faiss_dim: int | None = None
    faiss_metric: str = "cosine"
    faiss_shard_id: str = "default"
    top_k: int | None = None

    @classmethod
    def from_env(cls) -> "StorageRuntimeConfig":
        dim = os.getenv("MLCACHE_FAISS_DIM")
        top_k = os.getenv("MLCACHE_TOP_K")
        return cls(
            storage_backend=os.getenv("MLCACHE_STORAGE_BACKEND"),
            vector_backend=os.getenv("MLCACHE_VECTOR_BACKEND"),
            database_url=os.getenv("MLCACHE_DATABASE_URL"),
            mysql_database_url=os.getenv("MLCACHE_MYSQL_DATABASE_URL"),
            sqlite_database_url=os.getenv("MLCACHE_SQLITE_DATABASE_URL"),
            faiss_index_path=os.getenv("MLCACHE_FAISS_INDEX_PATH"),
            faiss_dim=int(dim) if dim else None,
            faiss_metric=os.getenv("MLCACHE_FAISS_METRIC", "cosine"),
            faiss_shard_id=os.getenv("MLCACHE_FAISS_SHARD_ID", "default"),
            top_k=int(top_k) if top_k else None,
        )

    def resolved_storage_backend(self, *, use_file_persistence: bool) -> str:
        backend = self.storage_backend or ("file" if use_file_persistence else "inmemory")
        backend = backend.lower()
        if backend not in {"inmemory", "file", "postgres", "mysql", "sqlite"}:
            raise ValueError("MLCACHE_STORAGE_BACKEND must be inmemory, file, postgres, mysql, or sqlite")
        return backend

    def resolved_vector_backend(self, *, storage_backend: str) -> str:
        backend = self.vector_backend or ("faiss" if storage_backend in {"postgres", "mysql", "sqlite"} else "inmemory")
        backend = backend.lower()
        if backend not in {"inmemory", "faiss"}:
            raise ValueError("MLCACHE_VECTOR_BACKEND must be inmemory or faiss")
        return backend


@dataclass(frozen=True, slots=True)
class MLCacheRuntimeConfig:
    shadow: ShadowRuntimeConfig = field(default_factory=ShadowRuntimeConfig)
    refit: RuntimeRefitConfig = field(default_factory=RuntimeRefitConfig)
    query_level: QueryLevelRuntimeConfig = field(default_factory=QueryLevelRuntimeConfig)
    storage: StorageRuntimeConfig = field(default_factory=StorageRuntimeConfig)
    eviction: EvictionRuntimeConfig = field(default_factory=EvictionRuntimeConfig)
    namespace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
