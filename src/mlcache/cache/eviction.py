"""Configurable cache eviction policies.

These policies decide *what to remove* when a cache exceeds its capacity. They
are deliberately decoupled from the semantic scoring logic: the
``SemanticScorer`` / oracle decides HIT vs MISS, while an eviction policy only
chooses a victim cache key once the store is full. A policy never inspects
embeddings, scores, or thresholds — only per-entry bookkeeping (insertion
order, recency, and access frequency).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import ClassVar

from mlcache.semantic_types import CacheKey


class EvictionPolicyName(StrEnum):
    """Valid eviction policy identifiers accepted by config/CLI."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"


@dataclass(slots=True)
class EvictionEntryMetadata:
    """Per-entry bookkeeping consumed by the eviction policies.

    ``insertion_order`` and ``last_access_tick`` are monotonic counters owned by
    the policy; they make victim selection deterministic even when several
    operations share the same wall-clock timestamp. ``created_at`` and
    ``last_accessed`` are wall-clock datetimes kept for observability and to
    satisfy the documented entry metadata contract.
    """

    cache_key: CacheKey
    insertion_order: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    last_access_tick: int


class CacheEvictionPolicy(ABC):
    """Tracks entry metadata and selects a victim when the cache is full.

    The policy owns the authoritative set of live cache keys: ``record_insert``
    adds one, ``record_removal`` drops one, and ``record_hit`` refreshes recency
    and frequency. ``select_victim`` returns the key that should be removed next
    according to the concrete policy.
    """

    name: ClassVar[EvictionPolicyName]

    def __init__(self) -> None:
        self._entries: dict[CacheKey, EvictionEntryMetadata] = {}
        self._insertion_counter = 0
        self._clock = 0

    def _next_tick(self) -> int:
        self._clock += 1
        return self._clock

    def record_insert(self, cache_key: CacheKey) -> None:
        """Register a freshly inserted entry.

        On insert we initialize ``created_at`` / ``last_accessed`` to now and
        ``access_count`` to 1. Re-inserting an existing key is treated as a new
        insertion (fresh insertion order, reset counters).
        """

        key = CacheKey(str(cache_key))
        now = datetime.now(UTC)
        self._insertion_counter += 1
        self._entries[key] = EvictionEntryMetadata(
            cache_key=key,
            insertion_order=self._insertion_counter,
            created_at=now,
            last_accessed=now,
            access_count=1,
            last_access_tick=self._next_tick(),
        )

    def record_hit(self, cache_key: CacheKey) -> None:
        """Update recency and frequency for an entry on a cache HIT."""

        meta = self._entries.get(CacheKey(str(cache_key)))
        if meta is None:
            return
        meta.access_count += 1
        meta.last_accessed = datetime.now(UTC)
        meta.last_access_tick = self._next_tick()

    def record_removal(self, cache_key: CacheKey) -> None:
        """Drop bookkeeping for an entry that has left the cache."""

        self._entries.pop(CacheKey(str(cache_key)), None)

    def select_victim(self) -> CacheKey | None:
        """Return the cache key to evict, or ``None`` when empty."""

        if not self._entries:
            return None
        victim = min(self._entries.values(), key=self._eviction_sort_key)
        return victim.cache_key

    def metadata(self, cache_key: CacheKey) -> EvictionEntryMetadata | None:
        return self._entries.get(CacheKey(str(cache_key)))

    def tracked_keys(self) -> tuple[CacheKey, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, cache_key: object) -> bool:
        return CacheKey(str(cache_key)) in self._entries

    @abstractmethod
    def _eviction_sort_key(self, meta: EvictionEntryMetadata) -> tuple[int, ...] | int:
        """Sort key whose minimum identifies the next victim."""

        raise NotImplementedError


class FIFOEvictionPolicy(CacheEvictionPolicy):
    """Evict the oldest *inserted* entry, regardless of access pattern."""

    name = EvictionPolicyName.FIFO

    def _eviction_sort_key(self, meta: EvictionEntryMetadata) -> int:
        return meta.insertion_order


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Evict the least *recently used* entry (oldest last access)."""

    name = EvictionPolicyName.LRU

    def _eviction_sort_key(self, meta: EvictionEntryMetadata) -> tuple[int, int]:
        # Oldest last access first; break ties by insertion order.
        return (meta.last_access_tick, meta.insertion_order)


class LFUEvictionPolicy(CacheEvictionPolicy):
    """Evict the least *frequently used* entry (lowest access count)."""

    name = EvictionPolicyName.LFU

    def _eviction_sort_key(self, meta: EvictionEntryMetadata) -> tuple[int, int, int]:
        # Lowest access count first; break ties by least-recently-used then
        # oldest insertion so selection stays deterministic.
        return (meta.access_count, meta.last_access_tick, meta.insertion_order)


_POLICY_CLASSES: dict[EvictionPolicyName, type[CacheEvictionPolicy]] = {
    EvictionPolicyName.LRU: LRUEvictionPolicy,
    EvictionPolicyName.LFU: LFUEvictionPolicy,
    EvictionPolicyName.FIFO: FIFOEvictionPolicy,
}


def build_eviction_policy(name: str | EvictionPolicyName) -> CacheEvictionPolicy:
    """Construct an eviction policy from its name (``lru``/``lfu``/``fifo``)."""

    key = name.value if isinstance(name, EvictionPolicyName) else str(name).strip().lower()
    try:
        policy_name = EvictionPolicyName(key)
    except ValueError as exc:
        valid = ", ".join(p.value for p in EvictionPolicyName)
        raise ValueError(f"Unknown eviction policy {name!r}; valid values: {valid}") from exc
    return _POLICY_CLASSES[policy_name]()


__all__ = [
    "CacheEvictionPolicy",
    "EvictionEntryMetadata",
    "EvictionPolicyName",
    "FIFOEvictionPolicy",
    "LFUEvictionPolicy",
    "LRUEvictionPolicy",
    "build_eviction_policy",
]
