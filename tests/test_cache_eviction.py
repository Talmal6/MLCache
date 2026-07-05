"""Tests for configurable cache eviction policies (LRU / LFU / FIFO).

These cover the eviction policy objects in isolation and their integration
through ``SemanticCacheGateway``. Eviction is orthogonal to the semantic
HIT/MISS decision, so the gateway tests use a ``FakeOracle`` that returns a HIT
for any indexed key — keeping the focus purely on *what gets removed* when the
cache is full.
"""

from __future__ import annotations

import unittest

from mlcache.cache import (
    FIFOEvictionPolicy,
    InMemoryKVStore,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    SemanticCacheGateway,
    build_eviction_policy,
)
from mlcache.cache.eviction import EvictionPolicyName
from mlcache.oracle.base import SemanticCacheOracle
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    OracleDecision,
    OracleDecisionStatus,
    Query,
    Response,
    Score,
    ScorerName,
)


class FakeOracle(SemanticCacheOracle):
    """Minimal oracle: a HIT for any currently-indexed key.

    The cache key is derived directly from the lookup query so tests can drive
    deterministic HITs without embeddings, scorers, or thresholds.
    """

    def __init__(self) -> None:
        self.live: set[CacheKey] = set()
        self.removed: list[CacheKey] = []

    def decide(self, request: CacheLookup) -> OracleDecision:
        key = CacheKey(str(request.query))
        if key in self.live:
            return OracleDecision(
                status=OracleDecisionStatus.HIT,
                accepted=True,
                cache_key=key,
                score=Score(1.0),
                threshold=None,
                scorer=ScorerName("fake"),
                candidate_count=1,
            )
        return OracleDecision(
            status=OracleDecisionStatus.MISS,
            accepted=False,
            cache_key=None,
            score=None,
            threshold=None,
            scorer=ScorerName("fake"),
        )

    def index(self, entry: CacheEntry) -> None:
        self.live.add(CacheKey(str(entry.cache_key)))

    def remove(self, cache_key: CacheKey) -> None:
        key = CacheKey(str(cache_key))
        self.removed.append(key)
        self.live.discard(key)

    # Unused by the gateway path under test.
    def fit(self, split):  # type: ignore[no-untyped-def] # pragma: no cover
        raise NotImplementedError

    def update_online(self, batch):  # type: ignore[no-untyped-def] # pragma: no cover
        raise NotImplementedError


def _entry(key: str) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(key),
        response=Response(f"response-{key}"),
        embedding=(0.0,),
    )


class EvictionPolicyUnitTests(unittest.TestCase):
    def test_build_eviction_policy_by_name(self) -> None:
        self.assertIsInstance(build_eviction_policy("fifo"), FIFOEvictionPolicy)
        self.assertIsInstance(build_eviction_policy("LRU"), LRUEvictionPolicy)
        self.assertIsInstance(build_eviction_policy(EvictionPolicyName.LFU), LFUEvictionPolicy)
        with self.assertRaises(ValueError):
            build_eviction_policy("nonsense")

    def test_insert_initializes_metadata(self) -> None:
        policy = FIFOEvictionPolicy()
        policy.record_insert(CacheKey("a"))
        meta = policy.metadata(CacheKey("a"))
        assert meta is not None
        self.assertEqual(meta.access_count, 1)
        self.assertEqual(meta.created_at, meta.last_accessed)
        self.assertEqual(len(policy), 1)

    def test_hit_updates_recency_and_count(self) -> None:
        policy = LRUEvictionPolicy()
        policy.record_insert(CacheKey("a"))
        before = policy.metadata(CacheKey("a"))
        assert before is not None
        before_tick = before.last_access_tick
        policy.record_hit(CacheKey("a"))
        after = policy.metadata(CacheKey("a"))
        assert after is not None
        self.assertEqual(after.access_count, 2)
        self.assertGreater(after.last_access_tick, before_tick)

    def test_fifo_selects_oldest_inserted_regardless_of_access(self) -> None:
        policy = FIFOEvictionPolicy()
        for key in ("a", "b", "c"):
            policy.record_insert(CacheKey(key))
        # Accessing the oldest entry must not protect it under FIFO.
        policy.record_hit(CacheKey("a"))
        self.assertEqual(policy.select_victim(), CacheKey("a"))

    def test_lru_selects_least_recently_used(self) -> None:
        policy = LRUEvictionPolicy()
        for key in ("a", "b", "c"):
            policy.record_insert(CacheKey(key))
        policy.record_hit(CacheKey("a"))  # a now most recently used
        # b is the least recently used (c was inserted after b).
        self.assertEqual(policy.select_victim(), CacheKey("b"))

    def test_lfu_selects_lowest_access_count(self) -> None:
        policy = LFUEvictionPolicy()
        for key in ("a", "b", "c"):
            policy.record_insert(CacheKey(key))
        policy.record_hit(CacheKey("a"))
        policy.record_hit(CacheKey("a"))
        policy.record_hit(CacheKey("b"))
        # counts: a=3, b=2, c=1 -> c is least frequently used.
        self.assertEqual(policy.select_victim(), CacheKey("c"))

    def test_lfu_ties_break_on_least_recently_used(self) -> None:
        policy = LFUEvictionPolicy()
        policy.record_insert(CacheKey("a"))
        policy.record_insert(CacheKey("b"))
        # Equal counts -> evict the older (least recently touched) entry.
        self.assertEqual(policy.select_victim(), CacheKey("a"))


class GatewayEvictionTests(unittest.TestCase):
    def _build(self, policy_name: str, *, max_entries: int) -> tuple[SemanticCacheGateway, InMemoryKVStore, FakeOracle]:
        kv = InMemoryKVStore()
        oracle = FakeOracle()
        gateway = SemanticCacheGateway(
            kv_store=kv,
            oracle=oracle,
            eviction_policy=build_eviction_policy(policy_name),
            max_entries=max_entries,
        )
        return gateway, kv, oracle

    def _hit(self, gateway: SemanticCacheGateway, key: str) -> bool:
        result = gateway.lookup_with_decision(CacheLookup(query=Query(key), embedding=(0.0,)))
        return result.response is not None

    def test_fifo_removes_oldest_inserted_entry(self) -> None:
        gateway, kv, oracle = self._build("fifo", max_entries=3)
        for key in ("a", "b", "c"):
            gateway.put(_entry(key))
        # Access does not influence FIFO.
        self.assertTrue(self._hit(gateway, "a"))
        self.assertTrue(self._hit(gateway, "b"))

        gateway.put(_entry("d"))  # exceeds capacity -> one eviction

        self.assertEqual(set(gateway.eviction_policy.tracked_keys()), {CacheKey("b"), CacheKey("c"), CacheKey("d")})
        self.assertFalse(kv.contains(CacheKey("a")))
        self.assertIn(CacheKey("a"), oracle.removed)
        self.assertNotIn(CacheKey("a"), oracle.live)

    def test_lru_removes_not_recently_accessed_entry(self) -> None:
        gateway, kv, oracle = self._build("lru", max_entries=3)
        for key in ("a", "b", "c"):
            gateway.put(_entry(key))
        # Touch a and b; c becomes the least recently used.
        self.assertTrue(self._hit(gateway, "a"))
        self.assertTrue(self._hit(gateway, "b"))

        gateway.put(_entry("d"))

        self.assertEqual(set(gateway.eviction_policy.tracked_keys()), {CacheKey("a"), CacheKey("b"), CacheKey("d")})
        self.assertFalse(kv.contains(CacheKey("c")))
        self.assertIn(CacheKey("c"), oracle.removed)

    def test_lfu_removes_lowest_access_count_entry(self) -> None:
        gateway, kv, oracle = self._build("lfu", max_entries=3)
        for key in ("a", "b", "c"):
            gateway.put(_entry(key))
        # counts after inserts: a=b=c=1. Bump a and b so c is the unique min.
        self.assertTrue(self._hit(gateway, "a"))
        self.assertTrue(self._hit(gateway, "a"))
        self.assertTrue(self._hit(gateway, "b"))

        gateway.put(_entry("d"))  # d has count 1, but c is older -> c evicted

        self.assertEqual(set(gateway.eviction_policy.tracked_keys()), {CacheKey("a"), CacheKey("b"), CacheKey("d")})
        self.assertFalse(kv.contains(CacheKey("c")))
        self.assertIn(CacheKey("c"), oracle.removed)

    def test_unbounded_cache_never_evicts(self) -> None:
        kv = InMemoryKVStore()
        oracle = FakeOracle()
        gateway = SemanticCacheGateway(kv_store=kv, oracle=oracle)  # no policy / no bound
        for key in ("a", "b", "c", "d", "e"):
            gateway.put(_entry(key))
        self.assertEqual(kv.size(), 5)
        self.assertEqual(oracle.removed, [])

    def test_invalidate_keeps_policy_in_sync(self) -> None:
        gateway, kv, _ = self._build("fifo", max_entries=3)
        gateway.put(_entry("a"))
        gateway.put(_entry("b"))
        gateway.invalidate(CacheKey("a"))
        self.assertEqual(set(gateway.eviction_policy.tracked_keys()), {CacheKey("b")})
        self.assertFalse(kv.contains(CacheKey("a")))

    def test_missing_kv_value_drops_policy_tracking(self) -> None:
        # If the external value vanished, the gateway self-heals by removing the
        # key from the oracle; the eviction policy must drop it too.
        gateway, kv, oracle = self._build("lru", max_entries=3)
        gateway.put(_entry("a"))
        kv.delete(CacheKey("a"))  # simulate external expiry
        result = gateway.lookup_with_decision(CacheLookup(query=Query("a"), embedding=(0.0,)))
        self.assertIsNone(result.response)
        self.assertNotIn(CacheKey("a"), gateway.eviction_policy.tracked_keys())
        self.assertIn(CacheKey("a"), oracle.removed)

    def test_rejects_non_positive_max_entries(self) -> None:
        with self.assertRaises(ValueError):
            SemanticCacheGateway(
                kv_store=InMemoryKVStore(),
                oracle=FakeOracle(),
                eviction_policy=build_eviction_policy("lru"),
                max_entries=0,
            )


if __name__ == "__main__":
    unittest.main()
