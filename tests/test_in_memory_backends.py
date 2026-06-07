import unittest

from mlcache.cache import InMemoryKVStore
from mlcache.calibration import InMemoryThresholdProvider, ThresholdScope
from mlcache.retrieval import InMemoryVectorStore
from mlcache.semantic_types import CacheEntry, CacheKey, CacheMetadata, Query, Response, ScorerName, Threshold


def entry(
    key: str,
    embedding: tuple[float, ...],
    *,
    namespace: str | None = None,
) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(f"query {key}"),
        response=Response(f"response {key}"),
        embedding=embedding,
        metadata=CacheMetadata(namespace=namespace),
    )


class InMemoryBackendTests(unittest.TestCase):
    def test_in_memory_kv_store_get_set_delete_contains(self) -> None:
        store = InMemoryKVStore()

        store.set(CacheKey("a"), Response("response a"))

        self.assertTrue(store.contains(CacheKey("a")))
        self.assertEqual(store.get(CacheKey("a")), Response("response a"))
        self.assertEqual(store.size(), 1)
        store.delete(CacheKey("a"))
        self.assertFalse(store.contains(CacheKey("a")))
        self.assertIsNone(store.get(CacheKey("a")))
        self.assertEqual(store.size(), 0)

    def test_in_memory_vector_store_upsert_search_delete(self) -> None:
        store = InMemoryVectorStore()
        store.upsert(entry("a", (1.0, 0.0)))

        results = store.search((1.0, 0.0), top_k=1)

        self.assertEqual(results[0].cache_key, CacheKey("a"))
        self.assertIsNotNone(store.get(CacheKey("a")))
        self.assertEqual(store.size(), 1)
        store.delete(CacheKey("a"))
        self.assertEqual(store.search((1.0, 0.0)), [])
        self.assertEqual(store.size(), 0)

    def test_in_memory_vector_store_returns_closest_vector_first(self) -> None:
        store = InMemoryVectorStore()
        store.upsert(entry("far", (0.0, 1.0)))
        store.upsert(entry("near", (0.99, 0.01)))
        store.upsert(entry("tie-a", (1.0, 0.0)))
        store.upsert(entry("tie-b", (1.0, 0.0)))

        results = store.search((1.0, 0.0), top_k=4)

        self.assertEqual([str(result.cache_key) for result in results], ["tie-a", "tie-b", "near", "far"])

    def test_in_memory_vector_store_namespace_filtering(self) -> None:
        store = InMemoryVectorStore()
        store.upsert(entry("a", (1.0, 0.0), namespace="left"))
        store.upsert(entry("b", (1.0, 0.0), namespace="right"))

        results = store.search((1.0, 0.0), namespace="right")

        self.assertEqual([result.cache_key for result in results], [CacheKey("b")])

    def test_in_memory_threshold_provider_stores_global_threshold(self) -> None:
        provider = InMemoryThresholdProvider()
        provider.set_threshold(Threshold(0.7), scorer=ScorerName("s"), scope=ThresholdScope.GLOBAL)

        self.assertEqual(
            provider.get_threshold(scorer=ScorerName("s"), scope=ThresholdScope.GLOBAL),
            Threshold(0.7),
        )
        self.assertEqual(provider.size(), 1)


if __name__ == "__main__":
    unittest.main()
