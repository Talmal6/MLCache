"""Example: running the semantic cache with each eviction policy.

The eviction policy decides *what to remove* when the cache exceeds
``max_entries``. It is independent of the semantic HIT/MISS decision (which the
scorer/oracle owns) — this example only varies which entry is dropped once the
cache is full.

Run with the project interpreter:

    .conda/bin/python scripts/run_eviction_policy_example.py

It builds three bounded caches (``max_entries=2``), replays the same access
pattern through each, and prints which entries survive under LRU, LFU, and
FIFO.
"""

from __future__ import annotations

import tempfile

from mlcache import MLCache
from mlcache.embeddings import HashingEmbeddingProvider
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, Query, Response

EMBEDDER = HashingEmbeddingProvider(dimensions=64)

# Three distinct, semantically-unrelated entries so each lookup only ever hits
# its own key (we are demonstrating eviction, not semantic reuse).
ITEMS: dict[str, str] = {
    "alpha": "Alpha is the first response.",
    "beta": "Beta is the second response.",
    "gamma": "Gamma is the third response.",
}


def _entry(key: str) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(key),
        response=Response(ITEMS[key]),
        embedding=EMBEDDER.embed(key),
    )


def _lookup(cache: MLCache, key: str) -> bool:
    result = cache.lookup_with_decision(
        CacheLookup(query=Query(key), embedding=EMBEDDER.embed(key))
    )
    return result.response is not None


def run_policy(policy: str) -> tuple[str, ...]:
    """Replay one access pattern under `policy` and return the surviving keys."""

    with tempfile.TemporaryDirectory(prefix=f"mlcache-evict-{policy}-") as root:
        cache = MLCache.from_preset(
            root_dir=root,
            scorer="cosine",
            threshold=0.99,          # exact-embedding match => HIT
            persistence=False,        # in-memory backend
            eviction_policy=policy,   # "lru" | "lfu" | "fifo"
            max_entries=2,            # capacity bound that triggers eviction
        )

        # Insert two entries, then access "alpha" so it is both the most
        # recently used and the most frequently used entry.
        cache.put(_entry("alpha"))
        cache.put(_entry("beta"))
        assert _lookup(cache, "alpha"), "expected a HIT on the exact embedding"

        # Insert a third entry into a size-2 cache => exactly one eviction.
        cache.put(_entry("gamma"))

        surviving = cache.runtime.gateway.eviction_policy.tracked_keys()
        return tuple(sorted(str(key) for key in surviving))


def main() -> None:
    print("Access pattern: put alpha, put beta, HIT alpha, put gamma (max_entries=2)\n")
    explanations = {
        "fifo": "drops the oldest *inserted* entry -> alpha",
        "lru": "drops the least *recently used* entry -> beta (alpha was just hit)",
        "lfu": "drops the least *frequently used* entry -> beta (alpha hit twice)",
    }
    for policy in ("fifo", "lru", "lfu"):
        survivors = run_policy(policy)
        print(f"{policy.upper():5s} survivors={survivors}  # {explanations[policy]}")


if __name__ == "__main__":
    main()
