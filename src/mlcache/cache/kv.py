"""External response storage contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mlcache.semantic_types import CacheKey, Response


class KVStore(ABC):
    """Response storage outside the semantic oracle."""

    @abstractmethod
    def get(self, cache_key: CacheKey) -> Response | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, cache_key: CacheKey, response: Response) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, cache_key: CacheKey) -> None:
        raise NotImplementedError

    @abstractmethod
    def contains(self, cache_key: CacheKey) -> bool:
        raise NotImplementedError

