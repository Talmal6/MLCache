"""File-backed local cache store."""

from __future__ import annotations

from pathlib import Path

from mlcache.cache.in_memory import InMemoryKVStore
from mlcache.persistence import atomic_write_json, read_json_or_default
from mlcache.semantic_types import CacheKey, Response


class FileKVStore(InMemoryKVStore):
    """JSON-backed response store for local experiments."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = read_json_or_default(self.path, {"responses": {}})
        super().__init__(
            {
                CacheKey(str(key)): Response(str(value))
                for key, value in dict(data.get("responses") or {}).items()
            }
        )

    def set(self, cache_key: CacheKey, response: Response) -> None:
        super().set(cache_key, response)
        self._persist()

    def delete(self, cache_key: CacheKey) -> None:
        super().delete(cache_key)
        self._persist()

    def clear(self) -> None:
        super().clear()
        self._persist()

    def _persist(self) -> None:
        atomic_write_json(
            self.path,
            {
                "format": "mlcache.file_kv_store.v1",
                "responses": {
                    str(key): str(self._responses[key])
                    for key in sorted(self._responses, key=str)
                },
            },
        )


__all__ = ["FileKVStore"]
