"""Threshold provider implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlcache.calibration.np_threshold import ThresholdProvider
from mlcache.calibration.types import ThresholdScope
from mlcache.persistence import atomic_write_json, json_safe, read_json_or_default
from mlcache.semantic_types import RegionId, ScorerName, Threshold


class InMemoryThresholdProvider(ThresholdProvider):
    """Deterministic in-memory threshold provider."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}

    def get_threshold(
        self,
        *,
        scorer: ScorerName,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
        region_id: RegionId | None = None,
        cluster_id: RegionId | None = None,
        context: dict[str, Any] | None = None,
    ) -> Threshold:
        del context
        key = self._key(scorer=scorer, scope=scope, region_id=region_id, cluster_id=cluster_id)
        if key not in self._records:
            raise KeyError(f"threshold not found for scorer={scorer} scope={scope}")
        return Threshold(float(self._records[key]["threshold"]))

    def set_threshold(
        self,
        threshold: Threshold,
        *,
        scorer: ScorerName,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
        region_id: RegionId | None = None,
        cluster_id: RegionId | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        key = self._key(scorer=scorer, scope=scope, region_id=region_id, cluster_id=cluster_id)
        self._records[key] = {
            "threshold": float(threshold),
            "scorer": str(scorer),
            "scope": ThresholdScope(scope).value,
            "region_id": None if region_id is None else str(region_id),
            "cluster_id": None if cluster_id is None else str(cluster_id),
            "context": json_safe(context or {}),
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def clear(self) -> None:
        self._records.clear()

    def size(self) -> int:
        return len(self._records)

    @staticmethod
    def _key(
        *,
        scorer: ScorerName,
        scope: ThresholdScope,
        region_id: RegionId | None,
        cluster_id: RegionId | None,
    ) -> tuple[str, str, str | None, str | None]:
        return (
            str(scorer),
            ThresholdScope(scope).value,
            None if region_id is None else str(region_id),
            None if cluster_id is None else str(cluster_id),
        )


class FileThresholdProvider(InMemoryThresholdProvider):
    """JSON-backed threshold provider for local runtime restart safety."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__()
        data = read_json_or_default(self.path, {"records": []})
        for record in data.get("records", ()):
            key = self._key(
                scorer=ScorerName(str(record["scorer"])),
                scope=ThresholdScope(record.get("scope", ThresholdScope.GLOBAL.value)),
                region_id=RegionId(record["region_id"]) if record.get("region_id") is not None else None,
                cluster_id=RegionId(record["cluster_id"]) if record.get("cluster_id") is not None else None,
            )
            self._records[key] = dict(record)

    def set_threshold(
        self,
        threshold: Threshold,
        *,
        scorer: ScorerName,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
        region_id: RegionId | None = None,
        cluster_id: RegionId | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().set_threshold(
            threshold,
            scorer=scorer,
            scope=scope,
            region_id=region_id,
            cluster_id=cluster_id,
            context=context,
        )
        self._persist()

    def clear(self) -> None:
        super().clear()
        self._persist()

    def _persist(self) -> None:
        atomic_write_json(
            self.path,
            {
                "format": "mlcache.file_threshold_provider.v1",
                "records": [
                    self._records[key]
                    for key in sorted(self._records)
                ],
            },
        )


__all__ = ["FileThresholdProvider", "InMemoryThresholdProvider"]
