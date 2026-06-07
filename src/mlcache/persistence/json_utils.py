"""Minimal JSON serialization helpers for local persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from mlcache.semantic_types import (
    CacheKey,
    CacheMetadata,
    OracleDecisionStatus,
    Query,
    RegionId,
    Score,
    Threshold,
)


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Atomically write deterministic JSON to path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    encoded = json.dumps(json_safe(data), indent=2, sort_keys=True)
    tmp.write_text(f"{encoded}\n", encoding="utf-8")
    os.replace(tmp, target)


def read_json_or_default(path: str | Path, default: Any) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    """Convert common MLCache values into JSON-serializable structures."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_safe(getattr(value, field.name)) for field in fields(value)}
    return str(value)


def encode_cache_metadata(metadata: CacheMetadata) -> dict[str, Any]:
    return {
        "namespace": metadata.namespace,
        "tenant_id": metadata.tenant_id,
        "model": metadata.model,
        "region_id": None if metadata.region_id is None else str(metadata.region_id),
        "cluster_id": None if metadata.cluster_id is None else str(metadata.cluster_id),
        "created_at": _datetime_to_json(metadata.created_at),
        "expires_at": _datetime_to_json(metadata.expires_at),
        "attributes": json_safe(metadata.attributes),
    }


def decode_cache_metadata(data: dict[str, Any] | None) -> CacheMetadata:
    data = data or {}
    return CacheMetadata(
        namespace=data.get("namespace"),
        tenant_id=data.get("tenant_id"),
        model=data.get("model"),
        region_id=RegionId(data["region_id"]) if data.get("region_id") is not None else None,
        cluster_id=RegionId(data["cluster_id"]) if data.get("cluster_id") is not None else None,
        created_at=_datetime_from_json(data.get("created_at")),
        expires_at=_datetime_from_json(data.get("expires_at")),
        attributes=dict(data.get("attributes") or {}),
    )


def encode_query_record(record: QueryCalibrationRecord) -> dict[str, Any]:
    return {
        "query_id": str(record.query_id),
        "query": None if record.query is None else str(record.query),
        "candidates": [
            {
                "score": float(candidate.score),
                "label": candidate.label,
                "candidate_rank": candidate.candidate_rank,
                "candidate_key": (
                    str(candidate.candidate_key) if candidate.candidate_key is not None else None
                ),
                "metadata": json_safe(candidate.metadata),
            }
            for candidate in record.candidates
        ],
        "metadata": json_safe(record.metadata),
    }


def decode_query_record(data: dict[str, Any]) -> QueryCalibrationRecord:
    from mlcache.calibration.query_level import QueryCalibrationCandidate, QueryCalibrationRecord

    return QueryCalibrationRecord(
        query_id=str(data["query_id"]),
        query=Query(data["query"]) if data.get("query") is not None else None,
        candidates=tuple(
            QueryCalibrationCandidate(
                score=Score(float(candidate["score"])),
                label=candidate.get("label"),
                candidate_rank=candidate.get("candidate_rank"),
                candidate_key=(
                    CacheKey(candidate["candidate_key"])
                    if candidate.get("candidate_key") is not None
                    else None
                ),
                metadata=dict(candidate.get("metadata") or {}),
            )
            for candidate in data.get("candidates", ())
        ),
        metadata=dict(data.get("metadata") or {}),
    )


def encode_query_level_policy_decision(decision: QueryLevelPolicyDecision) -> dict[str, Any]:
    return {
        "status": decision.status.value,
        "accepted": bool(decision.accepted),
        "selected_candidate_key": (
            str(decision.selected_candidate_key) if decision.selected_candidate_key is not None else None
        ),
        "selected_candidate_rank": decision.selected_candidate_rank,
        "selected_score": None if decision.selected_score is None else float(decision.selected_score),
        "threshold": None if decision.threshold is None else float(decision.threshold),
        "reason": decision.reason,
        "metadata": json_safe(decision.metadata),
    }


def decode_query_level_policy_decision(data: dict[str, Any]) -> QueryLevelPolicyDecision:
    from mlcache.policies.query_level import QueryLevelPolicyDecision

    return QueryLevelPolicyDecision(
        status=OracleDecisionStatus(data["status"]),
        accepted=bool(data["accepted"]),
        selected_candidate_key=(
            CacheKey(data["selected_candidate_key"])
            if data.get("selected_candidate_key") is not None
            else None
        ),
        selected_candidate_rank=data.get("selected_candidate_rank"),
        selected_score=Score(float(data["selected_score"])) if data.get("selected_score") is not None else None,
        threshold=Threshold(float(data["threshold"])) if data.get("threshold") is not None else None,
        reason=data.get("reason"),
        metadata=dict(data.get("metadata") or {}),
    )


def _datetime_to_json(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _datetime_from_json(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
