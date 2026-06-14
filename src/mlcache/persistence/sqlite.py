"""SQLite-backed durable storage adapters for MLCache.

SQLite needs no separate server process, so this backend gives the same
Postgres/MySQL source-of-truth contract (PENDING_INDEX -> ACTIVE -> TOMBSTONED
entries, an outbox that feeds FAISS, threshold/feedback/query-audit tables) while
running entirely in-process against a local database file. It reuses the encoding
and row-mapping helpers from the Postgres adapter; only the SQL dialect differs.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlcache.cache.kv import KVStore
from mlcache.calibration.np_threshold import ThresholdProvider
from mlcache.calibration.query_level import QueryCalibrationCandidate, QueryCalibrationRecord
from mlcache.calibration.query_record_store import QueryCalibrationRecordStore
from mlcache.calibration.types import ThresholdScope
from mlcache.feedback.store import JudgeTrainingStore, JudgedPairExample, SplitJudgeTrainingStore
from mlcache.persistence.binary import encode_float64_vector
from mlcache.persistence.postgres import (
    OutboxEvent,
    _UNSET,
    _cache_uuid,
    _candidate_evidence,
    _dict_from_json,
    _encode_judged_pair_metadata,
    _entry_uuid_and_metadata,
    _json_dumps,
    _label_to_int,
    _query_uuid,
    _response_from_payload,
    _row_to_judged_pair,
    _row_to_vector_result,
    _split_sql_statements,
)
from mlcache.retrieval.vector_store import VectorSearchResult
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    CacheMetadata,
    OracleDecision,
    Query,
    RegionId,
    Response,
    Score,
    ScorerName,
    Threshold,
)

ConnectionFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class SQLiteStorageConfig:
    database_url: str
    faiss_shard_id: str = "default"

    @classmethod
    def from_env(cls) -> "SQLiteStorageConfig":
        database_url = os.getenv("MLCACHE_SQLITE_DATABASE_URL") or os.getenv("MLCACHE_DATABASE_URL")
        if not database_url:
            raise ValueError("MLCACHE_SQLITE_DATABASE_URL or MLCACHE_DATABASE_URL is required for sqlite storage")
        return cls(
            database_url=database_url,
            faiss_shard_id=os.getenv("MLCACHE_FAISS_SHARD_ID", "default"),
        )


class SQLiteCacheRepository:
    """Shared SQLite access layer used by storage adapters."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        connection_factory: ConnectionFactory | None = None,
        shard_id: str = "default",
    ) -> None:
        if connection_factory is None and not database_url:
            raise ValueError("database_url or connection_factory is required")
        self.database_url = database_url
        self.connection_factory = connection_factory
        self.shard_id = shard_id

    def apply_migrations(self, *, migrations_dir: str | Path | None = None) -> None:
        directory = (
            Path(migrations_dir)
            if migrations_dir is not None
            else Path(__file__).with_name("sqlite_migrations")
        )
        statements: list[str] = []
        for path in sorted(directory.glob("*.sql")):
            statements.extend(_split_sql_statements(path.read_text(encoding="utf-8")))
        with self._connect() as conn:
            for statement in statements:
                conn.execute(statement)

    def insert_pending_entry(self, entry: CacheEntry, *, faiss_id: int | None = None) -> int:
        cache_uuid, metadata = _entry_uuid_and_metadata(entry)
        embedding = tuple(float(value) for value in entry.embedding)
        now = _sqlite_now()
        created_at = _sqlite_datetime(entry.metadata.created_at) or now
        expires_at = _sqlite_datetime(entry.metadata.expires_at)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT faiss_id FROM cache_entries WHERE cache_key = ?", (cache_uuid,)
            ).fetchone()
            if existing is not None:
                assigned_faiss_id = int(existing["faiss_id"])
                conn.execute(
                    """
                    UPDATE cache_entries SET
                        namespace = ?, tenant_id = ?, model_name = ?, query_text = ?,
                        embedding_dim = ?, embedding_blob = ?, metadata = ?, status = 'PENDING_INDEX',
                        expires_at = ?, indexed_at = NULL, deleted_at = NULL
                    WHERE cache_key = ?
                    """,
                    (
                        entry.metadata.namespace,
                        entry.metadata.tenant_id,
                        entry.metadata.model,
                        str(entry.query),
                        len(embedding),
                        encode_float64_vector(embedding),
                        _json_dumps(metadata),
                        expires_at,
                        cache_uuid,
                    ),
                )
            else:
                assigned_faiss_id = int(faiss_id) if faiss_id is not None else self._next_faiss_id(conn)
                conn.execute(
                    """
                    INSERT INTO cache_entries (
                        cache_key, faiss_id, namespace, tenant_id, model_name,
                        query_text, embedding_dim, embedding_blob, metadata, status,
                        created_at, expires_at, indexed_at, deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_INDEX', ?, ?, NULL, NULL)
                    """,
                    (
                        cache_uuid,
                        assigned_faiss_id,
                        entry.metadata.namespace,
                        entry.metadata.tenant_id,
                        entry.metadata.model,
                        str(entry.query),
                        len(embedding),
                        encode_float64_vector(embedding),
                        _json_dumps(metadata),
                        created_at,
                        expires_at,
                    ),
                )
            conn.execute(
                """
                INSERT INTO cache_payloads (cache_key, content_type, response_text, created_at)
                VALUES (?, 'text/plain', ?, ?)
                ON CONFLICT (cache_key) DO UPDATE SET
                    content_type = excluded.content_type,
                    response_text = excluded.response_text,
                    payload_json = NULL,
                    payload_bytes = NULL
                """,
                (cache_uuid, str(entry.response), now),
            )
            conn.execute(
                """
                INSERT INTO outbox_events (event_type, cache_key, faiss_id, shard_id, payload)
                VALUES ('UPSERT_ENTRY', ?, ?, ?, ?)
                """,
                (
                    cache_uuid,
                    assigned_faiss_id,
                    self.shard_id,
                    _json_dumps(
                        {
                            "embedding_dim": len(embedding),
                            "namespace": entry.metadata.namespace,
                            "tenant_id": entry.metadata.tenant_id,
                            "model_name": entry.metadata.model,
                        }
                    ),
                ),
            )
        return assigned_faiss_id

    def set_payload(self, cache_key: CacheKey, response: Response) -> None:
        cache_uuid = _cache_uuid(cache_key)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_payloads (cache_key, content_type, response_text, created_at)
                VALUES (?, 'text/plain', ?, ?)
                ON CONFLICT (cache_key) DO UPDATE SET
                    content_type = excluded.content_type,
                    response_text = excluded.response_text,
                    payload_json = NULL,
                    payload_bytes = NULL
                """,
                (cache_uuid, str(response), _sqlite_now()),
            )

    def get_payload(self, cache_key: CacheKey, *, require_active: bool = True) -> Response | None:
        cache_uuid = _cache_uuid(cache_key)
        now = _sqlite_now()
        if require_active:
            status_filter = "AND e.status = 'ACTIVE' AND (e.expires_at IS NULL OR e.expires_at > ?)"
            params: tuple[Any, ...] = (cache_uuid, now)
        else:
            status_filter = ""
            params = (cache_uuid,)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT p.response_text, p.payload_json, p.payload_bytes
                FROM cache_entries e
                JOIN cache_payloads p ON p.cache_key = e.cache_key
                WHERE e.cache_key = ? {status_filter}
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE cache_entries
                SET last_access_at = ?, hit_count = hit_count + 1
                WHERE cache_key = ?
                """,
                (now, cache_uuid),
            )
        return _response_from_payload(row)

    def contains_active_payload(self, cache_key: CacheKey) -> bool:
        return self.get_payload(cache_key) is not None

    def tombstone_entry(self, cache_key: CacheKey) -> None:
        cache_uuid = _cache_uuid(cache_key)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT faiss_id FROM cache_entries WHERE cache_key = ?", (cache_uuid,)
            ).fetchone()
            faiss_id = int(row["faiss_id"]) if row is not None else None
            conn.execute(
                """
                UPDATE cache_entries
                SET status = 'TOMBSTONED', deleted_at = ?
                WHERE cache_key = ?
                """,
                (_sqlite_now(), cache_uuid),
            )
            conn.execute(
                """
                INSERT INTO outbox_events (event_type, cache_key, faiss_id, shard_id, payload)
                VALUES ('DELETE_ENTRY', ?, ?, ?, '{}')
                """,
                (cache_uuid, faiss_id, self.shard_id),
            )

    def get_vector_result(self, cache_key: CacheKey, *, require_active: bool = True) -> VectorSearchResult | None:
        cache_uuid = _cache_uuid(cache_key)
        now = _sqlite_now()
        if require_active:
            status_filter = "AND e.status = 'ACTIVE' AND (e.expires_at IS NULL OR e.expires_at > ?)"
            params: tuple[Any, ...] = (cache_uuid, now)
        else:
            status_filter = ""
            params = (cache_uuid,)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT e.cache_key, e.faiss_id, e.query_text, e.embedding_blob, e.metadata,
                       p.response_text, p.payload_json, p.payload_bytes
                FROM cache_entries e
                LEFT JOIN cache_payloads p ON p.cache_key = e.cache_key
                WHERE e.cache_key = ? {status_filter}
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        return _row_to_vector_result(row, score=Score(1.0))

    def get_indexable_entry_by_faiss_id(self, faiss_id: int) -> VectorSearchResult | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT e.cache_key, e.faiss_id, e.query_text, e.embedding_blob, e.metadata,
                       p.response_text, p.payload_json, p.payload_bytes
                FROM cache_entries e
                LEFT JOIN cache_payloads p ON p.cache_key = e.cache_key
                WHERE e.faiss_id = ? AND e.status = 'PENDING_INDEX'
                """,
                (int(faiss_id),),
            ).fetchone()
        if row is None:
            return None
        return _row_to_vector_result(row, score=Score(1.0))

    def fetch_active_results_by_faiss_ids(
        self,
        faiss_scores: Sequence[tuple[int, float]],
        *,
        namespace: str | None = None,
        metadata: CacheMetadata | None = None,
    ) -> list[VectorSearchResult]:
        if not faiss_scores:
            return []
        faiss_ids = [int(item[0]) for item in faiss_scores]
        score_by_id = {int(faiss_id): float(score) for faiss_id, score in faiss_scores}
        namespace = namespace if namespace is not None else (metadata.namespace if metadata is not None else None)
        tenant_id = metadata.tenant_id if metadata is not None else None
        model_name = metadata.model if metadata is not None else None
        placeholders = ", ".join(["?"] * len(faiss_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.cache_key, e.faiss_id, e.query_text, e.embedding_blob, e.metadata,
                       p.response_text, p.payload_json, p.payload_bytes
                FROM cache_entries e
                LEFT JOIN cache_payloads p ON p.cache_key = e.cache_key
                WHERE e.faiss_id IN ({placeholders})
                  AND e.status = 'ACTIVE'
                  AND (e.expires_at IS NULL OR e.expires_at > ?)
                  AND (? IS NULL OR e.namespace = ?)
                  AND (? IS NULL OR e.tenant_id = ?)
                  AND (? IS NULL OR e.model_name = ?)
                """,
                (
                    *faiss_ids,
                    _sqlite_now(),
                    namespace,
                    namespace,
                    tenant_id,
                    tenant_id,
                    model_name,
                    model_name,
                ),
            ).fetchall()
        row_by_id = {int(row["faiss_id"]): row for row in rows}
        results: list[VectorSearchResult] = []
        for faiss_id in faiss_ids:
            row = row_by_id.get(faiss_id)
            if row is None:
                continue
            results.append(_row_to_vector_result(row, score=Score(score_by_id[faiss_id])))
        return results

    def mark_entry_indexed(self, *, cache_key: CacheKey | None = None, faiss_id: int | None = None) -> None:
        if cache_key is None and faiss_id is None:
            raise ValueError("cache_key or faiss_id is required")
        if cache_key is not None:
            where = "cache_key = ?"
            params: tuple[Any, ...] = (_cache_uuid(cache_key),)
        else:
            where = "faiss_id = ?"
            params = (int(faiss_id),)
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE cache_entries
                SET status = 'ACTIVE', indexed_at = ?
                WHERE {where} AND status = 'PENDING_INDEX'
                """,
                (_sqlite_now(), *params),
            )

    def pending_outbox_events(self, *, limit: int = 100) -> tuple[OutboxEvent, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, event_type, cache_key, faiss_id, shard_id, payload
                FROM outbox_events
                WHERE processed_at IS NULL
                ORDER BY event_id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return tuple(
            OutboxEvent(
                event_id=int(row["event_id"]),
                event_type=str(row["event_type"]),
                cache_key=CacheKey(str(row["cache_key"])) if row["cache_key"] is not None else None,
                faiss_id=int(row["faiss_id"]) if row["faiss_id"] is not None else None,
                shard_id=row["shard_id"],
                payload=_dict_from_json(row["payload"]),
            )
            for row in rows
        )

    def mark_outbox_processed(self, event_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE outbox_events SET processed_at = ? WHERE event_id = ?",
                (_sqlite_now(), int(event_id)),
            )

    def insert_faiss_snapshot(
        self,
        *,
        shard_id: str,
        generation: int,
        index_type: str,
        snapshot_uri: str,
        checksum: str,
        watermark_event_id: int,
        metadata: dict[str, Any] | None = None,
        active: bool = True,
    ) -> int:
        with self._connect() as conn:
            if active:
                conn.execute(
                    "UPDATE faiss_snapshots SET active = 0 WHERE shard_id = ?",
                    (shard_id,),
                )
            cur = conn.execute(
                """
                INSERT INTO faiss_snapshots (
                    shard_id, generation, index_type, snapshot_uri, checksum,
                    watermark_event_id, metadata, active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shard_id,
                    int(generation),
                    index_type,
                    snapshot_uri,
                    checksum,
                    int(watermark_event_id),
                    _json_dumps(metadata or {}),
                    1 if active else 0,
                ),
            )
            return int(cur.lastrowid)

    def latest_active_snapshot(self, *, shard_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT snapshot_id, shard_id, generation, index_type, snapshot_uri,
                       checksum, watermark_event_id, metadata, active, created_at
                FROM faiss_snapshots
                WHERE shard_id = ? AND active = 1
                ORDER BY generation DESC, snapshot_id DESC
                LIMIT 1
                """,
                (shard_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def get_active_threshold_version_id(
        self,
        *,
        scorer: ScorerName | str | None,
        scope: ThresholdScope = ThresholdScope.GLOBAL,
    ) -> int | None:
        if scorer is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT threshold_version_id
                FROM threshold_versions
                WHERE scorer_name = ? AND scope = ? AND active = 1
                ORDER BY created_at DESC, threshold_version_id DESC
                LIMIT 1
                """,
                (str(scorer), ThresholdScope(scope).value),
            ).fetchone()
        return int(row["threshold_version_id"]) if row is not None else None

    def record_query_event(
        self,
        request: CacheLookup,
        decision: OracleDecision,
        *,
        latency_ms: int | None = None,
    ) -> str:
        query_id = _query_uuid_text(
            request.metadata.attributes.get("query_id") or request.metadata.attributes.get("request_id")
        )
        if query_id is None:
            query_id = str(uuid.uuid4())
        candidates = _candidate_evidence(decision)
        threshold_version_id = self.get_active_threshold_version_id(scorer=decision.scorer)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_events (
                    query_id, namespace, tenant_id, model_name, query_text, query_embedding,
                    top_k, decision_status, accepted, accepted_cache_key, score, threshold,
                    scorer_name, threshold_version_id, reason, evidence, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (query_id) DO UPDATE SET
                    namespace = excluded.namespace,
                    tenant_id = excluded.tenant_id,
                    model_name = excluded.model_name,
                    query_text = excluded.query_text,
                    query_embedding = excluded.query_embedding,
                    top_k = excluded.top_k,
                    decision_status = excluded.decision_status,
                    accepted = excluded.accepted,
                    accepted_cache_key = excluded.accepted_cache_key,
                    score = excluded.score,
                    threshold = excluded.threshold,
                    scorer_name = excluded.scorer_name,
                    threshold_version_id = excluded.threshold_version_id,
                    reason = excluded.reason,
                    evidence = excluded.evidence,
                    latency_ms = excluded.latency_ms
                """,
                (
                    query_id,
                    request.namespace or request.metadata.namespace,
                    request.metadata.tenant_id,
                    request.metadata.model,
                    str(request.query),
                    encode_float64_vector(tuple(float(v) for v in request.embedding)),
                    len(candidates),
                    decision.status.value,
                    1 if decision.accepted else 0,
                    _cache_uuid(decision.cache_key) if decision.cache_key is not None else None,
                    float(decision.score) if decision.score is not None else None,
                    float(decision.threshold) if decision.threshold is not None else None,
                    str(decision.scorer) if decision.scorer is not None else None,
                    threshold_version_id,
                    decision.reason,
                    _json_dumps(decision.evidence),
                    latency_ms,
                ),
            )
            conn.execute("DELETE FROM query_candidates WHERE query_id = ?", (query_id,))
            for item in candidates:
                conn.execute(
                    """
                    INSERT INTO query_candidates (
                        query_id, candidate_rank, cache_key, faiss_id, vector_score,
                        scorer_score, label, candidate_metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        int(item["rank"]),
                        _cache_uuid(CacheKey(str(item["cache_key"]))) if item.get("cache_key") else None,
                        int(item["faiss_id"]) if item.get("faiss_id") is not None else None,
                        float(item["vector_score"]) if item.get("vector_score") is not None else None,
                        float(item["score"]) if item.get("score") is not None else None,
                        item.get("label"),
                        _json_dumps(item.get("metadata") or {}),
                    ),
                )
        return str(query_id)

    def add_query_calibration_record(self, record: QueryCalibrationRecord) -> None:
        query_id = _query_uuid_text(record.query_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_events (
                    query_id, query_text, top_k, decision_status, accepted, evidence
                )
                VALUES (?, ?, ?, 'CALIBRATION_RECORD', 0, ?)
                ON CONFLICT (query_id) DO UPDATE SET
                    query_text = excluded.query_text,
                    top_k = excluded.top_k,
                    decision_status = excluded.decision_status,
                    accepted = excluded.accepted,
                    evidence = excluded.evidence
                """,
                (
                    query_id,
                    str(record.query) if record.query is not None else "",
                    len(record.candidates),
                    _json_dumps(record.metadata),
                ),
            )
            conn.execute("DELETE FROM query_candidates WHERE query_id = ?", (query_id,))
            for idx, candidate in enumerate(record.candidates, start=1):
                rank = candidate.candidate_rank if candidate.candidate_rank is not None else idx
                conn.execute(
                    """
                    INSERT INTO query_candidates (
                        query_id, candidate_rank, cache_key, scorer_score, label, candidate_metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        int(rank),
                        _cache_uuid(candidate.candidate_key) if candidate.candidate_key is not None else None,
                        float(candidate.score),
                        candidate.label,
                        _json_dumps(candidate.metadata),
                    ),
                )

    def query_calibration_records(self, *, max_records: int = 100_000) -> tuple[QueryCalibrationRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.query_id, s.query_text, s.evidence,
                       c.candidate_rank, c.cache_key, c.scorer_score, c.vector_score,
                       c.label, c.candidate_metadata
                FROM (
                    SELECT query_id, query_text, evidence, created_at
                    FROM query_events
                    WHERE decision_status = 'CALIBRATION_RECORD'
                    ORDER BY created_at DESC
                    LIMIT ?
                ) AS s
                LEFT JOIN query_candidates c ON c.query_id = s.query_id
                ORDER BY s.created_at ASC, c.candidate_rank ASC
                """,
                (int(max_records),),
            ).fetchall()
        grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for row in rows:
            item = grouped.setdefault(
                str(row["query_id"]),
                {
                    "query": row["query_text"],
                    "metadata": _dict_from_json(row["evidence"]),
                    "candidates": [],
                },
            )
            if row["candidate_rank"] is None:
                continue
            score_value = row["scorer_score"] if row["scorer_score"] is not None else row["vector_score"]
            item["candidates"].append(
                QueryCalibrationCandidate(
                    score=Score(float(score_value or 0.0)),
                    label=row["label"],
                    candidate_rank=int(row["candidate_rank"]),
                    candidate_key=CacheKey(str(row["cache_key"])) if row["cache_key"] is not None else None,
                    metadata=_dict_from_json(row["candidate_metadata"]),
                )
            )
        return tuple(
            QueryCalibrationRecord(
                query_id=query_id,
                query=Query(data["query"]) if data["query"] else None,
                candidates=tuple(data["candidates"]),
                metadata=data["metadata"],
            )
            for query_id, data in grouped.items()
        )

    def clear_query_calibration_records(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM query_events WHERE decision_status = 'CALIBRATION_RECORD'")

    def add_feedback_pair(self, example: JudgedPairExample, *, split_name: str | None = None) -> None:
        label = _label_to_int(example.decision.label)
        if label is None:
            return
        metadata = _encode_judged_pair_metadata(example)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_pairs (
                    query_id, candidate_cache_key, label, features_blob, split_name, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _query_uuid_text(example.request.context.get("query_id")),
                    _cache_uuid(example.request.candidate_key) if example.request.candidate_key is not None else None,
                    label,
                    encode_float64_vector(example.features),
                    split_name,
                    _json_dumps(metadata),
                    _sqlite_datetime(example.created_at),
                ),
            )

    def feedback_pairs(self, *, label: int, split_name: str | None | object = _UNSET) -> tuple[JudgedPairExample, ...]:
        params: list[Any] = [int(label)]
        split_sql = ""
        if split_name is not _UNSET:
            split_sql = "AND split_name IS ?"
            params.append(split_name)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT label, features_blob, split_name, metadata, created_at
                FROM feedback_pairs
                WHERE label = ? {split_sql}
                ORDER BY created_at ASC, pair_id ASC
                """,
                tuple(params),
            ).fetchall()
        results: list[JudgedPairExample] = []
        for row in rows:
            row = dict(row)
            row["created_at"] = _parse_sqlite_datetime(row["created_at"])
            results.append(_row_to_judged_pair(row))
        return tuple(results)

    def set_threshold(
        self,
        threshold: Threshold,
        *,
        scorer: ScorerName,
        scope: ThresholdScope,
        region_id: RegionId | None = None,
        cluster_id: RegionId | None = None,
        metadata: dict[str, Any] | None = None,
        calibrated: bool = True,
    ) -> int:
        scope_value = ThresholdScope(scope).value
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE threshold_versions
                SET active = 0
                WHERE scorer_name = ?
                  AND scope = ?
                  AND region_id IS ?
                  AND cluster_id IS ?
                  AND active = 1
                """,
                (
                    str(scorer),
                    scope_value,
                    None if region_id is None else str(region_id),
                    None if cluster_id is None else str(cluster_id),
                ),
            )
            cur = conn.execute(
                """
                INSERT INTO threshold_versions (
                    scorer_name, scope, region_id, cluster_id, threshold,
                    calibrated, active, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    str(scorer),
                    scope_value,
                    None if region_id is None else str(region_id),
                    None if cluster_id is None else str(cluster_id),
                    float(threshold),
                    1 if calibrated else 0,
                    _json_dumps(metadata or {}),
                ),
            )
            return int(cur.lastrowid)

    def get_threshold(
        self,
        *,
        scorer: ScorerName,
        scope: ThresholdScope,
        region_id: RegionId | None = None,
        cluster_id: RegionId | None = None,
    ) -> Threshold:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT threshold
                FROM threshold_versions
                WHERE scorer_name = ?
                  AND scope = ?
                  AND region_id IS ?
                  AND cluster_id IS ?
                  AND active = 1
                ORDER BY created_at DESC, threshold_version_id DESC
                LIMIT 1
                """,
                (
                    str(scorer),
                    ThresholdScope(scope).value,
                    None if region_id is None else str(region_id),
                    None if cluster_id is None else str(cluster_id),
                ),
            ).fetchone()
        if row is None:
            raise KeyError(f"threshold not found for scorer={scorer} scope={scope}")
        return Threshold(float(row["threshold"]))

    def _next_faiss_id(self, conn: Any) -> int:
        cur = conn.execute("INSERT INTO faiss_id_seq DEFAULT VALUES")
        return int(cur.lastrowid)

    @contextmanager
    def _connect(self) -> Any:
        owns_connection = self.connection_factory is None
        conn = self.connection_factory() if self.connection_factory is not None else _connect_sqlite_url(self.database_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()


class SQLiteKVStore(KVStore):
    """KV adapter backed by `cache_payloads`, with `cache_entries` as truth."""

    def __init__(self, repository: SQLiteCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or SQLiteCacheRepository(database_url)

    def get(self, cache_key: CacheKey) -> Response | None:
        return self.repository.get_payload(cache_key, require_active=True)

    def set(self, cache_key: CacheKey, response: Response) -> None:
        self.repository.set_payload(cache_key, response)

    def delete(self, cache_key: CacheKey) -> None:
        self.repository.tombstone_entry(cache_key)

    def contains(self, cache_key: CacheKey) -> bool:
        return self.repository.contains_active_payload(cache_key)


class SQLiteThresholdProvider(ThresholdProvider):
    """Threshold provider backed by `threshold_versions`."""

    def __init__(self, repository: SQLiteCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or SQLiteCacheRepository(database_url)

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
        return self.repository.get_threshold(
            scorer=scorer,
            scope=scope,
            region_id=region_id,
            cluster_id=cluster_id,
        )

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
        self.repository.set_threshold(
            threshold,
            scorer=scorer,
            scope=scope,
            region_id=region_id,
            cluster_id=cluster_id,
            metadata=context or {},
        )


class SQLiteQueryCalibrationRecordStore(QueryCalibrationRecordStore):
    """Query calibration record store backed by query audit tables."""

    def __init__(
        self,
        repository: SQLiteCacheRepository | None = None,
        *,
        database_url: str | None = None,
        max_records: int = 100_000,
    ) -> None:
        if int(max_records) <= 0:
            raise ValueError("max_records must be positive")
        self.repository = repository or SQLiteCacheRepository(database_url)
        self.max_records = int(max_records)

    def add(self, record: QueryCalibrationRecord) -> None:
        self.repository.add_query_calibration_record(record)

    def records(self) -> tuple[QueryCalibrationRecord, ...]:
        return self.repository.query_calibration_records(max_records=self.max_records)

    def clear(self) -> None:
        self.repository.clear_query_calibration_records()


class SQLiteJudgeTrainingStore(JudgeTrainingStore):
    """Judge training store backed by `feedback_pairs`."""

    def __init__(self, repository: SQLiteCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or SQLiteCacheRepository(database_url)

    def add(self, example: JudgedPairExample) -> None:
        self.repository.add_feedback_pair(example)

    def h0(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=0)

    def h1(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=1)


class SQLiteSplitJudgeTrainingStore(SplitJudgeTrainingStore):
    """Split judge training store backed by `feedback_pairs.split_name`."""

    def __init__(self, repository: SQLiteCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or SQLiteCacheRepository(database_url)

    def add_train(self, example: JudgedPairExample) -> None:
        self.repository.add_feedback_pair(example, split_name="train")

    def add_calibration(self, example: JudgedPairExample) -> None:
        self.repository.add_feedback_pair(example, split_name="calibration")

    def h0(self) -> tuple[JudgedPairExample, ...]:
        return self.h0_train() + self.h0_calibration()

    def h1(self) -> tuple[JudgedPairExample, ...]:
        return self.h1_train() + self.h1_calibration()

    def h0_train(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=0, split_name="train")

    def h1_train(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=1, split_name="train")

    def h0_calibration(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=0, split_name="calibration")

    def h1_calibration(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=1, split_name="calibration")


class SQLiteOutboxStore:
    """Small outbox adapter over `outbox_events`."""

    def __init__(self, repository: SQLiteCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or SQLiteCacheRepository(database_url)

    def pending_events(self, *, limit: int = 100) -> tuple[OutboxEvent, ...]:
        return self.repository.pending_outbox_events(limit=limit)

    def mark_processed(self, event_id: int) -> None:
        self.repository.mark_outbox_processed(event_id)


def _dict_factory(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    return {column[0]: row[index] for index, column in enumerate(cursor.description)}


def _sqlite_path(database_url: str | None) -> str:
    if not database_url:
        raise ValueError("database_url is required")
    if not database_url.startswith("sqlite:"):
        return database_url
    rest = database_url[len("sqlite:") :]
    if rest.startswith("///"):
        path = rest[3:]
    elif rest.startswith("//"):
        path = rest[2:]
    else:
        path = rest.lstrip("/")
    if path in ("", ":memory:"):
        return ":memory:"
    # "sqlite:///C:/path" splits to "/C:/path" on some forms; drop the slash
    # that precedes a Windows drive letter so sqlite gets a usable path.
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return path


def _connect_sqlite_url(database_url: str | None) -> sqlite3.Connection:
    path = _sqlite_path(database_url)
    if path != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.row_factory = _dict_factory
    conn.execute("PRAGMA foreign_keys = ON")
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _sqlite_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def _sqlite_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")


def _parse_sqlite_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def _query_uuid_text(query_id: object | None) -> str | None:
    value = _query_uuid(query_id)
    return None if value is None else str(value)


__all__ = [
    "SQLiteCacheRepository",
    "SQLiteJudgeTrainingStore",
    "SQLiteKVStore",
    "SQLiteOutboxStore",
    "SQLiteQueryCalibrationRecordStore",
    "SQLiteSplitJudgeTrainingStore",
    "SQLiteStorageConfig",
    "SQLiteThresholdProvider",
]
