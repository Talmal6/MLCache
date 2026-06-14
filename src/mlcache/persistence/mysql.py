"""MySQL-backed durable storage adapters for MLCache."""

from __future__ import annotations

import os
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

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
class MySQLStorageConfig:
    database_url: str
    faiss_shard_id: str = "default"

    @classmethod
    def from_env(cls) -> "MySQLStorageConfig":
        database_url = os.getenv("MLCACHE_MYSQL_DATABASE_URL") or os.getenv("MLCACHE_DATABASE_URL")
        if not database_url:
            raise ValueError("MLCACHE_MYSQL_DATABASE_URL or MLCACHE_DATABASE_URL is required for mysql storage")
        return cls(
            database_url=database_url,
            faiss_shard_id=os.getenv("MLCACHE_FAISS_SHARD_ID", "default"),
        )


class MySQLCacheRepository:
    """Shared MySQL access layer used by storage adapters."""

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
            else Path(__file__).with_name("mysql_migrations")
        )
        statements: list[str] = []
        for path in sorted(directory.glob("*.sql")):
            statements.extend(_split_sql_statements(path.read_text(encoding="utf-8")))
        with self._connect() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)

    def insert_pending_entry(self, entry: CacheEntry, *, faiss_id: int | None = None) -> int:
        cache_uuid, metadata = _entry_uuid_and_metadata(entry)
        embedding = tuple(float(value) for value in entry.embedding)
        now = _mysql_datetime(datetime.now(UTC))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cache_entries (
                        cache_key, faiss_id, namespace, tenant_id, model_name,
                        query_text, embedding_dim, embedding_blob, metadata, status,
                        created_at, expires_at, indexed_at, deleted_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING_INDEX',
                        COALESCE(%s, CURRENT_TIMESTAMP(6)), %s, NULL, NULL
                    )
                    ON DUPLICATE KEY UPDATE
                        namespace = VALUES(namespace),
                        tenant_id = VALUES(tenant_id),
                        model_name = VALUES(model_name),
                        query_text = VALUES(query_text),
                        embedding_dim = VALUES(embedding_dim),
                        embedding_blob = VALUES(embedding_blob),
                        metadata = VALUES(metadata),
                        status = 'PENDING_INDEX',
                        expires_at = VALUES(expires_at),
                        indexed_at = NULL,
                        deleted_at = NULL
                    """,
                    (
                        cache_uuid,
                        faiss_id,
                        entry.metadata.namespace,
                        entry.metadata.tenant_id,
                        entry.metadata.model,
                        str(entry.query),
                        len(embedding),
                        encode_float64_vector(embedding),
                        _json_dumps(metadata),
                        _mysql_datetime(entry.metadata.created_at),
                        _mysql_datetime(entry.metadata.expires_at),
                    ),
                )
                cur.execute("SELECT faiss_id FROM cache_entries WHERE cache_key = %s", (cache_uuid,))
                row = cur.fetchone()
                assigned_faiss_id = int(_row_get(row, "faiss_id", 0))
                cur.execute(
                    """
                    INSERT INTO cache_payloads (cache_key, content_type, response_text, created_at)
                    VALUES (%s, 'text/plain', %s, %s)
                    ON DUPLICATE KEY UPDATE
                        content_type = VALUES(content_type),
                        response_text = VALUES(response_text),
                        payload_json = NULL,
                        payload_bytes = NULL
                    """,
                    (cache_uuid, str(entry.response), now),
                )
                cur.execute(
                    """
                    INSERT INTO outbox_events (event_type, cache_key, faiss_id, shard_id, payload)
                    VALUES ('UPSERT_ENTRY', %s, %s, %s, %s)
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cache_payloads (cache_key, content_type, response_text)
                    VALUES (%s, 'text/plain', %s)
                    ON DUPLICATE KEY UPDATE
                        content_type = VALUES(content_type),
                        response_text = VALUES(response_text),
                        payload_json = NULL,
                        payload_bytes = NULL
                    """,
                    (cache_uuid, str(response)),
                )

    def get_payload(self, cache_key: CacheKey, *, require_active: bool = True) -> Response | None:
        cache_uuid = _cache_uuid(cache_key)
        status_filter = (
            "AND e.status = 'ACTIVE' AND (e.expires_at IS NULL OR e.expires_at > CURRENT_TIMESTAMP(6))"
        )
        if not require_active:
            status_filter = ""
        with self._connect() as conn:
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    f"""
                    SELECT p.response_text, p.payload_json, p.payload_bytes
                    FROM cache_entries e
                    JOIN cache_payloads p ON p.cache_key = e.cache_key
                    WHERE e.cache_key = %s {status_filter}
                    """,
                    (cache_uuid,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cur.execute(
                    """
                    UPDATE cache_entries
                    SET last_access_at = CURRENT_TIMESTAMP(6), hit_count = hit_count + 1
                    WHERE cache_key = %s
                    """,
                    (cache_uuid,),
                )
        return _response_from_payload(row)

    def contains_active_payload(self, cache_key: CacheKey) -> bool:
        return self.get_payload(cache_key) is not None

    def tombstone_entry(self, cache_key: CacheKey) -> None:
        cache_uuid = _cache_uuid(cache_key)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT faiss_id FROM cache_entries WHERE cache_key = %s", (cache_uuid,))
                row = cur.fetchone()
                faiss_id = int(_row_get(row, "faiss_id", 0)) if row is not None else None
                cur.execute(
                    """
                    UPDATE cache_entries
                    SET status = 'TOMBSTONED', deleted_at = CURRENT_TIMESTAMP(6)
                    WHERE cache_key = %s
                    """,
                    (cache_uuid,),
                )
                cur.execute(
                    """
                    INSERT INTO outbox_events (event_type, cache_key, faiss_id, shard_id, payload)
                    VALUES ('DELETE_ENTRY', %s, %s, %s, JSON_OBJECT())
                    """,
                    (cache_uuid, faiss_id, self.shard_id),
                )

    def get_vector_result(self, cache_key: CacheKey, *, require_active: bool = True) -> VectorSearchResult | None:
        cache_uuid = _cache_uuid(cache_key)
        status_filter = (
            "AND e.status = 'ACTIVE' AND (e.expires_at IS NULL OR e.expires_at > CURRENT_TIMESTAMP(6))"
        )
        if not require_active:
            status_filter = ""
        with self._connect() as conn:
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    f"""
                    SELECT e.cache_key, e.faiss_id, e.query_text, e.embedding_blob, e.metadata,
                           p.response_text, p.payload_json, p.payload_bytes
                    FROM cache_entries e
                    LEFT JOIN cache_payloads p ON p.cache_key = e.cache_key
                    WHERE e.cache_key = %s {status_filter}
                    """,
                    (cache_uuid,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return _row_to_vector_result(row, score=Score(1.0))

    def get_indexable_entry_by_faiss_id(self, faiss_id: int) -> VectorSearchResult | None:
        with self._connect() as conn:
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    """
                    SELECT e.cache_key, e.faiss_id, e.query_text, e.embedding_blob, e.metadata,
                           p.response_text, p.payload_json, p.payload_bytes
                    FROM cache_entries e
                    LEFT JOIN cache_payloads p ON p.cache_key = e.cache_key
                    WHERE e.faiss_id = %s AND e.status = 'PENDING_INDEX'
                    """,
                    (int(faiss_id),),
                )
                row = cur.fetchone()
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
        placeholders = ", ".join(["%s"] * len(faiss_ids))
        with self._connect() as conn:
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    f"""
                    SELECT e.cache_key, e.faiss_id, e.query_text, e.embedding_blob, e.metadata,
                           p.response_text, p.payload_json, p.payload_bytes
                    FROM cache_entries e
                    LEFT JOIN cache_payloads p ON p.cache_key = e.cache_key
                    WHERE e.faiss_id IN ({placeholders})
                      AND e.status = 'ACTIVE'
                      AND (e.expires_at IS NULL OR e.expires_at > CURRENT_TIMESTAMP(6))
                      AND (%s IS NULL OR e.namespace = %s)
                      AND (%s IS NULL OR e.tenant_id = %s)
                      AND (%s IS NULL OR e.model_name = %s)
                    """,
                    tuple(faiss_ids) + (namespace, namespace, tenant_id, tenant_id, model_name, model_name),
                )
                rows = cur.fetchall()
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
            where = "cache_key = %s"
            params = (_cache_uuid(cache_key),)
        else:
            where = "faiss_id = %s"
            params = (int(faiss_id),)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE cache_entries
                    SET status = 'ACTIVE', indexed_at = CURRENT_TIMESTAMP(6)
                    WHERE {where} AND status = 'PENDING_INDEX'
                    """,
                    params,
                )

    def pending_outbox_events(self, *, limit: int = 100) -> tuple[OutboxEvent, ...]:
        with self._connect() as conn:
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    """
                    SELECT event_id, event_type, cache_key, faiss_id, shard_id, payload
                    FROM outbox_events
                    WHERE processed_at IS NULL
                    ORDER BY event_id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall()
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
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE outbox_events SET processed_at = CURRENT_TIMESTAMP(6) WHERE event_id = %s",
                    (int(event_id),),
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
            with conn.cursor() as cur:
                if active:
                    cur.execute(
                        "UPDATE faiss_snapshots SET active = false WHERE shard_id = %s",
                        (shard_id,),
                    )
                cur.execute(
                    """
                    INSERT INTO faiss_snapshots (
                        shard_id, generation, index_type, snapshot_uri, checksum,
                        watermark_event_id, metadata, active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        shard_id,
                        int(generation),
                        index_type,
                        snapshot_uri,
                        checksum,
                        int(watermark_event_id),
                        _json_dumps(metadata or {}),
                        bool(active),
                    ),
                )
                return int(cur.lastrowid)

    def latest_active_snapshot(self, *, shard_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    """
                    SELECT snapshot_id, shard_id, generation, index_type, snapshot_uri,
                           checksum, watermark_event_id, metadata, active, created_at
                    FROM faiss_snapshots
                    WHERE shard_id = %s AND active = true
                    ORDER BY generation DESC, snapshot_id DESC
                    LIMIT 1
                    """,
                    (shard_id,),
                )
                row = cur.fetchone()
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT threshold_version_id
                    FROM threshold_versions
                    WHERE scorer_name = %s AND scope = %s AND active = true
                    ORDER BY created_at DESC, threshold_version_id DESC
                    LIMIT 1
                    """,
                    (str(scorer), ThresholdScope(scope).value),
                )
                row = cur.fetchone()
        return int(_row_get(row, 0, 0)) if row is not None else None

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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_events (
                        query_id, namespace, tenant_id, model_name, query_text, query_embedding,
                        top_k, decision_status, accepted, accepted_cache_key, score, threshold,
                        scorer_name, threshold_version_id, reason, evidence, latency_ms
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        namespace = VALUES(namespace),
                        tenant_id = VALUES(tenant_id),
                        model_name = VALUES(model_name),
                        query_text = VALUES(query_text),
                        query_embedding = VALUES(query_embedding),
                        top_k = VALUES(top_k),
                        decision_status = VALUES(decision_status),
                        accepted = VALUES(accepted),
                        accepted_cache_key = VALUES(accepted_cache_key),
                        score = VALUES(score),
                        threshold = VALUES(threshold),
                        scorer_name = VALUES(scorer_name),
                        threshold_version_id = VALUES(threshold_version_id),
                        reason = VALUES(reason),
                        evidence = VALUES(evidence),
                        latency_ms = VALUES(latency_ms)
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
                        bool(decision.accepted),
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
                cur.execute("DELETE FROM query_candidates WHERE query_id = %s", (query_id,))
                for item in candidates:
                    cur.execute(
                        """
                        INSERT INTO query_candidates (
                            query_id, candidate_rank, cache_key, faiss_id, vector_score,
                            scorer_score, label, candidate_metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_events (
                        query_id, query_text, top_k, decision_status, accepted, evidence
                    )
                    VALUES (%s, %s, %s, 'CALIBRATION_RECORD', false, %s)
                    ON DUPLICATE KEY UPDATE
                        query_text = VALUES(query_text),
                        top_k = VALUES(top_k),
                        decision_status = VALUES(decision_status),
                        accepted = VALUES(accepted),
                        evidence = VALUES(evidence)
                    """,
                    (
                        query_id,
                        str(record.query) if record.query is not None else "",
                        len(record.candidates),
                        _json_dumps(record.metadata),
                    ),
                )
                cur.execute("DELETE FROM query_candidates WHERE query_id = %s", (query_id,))
                for idx, candidate in enumerate(record.candidates, start=1):
                    rank = candidate.candidate_rank if candidate.candidate_rank is not None else idx
                    cur.execute(
                        """
                        INSERT INTO query_candidates (
                            query_id, candidate_rank, cache_key, scorer_score, label, candidate_metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
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
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    """
                    SELECT s.query_id, s.query_text, s.evidence,
                           c.candidate_rank, c.cache_key, c.scorer_score, c.vector_score,
                           c.label, c.candidate_metadata
                    FROM (
                        SELECT query_id, query_text, evidence, created_at
                        FROM query_events
                        WHERE decision_status = 'CALIBRATION_RECORD'
                        ORDER BY created_at DESC
                        LIMIT %s
                    ) AS s
                    LEFT JOIN query_candidates c ON c.query_id = s.query_id
                    ORDER BY s.created_at ASC, c.candidate_rank ASC
                    """,
                    (int(max_records),),
                )
                rows = cur.fetchall()
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
            with conn.cursor() as cur:
                cur.execute("DELETE FROM query_events WHERE decision_status = 'CALIBRATION_RECORD'")

    def add_feedback_pair(self, example: JudgedPairExample, *, split_name: str | None = None) -> None:
        label = _label_to_int(example.decision.label)
        if label is None:
            return
        metadata = _encode_judged_pair_metadata(example)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback_pairs (
                        query_id, candidate_cache_key, label, features_blob, split_name, metadata, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        _query_uuid_text(example.request.context.get("query_id")),
                        _cache_uuid(example.request.candidate_key) if example.request.candidate_key is not None else None,
                        label,
                        encode_float64_vector(example.features),
                        split_name,
                        _json_dumps(metadata),
                        _mysql_datetime(example.created_at),
                    ),
                )

    def feedback_pairs(self, *, label: int, split_name: str | None | object = _UNSET) -> tuple[JudgedPairExample, ...]:
        params: list[Any] = [int(label)]
        split_sql = ""
        if split_name is not _UNSET:
            split_sql = "AND split_name <=> %s"
            params.append(split_name)
        with self._connect() as conn:
            with conn.cursor(_dict_cursor()) as cur:
                cur.execute(
                    f"""
                    SELECT label, features_blob, split_name, metadata, created_at
                    FROM feedback_pairs
                    WHERE label = %s {split_sql}
                    ORDER BY created_at ASC, pair_id ASC
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()
        return tuple(_row_to_judged_pair(row) for row in rows)

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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE threshold_versions
                    SET active = false
                    WHERE scorer_name = %s
                      AND scope = %s
                      AND region_id <=> %s
                      AND cluster_id <=> %s
                      AND active = true
                    """,
                    (
                        str(scorer),
                        scope_value,
                        None if region_id is None else str(region_id),
                        None if cluster_id is None else str(cluster_id),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO threshold_versions (
                        scorer_name, scope, region_id, cluster_id, threshold,
                        calibrated, active, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, true, %s)
                    """,
                    (
                        str(scorer),
                        scope_value,
                        None if region_id is None else str(region_id),
                        None if cluster_id is None else str(cluster_id),
                        float(threshold),
                        bool(calibrated),
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT threshold
                    FROM threshold_versions
                    WHERE scorer_name = %s
                      AND scope = %s
                      AND region_id <=> %s
                      AND cluster_id <=> %s
                      AND active = true
                    ORDER BY created_at DESC, threshold_version_id DESC
                    LIMIT 1
                    """,
                    (
                        str(scorer),
                        ThresholdScope(scope).value,
                        None if region_id is None else str(region_id),
                        None if cluster_id is None else str(cluster_id),
                    ),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(f"threshold not found for scorer={scorer} scope={scope}")
        return Threshold(float(_row_get(row, 0, "threshold")))

    @contextmanager
    def _connect(self) -> Any:
        owns_connection = self.connection_factory is None
        conn = self.connection_factory() if self.connection_factory is not None else _connect_mysql_url(self.database_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if owns_connection:
                conn.close()


class MySQLKVStore(KVStore):
    """KV adapter backed by `cache_payloads`, with `cache_entries` as truth."""

    def __init__(self, repository: MySQLCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or MySQLCacheRepository(database_url)

    def get(self, cache_key: CacheKey) -> Response | None:
        return self.repository.get_payload(cache_key, require_active=True)

    def set(self, cache_key: CacheKey, response: Response) -> None:
        self.repository.set_payload(cache_key, response)

    def delete(self, cache_key: CacheKey) -> None:
        self.repository.tombstone_entry(cache_key)

    def contains(self, cache_key: CacheKey) -> bool:
        return self.repository.contains_active_payload(cache_key)


class MySQLThresholdProvider(ThresholdProvider):
    """Threshold provider backed by `threshold_versions`."""

    def __init__(self, repository: MySQLCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or MySQLCacheRepository(database_url)

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


class MySQLQueryCalibrationRecordStore(QueryCalibrationRecordStore):
    """Query calibration record store backed by query audit tables."""

    def __init__(
        self,
        repository: MySQLCacheRepository | None = None,
        *,
        database_url: str | None = None,
        max_records: int = 100_000,
    ) -> None:
        if int(max_records) <= 0:
            raise ValueError("max_records must be positive")
        self.repository = repository or MySQLCacheRepository(database_url)
        self.max_records = int(max_records)

    def add(self, record: QueryCalibrationRecord) -> None:
        self.repository.add_query_calibration_record(record)

    def records(self) -> tuple[QueryCalibrationRecord, ...]:
        return self.repository.query_calibration_records(max_records=self.max_records)

    def clear(self) -> None:
        self.repository.clear_query_calibration_records()


class MySQLJudgeTrainingStore(JudgeTrainingStore):
    """Judge training store backed by `feedback_pairs`."""

    def __init__(self, repository: MySQLCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or MySQLCacheRepository(database_url)

    def add(self, example: JudgedPairExample) -> None:
        self.repository.add_feedback_pair(example)

    def h0(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=0)

    def h1(self) -> tuple[JudgedPairExample, ...]:
        return self.repository.feedback_pairs(label=1)


class MySQLSplitJudgeTrainingStore(SplitJudgeTrainingStore):
    """Split judge training store backed by `feedback_pairs.split_name`."""

    def __init__(self, repository: MySQLCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or MySQLCacheRepository(database_url)

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


class MySQLOutboxStore:
    """Small outbox adapter over `outbox_events`."""

    def __init__(self, repository: MySQLCacheRepository | None = None, *, database_url: str | None = None) -> None:
        self.repository = repository or MySQLCacheRepository(database_url)

    def pending_events(self, *, limit: int = 100) -> tuple[OutboxEvent, ...]:
        return self.repository.pending_outbox_events(limit=limit)

    def mark_processed(self, event_id: int) -> None:
        self.repository.mark_outbox_processed(event_id)


def _pymysql() -> Any:
    try:
        import pymysql
    except ImportError as exc:
        raise ImportError("MySQL storage requires: pip install -e '.[mysql]'") from exc
    return pymysql


def _dict_cursor() -> Any:
    try:
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise ImportError("MySQL storage requires: pip install -e '.[mysql]'") from exc
    return DictCursor


def _connect_mysql_url(database_url: str | None) -> Any:
    if not database_url:
        raise ValueError("database_url is required")
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError("MySQL database URL must use mysql:// or mysql+pymysql://")
    query = parse_qs(parsed.query)
    database = parsed.path.lstrip("/") or None
    charset = query.get("charset", ["utf8mb4"])[0]
    return _pymysql().connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        database=unquote(database) if database else None,
        charset=charset,
        autocommit=False,
    )


def _mysql_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _query_uuid_text(query_id: object | None) -> str | None:
    value = _query_uuid(query_id)
    return None if value is None else str(value)


def _row_get(row: Any, key: str | int, fallback_key: str | int) -> Any:
    if isinstance(row, dict):
        return row[key] if key in row else row[fallback_key]
    if isinstance(key, int):
        return row[key]
    if isinstance(fallback_key, int):
        return row[fallback_key]
    return row[0]


__all__ = [
    "MySQLCacheRepository",
    "MySQLJudgeTrainingStore",
    "MySQLKVStore",
    "MySQLOutboxStore",
    "MySQLQueryCalibrationRecordStore",
    "MySQLSplitJudgeTrainingStore",
    "MySQLStorageConfig",
    "MySQLThresholdProvider",
]
