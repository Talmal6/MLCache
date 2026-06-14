from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

from mlcache import (
    CacheEntry,
    CacheKey,
    CacheMetadata,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgedPairExample,
    MySQLCacheRepository,
    MySQLJudgeTrainingStore,
    MySQLKVStore,
    MySQLOutboxStore,
    MySQLQueryCalibrationRecordStore,
    MySQLThresholdProvider,
    PostgresCacheRepository,
    PostgresJudgeTrainingStore,
    PostgresKVStore,
    PostgresQueryCalibrationRecordStore,
    PostgresThresholdProvider,
    Query,
    QueryCalibrationCandidate,
    QueryCalibrationRecord,
    Response,
    Score,
    ScorerName,
    Threshold,
    ThresholdScope,
)
from mlcache.persistence.mysql import _connect_mysql_url


pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class SQLBackend:
    name: str
    repository_type: type
    kv_type: type
    threshold_type: type
    query_store_type: type
    judge_store_type: type


BACKENDS = (
    SQLBackend(
        name="postgres",
        repository_type=PostgresCacheRepository,
        kv_type=PostgresKVStore,
        threshold_type=PostgresThresholdProvider,
        query_store_type=PostgresQueryCalibrationRecordStore,
        judge_store_type=PostgresJudgeTrainingStore,
    ),
    SQLBackend(
        name="mysql",
        repository_type=MySQLCacheRepository,
        kv_type=MySQLKVStore,
        threshold_type=MySQLThresholdProvider,
        query_store_type=MySQLQueryCalibrationRecordStore,
        judge_store_type=MySQLJudgeTrainingStore,
    ),
)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend.name)
def test_sql_backend_contract_preserves_source_of_truth_status(backend: SQLBackend) -> None:
    with _fresh_database_url(backend) as database_url:
        repository = backend.repository_type(database_url, shard_id="contract")
        repository.apply_migrations()
        kv_store = backend.kv_type(repository)
        entry = _entry()

        faiss_id = repository.insert_pending_entry(entry)

        assert kv_store.get(entry.cache_key) is None
        assert not repository.fetch_active_results_by_faiss_ids(
            [(faiss_id, 1.0)],
            namespace="contract",
            metadata=CacheMetadata(namespace="contract", model="contract-model"),
        )
        assert backend.repository_type(database_url, shard_id="contract").pending_outbox_events(limit=10)[0].event_type == "UPSERT_ENTRY"

        repository.mark_entry_indexed(faiss_id=faiss_id)

        assert kv_store.get(entry.cache_key) == Response("contract response")
        active_results = repository.fetch_active_results_by_faiss_ids(
            [(faiss_id, 1.0)],
            namespace="contract",
            metadata=CacheMetadata(namespace="contract", model="contract-model"),
        )
        assert [result.cache_key for result in active_results] == [entry.cache_key]

        repository.tombstone_entry(entry.cache_key)

        assert kv_store.get(entry.cache_key) is None
        assert not repository.fetch_active_results_by_faiss_ids(
            [(faiss_id, 1.0)],
            namespace="contract",
            metadata=CacheMetadata(namespace="contract", model="contract-model"),
        )


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend.name)
def test_sql_backend_contract_persists_threshold_query_records_judge_pairs_and_outbox(
    backend: SQLBackend,
) -> None:
    with _fresh_database_url(backend) as database_url:
        repository = backend.repository_type(database_url, shard_id="contract")
        repository.apply_migrations()

        threshold_provider = backend.threshold_type(repository)
        threshold_provider.set_threshold(
            Threshold(0.42),
            scorer=ScorerName("contract-scorer"),
            scope=ThresholdScope.GLOBAL,
        )
        reloaded_threshold_provider = backend.threshold_type(backend.repository_type(database_url, shard_id="contract"))
        assert reloaded_threshold_provider.get_threshold(
            scorer=ScorerName("contract-scorer"),
            scope=ThresholdScope.GLOBAL,
        ) == Threshold(0.42)

        query_store = backend.query_store_type(repository)
        query_store.add(_query_record())
        reloaded_query_store = backend.query_store_type(backend.repository_type(database_url, shard_id="contract"))
        records = reloaded_query_store.records()
        assert len(records) == 1
        assert records[0].candidates[0].candidate_key == CacheKey("00000000-0000-0000-0000-000000000002")

        judge_store = backend.judge_store_type(repository)
        judge_store.add(_judged_pair(JudgeLabel.REUSABLE))
        judge_store.add(_judged_pair(JudgeLabel.NOT_REUSABLE))
        reloaded_judge_store = backend.judge_store_type(backend.repository_type(database_url, shard_id="contract"))
        assert len(reloaded_judge_store.h1()) == 1
        assert len(reloaded_judge_store.h0()) == 1

        entry = _entry()
        repository.insert_pending_entry(entry)
        outbox = MySQLOutboxStore(repository) if backend.name == "mysql" else None
        events = outbox.pending_events(limit=10) if outbox is not None else repository.pending_outbox_events(limit=10)
        assert len(events) == 1
        if outbox is not None:
            outbox.mark_processed(events[0].event_id)
        else:
            repository.mark_outbox_processed(events[0].event_id)
        assert not repository.pending_outbox_events(limit=10)


@contextmanager
def _fresh_database_url(backend: SQLBackend):
    if backend.name == "postgres":
        if importlib.util.find_spec("psycopg") is None:
            pytest.skip("requires psycopg; install with .[postgres]")
        admin_url = os.getenv("MLCACHE_TEST_ADMIN_DATABASE_URL")
        if not admin_url:
            pytest.skip("requires MLCACHE_TEST_ADMIN_DATABASE_URL for postgres storage contract tests")
        import psycopg
        from psycopg import sql

        db_name = f"mlcache_contract_{uuid.uuid4().hex}"
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        try:
            yield _database_url_for_name(admin_url, db_name)
        finally:
            with psycopg.connect(admin_url, autocommit=True) as conn:
                conn.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (db_name,),
                )
                conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
        return

    if importlib.util.find_spec("pymysql") is None:
        pytest.skip("requires pymysql; install with .[mysql]")
    admin_url = os.getenv("MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL")
    if not admin_url:
        pytest.skip("requires MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL for mysql storage contract tests")
    db_name = f"mlcache_contract_{uuid.uuid4().hex}"
    with _connect_mysql_url(admin_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
        conn.commit()
    try:
        yield _database_url_for_name(admin_url, db_name)
    finally:
        with _connect_mysql_url(admin_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
            conn.commit()


def _database_url_for_name(admin_url: str, database_name: str) -> str:
    parsed = urlsplit(admin_url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment))


def _entry() -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey("00000000-0000-0000-0000-000000000001"),
        query=Query("contract query"),
        response=Response("contract response"),
        embedding=(1.0, 0.0),
        metadata=CacheMetadata(namespace="contract", model="contract-model"),
    )


def _query_record() -> QueryCalibrationRecord:
    return QueryCalibrationRecord(
        query_id="00000000-0000-0000-0000-000000000003",
        query=Query("contract incoming"),
        candidates=(
            QueryCalibrationCandidate(
                score=Score(0.9),
                label=1,
                candidate_rank=1,
                candidate_key=CacheKey("00000000-0000-0000-0000-000000000002"),
                metadata={"source": "contract"},
            ),
        ),
        metadata={"suite": "contract"},
    )


def _judged_pair(label: JudgeLabel) -> JudgedPairExample:
    return JudgedPairExample(
        features=(0.1, 0.2, 0.3),
        request=JudgeRequest(
            query=Query(f"judge query {label.value}"),
            candidate_query=Query("candidate query"),
            candidate_response=Response("candidate response"),
            candidate_key=CacheKey(str(uuid.uuid4())),
            context={"query_id": str(uuid.uuid4())},
        ),
        decision=JudgeDecision(label=label, confidence=Score(0.8)),
        metadata={"suite": "contract"},
    )
