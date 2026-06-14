"""Server-free storage-contract tests for the SQLite backend.

The Postgres/MySQL equivalents live in `test_sql_storage_contracts.py` and are
marked `integration` because they need a running database server. SQLite needs
no server, so these run by default and guard the same source-of-truth and
persistence behaviour for the new backend.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from mlcache import (
    CacheEntry,
    CacheKey,
    CacheMetadata,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgedPairExample,
    Query,
    QueryCalibrationCandidate,
    QueryCalibrationRecord,
    Response,
    Score,
    ScorerName,
    SQLiteCacheRepository,
    SQLiteJudgeTrainingStore,
    SQLiteKVStore,
    SQLiteOutboxStore,
    SQLiteQueryCalibrationRecordStore,
    SQLiteThresholdProvider,
    Threshold,
    ThresholdScope,
)


def _database_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'contract.db').as_posix()}"


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


def test_sqlite_backend_contract_preserves_source_of_truth_status(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    repository = SQLiteCacheRepository(database_url, shard_id="contract")
    repository.apply_migrations()
    kv_store = SQLiteKVStore(repository)
    entry = _entry()
    metadata = CacheMetadata(namespace="contract", model="contract-model")

    faiss_id = repository.insert_pending_entry(entry)

    # PENDING_INDEX: not yet servable and not yet active in the index.
    assert kv_store.get(entry.cache_key) is None
    assert not repository.fetch_active_results_by_faiss_ids([(faiss_id, 1.0)], namespace="contract", metadata=metadata)
    assert repository.pending_outbox_events(limit=10)[0].event_type == "UPSERT_ENTRY"

    repository.mark_entry_indexed(faiss_id=faiss_id)

    # ACTIVE: servable and resolvable through the source of truth.
    assert kv_store.get(entry.cache_key) == Response("contract response")
    active = repository.fetch_active_results_by_faiss_ids([(faiss_id, 1.0)], namespace="contract", metadata=metadata)
    assert [result.cache_key for result in active] == [entry.cache_key]

    repository.tombstone_entry(entry.cache_key)

    # TOMBSTONED: no longer servable, no longer active.
    assert kv_store.get(entry.cache_key) is None
    assert not repository.fetch_active_results_by_faiss_ids([(faiss_id, 1.0)], namespace="contract", metadata=metadata)


def test_sqlite_backend_contract_persists_threshold_query_records_judge_pairs_and_outbox(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path)
    repository = SQLiteCacheRepository(database_url, shard_id="contract")
    repository.apply_migrations()

    SQLiteThresholdProvider(repository).set_threshold(
        Threshold(0.42),
        scorer=ScorerName("contract-scorer"),
        scope=ThresholdScope.GLOBAL,
    )
    reloaded_threshold = SQLiteThresholdProvider(SQLiteCacheRepository(database_url, shard_id="contract"))
    assert reloaded_threshold.get_threshold(
        scorer=ScorerName("contract-scorer"),
        scope=ThresholdScope.GLOBAL,
    ) == Threshold(0.42)

    SQLiteQueryCalibrationRecordStore(repository).add(_query_record())
    reloaded_query_store = SQLiteQueryCalibrationRecordStore(SQLiteCacheRepository(database_url, shard_id="contract"))
    records = reloaded_query_store.records()
    assert len(records) == 1
    assert records[0].candidates[0].candidate_key == CacheKey("00000000-0000-0000-0000-000000000002")

    judge_store = SQLiteJudgeTrainingStore(repository)
    judge_store.add(_judged_pair(JudgeLabel.REUSABLE))
    judge_store.add(_judged_pair(JudgeLabel.NOT_REUSABLE))
    reloaded_judge_store = SQLiteJudgeTrainingStore(SQLiteCacheRepository(database_url, shard_id="contract"))
    assert len(reloaded_judge_store.h1()) == 1
    assert len(reloaded_judge_store.h0()) == 1

    repository.insert_pending_entry(_entry())
    outbox = SQLiteOutboxStore(repository)
    events = outbox.pending_events(limit=10)
    assert len(events) == 1
    outbox.mark_processed(events[0].event_id)
    assert not repository.pending_outbox_events(limit=10)
