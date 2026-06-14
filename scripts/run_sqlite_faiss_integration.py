"""Run the real SQLite + FAISS MLCache integration harness.

SQLite needs no separate database server, so this harness creates a throwaway
database file under the work directory, applies migrations, and exercises the
same PENDING_INDEX -> ACTIVE -> TOMBSTONED lifecycle (with a FAISS outbox
indexer and snapshot reload) that the Postgres/MySQL harnesses cover.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from mlcache import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    CacheMetadata,
    FaissOutboxIndexer,
    MLCache,
    Query,
    Response,
    SQLiteCacheRepository,
    Threshold,
)
from mlcache.persistence.sqlite import _connect_sqlite_url


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _assert_optional_dependencies_available()
        work_dir = Path(args.work_dir)
        root = work_dir / f"mlcache-sqlite-faiss-{uuid.uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=False)
        database_url = args.database_url or f"sqlite:///{(root / 'mlcache.db').as_posix()}"
        result = run_harness(database_url=database_url, root_dir=root)
        result["pytest_command_recommendation"] = _pytest_command_recommendation()
        report_path = Path(args.report_path) if args.report_path else root / "sqlite_faiss_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        result["report_path"] = str(report_path)
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.getenv("MLCACHE_DEBUG"):
            raise
        return 1
    return 0


def run_harness(*, database_url: str, root_dir: Path) -> dict[str, Any]:
    faiss_path = root_dir / "faiss.index"
    os.environ.update(
        {
            "MLCACHE_STORAGE_BACKEND": "sqlite",
            "MLCACHE_VECTOR_BACKEND": "faiss",
            "MLCACHE_DATABASE_URL": database_url,
            "MLCACHE_SQLITE_DATABASE_URL": database_url,
            "MLCACHE_FAISS_DIM": "2",
            "MLCACHE_FAISS_INDEX_PATH": str(faiss_path),
            "MLCACHE_FAISS_METRIC": "cosine",
            "MLCACHE_FAISS_SHARD_ID": "integration",
            "MLCACHE_TOP_K": "5",
        }
    )

    _assert_no_user_tables(database_url)
    repository = SQLiteCacheRepository(database_url, shard_id="integration")
    try:
        repository.apply_migrations()
    except Exception as exc:
        raise RuntimeError(f"SQLite migrations failed: {exc}") from exc

    cache = MLCache.from_preset(root_dir=root_dir / "state", scorer="cosine", persistence=True)
    cache.set_threshold(Threshold(0.0))
    repository = cache.runtime.vector_store.repository
    assert repository is not None

    entry = _entry()
    cache.put(entry)
    pending_status = _entry_status(database_url, entry.cache_key)
    assert pending_status == "PENDING_INDEX", pending_status
    outbox_pending_before = _pending_outbox_count(database_url)
    assert outbox_pending_before == 1, outbox_pending_before

    pending_lookup = cache.lookup_with_decision(_lookup())
    assert not pending_lookup.decision.accepted, pending_lookup

    processed = FaissOutboxIndexer(repository=repository, vector_store=cache.runtime.vector_store).process_once()
    assert processed == 1, processed
    active_status = _entry_status(database_url, entry.cache_key)
    assert active_status == "ACTIVE", active_status

    active_lookup = cache.lookup_with_decision(_lookup())
    assert active_lookup.decision.accepted, active_lookup
    assert active_lookup.response == Response("response-active"), active_lookup
    _assert_hit_audit_row(database_url, entry.cache_key)

    snapshot = cache.runtime.vector_store.save_snapshot(faiss_path)
    restarted = MLCache.from_preset(root_dir=root_dir / "state-restarted", scorer="cosine", persistence=True)
    restarted.set_threshold(Threshold(0.0))
    restarted_lookup = restarted.lookup_with_decision(_lookup())
    assert restarted_lookup.decision.accepted, restarted_lookup
    assert restarted_lookup.response == Response("response-active"), restarted_lookup

    repository.tombstone_entry(entry.cache_key)
    tombstone_status = _entry_status(database_url, entry.cache_key)
    assert tombstone_status == "TOMBSTONED", tombstone_status
    tombstoned_lookup = restarted.lookup_with_decision(_lookup())
    assert not tombstoned_lookup.decision.accepted, tombstoned_lookup

    return {
        "database_url": database_url,
        "migration_command": "SQLiteCacheRepository.apply_migrations()",
        "faiss_snapshot": snapshot,
        "faiss_snapshot_path": snapshot.get("snapshot_uri"),
        "pending_status": pending_status,
        "active_status": active_status,
        "tombstone_status": tombstone_status,
        "outbox_processed": processed,
        "hit_cache_key": str(active_lookup.decision.cache_key),
        "restarted_hit_cache_key": str(restarted_lookup.decision.cache_key),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv("MLCACHE_TEST_SQLITE_DATABASE_URL") or os.getenv("MLCACHE_SQLITE_DATABASE_URL"),
        help="sqlite:///path/to/file.db; defaults to a throwaway file under --work-dir",
    )
    parser.add_argument("--work-dir", default="runs/sqlite_faiss_integration")
    parser.add_argument("--report-path")
    return parser.parse_args(argv)


def _assert_optional_dependencies_available() -> None:
    if importlib.util.find_spec("faiss") is None:
        extras = "python -m pip install -e \".[sqlite,faiss,dev]\""
        raise RuntimeError(f"missing optional dependencies: faiss; install with {extras}")


def _entry() -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(str(uuid.uuid4())),
        query=Query("anchor"),
        response=Response("response-active"),
        embedding=(1.0, 0.0),
        metadata=CacheMetadata(namespace="integration", model="test-model"),
    )


def _lookup() -> CacheLookup:
    return CacheLookup(
        query=Query("incoming"),
        embedding=(1.0, 0.0),
        namespace="integration",
        metadata=CacheMetadata(model="test-model", attributes={"query_id": str(uuid.uuid4())}),
    )


def _assert_no_user_tables(database_url: str) -> None:
    with closing(_connect_sqlite_url(database_url)) as conn:
        row = conn.execute(
            """
            SELECT count(*) AS n
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchone()
    if int(row["n"]) != 0:
        raise AssertionError(f"target database is not empty: {row['n']} tables")


def _entry_status(database_url: str, cache_key: CacheKey) -> str | None:
    with closing(_connect_sqlite_url(database_url)) as conn:
        row = conn.execute(
            "SELECT status FROM cache_entries WHERE cache_key = ?",
            (str(cache_key),),
        ).fetchone()
    return None if row is None else str(row["status"])


def _pending_outbox_count(database_url: str) -> int:
    with closing(_connect_sqlite_url(database_url)) as conn:
        row = conn.execute("SELECT count(*) AS n FROM outbox_events WHERE processed_at IS NULL").fetchone()
    return int(row["n"])


def _assert_hit_audit_row(database_url: str, cache_key: CacheKey) -> None:
    with closing(_connect_sqlite_url(database_url)) as conn:
        row = conn.execute(
            """
            SELECT accepted_cache_key, score, threshold, scorer_name, evidence
            FROM query_events
            WHERE accepted = 1
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise AssertionError("expected accepted query_events row")
    if str(row["accepted_cache_key"]) != str(cache_key):
        raise AssertionError(f"accepted_cache_key mismatch: {row['accepted_cache_key']}")
    if row["score"] is None or row["threshold"] is None or row["scorer_name"] is None:
        raise AssertionError("hit audit row is missing score, threshold, or scorer_name")
    if not json.loads(row["evidence"]).get("candidates"):
        raise AssertionError("hit audit evidence is missing candidate list")


def _pytest_command_recommendation() -> str:
    return "python -m pytest tests/test_sqlite_faiss_integration.py"


if __name__ == "__main__":
    sys.exit(main())
