"""Run the real MySQL + FAISS MLCache integration harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from mlcache import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    CacheMetadata,
    FaissOutboxIndexer,
    MLCache,
    MySQLCacheRepository,
    Query,
    Response,
    Threshold,
)
from mlcache.persistence.mysql import _connect_mysql_url


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _assert_optional_dependencies_available()
        work_dir = Path(args.work_dir)
        root = work_dir / f"mlcache-mysql-faiss-{uuid.uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=False)
        with _mysql_url(args) as database_url:
            result = run_harness(database_url=database_url, root_dir=root)
            result["pytest_command_recommendation"] = _pytest_command_recommendation(args)
            report_path = Path(args.report_path) if args.report_path else root / "mysql_faiss_report.json"
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
            "MLCACHE_STORAGE_BACKEND": "mysql",
            "MLCACHE_VECTOR_BACKEND": "faiss",
            "MLCACHE_DATABASE_URL": database_url,
            "MLCACHE_MYSQL_DATABASE_URL": database_url,
            "MLCACHE_FAISS_DIM": "2",
            "MLCACHE_FAISS_INDEX_PATH": str(faiss_path),
            "MLCACHE_FAISS_METRIC": "cosine",
            "MLCACHE_FAISS_SHARD_ID": "integration",
            "MLCACHE_TOP_K": "5",
        }
    )

    _assert_no_user_tables(database_url)
    repository = MySQLCacheRepository(database_url, shard_id="integration")
    try:
        repository.apply_migrations()
    except Exception as exc:
        raise RuntimeError(f"MySQL migrations failed: {exc}") from exc

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
        "database_url": _redact_url(database_url),
        "migration_command": "MySQLCacheRepository.apply_migrations()",
        "faiss_snapshot": snapshot,
        "faiss_snapshot_path": snapshot.get("snapshot_uri"),
        "pending_status": pending_status,
        "active_status": active_status,
        "tombstone_status": tombstone_status,
        "outbox_processed": processed,
        "hit_cache_key": str(active_lookup.decision.cache_key),
        "restarted_hit_cache_key": str(restarted_lookup.decision.cache_key),
    }


@contextmanager
def _mysql_url(args: argparse.Namespace) -> Iterator[str]:
    if args.admin_database_url:
        db_name = f"mlcache_it_{uuid.uuid4().hex}"
        with _connect_mysql_url(args.admin_database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
            conn.commit()
        try:
            yield _database_url_for_name(args.admin_database_url, db_name)
        finally:
            with _connect_mysql_url(args.admin_database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
                conn.commit()
        return

    if args.database_url:
        yield args.database_url
        return

    raise RuntimeError(
        "database URL is missing; set MLCACHE_TEST_MYSQL_DATABASE_URL, "
        "MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL, or MLCACHE_MYSQL_DATABASE_URL"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.getenv("MLCACHE_TEST_MYSQL_DATABASE_URL") or os.getenv("MLCACHE_MYSQL_DATABASE_URL"),
    )
    parser.add_argument("--admin-database-url", default=os.getenv("MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL"))
    parser.add_argument("--work-dir", default="runs/mysql_faiss_integration")
    parser.add_argument("--report-path")
    return parser.parse_args(argv)


def _assert_optional_dependencies_available() -> None:
    missing: list[str] = []
    if importlib.util.find_spec("pymysql") is None:
        missing.append("pymysql")
    if importlib.util.find_spec("faiss") is None:
        missing.append("faiss")
    if missing:
        extras = "python -m pip install -e \".[mysql,faiss,dev]\""
        raise RuntimeError(f"missing optional dependencies: {', '.join(missing)}; install with {extras}")


def _database_url_for_name(admin_url: str, database_name: str) -> str:
    parsed = urlsplit(admin_url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment))


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
    with _connect_mysql_url(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                  AND table_type = 'BASE TABLE'
                """
            )
            row = cur.fetchone()
    if int(row[0]) != 0:
        raise AssertionError(f"target database/schema is not empty: {row[0]} tables")


def _entry_status(database_url: str, cache_key: CacheKey) -> str | None:
    with _connect_mysql_url(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM cache_entries WHERE cache_key = %s",
                (str(cache_key),),
            )
            row = cur.fetchone()
    return None if row is None else str(row[0])


def _pending_outbox_count(database_url: str) -> int:
    with _connect_mysql_url(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM outbox_events WHERE processed_at IS NULL")
            row = cur.fetchone()
    return int(row[0])


def _assert_hit_audit_row(database_url: str, cache_key: CacheKey) -> None:
    with _connect_mysql_url(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT accepted_cache_key, score, threshold, scorer_name, evidence
                FROM query_events
                WHERE accepted = true
                ORDER BY created_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    if row is None:
        raise AssertionError("expected accepted query_events row")
    accepted_cache_key, score, threshold, scorer_name, evidence = row
    if str(accepted_cache_key) != str(cache_key):
        raise AssertionError(f"accepted_cache_key mismatch: {accepted_cache_key}")
    if score is None or threshold is None or scorer_name is None:
        raise AssertionError("hit audit row is missing score, threshold, or scorer_name")
    if not json.loads(evidence).get("candidates"):
        raise AssertionError("hit audit evidence is missing candidate list")


def _pytest_command_recommendation(args: argparse.Namespace) -> str:
    if args.admin_database_url:
        env_name = "MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL"
        value = _redact_url(args.admin_database_url)
    elif args.database_url:
        env_name = "MLCACHE_TEST_MYSQL_DATABASE_URL"
        value = _redact_url(args.database_url)
    else:
        env_name = "MLCACHE_TEST_MYSQL_DATABASE_URL"
        value = "<database-url>"
    if os.name == "nt":
        return f"$env:{env_name}='{value}'; python -m pytest tests/test_mysql_faiss_integration.py"
    return f"{env_name}='{value}' python -m pytest tests/test_mysql_faiss_integration.py"


def _redact_url(database_url: str) -> str:
    if "@" not in database_url:
        return database_url
    prefix, suffix = database_url.split("@", 1)
    scheme = prefix.split("://", 1)[0]
    return f"{scheme}://***@{suffix}"


if __name__ == "__main__":
    sys.exit(main())
