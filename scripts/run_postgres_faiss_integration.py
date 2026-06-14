"""Run the real PostgreSQL + FAISS MLCache integration harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from mlcache import CacheEntry, CacheKey, CacheLookup, CacheMetadata, FaissOutboxIndexer
from mlcache import MLCache, PostgresCacheRepository, Query, Response, Threshold


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _assert_optional_dependencies_available()
        work_dir = Path(args.work_dir)
        root = work_dir / f"mlcache-pg-faiss-{uuid.uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=False)
        with _postgres_url(args, root) as database_url:
            result = run_harness(database_url=database_url, root_dir=root)
            result["pytest_command_recommendation"] = _pytest_command_recommendation(args)
            report_path = Path(args.report_path) if args.report_path else root / "postgres_faiss_report.json"
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
            "MLCACHE_STORAGE_BACKEND": "postgres",
            "MLCACHE_VECTOR_BACKEND": "faiss",
            "MLCACHE_DATABASE_URL": database_url,
            "MLCACHE_FAISS_DIM": "2",
            "MLCACHE_FAISS_INDEX_PATH": str(faiss_path),
            "MLCACHE_FAISS_METRIC": "cosine",
            "MLCACHE_FAISS_SHARD_ID": "integration",
            "MLCACHE_TOP_K": "5",
        }
    )

    _assert_no_user_tables(database_url)
    repository = PostgresCacheRepository(database_url, shard_id="integration")
    try:
        repository.apply_migrations()
    except Exception as exc:
        raise RuntimeError(f"PostgreSQL migrations failed: {exc}") from exc

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
        "migration_command": "PostgresCacheRepository.apply_migrations()",
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
def _postgres_url(args: argparse.Namespace, root_dir: Path) -> Iterator[str]:
    if args.admin_database_url:
        psycopg, sql = _load_psycopg()
        db_name = f"mlcache_it_{uuid.uuid4().hex}"
        with psycopg.connect(args.admin_database_url, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        try:
            yield _database_url_for_name(args.admin_database_url, db_name)
        finally:
            with psycopg.connect(args.admin_database_url, autocommit=True) as conn:
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

    if args.database_url:
        yield args.database_url
        return

    if not args.pg_bin_dir:
        raise RuntimeError(
            "database URL is missing; set MLCACHE_TEST_DATABASE_URL or "
            "MLCACHE_TEST_ADMIN_DATABASE_URL, or pass --pg-bin-dir to start a temporary local cluster"
        )

    pg_bin = Path(args.pg_bin_dir) if args.pg_bin_dir else None
    initdb = _find_pg_binary("initdb", pg_bin)
    pg_ctl = _find_pg_binary("pg_ctl", pg_bin)
    data_dir = root_dir / "pgdata"
    log_path = root_dir / "postgres.log"
    port = _free_port()
    subprocess.run(
        [str(initdb), "-A", "trust", "-U", "postgres", "-D", str(data_dir)],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(pg_ctl),
            "-D",
            str(data_dir),
            "-l",
            str(log_path),
            "-o",
            f"-p {port}",
            "-w",
            "start",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    try:
        yield f"postgresql://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(
            [str(pg_ctl), "-D", str(data_dir), "-w", "stop"],
            check=False,
            text=True,
            capture_output=True,
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("MLCACHE_TEST_DATABASE_URL"))
    parser.add_argument("--admin-database-url", default=os.getenv("MLCACHE_TEST_ADMIN_DATABASE_URL"))
    parser.add_argument("--pg-bin-dir", default=os.getenv("POSTGRES_BIN"))
    parser.add_argument("--work-dir", default="runs/postgres_faiss_integration")
    parser.add_argument("--report-path")
    return parser.parse_args(argv)


def _assert_optional_dependencies_available() -> None:
    missing: list[str] = []
    if importlib.util.find_spec("psycopg") is None:
        missing.append("psycopg")
    if importlib.util.find_spec("faiss") is None:
        missing.append("faiss")
    if missing:
        extras = "python -m pip install -e \".[postgres,faiss,dev]\""
        raise RuntimeError(f"missing optional dependencies: {', '.join(missing)}; install with {extras}")


def _load_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is missing; install PostgreSQL support with "
            "python -m pip install -e \".[postgres,faiss,dev]\""
        ) from exc
    return psycopg, sql


def _database_url_for_name(admin_url: str, database_name: str) -> str:
    parsed = urlsplit(admin_url)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment))


def _find_pg_binary(name: str, pg_bin_dir: Path | None) -> Path:
    exe = f"{name}.exe" if os.name == "nt" else name
    if pg_bin_dir is not None:
        path = pg_bin_dir / exe
        if path.exists():
            return path
    found = shutil.which(name) or shutil.which(exe)
    if found:
        return Path(found)
    raise FileNotFoundError(f"{name} not found; pass --pg-bin-dir or set POSTGRES_BIN")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    psycopg, _ = _load_psycopg()
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_type = 'BASE TABLE'
            """
        ).fetchone()
    if int(row[0]) != 0:
        raise AssertionError(f"target database/schema is not empty: {row[0]} tables")


def _entry_status(database_url: str, cache_key: CacheKey) -> str | None:
    psycopg, _ = _load_psycopg()
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            "SELECT status FROM cache_entries WHERE cache_key = %s",
            (str(cache_key),),
        ).fetchone()
    return None if row is None else str(row[0])


def _pending_outbox_count(database_url: str) -> int:
    psycopg, _ = _load_psycopg()
    with psycopg.connect(database_url) as conn:
        row = conn.execute("SELECT count(*) FROM outbox_events WHERE processed_at IS NULL").fetchone()
    return int(row[0])


def _assert_hit_audit_row(database_url: str, cache_key: CacheKey) -> None:
    psycopg, _ = _load_psycopg()
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT accepted_cache_key, score, threshold, scorer_name, evidence
            FROM query_events
            WHERE accepted = true
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise AssertionError("expected accepted query_events row")
    accepted_cache_key, score, threshold, scorer_name, evidence = row
    if str(accepted_cache_key) != str(cache_key):
        raise AssertionError(f"accepted_cache_key mismatch: {accepted_cache_key}")
    if score is None or threshold is None or scorer_name is None:
        raise AssertionError("hit audit row is missing score, threshold, or scorer_name")
    if not dict(evidence).get("candidates"):
        raise AssertionError("hit audit evidence is missing candidate list")


def _redact_url(database_url: str) -> str:
    if "@" not in database_url:
        return database_url
    prefix, suffix = database_url.split("@", 1)
    scheme = prefix.split("://", 1)[0]
    return f"{scheme}://***@{suffix}"


def _pytest_command_recommendation(args: argparse.Namespace) -> str:
    if args.admin_database_url:
        env_name = "MLCACHE_TEST_ADMIN_DATABASE_URL"
        value = _redact_url(args.admin_database_url)
    elif args.database_url:
        env_name = "MLCACHE_TEST_DATABASE_URL"
        value = _redact_url(args.database_url)
    else:
        env_name = "MLCACHE_TEST_DATABASE_URL"
        value = "<database-url>"
    if os.name == "nt":
        return f"$env:{env_name}='{value}'; python -m pytest -m integration"
    return f"{env_name}='{value}' python -m pytest -m integration"


if __name__ == "__main__":
    sys.exit(main())
