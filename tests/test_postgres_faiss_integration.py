from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
import unittest
from unittest import mock
from urllib.parse import urlsplit, urlunsplit

import pytest

from mlcache import CacheKey, FaissOutboxIndexer, MLCache, PostgresCacheRepository, Response, Threshold
from scripts.run_postgres_faiss_integration import _entry, _lookup


pytestmark = pytest.mark.integration


def _missing_requirements() -> list[str]:
    missing: list[str] = []
    if not (os.getenv("MLCACHE_TEST_DATABASE_URL") or os.getenv("MLCACHE_TEST_ADMIN_DATABASE_URL")):
        missing.append("MLCACHE_TEST_DATABASE_URL or MLCACHE_TEST_ADMIN_DATABASE_URL")
    if importlib.util.find_spec("psycopg") is None:
        missing.append("psycopg")
    if importlib.util.find_spec("faiss") is None:
        missing.append("faiss")
    return missing


class PostgresFaissIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        missing = _missing_requirements()
        if missing:
            self.fail(
                "Postgres+FAISS integration requirements missing: "
                f"{', '.join(missing)}. Install with "
                "python -m pip install -e \".[postgres,faiss,dev]\" and set a test database URL."
            )

        import psycopg
        from psycopg import sql

        self.psycopg = psycopg
        self.sql = sql
        self.admin_url = os.getenv("MLCACHE_TEST_ADMIN_DATABASE_URL")
        self.database_name: str | None = None
        self.schema: str | None = None
        if self.admin_url:
            self.database_name = f"mlcache_test_{os.urandom(8).hex()}"
            with psycopg.connect(self.admin_url, autocommit=True) as conn:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database_name)))
            self.database_url = self._database_url_for_name(self.admin_url, self.database_name)
        else:
            self.database_url = os.environ["MLCACHE_TEST_DATABASE_URL"]
            self.schema = f"mlcache_test_{os.urandom(8).hex()}"
            with psycopg.connect(self.database_url) as conn:
                conn.execute(f'CREATE SCHEMA "{self.schema}"')
            self.database_url = self._database_url_with_search_path()

    def tearDown(self) -> None:
        if self.database_name is not None and self.admin_url is not None:
            with self.psycopg.connect(self.admin_url, autocommit=True) as conn:
                conn.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (self.database_name,),
                )
                conn.execute(
                    self.sql.SQL("DROP DATABASE IF EXISTS {}").format(self.sql.Identifier(self.database_name))
                )
            return
        base_url = os.environ["MLCACHE_TEST_DATABASE_URL"]
        with self.psycopg.connect(base_url) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    def test_public_mlcache_postgres_faiss_outbox_snapshot_and_tombstone_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-pg-faiss-") as tmp:
            root = Path(tmp)
            faiss_path = root / "faiss.index"
            with mock.patch.dict(
                os.environ,
                {
                    "MLCACHE_STORAGE_BACKEND": "postgres",
                    "MLCACHE_VECTOR_BACKEND": "faiss",
                    "MLCACHE_DATABASE_URL": self.database_url,
                    "MLCACHE_FAISS_DIM": "2",
                    "MLCACHE_FAISS_INDEX_PATH": str(faiss_path),
                    "MLCACHE_FAISS_METRIC": "cosine",
                    "MLCACHE_FAISS_SHARD_ID": "integration",
                    "MLCACHE_TOP_K": "5",
                },
                clear=False,
            ):
                repository = PostgresCacheRepository(self.database_url, shard_id="integration")
                self._assert_current_schema_is_empty()
                repository.apply_migrations()

                cache = MLCache.from_preset(root_dir=root / "state", scorer="cosine", persistence=True)
                cache.set_threshold(Threshold(0.0))
                repository = cache.runtime.vector_store.repository
                assert repository is not None

                entry = _entry()
                cache.put(entry)
                self.assertEqual(self._entry_status(entry.cache_key), "PENDING_INDEX")
                self.assertEqual(self._pending_outbox_count(), 1)

                pending = cache.lookup_with_decision(_lookup())
                self.assertFalse(pending.decision.accepted)

                processed = FaissOutboxIndexer(
                    repository=repository,
                    vector_store=cache.runtime.vector_store,
                ).process_once()
                self.assertEqual(processed, 1)
                self.assertEqual(self._entry_status(entry.cache_key), "ACTIVE")

                active = cache.lookup_with_decision(_lookup())
                self.assertTrue(active.decision.accepted)
                self.assertEqual(active.response, Response("response-active"))
                self._assert_hit_audit_row(entry.cache_key)

                cache.runtime.vector_store.save_snapshot(faiss_path)
                restarted = MLCache.from_preset(root_dir=root / "state-restarted", scorer="cosine", persistence=True)
                restarted.set_threshold(Threshold(0.0))
                restarted_hit = restarted.lookup_with_decision(_lookup())
                self.assertTrue(restarted_hit.decision.accepted)
                self.assertEqual(restarted_hit.response, Response("response-active"))

                repository.tombstone_entry(entry.cache_key)
                self.assertEqual(self._entry_status(entry.cache_key), "TOMBSTONED")
                stale_faiss_result = restarted.lookup_with_decision(_lookup())
                self.assertFalse(stale_faiss_result.decision.accepted)

    def _database_url_with_search_path(self) -> str:
        assert self.schema is not None
        separator = "&" if "?" in self.database_url else "?"
        return f"{self.database_url}{separator}options=-csearch_path%3D{self.schema}"

    @staticmethod
    def _database_url_for_name(admin_url: str, database_name: str) -> str:
        parsed = urlsplit(admin_url)
        return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment))

    def _assert_current_schema_is_empty(self) -> None:
        with self.psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_type = 'BASE TABLE'
                """
            ).fetchone()
        self.assertEqual(int(row[0]), 0)

    def _entry_status(self, cache_key: CacheKey) -> str | None:
        with self.psycopg.connect(self.database_url) as conn:
            row = conn.execute("SELECT status FROM cache_entries WHERE cache_key = %s", (str(cache_key),)).fetchone()
        return None if row is None else str(row[0])

    def _pending_outbox_count(self) -> int:
        with self.psycopg.connect(self.database_url) as conn:
            row = conn.execute("SELECT count(*) FROM outbox_events WHERE processed_at IS NULL").fetchone()
        return int(row[0])

    def _assert_hit_audit_row(self, cache_key: CacheKey) -> None:
        with self.psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                """
                SELECT accepted_cache_key, score, threshold, scorer_name, evidence
                FROM query_events
                WHERE accepted = true
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        self.assertIsNotNone(row)
        accepted_cache_key, score, threshold, scorer_name, evidence = row
        self.assertEqual(str(accepted_cache_key), str(cache_key))
        self.assertIsNotNone(score)
        self.assertIsNotNone(threshold)
        self.assertIsNotNone(scorer_name)
        self.assertTrue(dict(evidence).get("candidates"))


if __name__ == "__main__":
    unittest.main()
