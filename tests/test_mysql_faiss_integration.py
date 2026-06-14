from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path
import unittest

import pytest

from mlcache.persistence.mysql import _connect_mysql_url
from scripts.run_mysql_faiss_integration import _database_url_for_name, run_harness


pytestmark = pytest.mark.integration


def _missing_requirements() -> list[str]:
    missing: list[str] = []
    if not (
        os.getenv("MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL")
        or os.getenv("MLCACHE_TEST_MYSQL_DATABASE_URL")
        or os.getenv("MLCACHE_MYSQL_DATABASE_URL")
    ):
        missing.append(
            "MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL, MLCACHE_TEST_MYSQL_DATABASE_URL, or MLCACHE_MYSQL_DATABASE_URL"
        )
    if importlib.util.find_spec("pymysql") is None:
        missing.append("pymysql")
    if importlib.util.find_spec("faiss") is None:
        missing.append("faiss")
    return missing


class MySQLFaissIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        missing = _missing_requirements()
        if missing:
            self.skipTest(
                "MySQL+FAISS integration requirements missing: "
                f"{', '.join(missing)}. Install with "
                "python -m pip install -e \".[mysql,faiss,dev]\" and set a MySQL test database URL."
            )

        self.admin_url = os.getenv("MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL")
        self.database_name: str | None = None
        if self.admin_url:
            self.database_name = f"mlcache_test_{os.urandom(8).hex()}"
            with _connect_mysql_url(self.admin_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"CREATE DATABASE `{self.database_name}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
                    )
                conn.commit()
            self.database_url = _database_url_for_name(self.admin_url, self.database_name)
        else:
            self.database_url = (
                os.getenv("MLCACHE_TEST_MYSQL_DATABASE_URL")
                or os.getenv("MLCACHE_MYSQL_DATABASE_URL")
                or ""
            )

    def tearDown(self) -> None:
        if self.database_name is not None and self.admin_url is not None:
            with _connect_mysql_url(self.admin_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP DATABASE IF EXISTS `{self.database_name}`")
                conn.commit()

    def test_public_mlcache_mysql_faiss_outbox_snapshot_and_tombstone_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-mysql-faiss-") as tmp:
            result = run_harness(database_url=self.database_url, root_dir=Path(tmp))

        self.assertEqual(result["pending_status"], "PENDING_INDEX")
        self.assertEqual(result["active_status"], "ACTIVE")
        self.assertEqual(result["tombstone_status"], "TOMBSTONED")
        self.assertEqual(result["outbox_processed"], 1)
        self.assertEqual(result["hit_cache_key"], result["restarted_hit_cache_key"])


if __name__ == "__main__":
    unittest.main()
