from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pytest

from scripts.run_sqlite_faiss_integration import run_harness


def _missing_requirements() -> list[str]:
    missing: list[str] = []
    if importlib.util.find_spec("faiss") is None:
        missing.append("faiss")
    return missing


class SQLiteFaissIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        missing = _missing_requirements()
        if missing:
            self.skipTest(
                "SQLite+FAISS integration requirements missing: "
                f"{', '.join(missing)}. Install with "
                "python -m pip install -e \".[sqlite,faiss,dev]\"."
            )

    def test_public_mlcache_sqlite_faiss_outbox_snapshot_and_tombstone_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-sqlite-faiss-") as tmp:
            root = Path(tmp)
            database_url = f"sqlite:///{(root / 'mlcache.db').as_posix()}"
            result = run_harness(database_url=database_url, root_dir=root)

        self.assertEqual(result["pending_status"], "PENDING_INDEX")
        self.assertEqual(result["active_status"], "ACTIVE")
        self.assertEqual(result["tombstone_status"], "TOMBSTONED")
        self.assertEqual(result["outbox_processed"], 1)
        self.assertEqual(result["hit_cache_key"], result["restarted_hit_cache_key"])


if __name__ == "__main__":
    unittest.main()
