from __future__ import annotations

from pathlib import Path
import unittest

from mlcache.runtime import StorageRuntimeConfig


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "src" / "mlcache" / "persistence" / "migrations" / "001_postgres_storage.sql"
MYSQL_MIGRATION = ROOT / "src" / "mlcache" / "persistence" / "mysql_migrations" / "001_mysql_storage.sql"


class PersistentStorageSchemaTests(unittest.TestCase):
    def test_postgres_migration_declares_required_tables(self) -> None:
        sql = MIGRATION.read_text(encoding="utf-8")

        for table in (
            "cache_entries",
            "cache_payloads",
            "threshold_versions",
            "feedback_pairs",
            "query_events",
            "query_candidates",
            "outbox_events",
            "faiss_snapshots",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", sql)

    def test_mysql_migration_declares_required_tables(self) -> None:
        sql = MYSQL_MIGRATION.read_text(encoding="utf-8")

        for table in (
            "cache_entries",
            "cache_payloads",
            "threshold_versions",
            "feedback_pairs",
            "query_events",
            "query_candidates",
            "outbox_events",
            "faiss_snapshots",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", sql)

        self.assertIn("ON DELETE CASCADE", sql)
        self.assertIn("LONGBLOB", sql)
        self.assertIn("DATETIME(6)", sql)

    def test_storage_runtime_config_resolves_existing_defaults(self) -> None:
        config = StorageRuntimeConfig()

        self.assertEqual(config.resolved_storage_backend(use_file_persistence=True), "file")
        self.assertEqual(config.resolved_storage_backend(use_file_persistence=False), "inmemory")
        self.assertEqual(config.resolved_vector_backend(storage_backend="file"), "inmemory")

    def test_postgres_storage_defaults_to_faiss_vector_backend(self) -> None:
        config = StorageRuntimeConfig(storage_backend="postgres")

        self.assertEqual(config.resolved_vector_backend(storage_backend="postgres"), "faiss")

    def test_mysql_storage_defaults_to_faiss_vector_backend(self) -> None:
        config = StorageRuntimeConfig(storage_backend="mysql")

        self.assertEqual(config.resolved_vector_backend(storage_backend="mysql"), "faiss")


if __name__ == "__main__":
    unittest.main()
