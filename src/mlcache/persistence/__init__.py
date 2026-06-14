"""Local persistence helpers."""

from mlcache.persistence.json_utils import (
    atomic_write_json,
    decode_cache_metadata,
    decode_query_level_policy_decision,
    decode_query_record,
    encode_cache_metadata,
    encode_query_level_policy_decision,
    encode_query_record,
    json_safe,
    read_json_or_default,
)

_POSTGRES_EXPORTS = {
    "OutboxEvent",
    "PostgresCacheRepository",
    "PostgresJudgeTrainingStore",
    "PostgresKVStore",
    "PostgresQueryCalibrationRecordStore",
    "PostgresSplitJudgeTrainingStore",
    "PostgresStorageConfig",
    "PostgresThresholdProvider",
}

_MYSQL_EXPORTS = {
    "MySQLCacheRepository",
    "MySQLJudgeTrainingStore",
    "MySQLKVStore",
    "MySQLOutboxStore",
    "MySQLQueryCalibrationRecordStore",
    "MySQLSplitJudgeTrainingStore",
    "MySQLStorageConfig",
    "MySQLThresholdProvider",
}

_SQLITE_EXPORTS = {
    "SQLiteCacheRepository",
    "SQLiteJudgeTrainingStore",
    "SQLiteKVStore",
    "SQLiteOutboxStore",
    "SQLiteQueryCalibrationRecordStore",
    "SQLiteSplitJudgeTrainingStore",
    "SQLiteStorageConfig",
    "SQLiteThresholdProvider",
}


def __getattr__(name: str) -> object:
    if name in _POSTGRES_EXPORTS:
        from importlib import import_module

        module = import_module("mlcache.persistence.postgres")
        return getattr(module, name)
    if name in _MYSQL_EXPORTS:
        from importlib import import_module

        module = import_module("mlcache.persistence.mysql")
        return getattr(module, name)
    if name in _SQLITE_EXPORTS:
        from importlib import import_module

        module = import_module("mlcache.persistence.sqlite")
        return getattr(module, name)
    raise AttributeError(name)

__all__ = [
    "OutboxEvent",
    "MySQLCacheRepository",
    "MySQLJudgeTrainingStore",
    "MySQLKVStore",
    "MySQLOutboxStore",
    "MySQLQueryCalibrationRecordStore",
    "MySQLSplitJudgeTrainingStore",
    "MySQLStorageConfig",
    "MySQLThresholdProvider",
    "PostgresCacheRepository",
    "PostgresJudgeTrainingStore",
    "PostgresKVStore",
    "PostgresQueryCalibrationRecordStore",
    "PostgresSplitJudgeTrainingStore",
    "PostgresStorageConfig",
    "PostgresThresholdProvider",
    "SQLiteCacheRepository",
    "SQLiteJudgeTrainingStore",
    "SQLiteKVStore",
    "SQLiteOutboxStore",
    "SQLiteQueryCalibrationRecordStore",
    "SQLiteSplitJudgeTrainingStore",
    "SQLiteStorageConfig",
    "SQLiteThresholdProvider",
    "atomic_write_json",
    "decode_cache_metadata",
    "decode_query_level_policy_decision",
    "decode_query_record",
    "encode_cache_metadata",
    "encode_query_level_policy_decision",
    "encode_query_record",
    "json_safe",
    "read_json_or_default",
]
