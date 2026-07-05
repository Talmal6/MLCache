"""Local runtime factory for standalone experiments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mlcache.cache import (
    FileKVStore,
    InMemoryKVStore,
    KVStore,
    MySQLSemanticCacheGateway,
    PostgresSemanticCacheGateway,
    SQLiteSemanticCacheGateway,
)
from mlcache.calibration import (
    FileQueryCalibrationRecordStore,
    FileThresholdProvider,
    InMemoryQueryCalibrationRecordStore,
    InMemoryThresholdProvider,
    QueryCalibrationRecordStore,
    ThresholdProvider,
    ThresholdScope,
)
from mlcache.features import PairFeatureBuilder
from mlcache.feedback import InMemorySplitJudgeTrainingStore, JudgeTrainingStore, SemanticReuseJudge
from mlcache.persistence.postgres import (
    PostgresCacheRepository,
    PostgresKVStore,
    PostgresQueryCalibrationRecordStore,
    PostgresSplitJudgeTrainingStore,
    PostgresThresholdProvider,
)
from mlcache.persistence.mysql import (
    MySQLCacheRepository,
    MySQLKVStore,
    MySQLQueryCalibrationRecordStore,
    MySQLSplitJudgeTrainingStore,
    MySQLThresholdProvider,
)
from mlcache.persistence.sqlite import (
    SQLiteCacheRepository,
    SQLiteKVStore,
    SQLiteQueryCalibrationRecordStore,
    SQLiteSplitJudgeTrainingStore,
    SQLiteThresholdProvider,
)
from mlcache.policies import (
    FileQueryLevelShadowDecisionStore,
    InMemoryQueryLevelShadowDecisionStore,
    QueryLevelPolicyMode,
    QueryLevelShadowDecisionStore,
)
from mlcache.retrieval import FaissVectorStore, FileVectorStore, InMemoryVectorStore, VectorStore
from mlcache.runtime.config import EvictionRuntimeConfig, MLCacheRuntimeConfig, StorageRuntimeConfig
from mlcache.runtime.factory import build_mlcache_runtime
from mlcache.runtime.runtime import MLCacheRuntime
from mlcache.scorers import SemanticScorer


def build_local_mlcache_runtime(
    *,
    root_dir: str | Path,
    feature_builder: PairFeatureBuilder,
    scorer: SemanticScorer,
    judge: SemanticReuseJudge | None = None,
    config: MLCacheRuntimeConfig | None = None,
    use_file_persistence: bool = True,
) -> MLCacheRuntime:
    """Build an MLCache runtime from local in-memory or JSON-backed stores."""

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    runtime_config = config or MLCacheRuntimeConfig()
    storage_config = _resolve_storage_config(runtime_config.storage)
    runtime_config = replace(runtime_config, storage=storage_config)
    runtime_config = replace(runtime_config, eviction=_resolve_eviction_config(runtime_config.eviction))
    if storage_config.top_k is not None:
        runtime_config = replace(
            runtime_config,
            refit=replace(runtime_config.refit, top_k=storage_config.top_k),
        )
    storage_backend = storage_config.resolved_storage_backend(use_file_persistence=use_file_persistence)
    vector_backend = storage_config.resolved_vector_backend(storage_backend=storage_backend)
    repository: PostgresCacheRepository | MySQLCacheRepository | SQLiteCacheRepository | None = None
    gateway_factory = None

    if storage_backend == "postgres":
        if not storage_config.database_url:
            raise ValueError("MLCACHE_DATABASE_URL is required when MLCACHE_STORAGE_BACKEND=postgres")
        repository = PostgresCacheRepository(
            storage_config.database_url,
            shard_id=storage_config.faiss_shard_id,
        )
        kv_store = PostgresKVStore(repository)
        threshold_provider = PostgresThresholdProvider(repository)
        query_record_store = PostgresQueryCalibrationRecordStore(repository)
        query_shadow_store = InMemoryQueryLevelShadowDecisionStore()
        gateway_factory = lambda kv, oracle: PostgresSemanticCacheGateway(
            kv_store=kv,
            oracle=oracle,
            repository=repository,
        )
    elif storage_backend == "mysql":
        database_url = storage_config.mysql_database_url or storage_config.database_url
        if not database_url:
            raise ValueError(
                "MLCACHE_MYSQL_DATABASE_URL or MLCACHE_DATABASE_URL is required "
                "when MLCACHE_STORAGE_BACKEND=mysql"
            )
        repository = MySQLCacheRepository(
            database_url,
            shard_id=storage_config.faiss_shard_id,
        )
        kv_store = MySQLKVStore(repository)
        threshold_provider = MySQLThresholdProvider(repository)
        query_record_store = MySQLQueryCalibrationRecordStore(repository)
        query_shadow_store = InMemoryQueryLevelShadowDecisionStore()
        gateway_factory = lambda kv, oracle: MySQLSemanticCacheGateway(
            kv_store=kv,
            oracle=oracle,
            repository=repository,
        )
    elif storage_backend == "sqlite":
        database_url = storage_config.sqlite_database_url or storage_config.database_url
        if not database_url:
            raise ValueError(
                "MLCACHE_SQLITE_DATABASE_URL or MLCACHE_DATABASE_URL is required "
                "when MLCACHE_STORAGE_BACKEND=sqlite"
            )
        repository = SQLiteCacheRepository(
            database_url,
            shard_id=storage_config.faiss_shard_id,
        )
        kv_store = SQLiteKVStore(repository)
        threshold_provider = SQLiteThresholdProvider(repository)
        query_record_store = SQLiteQueryCalibrationRecordStore(repository)
        query_shadow_store = InMemoryQueryLevelShadowDecisionStore()
        gateway_factory = lambda kv, oracle: SQLiteSemanticCacheGateway(
            kv_store=kv,
            oracle=oracle,
            repository=repository,
        )
    elif storage_backend == "file":
        kv_store: KVStore = FileKVStore(root / "kv_store.json")
        threshold_provider: ThresholdProvider = FileThresholdProvider(root / "thresholds.json")
        query_record_store: QueryCalibrationRecordStore = FileQueryCalibrationRecordStore(
            root / "query_calibration_records.json"
        )
        query_shadow_store: QueryLevelShadowDecisionStore | None = FileQueryLevelShadowDecisionStore(
            root / "query_level_shadow_decisions.json"
        )
    else:
        kv_store = InMemoryKVStore()
        threshold_provider = InMemoryThresholdProvider()
        query_record_store = InMemoryQueryCalibrationRecordStore()
        query_shadow_store = InMemoryQueryLevelShadowDecisionStore()

    if vector_backend == "faiss":
        vector_store = FaissVectorStore(
            dim=storage_config.faiss_dim,
            index_path=storage_config.faiss_index_path,
            repository=repository,
            metric=storage_config.faiss_metric,
            shard_id=storage_config.faiss_shard_id,
        )
    elif storage_backend == "file":
        vector_store = FileVectorStore(root / "vector_store.json")
    else:
        vector_store = InMemoryVectorStore()

    runtime_config = _resolve_query_level_threshold(runtime_config, threshold_provider, scorer)
    judge_training_store = _build_local_judge_training_store(runtime_config, judge, repository=repository)
    query_level_mode = QueryLevelPolicyMode(runtime_config.query_level.mode)
    if not runtime_config.query_level.enabled or query_level_mode == QueryLevelPolicyMode.DISABLED:
        query_shadow_store = None

    return build_mlcache_runtime(
        kv_store=kv_store,
        vector_store=vector_store,
        feature_builder=feature_builder,
        scorer=scorer,
        threshold_provider=threshold_provider,
        judge=judge,
        judge_training_store=judge_training_store,
        query_record_store=query_record_store,
        query_level_shadow_store=query_shadow_store,
        gateway_factory=gateway_factory,
        config=runtime_config,
    )


def _resolve_storage_config(config: StorageRuntimeConfig) -> StorageRuntimeConfig:
    env = StorageRuntimeConfig.from_env()
    return StorageRuntimeConfig(
        storage_backend=config.storage_backend or env.storage_backend,
        vector_backend=config.vector_backend or env.vector_backend,
        database_url=config.database_url or env.database_url,
        mysql_database_url=config.mysql_database_url or env.mysql_database_url,
        sqlite_database_url=config.sqlite_database_url or env.sqlite_database_url,
        faiss_index_path=config.faiss_index_path or env.faiss_index_path,
        faiss_dim=config.faiss_dim if config.faiss_dim is not None else env.faiss_dim,
        faiss_metric=env.faiss_metric if config.faiss_metric == "cosine" else config.faiss_metric,
        faiss_shard_id=env.faiss_shard_id if config.faiss_shard_id == "default" else config.faiss_shard_id,
        top_k=config.top_k if config.top_k is not None else env.top_k,
    )


def _resolve_eviction_config(config: EvictionRuntimeConfig) -> EvictionRuntimeConfig:
    env = EvictionRuntimeConfig.from_env()
    return EvictionRuntimeConfig(
        policy=config.policy if config.policy != "lru" else env.policy,
        max_entries=config.max_entries if config.max_entries is not None else env.max_entries,
    )


def _resolve_query_level_threshold(
    config: MLCacheRuntimeConfig,
    threshold_provider: ThresholdProvider,
    scorer: SemanticScorer,
) -> MLCacheRuntimeConfig:
    query_level = config.query_level
    if not query_level.enabled or QueryLevelPolicyMode(query_level.mode) == QueryLevelPolicyMode.DISABLED:
        return config

    if query_level.threshold is not None:
        threshold_provider.set_threshold(
            query_level.threshold,
            scorer=scorer.name,
            scope=ThresholdScope.GLOBAL,
            context={"source": "local_runtime_query_level_config"},
        )
        return config

    try:
        threshold = threshold_provider.get_threshold(scorer=scorer.name, scope=ThresholdScope.GLOBAL)
    except Exception:
        return config

    return replace(config, query_level=replace(query_level, threshold=threshold))


def _build_local_judge_training_store(
    config: MLCacheRuntimeConfig,
    judge: SemanticReuseJudge | None,
    *,
    repository: PostgresCacheRepository | MySQLCacheRepository | SQLiteCacheRepository | None = None,
) -> JudgeTrainingStore | None:
    if not config.shadow.enabled or judge is None:
        return None
    if repository is not None:
        if isinstance(repository, MySQLCacheRepository):
            return MySQLSplitJudgeTrainingStore(repository)
        if isinstance(repository, SQLiteCacheRepository):
            return SQLiteSplitJudgeTrainingStore(repository)
        return PostgresSplitJudgeTrainingStore(repository)
    return InMemorySplitJudgeTrainingStore(
        max_h0_train=100_000,
        max_h1_train=100_000,
        max_h0_calibration=100_000,
        max_h1_calibration=100_000,
    )


__all__ = ["build_local_mlcache_runtime"]
