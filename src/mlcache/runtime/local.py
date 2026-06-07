"""Local runtime factory for standalone experiments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mlcache.cache import FileKVStore, InMemoryKVStore, KVStore
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
from mlcache.policies import (
    FileQueryLevelShadowDecisionStore,
    InMemoryQueryLevelShadowDecisionStore,
    QueryLevelPolicyMode,
    QueryLevelShadowDecisionStore,
)
from mlcache.retrieval import FileVectorStore, InMemoryVectorStore, VectorStore
from mlcache.runtime.config import MLCacheRuntimeConfig
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

    if use_file_persistence:
        kv_store: KVStore = FileKVStore(root / "kv_store.json")
        vector_store: VectorStore = FileVectorStore(root / "vector_store.json")
        threshold_provider: ThresholdProvider = FileThresholdProvider(root / "thresholds.json")
        query_record_store: QueryCalibrationRecordStore = FileQueryCalibrationRecordStore(
            root / "query_calibration_records.json"
        )
        query_shadow_store: QueryLevelShadowDecisionStore | None = FileQueryLevelShadowDecisionStore(
            root / "query_level_shadow_decisions.json"
        )
    else:
        kv_store = InMemoryKVStore()
        vector_store = InMemoryVectorStore()
        threshold_provider = InMemoryThresholdProvider()
        query_record_store = InMemoryQueryCalibrationRecordStore()
        query_shadow_store = InMemoryQueryLevelShadowDecisionStore()

    runtime_config = _resolve_query_level_threshold(runtime_config, threshold_provider, scorer)
    judge_training_store = _build_local_judge_training_store(runtime_config, judge)
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
        config=runtime_config,
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
) -> JudgeTrainingStore | None:
    if not config.shadow.enabled or judge is None:
        return None
    return InMemorySplitJudgeTrainingStore(
        max_h0_train=100_000,
        max_h1_train=100_000,
        max_h0_calibration=100_000,
        max_h1_calibration=100_000,
    )


__all__ = ["build_local_mlcache_runtime"]
