"""Factory helpers for explicit MLCache runtime composition."""

from __future__ import annotations

from mlcache.cache import KVStore, SemanticCacheGateway
from mlcache.calibration import ThresholdProvider
from mlcache.features import PairFeatureBuilder
from mlcache.feedback import (
    DefaultShadowTopKCollector,
    JudgeTrainingStore,
    SemanticReuseJudge,
    ShadowCollectionConfig,
    ShadowTopKCollector,
    SplitJudgeTrainingStore,
)
from mlcache.observability import AuditLogger, DiagnosticsReporter, MetricsSink
from mlcache.online import OnlineUpdater
from mlcache.oracle import TrainableSemanticCacheOracle
from mlcache.policies import RefitPolicy
from mlcache.retrieval import VectorStore
from mlcache.runtime.config import MLCacheRuntimeConfig
from mlcache.runtime.runtime import MLCacheRuntime
from mlcache.scorers import SemanticScorer


def build_mlcache_runtime(
    *,
    kv_store: KVStore,
    vector_store: VectorStore,
    feature_builder: PairFeatureBuilder,
    scorer: SemanticScorer,
    threshold_provider: ThresholdProvider | None = None,
    online_updater: OnlineUpdater | None = None,
    judge: SemanticReuseJudge | None = None,
    judge_training_store: JudgeTrainingStore | None = None,
    shadow_collector: ShadowTopKCollector | None = None,
    refit_policy: RefitPolicy | None = None,
    audit_logger: AuditLogger | None = None,
    metrics_sink: MetricsSink | None = None,
    diagnostics_reporter: DiagnosticsReporter | None = None,
    config: MLCacheRuntimeConfig | None = None,
) -> MLCacheRuntime:
    runtime_config = config or MLCacheRuntimeConfig()
    collector = shadow_collector

    if runtime_config.shadow.enabled and collector is None:
        if judge is None:
            raise ValueError("shadow collection requires a SemanticReuseJudge when no shadow_collector is provided")
        if not isinstance(judge_training_store, SplitJudgeTrainingStore):
            raise ValueError("shadow collection requires a SplitJudgeTrainingStore when no shadow_collector is provided")
        collector = DefaultShadowTopKCollector(
            feature_builder=feature_builder,
            judge=judge,
            store=judge_training_store,
            config=ShadowCollectionConfig(
                enabled=True,
                top_k=runtime_config.shadow.top_k,
                max_pairs_per_request=runtime_config.shadow.max_pairs_per_request,
                calibration_every_n=runtime_config.shadow.calibration_every_n,
                collect_uncertain=runtime_config.shadow.collect_uncertain,
            ),
        )

    oracle = TrainableSemanticCacheOracle(
        vector_store=vector_store,
        feature_builder=feature_builder,
        scorer=scorer,
        threshold_provider=threshold_provider,
        online_updater=online_updater,
        judge=judge,
        judge_training_store=judge_training_store,
        shadow_collector=collector,
        shadow_collection_enabled=runtime_config.shadow.enabled and collector is not None,
        refit_policy=refit_policy,
        auto_refit=runtime_config.refit.auto_refit,
        target_false_accept_rate=runtime_config.refit.target_false_accept_rate,
        top_k=runtime_config.refit.top_k,
    )
    gateway = SemanticCacheGateway(kv_store=kv_store, oracle=oracle)

    return MLCacheRuntime(
        gateway=gateway,
        oracle=oracle,
        kv_store=kv_store,
        vector_store=vector_store,
        shadow_collector=collector,
        judge_training_store=judge_training_store,
        audit_logger=audit_logger,
        metrics_sink=metrics_sink,
        diagnostics_reporter=diagnostics_reporter,
        config=runtime_config,
    )
