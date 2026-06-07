"""Runtime orchestrator boundary for MLCache."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mlcache.cache import CacheGatewayResult, KVStore, SemanticCacheGateway
from mlcache.feedback import JudgeTrainingStore, ShadowTopKCollector
from mlcache.observability import AuditEvent, AuditLogger, DiagnosticsReporter, MetricsSink
from mlcache.oracle import OracleFitResult, TrainableSemanticCacheOracle
from mlcache.retrieval import VectorStore
from mlcache.runtime.config import MLCacheRuntimeConfig
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, OracleDecisionStatus, Response, Threshold


class MLCacheRuntime:
    """Production composition boundary around the gateway and trainable oracle."""

    def __init__(
        self,
        *,
        gateway: SemanticCacheGateway,
        oracle: TrainableSemanticCacheOracle,
        kv_store: KVStore,
        vector_store: VectorStore,
        shadow_collector: ShadowTopKCollector | None = None,
        judge_training_store: JudgeTrainingStore | None = None,
        audit_logger: AuditLogger | None = None,
        metrics_sink: MetricsSink | None = None,
        diagnostics_reporter: DiagnosticsReporter | None = None,
        config: MLCacheRuntimeConfig | None = None,
    ) -> None:
        self.gateway = gateway
        self.oracle = oracle
        self.kv_store = kv_store
        self.vector_store = vector_store
        self.shadow_collector = shadow_collector
        self.judge_training_store = judge_training_store
        self.audit_logger = audit_logger
        self.metrics_sink = metrics_sink
        self.diagnostics_reporter = diagnostics_reporter
        self.config = config or MLCacheRuntimeConfig()
        self._diagnostics: list[dict[str, Any]] = []

    def lookup(self, request: CacheLookup) -> Response | None:
        return self.lookup_with_decision(request).response

    def lookup_with_decision(self, request: CacheLookup) -> CacheGatewayResult:
        result = self.gateway.lookup_with_decision(request)
        self._emit_observability(request, result)
        return result

    def put(self, entry: CacheEntry) -> CacheKey:
        return self.gateway.put(entry)

    def invalidate(self, cache_key: CacheKey) -> None:
        self.gateway.invalidate(cache_key)

    @property
    def activation_status(self) -> dict[str, Any]:
        return self.oracle.activation_status

    @property
    def shadow_snapshot(self) -> Any | None:
        collector = self.shadow_collector
        if collector is None or not hasattr(collector, "snapshot"):
            return None
        return collector.snapshot()

    @property
    def fit_result(self) -> OracleFitResult | None:
        return self.oracle.last_fit_result

    @property
    def threshold(self) -> Threshold | None:
        return self.oracle.threshold

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._diagnostics)

    @property
    def components(self) -> dict[str, str | None]:
        return {
            "gateway": self._component_name(self.gateway),
            "oracle": self._component_name(self.oracle),
            "kv_store": self._component_name(self.kv_store),
            "vector_store": self._component_name(self.vector_store),
            "feature_builder": self._component_name(getattr(self.oracle, "feature_builder", None)),
            "scorer": self._scorer_name(),
            "shadow_collector": self._component_name(self.shadow_collector),
            "judge_training_store": self._component_name(self.judge_training_store),
            "audit_logger": self._component_name(self.audit_logger),
            "metrics_sink": self._component_name(self.metrics_sink),
            "diagnostics_reporter": self._component_name(self.diagnostics_reporter),
        }

    def _emit_observability(self, request: CacheLookup, result: CacheGatewayResult) -> None:
        metadata = self._lookup_metadata(result)
        if self.audit_logger is not None:
            try:
                cache_key = result.decision.cache_key
                self.audit_logger.log(
                    AuditEvent(
                        timestamp=datetime.now(UTC),
                        query=request.query,
                        decision=result.decision,
                        candidate_keys=(cache_key,) if cache_key is not None else (),
                        metadata=metadata,
                    )
                )
            except Exception as exc:
                self._record_observability_failure("audit_logger", exc)

        if self.metrics_sink is not None:
            try:
                self._record_lookup_metrics(metadata, result)
            except Exception as exc:
                self._record_observability_failure("metrics_sink", exc)

        if self.diagnostics_reporter is not None:
            try:
                self.diagnostics_reporter.report("cache.lookup", metadata)
            except Exception as exc:
                self._record_observability_failure("diagnostics_reporter", exc)

    def _record_lookup_metrics(self, metadata: dict[str, Any], result: CacheGatewayResult) -> None:
        status = result.decision.status
        accepted = 1.0 if result.decision.accepted else 0.0
        self.metrics_sink.record("cache.lookup.accepted", accepted, metadata)
        self.metrics_sink.record("cache.lookup.hit", 1.0 if status == OracleDecisionStatus.HIT else 0.0, metadata)
        self.metrics_sink.record("cache.lookup.miss", 1.0 if status == OracleDecisionStatus.MISS else 0.0, metadata)
        self.metrics_sink.record("cache.lookup.abstain", 1.0 if status == OracleDecisionStatus.ABSTAIN else 0.0, metadata)
        self.metrics_sink.record("cache.lookup.candidate_count", float(result.decision.candidate_count), metadata)

    def _lookup_metadata(self, result: CacheGatewayResult) -> dict[str, Any]:
        decision = result.decision
        reason = result.metadata.get("reason", decision.reason)
        return {
            "decision_status": str(decision.status),
            "accepted": bool(decision.accepted),
            "reason": reason,
            "scorer": str(decision.scorer) if decision.scorer is not None else None,
            "threshold": float(decision.threshold) if decision.threshold is not None else None,
            "candidate_count": int(decision.candidate_count),
        }

    def _record_observability_failure(self, hook: str, exc: Exception) -> None:
        self._diagnostics.append(
            {
                "event": "observability_failure",
                "hook": hook,
                "error": repr(exc),
            }
        )

    def _scorer_name(self) -> str | None:
        scorer = getattr(self.oracle, "scorer", None)
        if scorer is None:
            return None
        name = getattr(scorer, "name", None)
        if name is None:
            return type(scorer).__name__
        return f"{type(scorer).__name__}:{name}"

    @staticmethod
    def _component_name(component: object | None) -> str | None:
        return type(component).__name__ if component is not None else None


SemanticCacheRuntime = MLCacheRuntime
