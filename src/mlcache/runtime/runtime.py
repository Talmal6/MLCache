"""Runtime orchestrator boundary for MLCache."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mlcache.cache import CacheGatewayResult, KVStore, SemanticCacheGateway
from mlcache.calibration import QueryCalibrationRecord, QueryCalibrationRecordStore
from mlcache.feedback import JudgeTrainingStore, ShadowTopKCollector
from mlcache.observability import AuditEvent, AuditLogger, DiagnosticsReporter, MetricsSink
from mlcache.oracle import OracleFitResult, TrainableSemanticCacheOracle
from mlcache.policies import (
    QueryLevelLearnedPolicy,
    QueryLevelPolicyDecision,
    QueryLevelPolicyMode,
    QueryLevelShadowDecisionStore,
)
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
        query_level_policy: QueryLevelLearnedPolicy | None = None,
        query_level_shadow_store: QueryLevelShadowDecisionStore | None = None,
        query_record_store: QueryCalibrationRecordStore | None = None,
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
        self.query_level_policy = query_level_policy
        self.query_level_shadow_store = query_level_shadow_store
        self.query_record_store = query_record_store
        self.config = config or MLCacheRuntimeConfig()
        if self.config.query_level.enabled and self._query_level_mode() == QueryLevelPolicyMode.ACTIVE:
            raise NotImplementedError("active query-level serving is not implemented")
        self._diagnostics: list[dict[str, Any]] = []

    def lookup(self, request: CacheLookup) -> Response | None:
        return self.lookup_with_decision(request).response

    def lookup_with_decision(self, request: CacheLookup) -> CacheGatewayResult:
        result = self.gateway.lookup_with_decision(request)
        self._emit_observability(request, result)
        self._maybe_evaluate_query_level_shadow(request, result)
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
    def query_level_shadow_decisions(self) -> tuple[QueryLevelPolicyDecision, ...]:
        store = self.query_level_shadow_store
        if store is None:
            return ()
        return store.decisions()

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
            "query_level_policy": self._component_name(self.query_level_policy),
            "query_level_shadow_store": self._component_name(self.query_level_shadow_store),
            "query_record_store": self._component_name(self.query_record_store),
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

    def _maybe_evaluate_query_level_shadow(self, request: CacheLookup, result: CacheGatewayResult) -> None:
        if not self._query_level_shadow_enabled():
            return
        if self.query_level_policy is None or self.query_level_shadow_store is None or self.query_record_store is None:
            return

        try:
            record = self._latest_query_calibration_record(request)
            if record is None:
                return
            threshold = self.config.query_level.threshold
            if threshold is None:
                threshold = self.query_level_policy.threshold
            shadow_decision = self.query_level_policy.evaluate(
                record,
                threshold=threshold,
                metadata={
                    "source": "runtime_query_level_shadow",
                    "serving_status": result.decision.status.value,
                    "serving_accepted": bool(result.decision.accepted),
                },
            )
            self.query_level_shadow_store.add(shadow_decision)
            self._emit_query_level_shadow_observability(result, shadow_decision)
        except Exception as exc:
            self._record_observability_failure("query_level_shadow", exc)

    def _latest_query_calibration_record(self, request: CacheLookup) -> QueryCalibrationRecord | None:
        store = self.query_record_store
        if store is None:
            return None

        explicit_query_id = request.metadata.attributes.get("query_id") or request.metadata.attributes.get("request_id")
        query_id = self._query_id(request)
        fallback_query = str(request.query)
        for record in reversed(store.records()):
            if str(record.query_id) == query_id:
                return record
            if explicit_query_id is None and record.query is not None and str(record.query) == fallback_query:
                return record
        return None

    def _emit_query_level_shadow_observability(
        self,
        result: CacheGatewayResult,
        shadow_decision: QueryLevelPolicyDecision,
    ) -> None:
        metadata = self._query_level_shadow_metadata(result, shadow_decision)
        if self.metrics_sink is not None:
            try:
                self._record_query_level_shadow_metrics(metadata, shadow_decision)
            except Exception as exc:
                self._record_observability_failure("query_level_shadow_metrics", exc)

        if self.diagnostics_reporter is not None:
            try:
                self.diagnostics_reporter.report("cache.query_level_shadow", metadata)
            except Exception as exc:
                self._record_observability_failure("query_level_shadow_diagnostics", exc)

    def _record_query_level_shadow_metrics(
        self,
        metadata: dict[str, Any],
        shadow_decision: QueryLevelPolicyDecision,
    ) -> None:
        self.metrics_sink.record("cache.query_level_shadow.evaluated", 1.0, metadata)
        self.metrics_sink.record(
            "cache.query_level_shadow.accepted",
            1.0 if shadow_decision.accepted else 0.0,
            metadata,
        )
        self.metrics_sink.record(
            "cache.query_level_shadow.abstain",
            1.0 if shadow_decision.status == OracleDecisionStatus.ABSTAIN else 0.0,
            metadata,
        )
        self.metrics_sink.record(
            "cache.query_level_shadow.miss",
            1.0 if shadow_decision.status == OracleDecisionStatus.MISS else 0.0,
            metadata,
        )
        if shadow_decision.selected_candidate_rank is not None:
            self.metrics_sink.record(
                "cache.query_level_shadow.selected_rank",
                float(shadow_decision.selected_candidate_rank),
                metadata,
            )
        if shadow_decision.selected_score is not None:
            self.metrics_sink.record("cache.query_level_shadow.score", float(shadow_decision.selected_score), metadata)
        if shadow_decision.threshold is not None:
            self.metrics_sink.record("cache.query_level_shadow.threshold", float(shadow_decision.threshold), metadata)

    @staticmethod
    def _query_level_shadow_metadata(
        result: CacheGatewayResult,
        shadow_decision: QueryLevelPolicyDecision,
    ) -> dict[str, Any]:
        return {
            "serving_status": result.decision.status.value,
            "serving_accepted": bool(result.decision.accepted),
            "shadow_status": shadow_decision.status.value,
            "shadow_accepted": bool(shadow_decision.accepted),
            "selected_candidate_rank": shadow_decision.selected_candidate_rank,
            "selected_candidate_key": (
                str(shadow_decision.selected_candidate_key)
                if shadow_decision.selected_candidate_key is not None
                else None
            ),
            "reason": shadow_decision.reason,
            "threshold": float(shadow_decision.threshold) if shadow_decision.threshold is not None else None,
            "score": float(shadow_decision.selected_score) if shadow_decision.selected_score is not None else None,
        }

    def _query_level_shadow_enabled(self) -> bool:
        query_level = self.config.query_level
        return bool(query_level.enabled and self._query_level_mode() == QueryLevelPolicyMode.SHADOW)

    def _query_level_mode(self) -> QueryLevelPolicyMode:
        return QueryLevelPolicyMode(self.config.query_level.mode)

    @staticmethod
    def _query_id(request: CacheLookup) -> str:
        attributes = request.metadata.attributes
        query_id = attributes.get("query_id") or attributes.get("request_id")
        if query_id:
            return str(query_id)
        query = str(request.query).strip()
        return query if query else "anonymous_query"

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
