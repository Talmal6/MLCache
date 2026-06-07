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
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    OracleDecision,
    OracleDecisionStatus,
    Response,
    ScorerName,
    Threshold,
)


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
        self._diagnostics: list[dict[str, Any]] = []

    def lookup(self, request: CacheLookup) -> Response | None:
        return self.lookup_with_decision(request).response

    def lookup_with_decision(self, request: CacheLookup) -> CacheGatewayResult:
        pair_result = self.gateway.lookup_with_decision(request)
        if self._query_level_active_enabled():
            result = self._lookup_with_query_level_active(request, pair_result)
            self._emit_observability(request, result)
            return result

        self._emit_observability(request, pair_result)
        self._maybe_evaluate_query_level_shadow(request, pair_result)
        return pair_result

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

    def _lookup_with_query_level_active(
        self,
        request: CacheLookup,
        pair_result: CacheGatewayResult,
    ) -> CacheGatewayResult:
        metadata = self._query_level_active_base_metadata(pair_result)
        if self.query_level_policy is None:
            return self._query_level_active_no_record_or_policy_result(
                pair_result=pair_result,
                metadata=metadata,
                reason="query_level_policy_missing",
            )
        if self.query_record_store is None:
            return self._query_level_active_no_record_or_policy_result(
                pair_result=pair_result,
                metadata=metadata,
                reason="query_level_record_missing",
            )

        try:
            record = self._latest_query_calibration_record(request)
            if record is None:
                return self._query_level_active_no_record_or_policy_result(
                    pair_result=pair_result,
                    metadata=metadata,
                    reason="query_level_record_missing",
                )

            threshold = self._query_level_threshold()
            query_decision = self.query_level_policy.evaluate(
                record,
                threshold=threshold,
                metadata={
                    "source": "runtime_query_level_active",
                    "serving_status": pair_result.decision.status.value,
                    "serving_accepted": bool(pair_result.decision.accepted),
                },
            )
            metadata.update(self._query_level_active_decision_metadata(query_decision))

            if not query_decision.accepted:
                return self._query_level_active_rejected_or_abstained_result(
                    pair_result=pair_result,
                    query_decision=query_decision,
                    metadata=metadata,
                )

            cache_key = query_decision.selected_candidate_key
            if cache_key is None:
                return self._query_level_active_kv_miss_result(
                    pair_result=pair_result,
                    query_decision=query_decision,
                    metadata=metadata,
                    reason="query_level_selected_candidate_key_missing",
                )

            response = self.kv_store.get(cache_key)
            if response is None:
                self._remove_query_level_kv_miss_candidate(cache_key)
                return self._query_level_active_kv_miss_result(
                    pair_result=pair_result,
                    query_decision=query_decision,
                    metadata=metadata,
                    reason="query_level_kv_key_missing_or_expired",
                )

            metadata.update(
                {
                    "query_level_reason": "query_level_policy_accepted",
                    "reason": "query_level_policy_accepted",
                    "fallback_used": False,
                    "final_decision_source": "query_level_active",
                    "query_level_kv_miss": False,
                }
            )
            decision = self._oracle_decision_from_query_level(
                query_decision,
                accepted=True,
                cache_key=cache_key,
                status=OracleDecisionStatus.HIT,
                reason="query_level_policy_accepted",
            )
            result = CacheGatewayResult(response=response, decision=decision, metadata=metadata)
            self._emit_query_level_active_observability(metadata)
            return result
        except Exception as exc:
            self._record_observability_failure("query_level_active", exc)
            metadata.update(
                {
                    "query_level_status": OracleDecisionStatus.ABSTAIN.value,
                    "query_level_accepted": False,
                    "query_level_reason": "query_level_active_failed",
                    "reason": "query_level_active_failed",
                    "fallback_used": True,
                    "final_decision_source": "query_level_active_fallback_pair_level",
                    "query_level_error": repr(exc),
                }
            )
            result = self._fallback_pair_level_result(pair_result, metadata)
            self._emit_query_level_active_observability(metadata)
            return result

    def _query_level_active_no_record_or_policy_result(
        self,
        *,
        pair_result: CacheGatewayResult,
        metadata: dict[str, Any],
        reason: str,
    ) -> CacheGatewayResult:
        metadata.update(
            {
                "query_level_status": OracleDecisionStatus.ABSTAIN.value,
                "query_level_accepted": False,
                "query_level_selected_candidate_key": None,
                "query_level_selected_candidate_rank": None,
                "query_level_selected_score": None,
                "query_level_threshold": self._threshold_metadata_value(self._query_level_threshold()),
                "query_level_reason": reason,
                "reason": reason,
                "query_level_kv_miss": False,
            }
        )
        if self.config.query_level.fallback_to_pair_level_on_missing_record:
            metadata.update(
                {
                    "fallback_used": True,
                    "final_decision_source": "query_level_active_fallback_pair_level",
                }
            )
            result = self._fallback_pair_level_result(pair_result, metadata)
            self._emit_query_level_active_observability(metadata)
            return result

        metadata.update(
            {
                "fallback_used": False,
                "final_decision_source": "query_level_active_no_fallback",
            }
        )
        decision = self._query_level_no_response_decision(
            status=OracleDecisionStatus.ABSTAIN,
            reason=reason,
            threshold=self._query_level_threshold(),
        )
        result = CacheGatewayResult(response=None, decision=decision, metadata=metadata)
        self._emit_query_level_active_observability(metadata)
        return result

    def _query_level_active_rejected_or_abstained_result(
        self,
        *,
        pair_result: CacheGatewayResult,
        query_decision: QueryLevelPolicyDecision,
        metadata: dict[str, Any],
    ) -> CacheGatewayResult:
        reason = query_decision.reason or "query_level_policy_not_accepted"
        metadata.update(
            {
                "query_level_reason": reason,
                "reason": reason,
                "query_level_kv_miss": False,
            }
        )
        if self.config.query_level.fallback_to_pair_level_on_abstain:
            metadata.update(
                {
                    "fallback_used": True,
                    "final_decision_source": "query_level_active_fallback_pair_level",
                }
            )
            result = self._fallback_pair_level_result(pair_result, metadata)
            self._emit_query_level_active_observability(metadata)
            return result

        metadata.update(
            {
                "fallback_used": False,
                "final_decision_source": "query_level_active_no_fallback",
            }
        )
        status = query_decision.status
        if status == OracleDecisionStatus.HIT:
            status = OracleDecisionStatus.MISS
        decision = self._oracle_decision_from_query_level(
            query_decision,
            accepted=False,
            cache_key=None,
            status=status,
            reason=reason,
        )
        result = CacheGatewayResult(response=None, decision=decision, metadata=metadata)
        self._emit_query_level_active_observability(metadata)
        return result

    def _query_level_active_kv_miss_result(
        self,
        *,
        pair_result: CacheGatewayResult,
        query_decision: QueryLevelPolicyDecision,
        metadata: dict[str, Any],
        reason: str,
    ) -> CacheGatewayResult:
        metadata.update(
            {
                "query_level_reason": reason,
                "reason": reason,
                "query_level_kv_miss": True,
            }
        )
        if self.config.query_level.fallback_to_pair_level_on_kv_miss:
            metadata.update(
                {
                    "fallback_used": True,
                    "final_decision_source": "query_level_active_fallback_pair_level",
                }
            )
            result = self._fallback_pair_level_result(pair_result, metadata)
            self._emit_query_level_active_observability(metadata)
            return result

        metadata.update(
            {
                "fallback_used": False,
                "final_decision_source": "query_level_active_no_fallback",
            }
        )
        decision = self._oracle_decision_from_query_level(
            query_decision,
            accepted=False,
            cache_key=None,
            status=OracleDecisionStatus.MISS,
            reason=reason,
        )
        result = CacheGatewayResult(response=None, decision=decision, metadata=metadata)
        self._emit_query_level_active_observability(metadata)
        return result

    def _fallback_pair_level_result(
        self,
        pair_result: CacheGatewayResult,
        metadata: dict[str, Any],
    ) -> CacheGatewayResult:
        return CacheGatewayResult(
            response=pair_result.response,
            decision=pair_result.decision,
            metadata={**pair_result.metadata, **metadata},
        )

    def _oracle_decision_from_query_level(
        self,
        query_decision: QueryLevelPolicyDecision,
        *,
        accepted: bool,
        cache_key: CacheKey | None,
        status: OracleDecisionStatus,
        reason: str,
    ) -> OracleDecision:
        return OracleDecision(
            status=status,
            accepted=accepted,
            cache_key=cache_key,
            score=query_decision.selected_score,
            threshold=query_decision.threshold,
            scorer=ScorerName("QueryLevelLearnedPolicy"),
            candidate_count=int(query_decision.metadata.get("candidate_count", 0)),
            accepted_candidate_rank=query_decision.selected_candidate_rank if accepted else None,
            reason=reason,
            evidence={"query_level_policy": dict(query_decision.metadata)},
        )

    @staticmethod
    def _query_level_no_response_decision(
        *,
        status: OracleDecisionStatus,
        reason: str,
        threshold: Threshold | None,
    ) -> OracleDecision:
        return OracleDecision(
            status=status,
            accepted=False,
            cache_key=None,
            score=None,
            threshold=threshold,
            scorer=ScorerName("QueryLevelLearnedPolicy"),
            reason=reason,
            evidence={"query_level_active": True},
        )

    def _remove_query_level_kv_miss_candidate(self, cache_key: CacheKey) -> None:
        try:
            self.oracle.remove(cache_key)
        except Exception as exc:
            self._record_observability_failure("query_level_active_invalidate", exc)

    def _query_level_active_base_metadata(self, pair_result: CacheGatewayResult) -> dict[str, Any]:
        return {
            "query_level_active": True,
            "pair_level_status": pair_result.decision.status.value,
            "pair_level_accepted": bool(pair_result.decision.accepted),
            "query_level_status": None,
            "query_level_accepted": None,
            "query_level_selected_candidate_key": None,
            "query_level_selected_candidate_rank": None,
            "query_level_selected_score": None,
            "query_level_threshold": self._threshold_metadata_value(self._query_level_threshold()),
            "query_level_reason": None,
            "fallback_used": False,
            "final_decision_source": None,
        }

    @staticmethod
    def _query_level_active_decision_metadata(
        query_decision: QueryLevelPolicyDecision,
    ) -> dict[str, Any]:
        return {
            "query_level_status": query_decision.status.value,
            "query_level_accepted": bool(query_decision.accepted),
            "query_level_selected_candidate_key": (
                str(query_decision.selected_candidate_key)
                if query_decision.selected_candidate_key is not None
                else None
            ),
            "query_level_selected_candidate_rank": query_decision.selected_candidate_rank,
            "query_level_selected_score": (
                float(query_decision.selected_score) if query_decision.selected_score is not None else None
            ),
            "query_level_threshold": (
                float(query_decision.threshold) if query_decision.threshold is not None else None
            ),
        }

    def _query_level_threshold(self) -> Threshold | None:
        threshold = self.config.query_level.threshold
        if threshold is not None:
            return threshold
        if self.query_level_policy is None:
            return None
        return self.query_level_policy.threshold

    @staticmethod
    def _threshold_metadata_value(threshold: Threshold | None) -> float | None:
        return float(threshold) if threshold is not None else None

    def _emit_query_level_active_observability(self, metadata: dict[str, Any]) -> None:
        if self.metrics_sink is None:
            return
        try:
            self._record_query_level_active_metrics(metadata)
        except Exception as exc:
            self._record_observability_failure("query_level_active_metrics", exc)

    def _record_query_level_active_metrics(self, metadata: dict[str, Any]) -> None:
        status = metadata.get("query_level_status")
        query_accepted = bool(metadata.get("query_level_accepted"))
        fallback_used = bool(metadata.get("fallback_used"))
        kv_miss = bool(metadata.get("query_level_kv_miss"))
        self.metrics_sink.record("cache.query_level_active.evaluated", 1.0, metadata)
        self.metrics_sink.record("cache.query_level_active.accepted", 1.0 if query_accepted else 0.0, metadata)
        self.metrics_sink.record(
            "cache.query_level_active.rejected",
            1.0 if status == OracleDecisionStatus.MISS.value else 0.0,
            metadata,
        )
        self.metrics_sink.record(
            "cache.query_level_active.abstain",
            1.0 if status == OracleDecisionStatus.ABSTAIN.value else 0.0,
            metadata,
        )
        self.metrics_sink.record("cache.query_level_active.fallback_used", 1.0 if fallback_used else 0.0, metadata)
        self.metrics_sink.record("cache.query_level_active.kv_miss", 1.0 if kv_miss else 0.0, metadata)
        selected_rank = metadata.get("query_level_selected_candidate_rank")
        if selected_rank is not None:
            self.metrics_sink.record("cache.query_level_active.selected_rank", float(selected_rank), metadata)
        score = metadata.get("query_level_selected_score")
        if score is not None:
            self.metrics_sink.record("cache.query_level_active.score", float(score), metadata)

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

    def _query_level_active_enabled(self) -> bool:
        query_level = self.config.query_level
        return bool(query_level.enabled and self._query_level_mode() == QueryLevelPolicyMode.ACTIVE)

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
