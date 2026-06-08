"""Trainable semantic cache oracle implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import RLock, Thread
from typing import Any, Sequence

from mlcache.calibration import ThresholdCalibrationRequest, ThresholdProvider, ThresholdScope, wilson_upper_bound
from mlcache.features import PairFeatureBuilder, PairFeatures
from mlcache.feedback import (
    JudgeLabel,
    JudgeRequest,
    JudgedPairExample,
    JudgeTrainingStore,
    SemanticReuseJudge,
    ShadowCollectionResult,
    ShadowTopKCollector,
    SplitJudgeTrainingStore,
)
from mlcache.online import OnlineBatch, OnlineUpdater, StopStatus
from mlcache.oracle.base import SemanticCacheOracle
from mlcache.oracle.fit import ActivationGateResult, OracleFitResult
from mlcache.oracle.runtime import OracleJudgeFeedback, OracleRuntimeSnapshot, OracleScoredResult
from mlcache.policies.refit import (
    ConservativeRefitConfig,
    ConservativeRefitPolicy,
    RefitAction,
    RefitPolicy,
    RefitPolicyContext,
    RefitPolicyDecision,
)
from mlcache.retrieval import VectorSearchResult, VectorStore
from mlcache.scorers import SemanticScorer
from mlcache.semantic_types import (
    CacheEntry,
    CacheKey,
    CacheLookup,
    LabeledPairBatch,
    OracleDecision,
    OracleDecisionStatus,
    Score,
    Threshold,
    TieMode,
    TrainCalibEvalSplit,
)


@dataclass(frozen=True, slots=True)
class _RefitRows:
    h0_rows: tuple[tuple[float, ...], ...]
    h1_rows: tuple[tuple[float, ...], ...]
    h0_train: tuple[tuple[float, ...], ...] = ()
    h1_train: tuple[tuple[float, ...], ...] = ()
    h0_calibration: tuple[tuple[float, ...], ...] = ()
    h1_calibration: tuple[tuple[float, ...], ...] = ()
    explicit_split: bool = False


class TrainableSemanticCacheOracle(SemanticCacheOracle):
    """Trainable oracle that owns scorer fitting, NP thresholding, and candidate decisions."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        feature_builder: PairFeatureBuilder,
        scorer: SemanticScorer,
        threshold_provider: ThresholdProvider | None = None,
        online_updater: OnlineUpdater | None = None,
        judge: SemanticReuseJudge | None = None,
        judge_training_store: JudgeTrainingStore | None = None,
        shadow_collector: ShadowTopKCollector | None = None,
        shadow_collection_enabled: bool = False,
        refit_policy: RefitPolicy | None = None,
        auto_refit: bool = True,
        judge_for_feedback: bool = True,
        judge_can_override_decision: bool = False,
        sync_recalibration_max_h0: int = 10_000,
        monitor_window_min_examples: int = 30,
        target_false_accept_rate: float = 0.05,
        tie_mode: TieMode = TieMode.GE,
        top_k: int = 1,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if sync_recalibration_max_h0 <= 0:
            raise ValueError("sync_recalibration_max_h0 must be positive")
        if monitor_window_min_examples <= 0:
            raise ValueError("monitor_window_min_examples must be positive")
        if not 0.0 < target_false_accept_rate < 1.0:
            raise ValueError("target_false_accept_rate must be in (0,1)")

        self.vector_store = vector_store
        self.feature_builder = feature_builder
        self.scorer = scorer
        self.threshold_provider = threshold_provider
        self.online_updater = online_updater
        self.judge = judge
        self.judge_training_store = judge_training_store
        self.shadow_collector = shadow_collector
        self.shadow_collection_enabled = bool(shadow_collection_enabled)
        self.refit_policy = refit_policy or ConservativeRefitPolicy()
        self.auto_refit = bool(auto_refit)
        self.judge_for_feedback = bool(judge_for_feedback)
        self.judge_can_override_decision = bool(judge_can_override_decision)
        self.sync_recalibration_max_h0 = int(sync_recalibration_max_h0)
        self.monitor_window_min_examples = int(monitor_window_min_examples)
        self.target_false_accept_rate = float(target_false_accept_rate)
        self.tie_mode = tie_mode
        self.top_k = int(top_k)
        self._threshold: Threshold | None = None
        self._threshold_version = 0
        self._semantic_hits_disabled = False
        self._fit_result: OracleFitResult | None = None
        self._activation_metadata: dict[str, Any] = {}
        self._last_refit_decision: RefitPolicyDecision | None = None
        self._new_h0_since_fit = 0
        self._new_h1_since_fit = 0
        self._new_h0_since_calibration = 0
        self._decisions_since_fit = 0
        self._decisions_since_calibration = 0
        self._monitor_h0_count = 0
        self._monitor_h1_count = 0
        self._monitor_false_accepts = 0
        self._monitor_true_accepts = 0
        self._previous_monitor_tpr: float | None = None
        self._fit_lock = RLock()
        self._fit_thread: Thread | None = None
        self._fit_exception: BaseException | None = None
        self._recalibration_thread: Thread | None = None
        self._recalibration_exception: BaseException | None = None

    @property
    def fit_result(self) -> OracleFitResult | None:
        with self._fit_lock:
            return self._fit_result

    @property
    def last_fit_result(self) -> OracleFitResult | None:
        return self.fit_result

    @property
    def threshold(self) -> Threshold | None:
        with self._fit_lock:
            return self._threshold

    @property
    def semantic_hits_disabled(self) -> bool:
        with self._fit_lock:
            return self._semantic_hits_disabled

    @property
    def activation_status(self) -> dict[str, Any]:
        with self._fit_lock:
            metadata = dict(self._fit_result.metadata) if self._fit_result is not None else {}
            metadata.update(self._activation_metadata)
            metadata.update(
                {
                    "threshold": None if self._threshold is None else float(self._threshold),
                    "threshold_version": self._threshold_version,
                    "semantic_hits_disabled": self._semantic_hits_disabled,
                }
            )
            return metadata

    @property
    def last_refit_decision(self) -> RefitPolicyDecision | None:
        with self._fit_lock:
            return self._last_refit_decision

    @property
    def fit_in_progress(self) -> bool:
        thread = self._fit_thread
        return thread is not None and thread.is_alive()

    def decide(self, request: CacheLookup) -> OracleDecision:
        # TODO: This currently implements pair-level direct threshold serving.
        # Production default should eventually be fallback-first with a query-level calibrated policy.
        if self.shadow_collection_enabled and self.shadow_collector is not None:
            return self._decide_with_shadow_collection(request)

        snapshot = self._snapshot_runtime_state(request)
        scored = self._score_request_with_snapshot(request, snapshot)
        feedback = self._maybe_collect_judge_feedback(
            request=request,
            scored=scored,
            snapshot=snapshot,
        )
        refit_decision = self._apply_feedback_and_maybe_maintain(
            feedback=feedback,
            scorer_decision=scored.decision,
            snapshot=snapshot,
        )
        return self._finalize_decision(
            scorer_decision=scored.decision,
            feedback=feedback,
            refit_decision=refit_decision,
        )

    def _decide_with_shadow_collection(self, request: CacheLookup) -> OracleDecision:
        snapshot = self._snapshot_runtime_state(request)
        scored = self._score_request_with_snapshot(request, snapshot)

        # Shadow top-k collection independently judges *every* retrieved
        # candidate -- including ones the active policy rejected -- and is
        # the sole source of judged-pair accumulation and refit/monitoring
        # signal on this path, so calibration sees hard negatives from
        # retrieved-but-rejected candidates rather than only from whichever
        # single candidate the policy happened to serve. The legacy
        # single-candidate judge-feedback path is only consulted when the
        # judge is allowed to override the serving decision; even then its
        # verdict is used solely to finalize the decision, not stored again,
        # so the served candidate is never double-judged or double-counted.
        feedback = None
        if self.judge_can_override_decision:
            feedback = self._maybe_collect_judge_feedback(
                request=request,
                scored=scored,
                snapshot=snapshot,
            )

        serving_decision = self._finalize_decision(
            scorer_decision=scored.decision,
            feedback=feedback,
            refit_decision=None,
        )
        shadow_result = self._maybe_collect_shadow_feedback(
            request=request,
            candidates=scored.candidates,
            served_decision=serving_decision,
        )
        if shadow_result is not None:
            self._record_shadow_collection_for_monitoring(
                result=shadow_result,
                served_decision=serving_decision,
            )
        self._maybe_auto_refit(current_threshold=snapshot.threshold)
        return serving_decision

    def _snapshot_runtime_state(self, request: CacheLookup) -> OracleRuntimeSnapshot:
        top_k = self._resolve_top_k(request)
        with self._fit_lock:
            self._decisions_since_fit += 1
            self._decisions_since_calibration += 1
            scorer = self.scorer
            local_threshold = self._threshold
            threshold_provider = self.threshold_provider
            tie_mode = self.tie_mode
            disabled = self._semantic_hits_disabled
            threshold_version = str(self._threshold_version)

        threshold = local_threshold
        if threshold_provider is not None and not disabled:
            try:
                threshold = threshold_provider.get_threshold(
                    scorer=scorer.name,
                    scope=ThresholdScope.GLOBAL,
                    context={"request": request},
                )
            except Exception:
                threshold = local_threshold

        return OracleRuntimeSnapshot(
            scorer=scorer,
            threshold=threshold,
            top_k=top_k,
            tie_mode=tie_mode,
            scorer_name=str(scorer.name),
            threshold_version=threshold_version,
            semantic_hits_disabled=disabled,
        )

    def _score_request_with_snapshot(
        self,
        request: CacheLookup,
        snapshot: OracleRuntimeSnapshot,
    ) -> OracleScoredResult:
        candidates = self.vector_store.search(
            request.embedding,
            namespace=request.namespace,
            top_k=snapshot.top_k,
        )
        if not candidates:
            return OracleScoredResult(
                decision=OracleDecision(
                    status=OracleDecisionStatus.MISS,
                    accepted=False,
                    cache_key=None,
                    score=None,
                    threshold=snapshot.threshold,
                    scorer=snapshot.scorer.name,
                    reason="no_neighbors",
                )
            )

        feedback_candidate = candidates[0]
        feedback_rank = 1
        feedback_features = self.feature_builder.build(request.embedding, feedback_candidate.embedding)
        feedback_score = self._safe_score(snapshot.scorer, feedback_features)

        if snapshot.semantic_hits_disabled:
            return OracleScoredResult(
                decision=OracleDecision(
                    status=OracleDecisionStatus.MISS,
                    accepted=False,
                    cache_key=None,
                    score=feedback_score,
                    threshold=snapshot.threshold,
                    scorer=snapshot.scorer.name,
                    candidate_count=len(candidates),
                    reason="semantic_hits_disabled",
                    evidence={"candidate_key": str(feedback_candidate.cache_key)},
                ),
                candidates=tuple(candidates),
                feedback_candidate=feedback_candidate,
                feedback_candidate_rank=feedback_rank,
                feedback_features=feedback_features,
                feedback_score=feedback_score,
            )

        if snapshot.threshold is None:
            return OracleScoredResult(
                decision=OracleDecision(
                    status=OracleDecisionStatus.ABSTAIN,
                    accepted=False,
                    cache_key=None,
                    score=feedback_score,
                    threshold=None,
                    scorer=snapshot.scorer.name,
                    candidate_count=len(candidates),
                    reason="oracle_not_fitted",
                    evidence={
                        "candidate_key": str(feedback_candidate.cache_key),
                        "vector_score": float(feedback_candidate.score),
                    },
                ),
                candidates=tuple(candidates),
                feedback_candidate=feedback_candidate,
                feedback_candidate_rank=feedback_rank,
                feedback_features=feedback_features,
                feedback_score=feedback_score,
            )

        best_key: CacheKey | None = None
        best_candidate: VectorSearchResult | None = None
        best_features: PairFeatures | None = None
        best_score: Score | None = None
        best_rank: int | None = None
        scored: list[dict[str, Any]] = []

        for rank, candidate in enumerate(candidates, start=1):
            features = feedback_features if rank == 1 else self.feature_builder.build(
                request.embedding,
                candidate.embedding,
            )
            score = feedback_score if rank == 1 and feedback_score is not None else snapshot.scorer.score(features)
            accepted = self._predict_score(score, snapshot.threshold, snapshot.tie_mode)
            scored.append(
                {
                    "rank": rank,
                    "cache_key": str(candidate.cache_key),
                    "score": float(score),
                    "accepted": accepted,
                    "vector_score": float(candidate.score),
                }
            )
            if accepted and (best_score is None or float(score) > float(best_score)):
                best_key = candidate.cache_key
                best_candidate = candidate
                best_features = features
                best_score = score
                best_rank = rank

        if best_key is None:
            return OracleScoredResult(
                decision=OracleDecision(
                    status=OracleDecisionStatus.MISS,
                    accepted=False,
                    cache_key=None,
                    score=max((Score(item["score"]) for item in scored), default=None),
                    threshold=snapshot.threshold,
                    scorer=snapshot.scorer.name,
                    candidate_count=len(candidates),
                    reason="no_candidate_passed_threshold",
                    evidence={"candidates": scored},
                ),
                candidates=tuple(candidates),
                feedback_candidate=feedback_candidate,
                feedback_candidate_rank=feedback_rank,
                feedback_features=feedback_features,
                feedback_score=feedback_score,
            )

        return OracleScoredResult(
            decision=OracleDecision(
                status=OracleDecisionStatus.HIT,
                accepted=True,
                cache_key=best_key,
                score=best_score,
                threshold=snapshot.threshold,
                scorer=snapshot.scorer.name,
                candidate_count=len(candidates),
                accepted_candidate_rank=best_rank,
                evidence={"candidates": scored},
            ),
            candidates=tuple(candidates),
            feedback_candidate=best_candidate,
            feedback_candidate_rank=best_rank,
            feedback_features=best_features,
            feedback_score=best_score,
        )

    def fit(self, split: TrainCalibEvalSplit) -> None:
        thread = self.fit_async(split)
        thread.join()
        self._raise_fit_exception()

    def fit_async(self, split: TrainCalibEvalSplit) -> Thread:
        with self._fit_lock:
            if self.fit_in_progress:
                raise RuntimeError("fit() is already running")
            self._fit_exception = None
            thread = Thread(
                target=self._fit_worker,
                args=(split,),
                name=f"{self.__class__.__name__}.fit",
                daemon=False,
            )
            self._fit_thread = thread

        thread.start()
        return thread

    def wait_for_fit(self, timeout: float | None = None) -> bool:
        thread = self._fit_thread
        if thread is None:
            return True

        thread.join(timeout)
        if thread.is_alive():
            return False

        self._raise_fit_exception()
        return True

    def _fit_worker(self, split: TrainCalibEvalSplit) -> None:
        try:
            self._fit_in_current_thread(split)
        except BaseException as exc:
            with self._fit_lock:
                self._fit_exception = exc

    def _fit_in_current_thread(self, split: TrainCalibEvalSplit) -> None:
        with self._fit_lock:
            old_scorer = self.scorer
            alpha = self.target_false_accept_rate
            tie_mode = self.tie_mode
            threshold_provider = self.threshold_provider

        new_scorer = old_scorer.copy_for_refit()
        new_scorer.fit(
            LabeledPairBatch(
                h0=split.h0_train,
                h1=split.h1_train,
                weights=split.metadata.get("weights"),
                metadata=split.metadata,
            ),
            alpha=alpha,
        )

        h0_scores = [new_scorer.score(self._row_features(row)) for row in split.h0_calib]
        threshold = new_scorer.calibrate(
            ThresholdCalibrationRequest(
                h0_scores=h0_scores,
                target_false_accept_rate=alpha,
                tie_mode=tie_mode,
                context=split.metadata,
            )
        )
        gate = self._check_activation_gate(
            threshold=threshold,
            calibration_h0_scores=h0_scores,
            split=split,
            metadata=split.metadata,
        )
        fit_metadata_base = {
            **split.metadata,
            **gate.metadata,
            "candidate_threshold": threshold,
            "activation_gate_passed": gate.passed,
            "activation_gate_reason": gate.reason,
        }

        with self._fit_lock:
            active_threshold = threshold if gate.passed else self._threshold
            fit_metadata = {
                **fit_metadata_base,
                "active_threshold": active_threshold,
            }
            self._fit_result = OracleFitResult(
                scorer=str(new_scorer.name),
                threshold=threshold,
                target_false_accept_rate=alpha,
                n_h0_train=len(split.h0_train),
                n_h1_train=len(split.h1_train),
                n_h0_calib=len(split.h0_calib),
                n_h1_calib=len(split.h1_calib),
                metadata=fit_metadata,
            )
            self._activation_metadata = dict(fit_metadata)
            if not gate.passed:
                config = self._activation_config()
                if config.deactivate_on_failed_refit:
                    self._semantic_hits_disabled = True
                self._last_refit_decision = RefitPolicyDecision(
                    action=RefitAction.NOOP,
                    reason="activation_gate_failed",
                    metadata=fit_metadata,
                )
                return

            self.scorer = new_scorer
            self._threshold = threshold
            self._threshold_version += 1
            self._semantic_hits_disabled = False
            self._new_h0_since_fit = 0
            self._new_h1_since_fit = 0
            self._new_h0_since_calibration = 0
            self._decisions_since_fit = 0
            self._decisions_since_calibration = 0
            self._reset_monitor_counts()

        if threshold_provider is not None:
            try:
                threshold_provider.set_threshold(
                    threshold,
                    scorer=new_scorer.name,
                    scope=ThresholdScope.GLOBAL,
                    context=fit_metadata,
                )
            except Exception:
                pass

    def _raise_fit_exception(self) -> None:
        with self._fit_lock:
            exc = self._fit_exception
        if exc is not None:
            raise exc

    def _check_activation_gate(
        self,
        *,
        threshold: Threshold,
        calibration_h0_scores: Sequence[Score],
        split: TrainCalibEvalSplit,
        metadata: dict[str, Any],
    ) -> ActivationGateResult:
        del metadata
        return self._activation_gate_from_counts(
            threshold=threshold,
            calibration_h0_scores=calibration_h0_scores,
            n_h0_train=len(split.h0_train),
            n_h1_train=len(split.h1_train),
            n_h0_calib=len(split.h0_calib),
            n_h1_calib=len(split.h1_calib),
            require_train_gates=True,
            require_calibration_h1=True,
        )

    def _check_recalibration_gate(
        self,
        *,
        threshold: Threshold,
        calibration_h0_scores: Sequence[Score],
    ) -> ActivationGateResult:
        return self._activation_gate_from_counts(
            threshold=threshold,
            calibration_h0_scores=calibration_h0_scores,
            n_h0_train=0,
            n_h1_train=0,
            n_h0_calib=len(calibration_h0_scores),
            n_h1_calib=0,
            require_train_gates=False,
            require_calibration_h1=False,
        )

    def _activation_gate_from_counts(
        self,
        *,
        threshold: Threshold,
        calibration_h0_scores: Sequence[Score],
        n_h0_train: int,
        n_h1_train: int,
        n_h0_calib: int,
        n_h1_calib: int,
        require_train_gates: bool,
        require_calibration_h1: bool,
    ) -> ActivationGateResult:
        config = self._activation_config()
        train_total = int(n_h0_train) + int(n_h1_train)
        threshold_is_finite = isfinite(float(threshold))
        calibration_h0_accepts = sum(
            1 for score in calibration_h0_scores if self._predict_score(score, threshold, self.tie_mode)
        )
        empirical_fpr = (
            None if n_h0_calib <= 0 else float(calibration_h0_accepts) / float(n_h0_calib)
        )
        wilson_upper = wilson_upper_bound(
            int(calibration_h0_accepts),
            int(n_h0_calib),
            z=config.wilson_confidence_z,
        )
        allowed_fpr_bound = self.target_false_accept_rate + config.fpr_wilson_margin

        reason: str | None = None
        if require_train_gates and train_total < config.min_train_total:
            reason = "min_train_total_not_met"
        elif require_train_gates and n_h0_train < config.min_train_h0:
            reason = "min_train_h0_not_met"
        elif require_train_gates and n_h1_train < config.min_train_h1:
            reason = "min_train_h1_not_met"
        elif n_h0_calib < config.min_calibration_h0:
            reason = "min_calibration_h0_not_met"
        elif require_calibration_h1 and n_h1_calib < config.min_calibration_h1:
            reason = "min_calibration_h1_not_met"
        elif config.require_finite_threshold and not threshold_is_finite:
            reason = "threshold_not_finite"
        elif wilson_upper is None:
            reason = "wilson_upper_fpr_unavailable"
        elif float(wilson_upper) > float(allowed_fpr_bound):
            reason = "wilson_upper_fpr_exceeds_bound"

        passed = reason is None
        diagnostics = {
            "train_total": train_total,
            "n_h0_train": int(n_h0_train),
            "n_h1_train": int(n_h1_train),
            "n_h0_calib": int(n_h0_calib),
            "n_h1_calib": int(n_h1_calib),
            "calibration_h0_accepts": int(calibration_h0_accepts),
            "empirical_calibration_fpr": empirical_fpr,
            "wilson_upper_fpr": wilson_upper,
            "allowed_fpr_bound": allowed_fpr_bound,
            "threshold_is_finite": threshold_is_finite,
            "activation_gate_passed": passed,
            "activation_gate_reason": reason,
        }
        return ActivationGateResult(passed=passed, reason=reason, metadata=diagnostics)

    def _activation_config(self) -> ConservativeRefitConfig:
        config = getattr(self.refit_policy, "config", None)
        if isinstance(config, ConservativeRefitConfig):
            return config
        return ConservativeRefitConfig()

    def update_online(self, batch: OnlineBatch) -> StopStatus:
        if self.online_updater is None:
            return StopStatus(
                stopped=False,
                reason="online_updater_not_configured",
                metadata={"batch_size": len(batch.features)},
            )
        self.online_updater.update_batch(batch)
        return StopStatus(
            stopped=False,
            reason="online_batch_applied",
            metadata={"batch_size": len(batch.features)},
        )

    def index(self, entry: CacheEntry) -> None:
        self.vector_store.upsert(entry)

    def remove(self, cache_key: CacheKey) -> None:
        self.vector_store.delete(cache_key)

    def _maybe_auto_refit(self, *, current_threshold: Threshold | None) -> RefitPolicyDecision | None:
        if not self.auto_refit or self.judge_training_store is None:
            return None

        try:
            refit_rows = self._refit_rows_from_training_store(self.judge_training_store)
            with self._fit_lock:
                monitor_fpr = self._monitor_fpr_locked()
                monitor_tpr = self._monitor_tpr_locked()
                monitor_fpr_upper_bound = self._monitor_fpr_upper_bound_locked()
                context = RefitPolicyContext(
                    total_h0=len(refit_rows.h0_rows),
                    total_h1=len(refit_rows.h1_rows),
                    target_false_accept_rate=self.target_false_accept_rate,
                    current_threshold=current_threshold,
                    new_h0_since_fit=self._new_h0_since_fit,
                    new_h1_since_fit=self._new_h1_since_fit,
                    new_h0_since_calibration=self._new_h0_since_calibration,
                    monitor_fpr=monitor_fpr,
                    monitor_fpr_upper_bound=monitor_fpr_upper_bound,
                    monitor_tpr=monitor_tpr,
                    previous_monitor_tpr=self._previous_monitor_tpr,
                    decisions_since_fit=self._decisions_since_fit,
                    decisions_since_calibration=self._decisions_since_calibration,
                    fit_in_progress=self.fit_in_progress,
                    metadata={"source": "decide"},
                )

            decision = self.refit_policy.decide(context)
            return self._execute_refit_decision(decision, refit_rows=refit_rows)
        except Exception as exc:
            failed = RefitPolicyDecision(
                action=RefitAction.NOOP,
                reason="auto_refit_failed",
                metadata={"error": repr(exc)},
            )
            with self._fit_lock:
                self._last_refit_decision = failed
            return failed

    def _execute_refit_decision(
        self,
        decision: RefitPolicyDecision,
        *,
        refit_rows: _RefitRows,
    ) -> RefitPolicyDecision:
        with self._fit_lock:
            self._last_refit_decision = decision

        if decision.action == RefitAction.RECALIBRATE_THRESHOLD:
            h0_recalibration_rows = (
                refit_rows.h0_calibration if refit_rows.explicit_split else refit_rows.h0_rows
            )
            if len(h0_recalibration_rows) > self.sync_recalibration_max_h0:
                self._start_recalibration_async(h0_recalibration_rows, metadata=decision.metadata)
                return decision
            threshold = self._recalibrate_threshold_from_rows(h0_recalibration_rows, metadata=decision.metadata)
            if threshold is not None:
                return decision
            skipped = RefitPolicyDecision(
                action=RefitAction.NOOP,
                reason="threshold_recalibration_not_activated",
                metadata=self.activation_status,
            )
            with self._fit_lock:
                self._last_refit_decision = skipped
            return skipped

        if decision.action == RefitAction.REFIT_SCORER:
            if refit_rows.explicit_split:
                split = self._build_explicit_refit_split(refit_rows, metadata=decision.metadata)
            else:
                split = self._build_refit_split(
                    refit_rows.h0_rows,
                    refit_rows.h1_rows,
                    metadata=decision.metadata,
                )
            if split is None:
                skipped = RefitPolicyDecision(
                    action=RefitAction.NOOP,
                    reason="refit_split_not_ready",
                    metadata=decision.metadata,
                )
                with self._fit_lock:
                    self._last_refit_decision = skipped
                return skipped
            self.fit_async(split)

        if decision.action == RefitAction.DISABLE_SEMANTIC_HITS:
            with self._fit_lock:
                self._semantic_hits_disabled = True
                self._threshold = Threshold(float("inf"))
                self._threshold_version += 1

        return decision

    def _refit_rows_from_training_store(self, store: JudgeTrainingStore) -> _RefitRows:
        if isinstance(store, SplitJudgeTrainingStore):
            h0_train = self._feature_rows(store.h0_train())
            h1_train = self._feature_rows(store.h1_train())
            h0_calibration = self._feature_rows(store.h0_calibration())
            h1_calibration = self._feature_rows(store.h1_calibration())
            return _RefitRows(
                h0_rows=h0_train + h0_calibration,
                h1_rows=h1_train + h1_calibration,
                h0_train=h0_train,
                h1_train=h1_train,
                h0_calibration=h0_calibration,
                h1_calibration=h1_calibration,
                explicit_split=True,
            )

        return _RefitRows(
            h0_rows=self._feature_rows(store.h0()),
            h1_rows=self._feature_rows(store.h1()),
        )

    def _start_recalibration_async(
        self,
        h0_rows: tuple[tuple[float, ...], ...],
        *,
        metadata: dict[str, Any],
    ) -> None:
        with self._fit_lock:
            if self._recalibration_thread is not None and self._recalibration_thread.is_alive():
                return
            self._recalibration_exception = None
            thread = Thread(
                target=self._recalibration_worker,
                args=(h0_rows, metadata),
                name=f"{self.__class__.__name__}.recalibrate",
                daemon=True,
            )
            self._recalibration_thread = thread
        thread.start()

    def _recalibration_worker(
        self,
        h0_rows: tuple[tuple[float, ...], ...],
        metadata: dict[str, Any],
    ) -> None:
        try:
            self._recalibrate_threshold_from_rows(h0_rows, metadata=metadata)
        except BaseException as exc:
            failed = RefitPolicyDecision(
                action=RefitAction.NOOP,
                reason="auto_recalibration_failed",
                metadata={"error": repr(exc)},
            )
            with self._fit_lock:
                self._recalibration_exception = exc
                self._last_refit_decision = failed

    def _recalibrate_threshold_from_rows(
        self,
        h0_rows: tuple[tuple[float, ...], ...],
        *,
        metadata: dict[str, Any],
    ) -> Threshold | None:
        if not h0_rows:
            return None

        with self._fit_lock:
            scorer = self.scorer
            scorer_name = scorer.name
            alpha = self.target_false_accept_rate
            tie_mode = self.tie_mode
            threshold_provider = self.threshold_provider

        h0_scores = [scorer.score(self._row_features(row)) for row in h0_rows]
        threshold = scorer.calibrate(
            ThresholdCalibrationRequest(
                h0_scores=h0_scores,
                target_false_accept_rate=alpha,
                tie_mode=tie_mode,
                context=metadata,
            )
        )
        gate = self._check_recalibration_gate(
            threshold=threshold,
            calibration_h0_scores=h0_scores,
        )
        recalibration_metadata_base = {
            **metadata,
            **gate.metadata,
            "candidate_threshold": threshold,
            "activation_gate_passed": gate.passed,
            "activation_gate_reason": gate.reason,
        }

        if not gate.passed:
            with self._fit_lock:
                recalibration_metadata = {
                    **recalibration_metadata_base,
                    "active_threshold": self._threshold,
                }
                config = self._activation_config()
                if config.deactivate_on_failed_refit and self._threshold is None:
                    self._semantic_hits_disabled = True
                self._activation_metadata = dict(recalibration_metadata)
                self._last_refit_decision = RefitPolicyDecision(
                    action=RefitAction.NOOP,
                    reason="threshold_activation_gate_failed",
                    metadata=recalibration_metadata,
                )
            return None

        with self._fit_lock:
            recalibration_metadata = {
                **recalibration_metadata_base,
                "active_threshold": threshold,
            }
            self._threshold = threshold
            self._threshold_version += 1
            self._semantic_hits_disabled = False
            self._activation_metadata = dict(recalibration_metadata)
            self._new_h0_since_calibration = 0
            self._decisions_since_calibration = 0
            self._reset_monitor_counts()

        if threshold_provider is not None:
            try:
                threshold_provider.set_threshold(
                    threshold,
                    scorer=scorer_name,
                    scope=ThresholdScope.GLOBAL,
                    context=recalibration_metadata,
                )
            except Exception:
                pass

        return threshold

    def _build_refit_split(
        self,
        h0_rows: tuple[tuple[float, ...], ...],
        h1_rows: tuple[tuple[float, ...], ...],
        *,
        metadata: dict[str, Any],
    ) -> TrainCalibEvalSplit | None:
        if not h0_rows or not h1_rows:
            return None

        h0_train, h0_calib, h0_eval = self._split_rows(h0_rows)
        h1_train, h1_calib, h1_eval = self._split_rows(h1_rows)
        if not h0_train or not h1_train or not h0_calib:
            return None

        return TrainCalibEvalSplit(
            h0_train=h0_train,
            h1_train=h1_train,
            h0_calib=h0_calib,
            h1_calib=h1_calib,
            h0_eval=h0_eval,
            h1_eval=h1_eval,
            metadata={
                **metadata,
                "source": "decide_auto_refit",
                "split_source": "legacy_resplit",
                "n_h0_total": len(h0_rows),
                "n_h1_total": len(h1_rows),
            },
        )

    def _build_explicit_refit_split(
        self,
        refit_rows: _RefitRows,
        *,
        metadata: dict[str, Any],
    ) -> TrainCalibEvalSplit | None:
        if not refit_rows.h0_train or not refit_rows.h1_train or not refit_rows.h0_calibration:
            return None

        return TrainCalibEvalSplit(
            h0_train=refit_rows.h0_train,
            h1_train=refit_rows.h1_train,
            h0_calib=refit_rows.h0_calibration,
            h1_calib=refit_rows.h1_calibration,
            h0_eval=(),
            h1_eval=(),
            metadata={
                **metadata,
                "source": "decide_auto_refit",
                "split_source": "split_judge_training_store",
                "n_h0_total": len(refit_rows.h0_rows),
                "n_h1_total": len(refit_rows.h1_rows),
            },
        )

    @staticmethod
    def _feature_rows(examples: tuple[JudgedPairExample, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(example.features for example in examples if example.features)

    @staticmethod
    def _split_rows(
        rows: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        n = len(rows)
        if n == 0:
            return (), (), ()
        if n == 1:
            return rows, (), ()
        if n == 2:
            return rows[:1], rows[1:], ()

        train_count = max(1, int(n * 0.7))
        calib_count = max(1, int(n * 0.2))
        if train_count + calib_count > n:
            train_count = n - calib_count
        eval_start = train_count + calib_count
        return rows[:train_count], rows[train_count:eval_start], rows[eval_start:]

    def _monitor_fpr_locked(self) -> float | None:
        if self._monitor_h0_count <= 0:
            return None
        return float(self._monitor_false_accepts) / float(self._monitor_h0_count)

    def _monitor_fpr_upper_bound_locked(self) -> float | None:
        return wilson_upper_bound(int(self._monitor_false_accepts), int(self._monitor_h0_count))

    def _monitor_tpr_locked(self) -> float | None:
        if self._monitor_h1_count <= 0:
            return None
        return float(self._monitor_true_accepts) / float(self._monitor_h1_count)

    def _reset_monitor_counts(self) -> None:
        if self._monitor_h0_count + self._monitor_h1_count >= self.monitor_window_min_examples:
            monitor_tpr = self._monitor_tpr_locked()
            if monitor_tpr is not None:
                self._previous_monitor_tpr = monitor_tpr
        self._monitor_h0_count = 0
        self._monitor_h1_count = 0
        self._monitor_false_accepts = 0
        self._monitor_true_accepts = 0

    def _resolve_top_k(self, request: CacheLookup) -> int:
        raw_top_k = request.metadata.attributes.get("top_k")
        if raw_top_k is None:
            return self.top_k
        top_k = int(raw_top_k)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return top_k

    @staticmethod
    def _row_features(row: Any) -> PairFeatures:
        return PairFeatures(hadamard=tuple(float(value) for value in row))

    def _maybe_collect_judge_feedback(
        self,
        *,
        request: CacheLookup,
        scored: OracleScoredResult,
        snapshot: OracleRuntimeSnapshot,
    ) -> OracleJudgeFeedback | None:
        # Single-candidate judge feedback on the served/top candidate. This is
        # the *non*-shadow `decide()` path's only source of judged pairs and
        # refit/monitoring signal -- intentionally tied to the served
        # candidate, since that's the only candidate whose accept/reject
        # outcome it can evaluate. When shadow collection is enabled,
        # `_decide_with_shadow_collection` judges and accumulates the full
        # top-k independently of serving (see `_record_shadow_collection_for_monitoring`)
        # and only reaches this method when `judge_can_override_decision` needs
        # a verdict to finalize the decision -- never to store or count it
        # again, so the served candidate is never double-judged.
        if not self._should_call_judge() or scored.feedback_candidate is None:
            return None

        candidate = scored.feedback_candidate
        features = scored.feedback_features or self.feature_builder.build(request.embedding, candidate.embedding)
        score = scored.feedback_score
        if score is None:
            score = self._safe_score(snapshot.scorer, features)

        judge_request = JudgeRequest(
            query=request.query,
            candidate_query=candidate.query,
            candidate_key=candidate.cache_key,
            metadata=candidate.metadata,
            context={
                "request_metadata": request.metadata,
                "vector_score": float(candidate.score),
            },
        )
        judge_result = self.judge.judge(judge_request)
        return OracleJudgeFeedback(
            result=judge_result,
            candidate=candidate,
            features=features,
            score=score,
            scorer_decision=scored.decision,
            candidate_rank=scored.feedback_candidate_rank,
        )

    def _maybe_collect_shadow_feedback(
        self,
        *,
        request: CacheLookup,
        candidates: tuple[VectorSearchResult, ...],
        served_decision: OracleDecision,
    ) -> ShadowCollectionResult | None:
        if not self.shadow_collection_enabled or self.shadow_collector is None:
            return None
        try:
            return self.shadow_collector.collect(request, candidates, served_decision)
        except Exception:
            return None

    def _record_shadow_collection_for_monitoring(
        self,
        *,
        result: ShadowCollectionResult,
        served_decision: OracleDecision,
    ) -> None:
        """Feed refit-triggering counters and online monitoring from shadow top-k judgments.

        The shadow collector judges the full top-k independently of serving,
        so its aggregate H0/H1 counts are a richer, unbiased signal for "have
        we accumulated enough new pairs to refit/recalibrate" than any single
        served-candidate verdict could be. Only the served candidate can ever
        be a true/false accept (at most one candidate is served per request),
        so its judged label -- if the shadow batch judged it -- is what drives
        TPR/FPR monitoring.
        """

        with self._fit_lock:
            self._new_h0_since_fit += result.h0_added
            self._new_h1_since_fit += result.h1_added
            self._new_h0_since_calibration += result.h0_added
            self._monitor_h0_count += result.h0_added
            self._monitor_h1_count += result.h1_added
            if served_decision.accepted and result.served_candidate_label is not None:
                if result.served_candidate_label == 1:
                    self._monitor_true_accepts += 1
                elif result.served_candidate_label == 0:
                    self._monitor_false_accepts += 1

    def _apply_feedback_and_maybe_maintain(
        self,
        *,
        feedback: OracleJudgeFeedback | None,
        scorer_decision: OracleDecision,
        snapshot: OracleRuntimeSnapshot,
    ) -> RefitPolicyDecision | None:
        if feedback is not None and self.judge_for_feedback:
            self._record_judge_feedback(feedback)
        return self._maybe_auto_refit(current_threshold=snapshot.threshold)

    def _record_judge_feedback(self, feedback: OracleJudgeFeedback) -> None:
        decision = feedback.result.decision
        if self.judge_training_store is not None:
            self.judge_training_store.add(
                JudgedPairExample(
                    features=self._features_to_tuple(feedback.features),
                    request=feedback.result.request,
                    decision=decision,
                    metadata={"oracle_score": float(feedback.score) if feedback.score is not None else None},
                )
            )

        if decision.label == JudgeLabel.UNCERTAIN:
            return

        with self._fit_lock:
            if decision.label == JudgeLabel.REUSABLE:
                self._new_h1_since_fit += 1
                self._monitor_h1_count += 1
                if feedback.scorer_decision.accepted:
                    self._monitor_true_accepts += 1
            elif decision.label == JudgeLabel.NOT_REUSABLE:
                self._new_h0_since_fit += 1
                self._new_h0_since_calibration += 1
                self._monitor_h0_count += 1
                if feedback.scorer_decision.accepted:
                    self._monitor_false_accepts += 1

    def _finalize_decision(
        self,
        *,
        scorer_decision: OracleDecision,
        feedback: OracleJudgeFeedback | None,
        refit_decision: RefitPolicyDecision | None,
    ) -> OracleDecision:
        if feedback is None or not self.judge_can_override_decision:
            return scorer_decision
        return self._decision_from_judge_feedback(feedback, refit_decision=refit_decision)

    def _decision_from_judge_feedback(
        self,
        feedback: OracleJudgeFeedback,
        *,
        refit_decision: RefitPolicyDecision | None,
    ) -> OracleDecision:
        judge_result = feedback.result
        scorer_decision = feedback.scorer_decision
        decision_score = judge_result.decision.confidence
        if decision_score is None:
            decision_score = feedback.score

        evidence = {
            "candidate_key": str(feedback.candidate.cache_key),
            "judge": self.judge.name if self.judge is not None else None,
            "judge_label": judge_result.decision.label.value,
            "judge_rationale": judge_result.decision.rationale,
            "vector_score": float(feedback.candidate.score),
            "scorer_decision": {
                "status": scorer_decision.status.value,
                "accepted": scorer_decision.accepted,
                "cache_key": str(scorer_decision.cache_key) if scorer_decision.cache_key is not None else None,
                "score": float(scorer_decision.score) if scorer_decision.score is not None else None,
                "threshold": float(scorer_decision.threshold) if scorer_decision.threshold is not None else None,
                "reason": scorer_decision.reason,
            },
        }
        if refit_decision is not None:
            evidence["refit_policy"] = {
                "action": refit_decision.action.value,
                "reason": refit_decision.reason,
            }

        if judge_result.decision.label == JudgeLabel.REUSABLE:
            return OracleDecision(
                status=OracleDecisionStatus.HIT,
                accepted=True,
                cache_key=feedback.candidate.cache_key,
                score=decision_score,
                threshold=scorer_decision.threshold,
                scorer=scorer_decision.scorer,
                candidate_count=scorer_decision.candidate_count,
                accepted_candidate_rank=feedback.candidate_rank,
                reason="judge_labeled_h1",
                evidence=evidence,
            )

        if judge_result.decision.label == JudgeLabel.NOT_REUSABLE:
            return OracleDecision(
                status=OracleDecisionStatus.MISS,
                accepted=False,
                cache_key=None,
                score=decision_score,
                threshold=scorer_decision.threshold,
                scorer=scorer_decision.scorer,
                candidate_count=scorer_decision.candidate_count,
                reason="judge_labeled_h0",
                evidence=evidence,
            )

        return OracleDecision(
            status=OracleDecisionStatus.ABSTAIN,
            accepted=False,
            cache_key=None,
            score=decision_score,
            threshold=scorer_decision.threshold,
            scorer=scorer_decision.scorer,
            candidate_count=scorer_decision.candidate_count,
            reason="judge_uncertain",
            evidence=evidence,
        )

    def _should_call_judge(self) -> bool:
        return bool(self.judge is not None and (self.judge_for_feedback or self.judge_can_override_decision))

    @staticmethod
    def _predict_score(score: Score, threshold: Threshold, tie_mode: TieMode) -> bool:
        if tie_mode == TieMode.GT:
            return float(score) > float(threshold)
        return float(score) >= float(threshold)

    @staticmethod
    def _safe_score(scorer: SemanticScorer, features: PairFeatures) -> Score | None:
        try:
            return scorer.score(features)
        except Exception:
            return None

    @staticmethod
    def _features_to_tuple(features: PairFeatures) -> tuple[float, ...]:
        if features.concat:
            return tuple(float(v) for v in features.concat)
        if features.hadamard:
            return tuple(float(v) for v in features.hadamard)
        if features.abs_diff:
            return tuple(float(v) for v in features.abs_diff)
        if features.cosine is not None:
            return (float(features.cosine),)
        main = features.values.get("main")
        if main is not None:
            return tuple(float(v) for v in main)
        return ()
