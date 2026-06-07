"""Trainable semantic cache oracle implementation."""

from __future__ import annotations

from threading import RLock, Thread
from typing import Any

from mlcache.calibration import ThresholdCalibrationRequest, ThresholdProvider, ThresholdScope, wilson_upper_bound
from mlcache.features import PairFeatureBuilder, PairFeatures
from mlcache.feedback import (
    JudgeLabel,
    JudgeRequest,
    JudgedPairExample,
    JudgeTrainingStore,
    SemanticReuseJudge,
)
from mlcache.online import OnlineBatch, OnlineUpdater, StopStatus
from mlcache.oracle.base import SemanticCacheOracle
from mlcache.oracle.fit import OracleFitResult
from mlcache.oracle.runtime import OracleJudgeFeedback, OracleRuntimeSnapshot, OracleScoredResult
from mlcache.policies.refit import (
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
    def threshold(self) -> Threshold | None:
        with self._fit_lock:
            return self._threshold

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

        with self._fit_lock:
            self.scorer = new_scorer
            self._threshold = threshold
            self._threshold_version += 1
            self._semantic_hits_disabled = False
            self._fit_result = OracleFitResult(
                scorer=str(new_scorer.name),
                threshold=threshold,
                target_false_accept_rate=alpha,
                n_h0_train=len(split.h0_train),
                n_h1_train=len(split.h1_train),
                n_h0_calib=len(split.h0_calib),
                n_h1_calib=len(split.h1_calib),
                metadata=split.metadata,
            )
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
                    context=split.metadata,
                )
            except Exception:
                pass

    def _raise_fit_exception(self) -> None:
        with self._fit_lock:
            exc = self._fit_exception
        if exc is not None:
            raise exc

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
            h0_rows = self._feature_rows(self.judge_training_store.h0())
            h1_rows = self._feature_rows(self.judge_training_store.h1())
            with self._fit_lock:
                monitor_fpr = self._monitor_fpr_locked()
                monitor_tpr = self._monitor_tpr_locked()
                monitor_fpr_upper_bound = self._monitor_fpr_upper_bound_locked()
                context = RefitPolicyContext(
                    total_h0=len(h0_rows),
                    total_h1=len(h1_rows),
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
            return self._execute_refit_decision(decision, h0_rows=h0_rows, h1_rows=h1_rows)
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
        h0_rows: tuple[tuple[float, ...], ...],
        h1_rows: tuple[tuple[float, ...], ...],
    ) -> RefitPolicyDecision:
        with self._fit_lock:
            self._last_refit_decision = decision

        if decision.action == RefitAction.RECALIBRATE_THRESHOLD:
            if len(h0_rows) > self.sync_recalibration_max_h0:
                self._start_recalibration_async(h0_rows, metadata=decision.metadata)
                return decision
            threshold = self._recalibrate_threshold_from_rows(h0_rows, metadata=decision.metadata)
            if threshold is not None:
                return decision
            skipped = RefitPolicyDecision(
                action=RefitAction.NOOP,
                reason="threshold_recalibration_not_ready",
                metadata=decision.metadata,
            )
            with self._fit_lock:
                self._last_refit_decision = skipped
            return skipped

        if decision.action == RefitAction.REFIT_SCORER:
            split = self._build_refit_split(h0_rows, h1_rows, metadata=decision.metadata)
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

        with self._fit_lock:
            self._threshold = threshold
            self._threshold_version += 1
            self._semantic_hits_disabled = False
            self._new_h0_since_calibration = 0
            self._decisions_since_calibration = 0
            self._reset_monitor_counts()

        if threshold_provider is not None:
            try:
                threshold_provider.set_threshold(
                    threshold,
                    scorer=scorer_name,
                    scope=ThresholdScope.GLOBAL,
                    context=metadata,
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
                "n_h0_total": len(h0_rows),
                "n_h1_total": len(h1_rows),
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
        # TODO: Shadow top-k collection should be independent of serving.
        # This feedback path remains tied to the served candidate for compatibility.
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
