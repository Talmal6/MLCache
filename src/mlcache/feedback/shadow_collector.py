"""Shadow top-k feedback collection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Sequence

from mlcache.calibration.query_record_store import QueryCalibrationRecordStore
from mlcache.calibration.query_records import QueryCalibrationRecordBuilder
from mlcache.features import PairFeatureBuilder, PairFeatures
from mlcache.feedback.judges import SemanticReuseJudge
from mlcache.feedback.store import JudgedPairExample, SplitJudgeTrainingStore
from mlcache.feedback.types import JudgeDecision, JudgeLabel, JudgeRequest
from mlcache.retrieval import VectorSearchResult
from mlcache.semantic_types import CacheKey, CacheLookup, OracleDecision


@dataclass(frozen=True, slots=True)
class ShadowCollectionConfig:
    enabled: bool = True
    top_k: int = 5
    max_pairs_per_request: int | None = None
    calibration_every_n: int = 5
    collect_uncertain: bool = False
    include_served_candidate: bool = True
    include_unserved_candidates: bool = True
    skip_self_pairs: bool = True
    skip_duplicate_candidates: bool = True
    raise_on_failure: bool = False


@dataclass(frozen=True, slots=True)
class ShadowCollectionResult:
    pairs_observed: int
    judge_calls: int
    h0_added: int
    h1_added: int
    uncertain: int
    skipped: int
    failures: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShadowCollectorSnapshot:
    pairs_observed: int
    judge_calls: int
    h0_observed: int
    h1_observed: int
    uncertain_observed: int
    skipped: int
    failures: int
    train_h0: int
    train_h1: int
    calibration_h0: int
    calibration_h1: int


class ShadowTopKCollector(ABC):
    """Collects labels for top-k candidates independently of serving decisions."""

    @abstractmethod
    def collect(
        self,
        request: CacheLookup,
        candidates: Sequence[VectorSearchResult],
        served_decision: OracleDecision,
    ) -> ShadowCollectionResult:
        raise NotImplementedError


class DefaultShadowTopKCollector(ShadowTopKCollector):
    """Synchronous shadow collector for online hard H0/H1 mining."""

    def __init__(
        self,
        *,
        feature_builder: PairFeatureBuilder,
        judge: SemanticReuseJudge,
        store: SplitJudgeTrainingStore,
        config: ShadowCollectionConfig | None = None,
        query_record_builder: QueryCalibrationRecordBuilder | None = None,
        query_record_store: QueryCalibrationRecordStore | None = None,
        record_query_calibration: bool = False,
    ) -> None:
        self.feature_builder = feature_builder
        self.judge = judge
        self.store = store
        self.config = config or ShadowCollectionConfig()
        self.query_record_builder = query_record_builder
        self.query_record_store = query_record_store
        self.record_query_calibration = bool(record_query_calibration)
        if self.record_query_calibration and (
            self.query_record_builder is None or self.query_record_store is None
        ):
            raise ValueError(
                "record_query_calibration=True requires both query_record_builder and query_record_store"
            )
        self._validate_config()
        self._lock = RLock()
        self._pairs_observed = 0
        self._judge_calls = 0
        self._h0_observed = 0
        self._h1_observed = 0
        self._uncertain_observed = 0
        self._skipped = 0
        self._failures = 0
        self._h0_split_count = 0
        self._h1_split_count = 0

    def collect(
        self,
        request: CacheLookup,
        candidates: Sequence[VectorSearchResult],
        served_decision: OracleDecision,
    ) -> ShadowCollectionResult:
        if not self.config.enabled:
            return ShadowCollectionResult(
                pairs_observed=0,
                judge_calls=0,
                h0_added=0,
                h1_added=0,
                uncertain=0,
                skipped=0,
                failures=0,
                metadata={"enabled": False},
            )

        local = {
            "pairs_observed": 0,
            "judge_calls": 0,
            "h0_added": 0,
            "h1_added": 0,
            "uncertain": 0,
            "skipped": 0,
            "failures": 0,
        }
        seen_candidate_keys: set[str] = set()
        candidate_labels: dict[CacheKey, int | None] = {}
        max_pairs = self.config.max_pairs_per_request
        selected = tuple(candidates)[: self.config.top_k]

        for rank, candidate in enumerate(selected, start=1):
            if max_pairs is not None and local["pairs_observed"] >= max_pairs:
                break
            if self._should_skip_candidate(
                request=request,
                candidate=candidate,
                served_decision=served_decision,
                seen_candidate_keys=seen_candidate_keys,
            ):
                local["skipped"] += 1
                continue

            local["pairs_observed"] += 1
            try:
                features = self.feature_builder.build(request.embedding, candidate.embedding)
            except Exception:
                local["failures"] += 1
                if self.config.raise_on_failure:
                    self._commit_local_counts(local)
                    raise
                continue

            judge_request = self._judge_request(request, candidate, served_decision, rank)
            local["judge_calls"] += 1
            try:
                judge_result = self.judge.judge(judge_request)
            except Exception:
                local["failures"] += 1
                if self.config.raise_on_failure:
                    self._commit_local_counts(local)
                    raise
                continue

            label = judge_result.decision.label
            candidate_labels[candidate.cache_key] = self._label_to_int(label)
            if label == JudgeLabel.UNCERTAIN:
                local["uncertain"] += 1
                if self.config.collect_uncertain:
                    self._store_example(
                        features=features,
                        request=judge_result.request,
                        decision=judge_result.decision,
                        candidate=candidate,
                        rank=rank,
                        split="train",
                        served_decision=served_decision,
                    )
                continue

            split = self._next_split(label)
            try:
                self._store_example(
                    features=features,
                    request=judge_result.request,
                    decision=judge_result.decision,
                    candidate=candidate,
                    rank=rank,
                    split=split,
                    served_decision=served_decision,
                )
            except Exception:
                local["failures"] += 1
                if self.config.raise_on_failure:
                    self._commit_local_counts(local)
                    raise
                continue

            if label == JudgeLabel.REUSABLE:
                local["h1_added"] += 1
            else:
                local["h0_added"] += 1

        if not self._maybe_store_query_record(
            request=request,
            candidates=selected,
            served_decision=served_decision,
            candidate_labels=candidate_labels,
        ):
            local["failures"] += 1

        self._commit_local_counts(local)
        return ShadowCollectionResult(
            pairs_observed=local["pairs_observed"],
            judge_calls=local["judge_calls"],
            h0_added=local["h0_added"],
            h1_added=local["h1_added"],
            uncertain=local["uncertain"],
            skipped=local["skipped"],
            failures=local["failures"],
            metadata={
                "enabled": True,
                "candidate_count": len(selected),
                "top_k": self.config.top_k,
            },
        )

    def snapshot(self) -> ShadowCollectorSnapshot:
        with self._lock:
            return ShadowCollectorSnapshot(
                pairs_observed=self._pairs_observed,
                judge_calls=self._judge_calls,
                h0_observed=self._h0_observed,
                h1_observed=self._h1_observed,
                uncertain_observed=self._uncertain_observed,
                skipped=self._skipped,
                failures=self._failures,
                train_h0=len(self.store.h0_train()),
                train_h1=len(self.store.h1_train()),
                calibration_h0=len(self.store.h0_calibration()),
                calibration_h1=len(self.store.h1_calibration()),
            )

    def _should_skip_candidate(
        self,
        *,
        request: CacheLookup,
        candidate: VectorSearchResult,
        served_decision: OracleDecision,
        seen_candidate_keys: set[str],
    ) -> bool:
        candidate_key = str(candidate.cache_key)
        if self.config.skip_duplicate_candidates:
            if candidate_key in seen_candidate_keys:
                return True
            seen_candidate_keys.add(candidate_key)

        if self.config.skip_self_pairs and self._is_self_pair(request, candidate):
            return True

        is_served = served_decision.cache_key is not None and candidate.cache_key == served_decision.cache_key
        if is_served and not self.config.include_served_candidate:
            return True
        if not is_served and not self.config.include_unserved_candidates:
            return True
        return False

    def _maybe_store_query_record(
        self,
        *,
        request: CacheLookup,
        candidates: tuple[VectorSearchResult, ...],
        served_decision: OracleDecision,
        candidate_labels: dict[CacheKey, int | None],
    ) -> bool:
        if not self.record_query_calibration:
            return True
        if self.query_record_builder is None or self.query_record_store is None:
            return True
        try:
            record = self.query_record_builder.build_record(
                query_id=self._query_id(request),
                request=request,
                candidates=candidates,
                candidate_labels=candidate_labels,
                metadata={
                    "source": "shadow_top_k",
                    "served_decision_status": served_decision.status.value,
                    "served_decision_accepted": bool(served_decision.accepted),
                    "candidate_count": len(candidates),
                    "top_k": self.config.top_k,
                },
            )
            self.query_record_store.add(record)
            return True
        except Exception:
            if self.config.raise_on_failure:
                raise
            return False

    @staticmethod
    def _query_id(request: CacheLookup) -> str:
        attributes = request.metadata.attributes
        query_id = attributes.get("query_id") or attributes.get("request_id")
        if query_id:
            return str(query_id)
        query = str(request.query).strip()
        return query if query else "anonymous_query"

    @staticmethod
    def _label_to_int(label: JudgeLabel) -> int | None:
        if label == JudgeLabel.REUSABLE:
            return 1
        if label == JudgeLabel.NOT_REUSABLE:
            return 0
        return None

    @staticmethod
    def _is_self_pair(request: CacheLookup, candidate: VectorSearchResult) -> bool:
        return candidate.query is not None and str(request.query).strip() == str(candidate.query).strip()

    @staticmethod
    def _judge_request(
        request: CacheLookup,
        candidate: VectorSearchResult,
        served_decision: OracleDecision,
        rank: int,
    ) -> JudgeRequest:
        return JudgeRequest(
            query=request.query,
            candidate_query=candidate.query,
            candidate_key=candidate.cache_key,
            metadata=candidate.metadata,
            context={
                "request_metadata": request.metadata,
                "candidate_rank": rank,
                "vector_score": float(candidate.score),
                "served_decision_status": served_decision.status.value,
                "served_decision_accepted": bool(served_decision.accepted),
            },
        )

    def _next_split(self, label: JudgeLabel) -> str:
        with self._lock:
            if label == JudgeLabel.REUSABLE:
                self._h1_split_count += 1
                count = self._h1_split_count
            else:
                self._h0_split_count += 1
                count = self._h0_split_count
        if count % self.config.calibration_every_n == 0:
            return "calibration"
        return "train"

    def _store_example(
        self,
        *,
        features: PairFeatures,
        request: JudgeRequest,
        decision: JudgeDecision,
        candidate: VectorSearchResult,
        rank: int,
        split: str,
        served_decision: OracleDecision | None = None,
    ) -> None:
        metadata = {
            "source": "shadow_top_k",
            "candidate_rank": rank,
            "candidate_key": str(candidate.cache_key),
            "vector_score": float(candidate.score),
            "split": split,
        }
        if served_decision is not None:
            metadata.update(
                {
                    "served_decision_status": served_decision.status.value,
                    "served_decision_accepted": bool(served_decision.accepted),
                }
            )

        example = JudgedPairExample(
            features=self._features_to_tuple(features),
            request=request,
            decision=decision,
            metadata=metadata,
        )
        if split == "calibration":
            self.store.add_calibration(example)
        else:
            self.store.add_train(example)

    def _commit_local_counts(self, local: dict[str, int]) -> None:
        with self._lock:
            self._pairs_observed += local["pairs_observed"]
            self._judge_calls += local["judge_calls"]
            self._h0_observed += local["h0_added"]
            self._h1_observed += local["h1_added"]
            self._uncertain_observed += local["uncertain"]
            self._skipped += local["skipped"]
            self._failures += local["failures"]

    def _validate_config(self) -> None:
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.max_pairs_per_request is not None and self.config.max_pairs_per_request < 0:
            raise ValueError("max_pairs_per_request must be non-negative when set")
        if self.config.calibration_every_n <= 0:
            raise ValueError("calibration_every_n must be positive")

    @staticmethod
    def _features_to_tuple(features: PairFeatures) -> tuple[float, ...]:
        if features.concat:
            return tuple(float(value) for value in features.concat)
        if features.hadamard:
            return tuple(float(value) for value in features.hadamard)
        if features.abs_diff:
            return tuple(float(value) for value in features.abs_diff)
        if features.cosine is not None:
            return (float(features.cosine),)
        main = features.values.get("main")
        if main is not None:
            return tuple(float(value) for value in main)
        return ()
