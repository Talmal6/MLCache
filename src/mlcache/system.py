"""`SemanticCacheSystem`: a complete, self-running online semantic-cache.

Experiments that use `MLCache` directly tend to re-implement pieces of the
calibration/training lifecycle themselves (manually scoring pairs, calling
`scorer.calibrate(...)`, setting thresholds by hand). `SemanticCacheSystem` is
the alternative: it owns the *entire* lifecycle — serving (lookup, retrieval,
scoring, HIT/MISS decisions, LLM fallback, cache writes) and learning (shadow
judging of top-k candidates, pair accumulation, training, threshold
calibration, atomic policy activation, and convergence-based freezing).

Experiments configure a system and read its metrics; they do not implement
calibration or training:

```python
system = SemanticCacheSystem(llm=llm, stream=stream, scorer="ensemble", target_fpr=0.05)
report = system.run()
```

Internally this leans on the runtime's existing online-learning loop
(`MLCacheRuntimeConfig(shadow=..., refit=...)` drives
`TrainableSemanticCacheOracle._decide_with_shadow_collection`, which already
judges top-k candidates in shadow mode, accumulates judged pairs, retrains and
recalibrates atomically, and gates an untrained scorer behind `ABSTAIN`/MISS)
— `SemanticCacheSystem` wires that loop up correctly, adds the serving
orchestration around it, and adds convergence-based freezing on top.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Iterator, Literal

from mlcache.builder import MLCache
from mlcache.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from mlcache.feedback import SemanticReuseJudge
from mlcache.llm_wrapper import LLMClient, LLMJudge
from mlcache.online import OnlineMetrics, OnlineStoppingConfig, WindowedOnlineStoppingController
from mlcache.policies import ConservativeRefitConfig, ConservativeRefitPolicy
from mlcache.runtime.config import MLCacheRuntimeConfig, RuntimeRefitConfig, ShadowRuntimeConfig
from mlcache.scorers import CosineScorer, SemanticScorer
from mlcache.scorers.utils import score_rows_with_scorer
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, CacheMetadata, Query, Response


@dataclass(frozen=True)
class ActivePolicy:
    """An immutable snapshot of the scorer + threshold currently serving requests.

    Scorer and threshold are always activated together as a single version
    pair — see `SemanticCacheSystem.policy`, which builds this from the
    oracle's atomically-swapped state so callers never observe a scorer from
    one generation paired with a threshold from another.
    """

    scorer: SemanticScorer
    scorer_name: str
    scorer_version: int
    threshold: float | None
    threshold_version: int
    calibrated: bool
    trained: bool
    frozen: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemResponse:
    """Result of serving one request through `SemanticCacheSystem.handle`."""

    text: str
    source: Literal["cache", "llm"]
    cache_key: str | None
    score: float | None
    threshold: float | None
    policy: ActivePolicy
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticCacheSystem:
    """Owns the full online semantic-cache lifecycle around an `LLMClient`.

    Serving loop (per request): embed -> `cache.lookup_with_decision` (which
    retrieves top-k candidates, scores with the active policy, and renders a
    HIT/MISS decision) -> HIT returns the cached response, MISS calls the LLM,
    writes the response through, and returns it.

    Learning loop: driven by the runtime's shadow collector + refit pipeline
    (configured here via `ShadowRuntimeConfig`/`RuntimeRefitConfig`), which
    judges top-k candidates, accumulates judged pairs, and atomically retrains
    + recalibrates the active policy in the background as batches fill. Until
    a calibrated policy exists, the oracle abstains (`ABSTAIN` -> MISS), so an
    untrained scorer never makes semantic HIT decisions — every request is a
    MISS, served by the LLM and written through, while the judge keeps
    labelling candidates for learning. `freeze()` (automatic once the stopping
    controller converges, or manual) stops further training/calibration while
    serving continues on the frozen policy.
    """

    def __init__(
        self,
        llm: LLMClient,
        stream: Iterable[Any] | None = None,
        *,
        scorer: str | SemanticScorer = "ensemble",
        scorers: Iterable[str] | None = None,
        target_fpr: float = 0.05,
        top_k: int = 5,
        root_dir: str | Path = "cache_state",
        judge: SemanticReuseJudge | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        batch_size: int = 100,
        min_h0: int = 50,
        min_h1: int = 20,
        max_updates: int | None = None,
        parallelism: int = 8,
        namespace: str | None = "semantic-cache-system",
        persistence: bool = True,
        stopping_config: OnlineStoppingConfig | None = None,
    ) -> None:
        self.llm = llm
        self.stream = stream
        self.target_fpr = float(target_fpr)
        self.top_k = int(top_k)
        self.batch_size = int(batch_size)
        self.max_updates = max_updates
        self.namespace = namespace

        # `LLMClient` generates responses; `SemanticReuseJudge` labels
        # query/candidate pairs. The same model may power both, but a default
        # judge is only synthesized (via `LLMJudge`) when the caller doesn't
        # supply one — the two interfaces stay distinct either way.
        self.judge: SemanticReuseJudge = judge if judge is not None else LLMJudge(llm)
        self._embeddings = embedding_provider if embedding_provider is not None else HashingEmbeddingProvider()

        self._lock = RLock()
        self._frozen = False
        self._freeze_reason: str | None = None
        self._counts: dict[str, int] = {"cache": 0, "llm": 0}
        self._scorer_version = 0
        self._last_scorer_identity: object | None = None
        # Used to tell whether the active scorer has actually been
        # fit-and-activated yet (vs. still being the cold instance built at
        # construction time). Trainable scorers swap to a new instance on
        # activation; cosine's `fit` is a no-op, so it never swaps and is
        # instead considered "trained" once it's calibrated.
        self._cold_scorer: object | None = None
        self._update_count = 0
        self._stopping = WindowedOnlineStoppingController(
            stopping_config or OnlineStoppingConfig(min_monitor_h0=int(min_h0), min_monitor_h1=int(min_h1))
        )

        # Everything that can run independently — candidate scoring/judging
        # inside the shadow collector, LLM calls, future background work — is
        # dispatched onto this single bounded pool so serving never blocks on
        # learning, and the two never block each other after retrieval.
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(parallelism)))

        self.cache = MLCache.from_preset(
            root_dir=root_dir,
            scorer=scorer,
            scorers=scorers,
            top_k=self.top_k,
            persistence=persistence,
            judge=self.judge,
            config=MLCacheRuntimeConfig(
                shadow=ShadowRuntimeConfig(
                    enabled=True,
                    top_k=self.top_k,
                    calibration_every_n=max(1, self.batch_size // 10),
                ),
                refit=RuntimeRefitConfig(
                    auto_refit=True,
                    target_false_accept_rate=self.target_fpr,
                    top_k=self.top_k,
                ),
            ),
        )

        # Parallelize judge calls across top-k shadow candidates: each call is
        # an independent, often LLM-latency-bound operation, so dispatching
        # them onto the shared pool lets retrieval/scoring/serving continue
        # without waiting on the judge (and vice versa).
        self._cold_scorer = self.cache.scorer

        oracle = self.cache.runtime.oracle
        collector = oracle.shadow_collector
        if collector is not None:
            collector.judge_executor = self._executor

        # The oracle's default `ConservativeRefitConfig` gates training and
        # calibration on production-scale pair counts (hundreds to
        # thousands). Replace it with one keyed off `min_h0`/`min_h1`/
        # `batch_size` so the configured stopping/update cadence actually
        # governs when the system trains, calibrates, and activates.
        oracle.refit_policy = ConservativeRefitPolicy(
            config=ConservativeRefitConfig(
                min_h0_for_fit=int(min_h0),
                min_h1_for_fit=int(min_h1),
                min_h0_for_calibration=int(min_h0),
                min_train_total=int(min_h0) + int(min_h1),
                min_train_h0=int(min_h0),
                min_train_h1=int(min_h1),
                min_calibration_h0=int(min_h0),
                min_calibration_h1=int(min_h1),
                min_new_h0_for_calibration=max(1, int(min_h0) // 2),
                min_new_h0_for_refit=max(1, int(min_h0) // 2),
                min_new_h1_for_refit=max(1, int(min_h1) // 2),
                # The default 0.03 Wilson margin demands hundreds of H0
                # calibration examples before any threshold can be activated
                # confidently. Scale it with the FPR budget so the activation
                # gate stays meaningfully conservative without requiring
                # production-scale traffic before the configured `min_h0`
                # calibration pairs can ever activate a policy.
                fpr_wilson_margin=max(0.03, 0.5 * float(target_fpr)),
            )
        )

    @classmethod
    def from_llm_and_stream(cls, *, llm: LLMClient, stream: Iterable[Any], **kwargs: Any) -> "SemanticCacheSystem":
        return cls(llm=llm, stream=stream, **kwargs)

    # -- serving -----------------------------------------------------------

    def handle(self, prompt: str) -> SystemResponse:
        """Serve a single request: HIT returns the cached response, MISS calls the LLM."""

        embedding = self._embeddings.embed(prompt)
        lookup = CacheLookup(query=Query(prompt), embedding=embedding, namespace=self.namespace)
        result = self.cache.lookup_with_decision(lookup)
        decision = result.decision
        policy = self.policy

        if decision.accepted and result.response is not None:
            self._record("cache")
            return SystemResponse(
                text=str(result.response),
                source="cache",
                cache_key=str(decision.cache_key) if decision.cache_key is not None else None,
                score=float(decision.score) if decision.score is not None else None,
                threshold=float(decision.threshold) if decision.threshold is not None else None,
                policy=policy,
                metadata=dict(result.metadata),
            )

        llm_response = self.llm.generate(prompt)
        cache_key = self._cache_key_for(prompt)
        self.cache.put(
            CacheEntry(
                cache_key=cache_key,
                query=Query(prompt),
                response=Response(llm_response.text),
                embedding=embedding,
                metadata=CacheMetadata(namespace=self.namespace),
            )
        )
        self._record("llm")
        return SystemResponse(
            text=llm_response.text,
            source="llm",
            cache_key=str(cache_key),
            score=float(decision.score) if decision.score is not None else None,
            threshold=float(decision.threshold) if decision.threshold is not None else None,
            policy=policy,
            metadata=dict(result.metadata),
        )

    def run(self, *, max_requests: int | None = None) -> dict[str, Any]:
        """Replay `stream` end to end, serving each prompt and tracking metrics.

        Periodically (every `batch_size` requests) checks the convergence
        stopping condition and freezes the policy once it's met; serving
        continues on the frozen policy afterwards (training/calibration stop,
        the judge keeps running only for monitoring).
        """

        served = 0
        for prompt in self._iter_prompts():
            self.handle(prompt)
            served += 1
            if served % max(1, self.batch_size) == 0:
                self._maybe_check_stopping()
            if max_requests is not None and served >= max_requests:
                break
            if self.max_updates is not None and self._update_count >= self.max_updates:
                break
        return self.report()

    def _iter_prompts(self) -> Iterator[str]:
        if self.stream is None:
            return iter(())
        return (prompt if isinstance(prompt, str) else str(prompt) for prompt in self.stream)

    def _cache_key_for(self, prompt: str) -> CacheKey:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        prefix = f"{self.namespace}:" if self.namespace else ""
        return CacheKey(f"{prefix}{digest}")

    def _record(self, source: Literal["cache", "llm"]) -> None:
        with self._lock:
            self._counts[source] = self._counts.get(source, 0) + 1

    # -- policy / lifecycle -------------------------------------------------

    @property
    def policy(self) -> ActivePolicy:
        """The currently active scorer+threshold pair, with lifecycle metadata.

        Built from the oracle's own atomically-swapped state (`scorer`,
        `threshold`, `activation_status`), so the returned scorer and
        threshold always belong to the same activation generation.
        """

        oracle = self.cache.runtime.oracle
        # `cache.scorer` is the scorer the cache was *built* with; trainable
        # oracles atomically swap in a freshly-fitted scorer instance on
        # retrain (`oracle.scorer`), so the live, currently-serving scorer
        # must be read from the oracle, not the cache facade.
        scorer = oracle.scorer
        with self._lock:
            if scorer is not self._last_scorer_identity:
                self._last_scorer_identity = scorer
                self._scorer_version += 1
            scorer_version = self._scorer_version
            frozen = self._frozen

        threshold = self.cache.threshold
        status = dict(oracle.activation_status)
        threshold_version = int(status.get("threshold_version") or 0)
        calibrated = threshold is not None and isfinite(float(threshold))
        # The oracle copies-and-swaps the active scorer on every accepted
        # refit cycle regardless of scorer type, so instance identity alone
        # can't distinguish "meaningfully trained" from "freshly copied but
        # never fit". Cosine's `fit` is a literal no-op, so a copied-but-unfit
        # instance is functionally identical to the cold one — for cosine,
        # "trained" is defined as "calibrated" instead. Trainable scorers only
        # count once an actual fit-and-activate cycle has swapped in a new,
        # *fitted* instance (never the cold one built at construction).
        if isinstance(scorer, CosineScorer):
            trained = bool(calibrated)
        else:
            trained = scorer is not self._cold_scorer

        return ActivePolicy(
            scorer=scorer,
            scorer_name=str(scorer.name),
            scorer_version=scorer_version,
            threshold=None if threshold is None else float(threshold),
            threshold_version=threshold_version,
            calibrated=bool(calibrated),
            trained=bool(trained),
            frozen=frozen,
            metadata=status,
        )

    def freeze(self, *, reason: str | None = None) -> None:
        """Stop training/calibration; serving continues on the frozen policy.

        The judge may keep running afterwards purely for monitoring/drift
        detection — only the auto-refit loop is disabled.
        """

        with self._lock:
            if self._frozen:
                return
            self._frozen = True
            self._freeze_reason = reason
        self.cache.runtime.oracle.auto_refit = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _maybe_check_stopping(self) -> None:
        """Feed empirical monitor TPR/FPR into the stopping controller; freeze on convergence."""

        if self._frozen:
            return

        store = self.cache.runtime.judge_training_store
        threshold = self.cache.threshold
        if store is None or threshold is None or not isfinite(float(threshold)):
            return

        h0 = store.h0_calibration()
        h1 = store.h1_calibration()
        if not h0 or not h1:
            return

        scorer = self.cache.runtime.oracle.scorer
        tau = float(threshold)
        h0_scores = score_rows_with_scorer([example.features for example in h0], scorer)
        h1_scores = score_rows_with_scorer([example.features for example in h1], scorer)
        monitor_fpr = float((h0_scores >= tau).mean())
        monitor_tpr = float((h1_scores >= tau).mean())

        with self._lock:
            self._update_count += 1
            batch_index = self._update_count

        status = self._stopping.observe(
            OnlineMetrics(
                batch_index=batch_index,
                monitor_tpr=monitor_tpr,
                monitor_fpr=monitor_fpr,
                threshold=threshold,
                target_false_accept_rate=self.target_fpr,
                monitor_h0_count=len(h0),
                monitor_h1_count=len(h1),
            )
        )
        if status.stopped:
            self.freeze(reason=status.reason)

    # -- reporting -----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Summary metrics an experiment can read without implementing its own bookkeeping."""

        policy = self.policy
        with self._lock:
            counts = dict(self._counts)
            freeze_reason = self._freeze_reason
        cache_hits = counts.get("cache", 0)
        llm_calls = counts.get("llm", 0)
        total = cache_hits + llm_calls
        return {
            "requests": total,
            "cache_hits": cache_hits,
            "llm_calls": llm_calls,
            "hit_rate": (cache_hits / total) if total else 0.0,
            "scorer_name": policy.scorer_name,
            "scorer_version": policy.scorer_version,
            "threshold": policy.threshold,
            "threshold_version": policy.threshold_version,
            "calibrated": policy.calibrated,
            "trained": policy.trained,
            "frozen": policy.frozen,
            "freeze_reason": freeze_reason,
            "stopping_updates_observed": self._update_count,
        }


__all__ = [
    "ActivePolicy",
    "SemanticCacheSystem",
    "SystemResponse",
]
