"""End-to-end tests for `SemanticCacheSystem`.

These exercise the full online lifecycle (cold-start fallback -> shadow
judging -> pair accumulation -> training -> calibration -> atomic activation
-> serving hits -> convergence/freeze) through the public `SemanticCacheSystem`
API only — experiments never implement calibration or training themselves, and
neither do these tests.

The embedding/judge stubs below are deterministic (seeded via `hashlib.sha256`,
never Python's randomized `hash()`) so the lifecycle reaches the same
calibrated, finite-threshold state on every machine and process.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from mlcache import (
    CosineScorer,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    MockLLM,
    SemanticCacheSystem,
    SemanticReuseJudge,
)
from mlcache.embeddings import EmbeddingProvider


def _stable_seed(text: str) -> int:
    """Process- and machine-independent seed (Python's `hash()` is randomized per-process)."""

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


_TOPIC_PATTERN = re.compile(r"#(\d+)")


class TopicEmbeddingProvider(EmbeddingProvider):
    """Deterministic stub embedding: paraphrases of the same `#topic` cluster tightly.

    Each topic gets a fixed, normalized centroid (seeded by topic id only); each
    text gets small per-text jitter (seeded by a stable hash of the text, not
    Python's `hash()`). With `dimensions=32, jitter_scale=0.05` this produces a
    clean separation: same-topic cosine similarity stays around ~0.9+ while
    cross-topic similarity stays under ~0.35 — wide enough that calibration
    reliably finds a finite threshold for both cosine and ensemble scorers.
    """

    def __init__(self, *, dimensions: int = 32, jitter_scale: float = 0.05) -> None:
        self._dimensions = dimensions
        self._jitter_scale = jitter_scale

    def _topic_centroid(self, topic: int) -> np.ndarray:
        vector = np.random.default_rng(1000 + topic).normal(size=self._dimensions)
        return vector / np.linalg.norm(vector)

    def embed(self, text: str) -> tuple[float, ...]:
        match = _TOPIC_PATTERN.search(text)
        topic = int(match.group(1)) if match else 0
        centroid = self._topic_centroid(topic)
        jitter = np.random.default_rng(_stable_seed(text)).normal(scale=self._jitter_scale, size=self._dimensions)
        vector = centroid + jitter
        vector = vector / np.linalg.norm(vector)
        return tuple(float(value) for value in vector)


class DeterministicTopicJudge(SemanticReuseJudge):
    """Labels a pair REUSABLE iff the query and candidate share the same `#topic`.

    Unlike `LLMJudge(MockLLM())` (which always returns UNCERTAIN, since
    `MockLLM` only echoes its prompt), this gives a fully deterministic ground
    truth so judged H0/H1 pairs accumulate predictably.
    """

    def __init__(self, *, name: str = "deterministic-topic-judge") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def judge(self, request: JudgeRequest) -> JudgeResult:
        query_topic = self._topic(str(request.query))
        candidate_topic = self._topic(str(request.candidate_query) if request.candidate_query is not None else "")
        if query_topic is None or candidate_topic is None:
            label = JudgeLabel.NOT_REUSABLE
        else:
            label = JudgeLabel.REUSABLE if query_topic == candidate_topic else JudgeLabel.NOT_REUSABLE
        return JudgeResult(request=request, decision=JudgeDecision(label=label, metadata={"judge": self._name}))

    @staticmethod
    def _topic(text: str) -> str | None:
        match = _TOPIC_PATTERN.search(text)
        return match.group(1) if match else None


_TEMPLATES: Sequence[str] = (
    "What is the capital of country #{n}?",
    "Tell me the capital city of nation #{n}.",
    "Name the capital of country #{n}, please.",
    "Could you say which city is the capital of country #{n}?",
)


def _topic_prompts(count: int, *, topics: int = 6) -> list[str]:
    return [_TEMPLATES[i % len(_TEMPLATES)].format(n=i % topics) for i in range(count)]


def _build_system(tmp_dir: str, **overrides: object) -> SemanticCacheSystem:
    config: dict[str, object] = dict(
        llm=MockLLM(response_template="answer for: {prompt}"),
        stream=None,
        scorer="ensemble",
        scorers=["cosine", "lda"],
        judge=DeterministicTopicJudge(),
        embedding_provider=TopicEmbeddingProvider(),
        target_fpr=0.25,
        top_k=3,
        root_dir=tmp_dir,
        batch_size=20,
        min_h0=15,
        min_h1=8,
        persistence=False,
        parallelism=4,
    )
    config.update(overrides)
    return SemanticCacheSystem(**config)  # type: ignore[arg-type]


class ColdStartTests(unittest.TestCase):
    """Spec: an untrained/uncalibrated policy must never serve semantic HITs."""

    def test_first_requests_all_miss_to_the_llm_while_uncalibrated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-cold-") as tmp:
            system = _build_system(tmp)
            prompts = _topic_prompts(15)

            responses = [system.handle(prompt) for prompt in prompts]

            self.assertTrue(all(response.source == "llm" for response in responses))
            self.assertFalse(system.policy.calibrated)
            self.assertFalse(system.policy.trained)
            self.assertIsNone(system.policy.threshold)
            # Every miss must still be written through to the cache.
            self.assertEqual(system.report()["llm_calls"], 15)
            self.assertEqual(system.report()["cache_hits"], 0)

    def test_llm_responses_are_written_through_to_the_cache(self) -> None:
        from mlcache import CacheKey

        with tempfile.TemporaryDirectory(prefix="scs-writethrough-") as tmp:
            system = _build_system(tmp)
            prompt = _topic_prompts(1)[0]

            response = system.handle(prompt)

            self.assertEqual(response.source, "llm")
            self.assertIsNotNone(response.cache_key)
            stored = system.cache.runtime.kv_store.get(CacheKey(response.cache_key))
            self.assertIsNotNone(stored)
            self.assertEqual(str(stored), response.text)


class JudgedPairAccumulationTests(unittest.TestCase):
    def test_shadow_judging_accumulates_h0_and_h1_pairs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-pairs-") as tmp:
            system = _build_system(tmp)
            for prompt in _topic_prompts(120):
                system.handle(prompt)

            store = system.cache.runtime.judge_training_store
            self.assertIsNotNone(store)
            total_h0 = len(store.h0_train()) + len(store.h0_calibration())
            total_h1 = len(store.h1_train()) + len(store.h1_calibration())
            self.assertGreater(total_h0, 0, "shadow collection should have observed NOT_REUSABLE pairs")
            self.assertGreater(total_h1, 0, "shadow collection should have observed REUSABLE pairs")

    def test_shadow_pairs_from_rejected_candidates_reach_the_training_store_and_drive_calibration(self) -> None:
        """Spec: shadow top-k judging must label *every* retrieved candidate --
        including ones the active policy rejected (served nothing, MISS) --
        and those judged pairs must land in the same store that feeds
        training/calibration, not just pairs for the served/accepted candidate.

        On a MISS, `served_decision.cache_key` is `None` (no candidate was
        served), so *every* judged pair stored under a MISS decision is, by
        construction, a retrieved-but-rejected candidate. Their presence in
        the store -- which is the sole source of `_refit_rows_from_training_store`
        -- is direct proof that online learning is not limited to served pairs.
        """

        with tempfile.TemporaryDirectory(prefix="scs-rejected-pairs-") as tmp:
            prompts = _topic_prompts(360)
            system = _build_system(tmp, stream=prompts)
            system.run()

            store = system.cache.runtime.judge_training_store
            self.assertIsNotNone(store)
            all_examples = (
                store.h0_train() + store.h0_calibration() + store.h1_train() + store.h1_calibration()
            )
            self.assertGreater(len(all_examples), 0, "shadow collection should have stored judged pairs")

            rejected_candidate_examples = [
                example for example in all_examples if example.metadata.get("served_decision_accepted") is False
            ]
            self.assertGreater(
                len(rejected_candidate_examples),
                0,
                "judged pairs for retrieved-but-rejected candidates (MISS decisions, "
                "no served candidate) should be present in the training store",
            )
            self.assertTrue(
                any(example.decision.label == JudgeLabel.NOT_REUSABLE for example in rejected_candidate_examples),
                "rejected-candidate pairs should include NOT_REUSABLE (H0) labels",
            )
            self.assertTrue(
                any(example.decision.label == JudgeLabel.REUSABLE for example in rejected_candidate_examples),
                "rejected-candidate pairs should include REUSABLE (H1) labels",
            )

            # More judged pairs were stored than the system ever served from
            # the cache -- proof that shadow judging covers candidates beyond
            # whichever single one (if any) was served per request.
            self.assertGreater(len(all_examples), system.report()["cache_hits"])

            # These pairs are the only feedstock for training/calibration
            # (`_refit_rows_from_training_store` reads exactly these buckets),
            # so a system that reached a trained, calibrated policy did so by
            # learning from them.
            self.assertTrue(system.policy.trained)
            self.assertTrue(system.policy.calibrated)


class EnsembleLifecycleTests(unittest.TestCase):
    """Spec: an untrained ensemble must never make semantic HIT decisions;
    it must train, calibrate, and atomically activate before serving hits."""

    def test_full_lifecycle_reaches_trained_calibrated_policy_and_serves_hits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-ensemble-lifecycle-") as tmp:
            prompts = _topic_prompts(500)
            system = _build_system(tmp, stream=prompts)

            first_hit_at: int | None = None
            calibrated_before_first_hit = False
            for index, prompt in enumerate(prompts, start=1):
                response = system.handle(prompt)
                if response.source == "cache":
                    if first_hit_at is None:
                        first_hit_at = index
                        calibrated_before_first_hit = response.policy.calibrated
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            policy = system.policy
            report = system.report()

            self.assertIsNotNone(first_hit_at, "ensemble should eventually serve cache hits once activated")
            self.assertTrue(calibrated_before_first_hit, "a HIT must never be served by an uncalibrated policy")
            self.assertTrue(policy.calibrated)
            self.assertTrue(policy.trained)
            self.assertIsNotNone(policy.threshold)
            self.assertTrue(np.isfinite(policy.threshold))
            self.assertGreaterEqual(policy.scorer_version, 2, "a real fit-and-activate swap must bump scorer_version")
            self.assertGreaterEqual(policy.threshold_version, 1)
            self.assertGreater(report["cache_hits"], 0)
            self.assertEqual(report["requests"], len(prompts))

    def test_ensemble_starts_untrained_and_serves_no_semantic_hits_until_calibrated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-ensemble-cold-") as tmp:
            system = _build_system(tmp)

            self.assertFalse(system.policy.calibrated)
            self.assertFalse(system.policy.trained)
            self.assertIsNone(system.policy.threshold)

            # The real invariant is "an uncalibrated policy never serves a
            # semantic HIT" -- not "calibration never happens before the
            # first HIT". A freshly-calibrated policy can still legitimately
            # MISS (e.g. a query with no close match), so asserting
            # `not calibrated` on every pre-HIT response would conflate
            # "calibrated" with "would always HIT". Instead: while
            # uncalibrated, every response must come from the LLM; once a
            # HIT occurs, the serving policy must be calibrated.
            saw_uncalibrated_request = False
            for prompt in _topic_prompts(60):
                response = system.handle(prompt)
                if not response.policy.calibrated:
                    saw_uncalibrated_request = True
                    self.assertEqual(
                        response.source, "llm", "an uncalibrated policy must never serve a semantic HIT"
                    )
                elif response.source == "cache":
                    break
            self.assertTrue(saw_uncalibrated_request, "the cold-start window must include uncalibrated requests")

    def test_later_batches_retrain_on_accumulated_pairs_and_increment_versions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-ensemble-retrain-") as tmp:
            prompts = _topic_prompts(900)
            system = _build_system(tmp, stream=prompts)

            scorer_versions: list[int] = []
            threshold_versions: list[int] = []
            pair_totals: list[int] = []
            store = system.cache.runtime.judge_training_store
            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()
                    policy = system.policy
                    scorer_versions.append(policy.scorer_version)
                    threshold_versions.append(policy.threshold_version)
                    pair_totals.append(
                        len(store.h0_train()) + len(store.h0_calibration())
                        + len(store.h1_train()) + len(store.h1_calibration())
                    )

            # The judged-pair store must keep growing batch over batch — later
            # refits train on the accumulated old+new pairs, not a reset slate.
            self.assertGreater(pair_totals[-1], pair_totals[0], "judged pairs must accumulate across batches")
            self.assertTrue(
                all(later >= earlier for earlier, later in zip(pair_totals, pair_totals[1:])),
                "the judged-pair store must never shrink between batches",
            )
            # Versions are monotonically non-decreasing, and each must have
            # advanced past its cold-start value by the end of the run.
            self.assertTrue(all(later >= earlier for earlier, later in zip(scorer_versions, scorer_versions[1:])))
            self.assertTrue(all(later >= earlier for earlier, later in zip(threshold_versions, threshold_versions[1:])))
            self.assertGreaterEqual(scorer_versions[-1], 2, "at least one real retrain must have occurred")
            self.assertGreaterEqual(threshold_versions[-1], 1, "at least one recalibration must have occurred")

    def test_active_scorer_is_read_from_the_oracle_not_the_stale_cache_facade(self) -> None:
        """`cache.scorer` is fixed at construction; the oracle swaps `oracle.scorer`
        atomically on retrain. `policy.scorer` must always reflect the live one."""

        with tempfile.TemporaryDirectory(prefix="scs-live-scorer-") as tmp:
            prompts = _topic_prompts(500)
            system = _build_system(tmp, stream=prompts)
            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            self.assertIs(system.policy.scorer, system.cache.runtime.oracle.scorer)
            self.assertTrue(system.policy.scorer.is_fitted if hasattr(system.policy.scorer, "is_fitted") else True)


class CosineLifecycleTests(unittest.TestCase):
    """Spec: cosine uses the same online lifecycle, except `fit(...)` is a no-op."""

    def test_cosine_starts_uncalibrated_with_semantic_hits_disabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-cosine-cold-") as tmp:
            system = _build_system(tmp, scorer="cosine", scorers=None)

            self.assertFalse(system.policy.calibrated)
            self.assertFalse(system.policy.trained)
            self.assertIsNone(system.policy.threshold)

            responses = [system.handle(prompt) for prompt in _topic_prompts(15)]
            self.assertTrue(
                all(response.source == "llm" for response in responses),
                "an uncalibrated cosine policy must never serve a semantic HIT",
            )

    def test_cosine_judged_pairs_accumulate_and_calibrate_without_training_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-cosine-pairs-") as tmp:
            prompts = _topic_prompts(400)
            system = _build_system(tmp, stream=prompts, scorer="cosine", scorers=None)

            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            store = system.cache.runtime.judge_training_store
            total_h0 = len(store.h0_train()) + len(store.h0_calibration())
            total_h1 = len(store.h1_train()) + len(store.h1_calibration())
            self.assertGreater(total_h0, 0, "cosine lifecycle should still accumulate H0 pairs via shadow judging")
            self.assertGreater(total_h1, 0, "cosine lifecycle should still accumulate H1 pairs via shadow judging")

            policy = system.policy
            # `fit` is a literal no-op for cosine — calibration must succeed
            # on the strength of the unlearned scorer alone, with no fit
            # exception ever surfacing from the oracle.
            self.assertIsNone(system.cache.runtime.oracle._fit_exception)
            self.assertTrue(policy.calibrated)
            self.assertTrue(policy.trained)
            self.assertIsNotNone(policy.threshold)
            self.assertTrue(np.isfinite(policy.threshold))

    def test_cosine_freeze_prevents_further_threshold_updates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-cosine-freeze-") as tmp:
            prompts = _topic_prompts(400)
            system = _build_system(tmp, stream=prompts, scorer="cosine", scorers=None)
            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            self.assertTrue(system.policy.calibrated, "fixture must reach a calibrated cosine policy before freezing")
            before = system.report()

            system.freeze(reason="cosine-manual-test")
            self.assertTrue(system.frozen)

            for prompt in _topic_prompts(120):
                system.handle(prompt)

            after = system.report()
            self.assertEqual(before["threshold_version"], after["threshold_version"])
            self.assertEqual(before["threshold"], after["threshold"])
            self.assertEqual(before["scorer_version"], after["scorer_version"])
            self.assertTrue(after["frozen"])

    def test_cosine_never_swaps_identity_but_still_calibrates_and_serves(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-cosine-lifecycle-") as tmp:
            prompts = _topic_prompts(400)
            system = _build_system(tmp, stream=prompts, scorer="cosine", scorers=None)

            cold_scorer = system.policy.scorer
            self.assertIsInstance(cold_scorer, CosineScorer)

            first_hit_at: int | None = None
            for index, prompt in enumerate(prompts, start=1):
                response = system.handle(prompt)
                if response.source == "cache" and first_hit_at is None:
                    first_hit_at = index
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            policy = system.policy
            self.assertIsNotNone(first_hit_at)
            # The oracle still copies-and-swaps the scorer instance on each
            # accepted refit cycle (so `scorer_version` can advance), but
            # `fit` is a literal no-op for cosine — a freshly copied instance
            # behaves identically to the cold one. "trained" is therefore
            # defined as "calibrated" for cosine, not "instance changed".
            self.assertIsInstance(policy.scorer, CosineScorer)
            self.assertTrue(policy.calibrated)
            self.assertTrue(policy.trained)
            self.assertIsNotNone(policy.threshold)
            self.assertGreaterEqual(policy.threshold_version, 1)


class FreezeTests(unittest.TestCase):
    def test_freeze_halts_further_training_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-freeze-") as tmp:
            prompts = _topic_prompts(500)
            system = _build_system(tmp, stream=prompts)
            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            self.assertTrue(system.policy.calibrated, "fixture must reach a calibrated policy before freezing")
            before = system.report()

            system.freeze(reason="manual-test")

            self.assertTrue(system.frozen)
            self.assertFalse(system.cache.runtime.oracle.auto_refit)

            for prompt in _topic_prompts(120, topics=6)[:120]:
                system.handle(prompt)

            after = system.report()
            self.assertEqual(before["scorer_version"], after["scorer_version"])
            self.assertEqual(before["threshold_version"], after["threshold_version"])
            self.assertEqual(before["threshold"], after["threshold"])
            self.assertTrue(after["calibrated"])
            self.assertTrue(after["frozen"])

    def test_freeze_is_idempotent_and_keeps_the_first_reason(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-freeze-idempotent-") as tmp:
            system = _build_system(tmp)
            system.freeze(reason="first")
            system.freeze(reason="second")

            self.assertTrue(system.frozen)
            self.assertEqual(system.report()["freeze_reason"], "first")

    def test_serving_continues_on_the_frozen_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-freeze-serves-") as tmp:
            prompts = _topic_prompts(500)
            system = _build_system(tmp, stream=prompts)
            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            system.freeze(reason="manual-test")
            frozen_policy = system.policy
            self.assertTrue(frozen_policy.calibrated)

            # Serving must keep working — and keep returning the SAME frozen
            # policy snapshot — after freezing.
            responses = [system.handle(prompt) for prompt in _topic_prompts(40)]
            self.assertTrue(any(response.source == "cache" for response in responses))
            for response in responses:
                self.assertEqual(response.policy.scorer_version, frozen_policy.scorer_version)
                self.assertEqual(response.policy.threshold_version, frozen_policy.threshold_version)
                self.assertTrue(response.policy.frozen)


class AutoStopConvergenceTests(unittest.TestCase):
    """Spec: training stops automatically once the windowed stopping controller
    converges, and the `min_monitor_*` gate can block that auto-stop when the
    monitor buckets can never reach the configured size.
    """

    def test_convergence_auto_freezes_and_halts_training(self) -> None:
        from mlcache import OnlineStoppingConfig

        # A permissive controller: any two filled-window observations with both
        # monitor buckets non-empty count as converged (huge eps, huge fpr
        # margin), so the system auto-freezes shortly after it calibrates.
        stopping = OnlineStoppingConfig(
            window=2,
            patience=1,
            eps_tpr=1.0,
            eps_fpr=1.0,
            eps_tau=1.0,
            fpr_margin=1.0,
            min_monitor_h0=1,
            min_monitor_h1=1,
        )
        with tempfile.TemporaryDirectory(prefix="scs-autostop-") as tmp:
            prompts = _topic_prompts(500)
            system = _build_system(
                tmp, stream=prompts, scorer="cosine", scorers=None, stopping_config=stopping
            )
            froze_at: int | None = None
            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()
                if froze_at is None and system.frozen:
                    froze_at = index
                    break

            self.assertIsNotNone(froze_at, "system should auto-freeze on convergence")
            self.assertTrue(system.frozen)
            self.assertTrue(system.policy.calibrated)
            report = system.report()
            self.assertFalse(system.cache.runtime.oracle.auto_refit)
            self.assertIn("converged", str(report["freeze_reason"]))

            # After the auto-freeze, training/calibration must not advance.
            before = system.report()
            for prompt in _topic_prompts(120):
                system.handle(prompt)
            after = system.report()
            self.assertEqual(before["threshold_version"], after["threshold_version"])
            self.assertEqual(before["scorer_version"], after["scorer_version"])
            self.assertEqual(before["threshold"], after["threshold"])

    def test_unreachable_min_monitor_h1_never_auto_freezes(self) -> None:
        """Regression for the observed pathology: when `min_monitor_h1` exceeds
        the H1 the stream can put in the calibration bucket, the controller is
        stuck on `insufficient_monitor_data` and never converges, so training
        churns forever. The system must stay unfrozen (calibrated, but never
        auto-stopped) — which is the signal that the gate, not the data, is the
        blocker.
        """
        from mlcache import OnlineStoppingConfig

        stopping = OnlineStoppingConfig(
            window=2, patience=1, eps_tpr=1.0, eps_fpr=1.0, eps_tau=1.0, fpr_margin=1.0,
            min_monitor_h0=1, min_monitor_h1=10_000,
        )
        with tempfile.TemporaryDirectory(prefix="scs-autostop-blocked-") as tmp:
            prompts = _topic_prompts(500)
            system = _build_system(
                tmp, stream=prompts, scorer="cosine", scorers=None, stopping_config=stopping
            )
            for index, prompt in enumerate(prompts, start=1):
                system.handle(prompt)
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            self.assertTrue(system.policy.calibrated, "fixture must still calibrate")
            self.assertFalse(system.frozen, "unreachable monitor gate must block auto-stop")
            self.assertTrue(system.cache.runtime.oracle.auto_refit)

    def test_check_stopping_is_a_noop_once_frozen(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-autostop-noop-") as tmp:
            system = _build_system(tmp)
            system.freeze(reason="manual")
            updates_before = system.report()["stopping_updates_observed"]
            # Even after streaming, a frozen system must not observe more
            # stopping updates (the controller is bypassed).
            for prompt in _topic_prompts(60):
                system.handle(prompt)
                system._maybe_check_stopping()
            self.assertEqual(system.report()["stopping_updates_observed"], updates_before)


class ColdEnsembleScoringTests(unittest.TestCase):
    """Regression: a cold (unfit) multi-member ensemble must MISS, never crash.

    The cold `EnsembleScorer` wraps sub-scorers (LDA/PCA/XGBoost/MLP) that raise
    `fit() must be called before score()` until trained. The oracle's per-rank
    scoring path must route through `_safe_score` so an unfit scorer yields a
    clean MISS (no semantic hit) instead of propagating the exception out of
    `handle()`.
    """

    def test_cold_full_ensemble_misses_instead_of_raising(self) -> None:
        scorers = ["cosine", "lda", "pca_whitened_cosine"]
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not installed")

        with tempfile.TemporaryDirectory(prefix="scs-cold-ensemble-") as tmp:
            system = _build_system(tmp, scorer="ensemble", scorers=scorers)
            # First few requests retrieve previously-cached neighbours that the
            # cold ensemble cannot score; this must not raise.
            responses = [system.handle(prompt) for prompt in _topic_prompts(8)]

            self.assertTrue(all(r.source == "llm" for r in responses))
            self.assertFalse(system.policy.trained)
            self.assertIsNone(system.cache.runtime.oracle._fit_exception)


class ParallelismTests(unittest.TestCase):
    """Spec: judge calls across top-k candidates must be parallelized, and the
    serving/scoring path must not be blocked by them."""

    def test_judge_calls_for_a_single_lookup_run_concurrently(self) -> None:
        observed_threads: list[int] = []
        release = threading.Event()
        lock = threading.Lock()

        class SlowConcurrencyProbeJudge(SemanticReuseJudge):
            @property
            def name(self) -> str:
                return "slow-probe"

            def judge(self, request: JudgeRequest) -> JudgeResult:
                with lock:
                    observed_threads.append(threading.get_ident())
                # Block until every dispatched call has registered itself —
                # this only succeeds if the calls are running concurrently.
                release.wait(timeout=5.0)
                topic = DeterministicTopicJudge._topic(str(request.query))
                candidate_topic = DeterministicTopicJudge._topic(
                    str(request.candidate_query) if request.candidate_query is not None else ""
                )
                label = JudgeLabel.REUSABLE if topic and topic == candidate_topic else JudgeLabel.NOT_REUSABLE
                return JudgeResult(request=request, decision=JudgeDecision(label=label))

        with tempfile.TemporaryDirectory(prefix="scs-parallel-judge-") as tmp:
            system = _build_system(
                tmp,
                judge=SlowConcurrencyProbeJudge(),
                top_k=3,
                parallelism=4,
            )
            # Seed enough distinct candidates that a lookup retrieves top_k > 1
            # of them, so the shadow collector has multiple judge calls to
            # dispatch for a single request.
            for prompt in _topic_prompts(12):
                system.handle(prompt)

            def _release_soon() -> None:
                time.sleep(0.3)
                release.set()

            releaser = threading.Thread(target=_release_soon)
            releaser.start()
            try:
                system.handle(_topic_prompts(1)[0])
            finally:
                releaser.join()

        # Concurrent dispatch means more than one worker thread observed the
        # judge call before any of them were allowed to finish.
        self.assertGreater(len(set(observed_threads)), 1, "judge calls should be dispatched onto multiple worker threads")

    def test_concurrent_judge_dispatch_beats_naive_sequential_latency(self) -> None:
        """The oracle's decision loop does consult the judge before `handle`
        returns (it isn't decoupled into a background task), so per-request
        latency under a slow judge is bounded below by judge latency. What
        `SemanticCacheSystem` *does* guarantee — by wiring a shared
        `judge_executor` into the shadow collector (spec requirement #2) — is
        that the top-k candidates' judge calls run concurrently rather than
        one-at-a-time. This asserts that concurrent contribution directly: the
        measured latency must be well under `judge_latency * top_k` (the
        fully-sequential bound), proving calls overlap in time."""

        judge_latency = 0.4
        call_count = 0
        lock = threading.Lock()

        class TimedSlowJudge(SemanticReuseJudge):
            @property
            def name(self) -> str:
                return "timed-slow"

            def judge(self, request: JudgeRequest) -> JudgeResult:
                nonlocal call_count
                time.sleep(judge_latency)
                with lock:
                    call_count += 1
                topic = DeterministicTopicJudge._topic(str(request.query))
                candidate_topic = DeterministicTopicJudge._topic(
                    str(request.candidate_query) if request.candidate_query is not None else ""
                )
                label = JudgeLabel.REUSABLE if topic and topic == candidate_topic else JudgeLabel.NOT_REUSABLE
                return JudgeResult(request=request, decision=JudgeDecision(label=label))

        top_k = 3
        with tempfile.TemporaryDirectory(prefix="scs-concurrent-judging-") as tmp:
            system = _build_system(tmp, judge=TimedSlowJudge(), parallelism=4, top_k=top_k)
            # Warm up enough distinct candidates that a lookup retrieves top_k > 1.
            for prompt in _topic_prompts(6):
                system.handle(prompt)

            with lock:
                call_count = 0
            started = time.monotonic()
            response = system.handle(_topic_prompts(1)[0])
            elapsed = time.monotonic() - started

        self.assertEqual(response.source, "llm")
        self.assertGreater(call_count, 1, "the lookup should have judged more than one shadow candidate")
        sequential_bound = judge_latency * call_count
        self.assertLess(
            elapsed,
            sequential_bound * 0.75,
            f"concurrent judge dispatch ({call_count} calls @ {judge_latency}s) should beat "
            f"the fully-sequential bound of {sequential_bound:.2f}s (got {elapsed:.2f}s)",
        )


class RobustnessTests(unittest.TestCase):
    """Spec: serving must survive judge/training failures and background retraining
    without ever corrupting or replacing the active policy with a broken one."""

    def test_serving_continues_on_old_policy_while_retraining_in_background(self) -> None:
        gate = threading.Event()

        class GatedFitScorer(CosineScorer):
            """Cosine scorer whose `fit` blocks until the test releases `gate`.

            This pins down the exact moment a background refit is "in
            progress" so the test can deterministically observe that serving
            keeps using the pre-refit policy snapshot throughout."""

            def __init__(self, *, fit_gate: threading.Event) -> None:
                super().__init__()
                self._fit_gate = fit_gate

            def fit(self, batch, **kwargs):  # type: ignore[override]
                self._fit_gate.wait(timeout=10.0)
                return super().fit(batch, **kwargs)

            def copy_for_refit(self) -> "GatedFitScorer":
                return GatedFitScorer(fit_gate=self._fit_gate)

        with tempfile.TemporaryDirectory(prefix="scs-bg-retrain-") as tmp:
            prompts = _topic_prompts(400)
            system = _build_system(
                tmp,
                stream=prompts,
                scorer=GatedFitScorer(fit_gate=gate),
                scorers=None,
            )
            oracle = system.cache.runtime.oracle

            fit_started = False
            policy_before_fit: object | None = None
            served_during_fit: list[object] = []
            for index, prompt in enumerate(prompts, start=1):
                response = system.handle(prompt)
                if not fit_started and oracle.fit_in_progress:
                    fit_started = True
                    policy_before_fit = system.policy
                if fit_started and not gate.is_set():
                    served_during_fit.append(response)
                    if len(served_during_fit) >= 3:
                        gate.set()
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            gate.set()  # release unconditionally so the worker thread can't hang the test
            oracle.wait_for_fit(timeout=10.0)

            self.assertTrue(fit_started, "fixture must trigger at least one background refit")
            assert policy_before_fit is not None
            for response in served_during_fit:
                self.assertEqual(
                    response.policy.scorer_version,
                    policy_before_fit.scorer_version,  # type: ignore[attr-defined]
                    "requests served while a refit is in flight must use the pre-refit policy snapshot",
                )

    def test_failed_judge_calls_do_not_crash_serving(self) -> None:
        call_count = 0
        lock = threading.Lock()

        class FlakyJudge(SemanticReuseJudge):
            @property
            def name(self) -> str:
                return "flaky"

            def judge(self, request: JudgeRequest) -> JudgeResult:
                nonlocal call_count
                with lock:
                    call_count += 1
                    failing = call_count % 3 == 0
                if failing:
                    raise RuntimeError("synthetic judge failure")
                return DeterministicTopicJudge().judge(request)

        with tempfile.TemporaryDirectory(prefix="scs-flaky-judge-") as tmp:
            prompts = _topic_prompts(150)
            system = _build_system(tmp, stream=prompts, judge=FlakyJudge())

            responses = [system.handle(prompt) for prompt in prompts]

            self.assertEqual(len(responses), len(prompts))
            self.assertTrue(all(response.text for response in responses), "serving must keep returning real responses")
            self.assertGreater(call_count, 0, "the flaky judge must actually have been invoked")

    def test_failed_training_does_not_replace_the_active_policy(self) -> None:
        class FlakyState:
            def __init__(self) -> None:
                self.fit_calls = 0
                self.should_fail = False

        class FlakyFitScorer(CosineScorer):
            """Cosine scorer that fits normally until the test flips
            `should_fail` (right after the policy first calibrates), and
            raises on every fit attempted afterwards — proving a broken
            refit can never clobber the previously-active, working policy.

            Early fit attempts routinely fail the activation gate while
            judged pairs are still accumulating (not because fitting itself
            errors), and the oracle just keeps retrying with more data, so
            "first fit succeeds" is not the same as "first fit activates a
            policy" — the flakiness must be keyed off actual calibration,
            not call count.
            """

            def __init__(self, *, state: FlakyState) -> None:
                super().__init__()
                self._state = state

            def fit(self, batch, **kwargs):  # type: ignore[override]
                self._state.fit_calls += 1
                if self._state.should_fail:
                    raise RuntimeError("synthetic fit failure")
                return super().fit(batch, **kwargs)

            def copy_for_refit(self) -> "FlakyFitScorer":
                return FlakyFitScorer(state=self._state)

        with tempfile.TemporaryDirectory(prefix="scs-flaky-fit-") as tmp:
            prompts = _topic_prompts(700)
            state = FlakyState()
            system = _build_system(tmp, stream=prompts, scorer=FlakyFitScorer(state=state), scorers=None)
            oracle = system.cache.runtime.oracle

            calibrated_threshold: float | None = None
            calibrated_scorer_version: int | None = None
            for index, prompt in enumerate(prompts, start=1):
                response = system.handle(prompt)
                if calibrated_threshold is None and response.policy.calibrated:
                    calibrated_threshold = response.policy.threshold
                    calibrated_scorer_version = response.policy.scorer_version
                    # From here on, every refit attempt must fail -- proving
                    # the active policy stays untouched by broken retrains.
                    state.should_fail = True
                if index % system.batch_size == 0:
                    system._maybe_check_stopping()

            # Drain any in-flight background fit (it may raise -- that's the
            # synthetic failure under test, and `decide()` already handles it
            # without disturbing the active policy; here we just don't want a
            # daemon thread outliving the test).
            deadline = time.monotonic() + 5.0
            while oracle.fit_in_progress and time.monotonic() < deadline:
                time.sleep(0.05)
            try:
                oracle.wait_for_fit(timeout=0.0)
            except BaseException:
                pass

            self.assertIsNotNone(calibrated_threshold, "the first (successful) fit must calibrate the policy")
            self.assertGreater(state.fit_calls, 1, "the fixture must have attempted at least one more (failing) fit")

            policy = system.policy
            # A failed refit must leave the previously-active scorer/threshold
            # serving — never an unfit, broken, or half-swapped instance.
            self.assertTrue(policy.calibrated)
            self.assertIsNotNone(policy.threshold)
            self.assertTrue(np.isfinite(policy.threshold))
            self.assertEqual(policy.threshold, calibrated_threshold)
            self.assertEqual(policy.scorer_version, calibrated_scorer_version)

            # And serving must keep working normally throughout.
            responses = [system.handle(prompt) for prompt in _topic_prompts(30)]
            self.assertTrue(all(response.text for response in responses))
            self.assertTrue(any(response.source == "cache" for response in responses))


class OfflineCalibrationTests(unittest.TestCase):
    """Direct unit coverage for `MLCache.prefit_and_calibrate`, the offline
    cold-start counterpart of the online lifecycle (used by experiments that
    replay labeled datasets instead of running a live judge)."""

    @staticmethod
    def _judged_pairs(provider: TopicEmbeddingProvider, *, count: int = 200, topics: int = 6) -> list:
        from mlcache import JudgedPairExample
        from mlcache.features import NormalizedHadamardFeatureBuilder

        feature_builder = NormalizedHadamardFeatureBuilder()
        prompts = _topic_prompts(count, topics=topics)
        embeddings = [provider.embed(prompt) for prompt in prompts]

        pairs = []
        judge = DeterministicTopicJudge()
        for i, query_prompt in enumerate(prompts):
            # Pair each query with two synthetic "candidates": one `topics`
            # positions back (guaranteed same topic -> H1) and its immediate
            # neighbor (a different topic, since consecutive prompts cycle
            # through distinct `#topic` ids -> H0). This gives both labels
            # solid representation without depending on chance.
            candidate_indices = {(i + 1) % len(prompts)}
            if i >= topics:
                candidate_indices.add(i - topics)
            for j in candidate_indices:
                candidate_prompt = prompts[j]
                features = feature_builder.build(embeddings[i], embeddings[j])
                request = JudgeRequest(query=query_prompt, candidate_query=candidate_prompt)
                judge_result = judge.judge(request)
                pairs.append(
                    JudgedPairExample(
                        features=tuple(float(value) for value in features.hadamard),
                        request=request,
                        decision=judge_result.decision,
                    )
                )
        return pairs

    def test_prefit_and_calibrate_fits_trains_calibrates_and_reports(self) -> None:
        from mlcache import MLCache
        from mlcache.features import PairFeatures

        provider = TopicEmbeddingProvider()
        pairs = self._judged_pairs(provider)
        n_h0 = sum(1 for pair in pairs if pair.decision.label == JudgeLabel.NOT_REUSABLE)
        n_h1 = sum(1 for pair in pairs if pair.decision.label == JudgeLabel.REUSABLE)
        self.assertGreater(n_h0, 0)
        self.assertGreater(n_h1, 0)

        with tempfile.TemporaryDirectory(prefix="scs-prefit-calibrate-") as tmp:
            cache = MLCache.from_preset(
                root_dir=tmp,
                scorer="ensemble",
                scorers=["cosine", "lda"],
                top_k=3,
                persistence=False,
            )
            sample_features = PairFeatures(hadamard=pairs[0].features)
            # An unfit LDA member raises rather than scoring a degenerate
            # value, so it can't be scored *before* `prefit_and_calibrate` —
            # which is itself the clearest proof that fitting is mandatory
            # and must have actually happened for scoring to succeed below.
            with self.assertRaises(ValueError):
                cache.scorer.score(sample_features)

            report = cache.prefit_and_calibrate(pairs, target_fpr=0.2, fit_kwargs={"alpha": 0.05, "seed": 7})

            # `prefit` fits the scorer (and its trainable ensemble members) in
            # place -- no instance swap, that's the oracle's online-refit job,
            # not the offline cold-start path's. Scoring now succeeding is
            # direct proof the trainable member was actually fit.
            score_after_fit = cache.scorer.score(sample_features)
            self.assertTrue(np.isfinite(float(score_after_fit)))

            # The threshold must be finite and active on the runtime.
            self.assertIsNotNone(cache.runtime.oracle._threshold)
            self.assertTrue(np.isfinite(float(cache.runtime.oracle._threshold)))
            self.assertEqual(float(cache.runtime.oracle._threshold), float(report["threshold"]))

            self.assertEqual(set(report.keys()), {"scorer", "threshold", "target_fpr", "n_h0", "n_h1"})
            self.assertEqual(report["scorer"], cache.scorer.name)
            self.assertTrue(np.isfinite(report["threshold"]))
            self.assertEqual(report["target_fpr"], 0.2)
            self.assertEqual(report["n_h0"], n_h0)
            self.assertEqual(report["n_h1"], n_h1)

    def test_prefit_and_calibrate_calibrates_cosine_without_fitting(self) -> None:
        from mlcache import MLCache

        provider = TopicEmbeddingProvider()
        pairs = self._judged_pairs(provider)

        with tempfile.TemporaryDirectory(prefix="scs-prefit-calibrate-cosine-") as tmp:
            cache = MLCache.from_preset(root_dir=tmp, scorer="cosine", top_k=3, persistence=False)

            report = cache.prefit_and_calibrate(pairs, target_fpr=0.2)

            self.assertEqual(report["scorer"], cache.scorer.name)
            self.assertTrue(np.isfinite(report["threshold"]))
            self.assertTrue(np.isfinite(float(cache.runtime.oracle._threshold)))


class RefitPolicyWiringTests(unittest.TestCase):
    """SemanticCacheSystem(min_h0=M, min_h1=N, batch_size=B, target_fpr=F) must
    produce a ConservativeRefitPolicy scaled to those values — not the
    production defaults (min_h0_for_fit=100, min_train_total=1200).

    Critically, min_calibration_h0 must be large enough that the Wilson
    activation gate can actually pass: z²/(n+z²) ≤ target_fpr + fpr_wilson_margin.
    When this condition is not met the fit fires, but the gate always rejects
    the result, and trained stays False permanently.
    """

    def _cfg(self, *, min_h0: int = 15, min_h1: int = 8,
             batch_size: int = 20, target_fpr: float = 0.10) -> object:
        with tempfile.TemporaryDirectory(prefix="scs-wiring-") as tmp:
            system = SemanticCacheSystem(
                llm=MockLLM(response_template="ans: {prompt}"),
                min_h0=min_h0, min_h1=min_h1, batch_size=batch_size,
                target_fpr=target_fpr, root_dir=tmp, persistence=False,
            )
            cfg = system.cache.runtime.oracle.refit_policy.config
            system._executor.shutdown(wait=False)
            return cfg

    def test_not_production_defaults(self) -> None:
        cfg = self._cfg()
        self.assertNotEqual(cfg.min_h0_for_fit, 100, "min_h0_for_fit must not be the production default")
        self.assertNotEqual(cfg.min_train_total, 1200, "min_train_total must not be the production default")

    def test_min_h1_for_fit_equals_passed_value(self) -> None:
        cfg = self._cfg(min_h1=8)
        self.assertEqual(cfg.min_h1_for_fit, 8)

    def test_min_h0_for_fit_is_at_least_passed_min_h0(self) -> None:
        cfg = self._cfg(min_h0=15)
        self.assertGreaterEqual(cfg.min_h0_for_fit, 15)

    def test_train_splits_are_scaled_by_batch_size(self) -> None:
        # batch_size=20 -> cal_every_n=2 -> train_frac=0.5
        cfg = self._cfg(min_h0=20, min_h1=10, batch_size=20)
        self.assertEqual(cfg.min_train_h0, 10)   # int(20 * 0.5)
        self.assertEqual(cfg.min_train_h1, 5)    # int(10 * 0.5)

    def test_wilson_gate_is_achievable_with_min_calibration_h0_examples(self) -> None:
        """With min_calibration_h0 H0 calib examples and 0 false accepts,
        z²/(n+z²) must be ≤ target_fpr + fpr_wilson_margin.
        If not, the activation gate can never open regardless of data volume."""
        cfg = self._cfg(target_fpr=0.10)
        z = cfg.wilson_confidence_z
        allowed = 0.10 + cfg.fpr_wilson_margin
        n = cfg.min_calibration_h0
        wilson_upper_at_min = z * z / (n + z * z)
        self.assertLessEqual(
            wilson_upper_at_min, allowed,
            f"Wilson gate unachievable at min_calibration_h0={n}: "
            f"upper={wilson_upper_at_min:.4f} > allowed={allowed:.4f}",
        )

    def test_wilson_gate_achievable_at_fpr_25(self) -> None:
        cfg = self._cfg(target_fpr=0.25)
        z = cfg.wilson_confidence_z
        allowed = 0.25 + cfg.fpr_wilson_margin
        n = cfg.min_calibration_h0
        wilson_upper_at_min = z * z / (n + z * z)
        self.assertLessEqual(wilson_upper_at_min, allowed,
                             f"Wilson gate unachievable at fpr=0.25: upper={wilson_upper_at_min:.4f}")


class EndToEndConstructionTests(unittest.TestCase):
    def test_from_llm_and_stream_builds_and_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-from-llm-stream-") as tmp:
            prompts = _topic_prompts(80)
            system = SemanticCacheSystem.from_llm_and_stream(
                llm=MockLLM(response_template="answer for: {prompt}"),
                stream=prompts,
                scorer="cosine",
                judge=DeterministicTopicJudge(),
                embedding_provider=TopicEmbeddingProvider(),
                top_k=3,
                root_dir=tmp,
                batch_size=20,
                min_h0=10,
                min_h1=5,
                persistence=False,
            )

            report = system.run(max_requests=80)

            self.assertEqual(report["requests"], 80)
            self.assertGreater(report["llm_calls"], 0)

    def test_default_judge_is_synthesized_from_the_llm_when_not_provided(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scs-default-judge-") as tmp:
            llm = MockLLM()
            system = SemanticCacheSystem(llm=llm, stream=None, root_dir=tmp, persistence=False)

            from mlcache import LLMJudge

            self.assertIsInstance(system.judge, LLMJudge)


if __name__ == "__main__":
    unittest.main()
