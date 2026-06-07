import unittest

from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.feedback import (
    DefaultShadowTopKCollector,
    InMemorySplitJudgeTrainingStore,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    SemanticReuseJudge,
    ShadowCollectionConfig,
)
from mlcache.retrieval import VectorSearchResult
from mlcache.semantic_types import (
    CacheKey,
    CacheLookup,
    OracleDecision,
    OracleDecisionStatus,
    Query,
    Score,
    ScorerName,
    Threshold,
)


class MappingJudge(SemanticReuseJudge):
    def __init__(self, labels: dict[str, JudgeLabel | Exception]) -> None:
        self.labels = labels
        self.requests: list[JudgeRequest] = []

    @property
    def name(self) -> str:
        return "mapping"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        key = str(request.candidate_key)
        label = self.labels[key]
        if isinstance(label, Exception):
            raise label
        return JudgeResult(request=request, decision=JudgeDecision(label=label))


def request(query: str = "new") -> CacheLookup:
    return CacheLookup(query=Query(query), embedding=(1.0, 0.0))


def candidate(key: str, *, query_text: str = "old", score: float = 0.9) -> VectorSearchResult:
    return VectorSearchResult(
        cache_key=CacheKey(key),
        embedding=(1.0, 0.0),
        score=Score(score),
        query=Query(query_text),
    )


def served_decision(key: str | None = None, *, accepted: bool = False) -> OracleDecision:
    return OracleDecision(
        status=OracleDecisionStatus.HIT if accepted else OracleDecisionStatus.MISS,
        accepted=accepted,
        cache_key=CacheKey(key) if key is not None else None,
        score=Score(1.0) if accepted else None,
        threshold=Threshold(0.5),
        scorer=ScorerName("fixed"),
    )


def split_store() -> InMemorySplitJudgeTrainingStore:
    return InMemorySplitJudgeTrainingStore(
        max_h0_train=20,
        max_h1_train=20,
        max_h0_calibration=20,
        max_h1_calibration=20,
    )


def collector(
    labels: dict[str, JudgeLabel | Exception],
    store: InMemorySplitJudgeTrainingStore,
    config: ShadowCollectionConfig | None = None,
) -> DefaultShadowTopKCollector:
    return DefaultShadowTopKCollector(
        feature_builder=NormalizedHadamardFeatureBuilder(),
        judge=MappingJudge(labels),
        store=store,
        config=config,
    )


class DefaultShadowTopKCollectorTests(unittest.TestCase):
    def test_skips_duplicate_candidates(self) -> None:
        store = split_store()
        shadow = collector({"a": JudgeLabel.REUSABLE}, store)

        result = shadow.collect(request(), [candidate("a"), candidate("a")], served_decision())

        self.assertEqual(result.pairs_observed, 1)
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.judge_calls, 1)
        self.assertEqual(len(store.h1_train()), 1)

    def test_skips_self_pairs(self) -> None:
        store = split_store()
        shadow = collector({"a": JudgeLabel.REUSABLE}, store)

        result = shadow.collect(request("same"), [candidate("a", query_text="same")], served_decision())

        self.assertEqual(result.pairs_observed, 0)
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.judge_calls, 0)
        self.assertEqual(store.h0(), ())
        self.assertEqual(store.h1(), ())

    def test_h0_and_h1_labels_are_stored(self) -> None:
        store = split_store()
        shadow = collector({"h0": JudgeLabel.NOT_REUSABLE, "h1": JudgeLabel.REUSABLE}, store)

        result = shadow.collect(request(), [candidate("h0"), candidate("h1")], served_decision())

        self.assertEqual(result.h0_added, 1)
        self.assertEqual(result.h1_added, 1)
        self.assertEqual(len(store.h0_train()), 1)
        self.assertEqual(len(store.h1_train()), 1)
        self.assertEqual(store.h0_train()[0].metadata["source"], "shadow_top_k")
        self.assertEqual(store.h1_train()[0].metadata["candidate_rank"], 2)

    def test_calibration_every_n_split_is_deterministic_per_label(self) -> None:
        store = split_store()
        labels = {
            "h0-1": JudgeLabel.NOT_REUSABLE,
            "h0-2": JudgeLabel.NOT_REUSABLE,
            "h0-3": JudgeLabel.NOT_REUSABLE,
            "h0-4": JudgeLabel.NOT_REUSABLE,
            "h1-1": JudgeLabel.REUSABLE,
            "h1-2": JudgeLabel.REUSABLE,
            "h1-3": JudgeLabel.REUSABLE,
            "h1-4": JudgeLabel.REUSABLE,
        }
        shadow = collector(labels, store, ShadowCollectionConfig(top_k=8, calibration_every_n=2))
        candidates = [candidate(key) for key in labels]

        result = shadow.collect(request(), candidates, served_decision())

        self.assertEqual(result.h0_added, 4)
        self.assertEqual(result.h1_added, 4)
        self.assertEqual(len(store.h0_train()), 2)
        self.assertEqual(len(store.h0_calibration()), 2)
        self.assertEqual(len(store.h1_train()), 2)
        self.assertEqual(len(store.h1_calibration()), 2)
        self.assertEqual([ex.metadata["split"] for ex in store.h0_train()], ["train", "train"])
        self.assertEqual([ex.metadata["split"] for ex in store.h0_calibration()], ["calibration", "calibration"])

    def test_uncertain_is_counted_and_not_stored_by_default(self) -> None:
        store = split_store()
        shadow = collector({"a": JudgeLabel.UNCERTAIN}, store)

        result = shadow.collect(request(), [candidate("a")], served_decision())

        self.assertEqual(result.uncertain, 1)
        self.assertEqual(result.judge_calls, 1)
        self.assertEqual(store.h0(), ())
        self.assertEqual(store.h1(), ())

    def test_judge_exception_counts_failure_and_continues(self) -> None:
        store = split_store()
        shadow = collector({"bad": RuntimeError("judge failed"), "good": JudgeLabel.REUSABLE}, store)

        result = shadow.collect(request(), [candidate("bad"), candidate("good")], served_decision())

        self.assertEqual(result.failures, 1)
        self.assertEqual(result.judge_calls, 2)
        self.assertEqual(result.h1_added, 1)
        self.assertEqual(len(store.h1_train()), 1)

    def test_snapshot_reports_lifetime_counts_and_store_sizes(self) -> None:
        store = split_store()
        shadow = collector(
            {"h0": JudgeLabel.NOT_REUSABLE, "h1": JudgeLabel.REUSABLE, "u": JudgeLabel.UNCERTAIN},
            store,
            ShadowCollectionConfig(top_k=3, calibration_every_n=2),
        )

        shadow.collect(request(), [candidate("h0"), candidate("h1"), candidate("u")], served_decision())
        snapshot = shadow.snapshot()

        self.assertEqual(snapshot.pairs_observed, 3)
        self.assertEqual(snapshot.judge_calls, 3)
        self.assertEqual(snapshot.h0_observed, 1)
        self.assertEqual(snapshot.h1_observed, 1)
        self.assertEqual(snapshot.uncertain_observed, 1)
        self.assertEqual(snapshot.train_h0, 1)
        self.assertEqual(snapshot.train_h1, 1)


if __name__ == "__main__":
    unittest.main()

