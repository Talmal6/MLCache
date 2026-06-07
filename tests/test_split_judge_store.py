import unittest

from mlcache.feedback import (
    InMemorySplitJudgeTrainingStore,
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgedPairExample,
)
from mlcache.semantic_types import CacheKey, Query


def example(label: JudgeLabel, key: str, value: float) -> JudgedPairExample:
    return JudgedPairExample(
        features=(value,),
        request=JudgeRequest(query=Query("q"), candidate_key=CacheKey(key)),
        decision=JudgeDecision(label=label),
    )


class InMemorySplitJudgeTrainingStoreTests(unittest.TestCase):
    def test_stores_h0_h1_train_and_calibration_buckets(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=3,
            max_h1_train=3,
            max_h0_calibration=3,
            max_h1_calibration=3,
        )

        h0_train = example(JudgeLabel.NOT_REUSABLE, "h0-train", 0.0)
        h1_train = example(JudgeLabel.REUSABLE, "h1-train", 1.0)
        h0_calib = example(JudgeLabel.NOT_REUSABLE, "h0-calib", 2.0)
        h1_calib = example(JudgeLabel.REUSABLE, "h1-calib", 3.0)
        store.add_train(h0_train)
        store.add_train(h1_train)
        store.add_calibration(h0_calib)
        store.add_calibration(h1_calib)

        self.assertEqual(store.h0_train(), (h0_train,))
        self.assertEqual(store.h1_train(), (h1_train,))
        self.assertEqual(store.h0_calibration(), (h0_calib,))
        self.assertEqual(store.h1_calibration(), (h1_calib,))

    def test_legacy_h0_h1_methods_return_all_splits(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=3,
            max_h1_train=3,
            max_h0_calibration=3,
            max_h1_calibration=3,
        )
        h0_train = example(JudgeLabel.NOT_REUSABLE, "h0-train", 0.0)
        h0_calib = example(JudgeLabel.NOT_REUSABLE, "h0-calib", 1.0)
        h1_train = example(JudgeLabel.REUSABLE, "h1-train", 2.0)
        h1_calib = example(JudgeLabel.REUSABLE, "h1-calib", 3.0)

        store.add_train(h0_train)
        store.add_calibration(h0_calib)
        store.add_train(h1_train)
        store.add_calibration(h1_calib)

        self.assertEqual(store.h0(), (h0_train, h0_calib))
        self.assertEqual(store.h1(), (h1_train, h1_calib))

    def test_train_and_calibration_do_not_overlap(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=3,
            max_h1_train=3,
            max_h0_calibration=3,
            max_h1_calibration=3,
        )
        row = example(JudgeLabel.NOT_REUSABLE, "same", 0.0)

        store.add_train(row)
        store.add_calibration(row)

        self.assertEqual(store.h0_train(), (row,))
        self.assertEqual(store.h0_calibration(), ())

    def test_uncertain_is_ignored_by_default(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=3,
            max_h1_train=3,
            max_h0_calibration=3,
            max_h1_calibration=3,
        )

        store.add_train(example(JudgeLabel.UNCERTAIN, "uncertain", 0.0))

        self.assertEqual(store.h0(), ())
        self.assertEqual(store.h1(), ())


if __name__ == "__main__":
    unittest.main()

