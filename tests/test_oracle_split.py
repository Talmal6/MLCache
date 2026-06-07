import unittest

from mlcache.feedback import InMemorySplitJudgeTrainingStore, JudgeDecision, JudgeLabel, JudgeRequest, JudgedPairExample
from mlcache.oracle import TrainableSemanticCacheOracle
from mlcache.semantic_types import CacheKey, Query


def _assert_disjoint(test_case: unittest.TestCase, *parts: tuple[tuple[float, ...], ...]) -> None:
    seen: set[tuple[float, ...]] = set()
    for part in parts:
        overlap = seen.intersection(part)
        test_case.assertFalse(overlap)
        seen.update(part)


class OracleSplitTests(unittest.TestCase):
    def test_split_rows_empty(self) -> None:
        train, calib, eval_rows = TrainableSemanticCacheOracle._split_rows(())

        self.assertEqual(train, ())
        self.assertEqual(calib, ())
        self.assertEqual(eval_rows, ())

    def test_split_rows_one_row_does_not_duplicate(self) -> None:
        rows = ((1.0,),)
        train, calib, eval_rows = TrainableSemanticCacheOracle._split_rows(rows)

        self.assertEqual(train, rows)
        self.assertEqual(calib, ())
        self.assertEqual(eval_rows, ())
        _assert_disjoint(self, train, calib, eval_rows)

    def test_split_rows_two_rows_are_disjoint_train_and_calib(self) -> None:
        rows = ((1.0,), (2.0,))
        train, calib, eval_rows = TrainableSemanticCacheOracle._split_rows(rows)

        self.assertEqual(train, ((1.0,),))
        self.assertEqual(calib, ((2.0,),))
        self.assertEqual(eval_rows, ())
        _assert_disjoint(self, train, calib, eval_rows)

    def test_split_rows_normal_split_has_no_overlap(self) -> None:
        rows = tuple((float(idx),) for idx in range(10))
        train, calib, eval_rows = TrainableSemanticCacheOracle._split_rows(rows)

        self.assertEqual(len(train), 7)
        self.assertEqual(len(calib), 2)
        self.assertEqual(len(eval_rows), 1)
        self.assertEqual(train + calib + eval_rows, rows)
        _assert_disjoint(self, train, calib, eval_rows)

    def test_build_refit_split_returns_none_when_calibration_is_insufficient(self) -> None:
        oracle = object.__new__(TrainableSemanticCacheOracle)

        split = oracle._build_refit_split(((1.0,),), ((2.0,),), metadata={})

        self.assertIsNone(split)

    def test_split_aware_training_store_is_not_resplit_for_refit(self) -> None:
        store = InMemorySplitJudgeTrainingStore(
            max_h0_train=3,
            max_h1_train=3,
            max_h0_calibration=3,
            max_h1_calibration=3,
        )
        h0_train = JudgedPairExample(
            features=(0.0,),
            request=JudgeRequest(query=Query("q"), candidate_key=CacheKey("h0-train")),
            decision=JudgeDecision(label=JudgeLabel.NOT_REUSABLE),
        )
        h1_train = JudgedPairExample(
            features=(1.0,),
            request=JudgeRequest(query=Query("q"), candidate_key=CacheKey("h1-train")),
            decision=JudgeDecision(label=JudgeLabel.REUSABLE),
        )
        h0_calib = JudgedPairExample(
            features=(2.0,),
            request=JudgeRequest(query=Query("q"), candidate_key=CacheKey("h0-calib")),
            decision=JudgeDecision(label=JudgeLabel.NOT_REUSABLE),
        )
        store.add_train(h0_train)
        store.add_train(h1_train)
        store.add_calibration(h0_calib)
        oracle = object.__new__(TrainableSemanticCacheOracle)

        rows = oracle._refit_rows_from_training_store(store)
        split = oracle._build_explicit_refit_split(rows, metadata={})

        self.assertIsNotNone(split)
        self.assertEqual(split.h0_train, ((0.0,),))
        self.assertEqual(split.h1_train, ((1.0,),))
        self.assertEqual(split.h0_calib, ((2.0,),))
        self.assertEqual(split.h1_calib, ())
        self.assertEqual(split.h0_eval, ())
        self.assertEqual(split.h1_eval, ())


if __name__ == "__main__":
    unittest.main()
