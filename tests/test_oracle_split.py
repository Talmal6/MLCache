import unittest

from mlcache.oracle import TrainableSemanticCacheOracle


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


if __name__ == "__main__":
    unittest.main()
