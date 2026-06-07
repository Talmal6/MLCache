import unittest

from mlcache.features import NormalizedHadamardFeatureBuilder, PairFeatureKind


class NormalizedHadamardFeatureBuilderTests(unittest.TestCase):
    def test_normalized_hadamard_features_store_cosine_separately(self) -> None:
        features = NormalizedHadamardFeatureBuilder().build([3, 4], [4, 0])

        self.assertAlmostEqual(features.hadamard[0], 0.6)
        self.assertAlmostEqual(features.hadamard[1], 0.0)
        self.assertAlmostEqual(float(features.cosine), 0.6)
        self.assertEqual(len(features.hadamard), 2)
        self.assertEqual(features.abs_diff, ())
        self.assertEqual(features.concat, ())

    def test_normalized_hadamard_features_include_metadata(self) -> None:
        features = NormalizedHadamardFeatureBuilder().build([3, 4], [4, 0])

        self.assertEqual(features.values["feature_type"], "normalized_hadamard")
        self.assertEqual(features.values["embedding_dim"], 2)
        self.assertEqual(features.values["dtype"], "float64")

    def test_normalized_hadamard_default_kind(self) -> None:
        self.assertEqual(NormalizedHadamardFeatureBuilder().default_kind(), PairFeatureKind.HADAMARD)

    def test_normalized_hadamard_rejects_zero_vectors(self) -> None:
        builder = NormalizedHadamardFeatureBuilder()

        with self.assertRaisesRegex(ValueError, "query_embedding must not be a zero vector"):
            builder.build([0.0, 0.0], [1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "candidate_embedding must not be a zero vector"):
            builder.build([1.0, 0.0], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
