from cache import SemanticCacheGateway
from features import NormalizedHadamardFeatureBuilder, PairFeatureBuilder
from oracle import TrainableSemanticCacheOracle
from thresholds import ThresholdCalibrationRequest
from vector_store import VectorStore

from mlcache.cache import SemanticCacheGateway as NewSemanticCacheGateway
from mlcache.calibration import ThresholdCalibrationRequest as NewThresholdCalibrationRequest
from mlcache.calibration import wilson_upper_bound
from mlcache.features import NormalizedHadamardFeatureBuilder as NewNormalizedHadamardFeatureBuilder
from mlcache.oracle import TrainableSemanticCacheOracle as NewTrainableSemanticCacheOracle
from mlcache.retrieval import VectorStore as NewVectorStore


import unittest


class ImportTests(unittest.TestCase):
    def test_old_imports_reexport_new_objects(self) -> None:
        self.assertIs(SemanticCacheGateway, NewSemanticCacheGateway)
        self.assertIs(TrainableSemanticCacheOracle, NewTrainableSemanticCacheOracle)
        self.assertIs(NormalizedHadamardFeatureBuilder, NewNormalizedHadamardFeatureBuilder)
        self.assertIs(ThresholdCalibrationRequest, NewThresholdCalibrationRequest)
        self.assertIs(VectorStore, NewVectorStore)
        self.assertTrue(issubclass(NormalizedHadamardFeatureBuilder, PairFeatureBuilder))

    def test_new_imports_are_available(self) -> None:
        self.assertEqual(NewSemanticCacheGateway.__name__, "SemanticCacheGateway")
        self.assertEqual(NewTrainableSemanticCacheOracle.__name__, "TrainableSemanticCacheOracle")
        self.assertEqual(NewNormalizedHadamardFeatureBuilder.__name__, "NormalizedHadamardFeatureBuilder")
        self.assertEqual(NewThresholdCalibrationRequest.__name__, "ThresholdCalibrationRequest")
        self.assertEqual(NewVectorStore.__name__, "VectorStore")
        self.assertTrue(callable(wilson_upper_bound))


if __name__ == "__main__":
    unittest.main()
