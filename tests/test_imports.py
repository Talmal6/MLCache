from cache import SemanticCacheGateway
from features import NormalizedHadamardFeatureBuilder, PairFeatureBuilder
from judges import DefaultShadowTopKCollector as OldDefaultShadowTopKCollector
from judges.store import InMemorySplitJudgeTrainingStore, SplitJudgeTrainingStore
from oracle import ActivationGateResult, TrainableSemanticCacheOracle
from thresholds import QueryLevelCalibrationConfig, ThresholdCalibrationRequest
from vector_store import VectorStore

from mlcache.cache import SemanticCacheGateway as NewSemanticCacheGateway
from mlcache.calibration import QueryLevelCalibrationConfig as NewQueryLevelCalibrationConfig
from mlcache.calibration import ThresholdCalibrationRequest as NewThresholdCalibrationRequest
from mlcache.calibration import wilson_upper_bound
from mlcache.feedback import DefaultShadowTopKCollector, InMemorySplitJudgeTrainingStore as NewSplitStore
from mlcache.feedback import SplitJudgeTrainingStore as NewSplitJudgeTrainingStore
from mlcache.features import NormalizedHadamardFeatureBuilder as NewNormalizedHadamardFeatureBuilder
from mlcache.oracle import ActivationGateResult as NewActivationGateResult
from mlcache.oracle import TrainableSemanticCacheOracle as NewTrainableSemanticCacheOracle
from mlcache.retrieval import VectorStore as NewVectorStore


import unittest


class ImportTests(unittest.TestCase):
    def test_old_imports_reexport_new_objects(self) -> None:
        self.assertIs(SemanticCacheGateway, NewSemanticCacheGateway)
        self.assertIs(TrainableSemanticCacheOracle, NewTrainableSemanticCacheOracle)
        self.assertIs(NormalizedHadamardFeatureBuilder, NewNormalizedHadamardFeatureBuilder)
        self.assertIs(ThresholdCalibrationRequest, NewThresholdCalibrationRequest)
        self.assertIs(QueryLevelCalibrationConfig, NewQueryLevelCalibrationConfig)
        self.assertIs(VectorStore, NewVectorStore)
        self.assertIs(InMemorySplitJudgeTrainingStore, NewSplitStore)
        self.assertIs(SplitJudgeTrainingStore, NewSplitJudgeTrainingStore)
        self.assertIs(OldDefaultShadowTopKCollector, DefaultShadowTopKCollector)
        self.assertIs(ActivationGateResult, NewActivationGateResult)
        self.assertTrue(issubclass(NormalizedHadamardFeatureBuilder, PairFeatureBuilder))

    def test_new_imports_are_available(self) -> None:
        self.assertEqual(NewSemanticCacheGateway.__name__, "SemanticCacheGateway")
        self.assertEqual(NewTrainableSemanticCacheOracle.__name__, "TrainableSemanticCacheOracle")
        self.assertEqual(NewNormalizedHadamardFeatureBuilder.__name__, "NormalizedHadamardFeatureBuilder")
        self.assertEqual(NewThresholdCalibrationRequest.__name__, "ThresholdCalibrationRequest")
        self.assertEqual(NewQueryLevelCalibrationConfig.__name__, "QueryLevelCalibrationConfig")
        self.assertEqual(NewVectorStore.__name__, "VectorStore")
        self.assertEqual(DefaultShadowTopKCollector.__name__, "DefaultShadowTopKCollector")
        self.assertEqual(NewSplitStore.__name__, "InMemorySplitJudgeTrainingStore")
        self.assertEqual(NewActivationGateResult.__name__, "ActivationGateResult")
        self.assertTrue(callable(wilson_upper_bound))


if __name__ == "__main__":
    unittest.main()
