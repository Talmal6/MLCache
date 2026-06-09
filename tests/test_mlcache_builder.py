from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mlcache import (
    EnsembleScorer,
    LabeledPairBatch,
    ML_DEPENDENCIES_ERROR,
    MLCache,
    Threshold,
    build_mlcache,
)
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, CacheMetadata, Query, Response


ROOT = Path(__file__).resolve().parents[1]
RUN_CACHE_SCRIPT = ROOT / "scripts" / "run_cache.py"
SPEC = importlib.util.spec_from_file_location("run_cache", RUN_CACHE_SCRIPT)
run_cache = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_cache)


def _entry(key: str, query: str, vector: np.ndarray) -> CacheEntry:
    return CacheEntry(
        cache_key=CacheKey(key),
        query=Query(query),
        response=Response(f"response-for-{key}"),
        embedding=tuple(float(v) for v in vector),
        metadata=CacheMetadata(),
    )


def _lookup(query: str, vector: np.ndarray) -> CacheLookup:
    return CacheLookup(query=Query(query), embedding=tuple(float(v) for v in vector))


class FromPresetCosineTests(unittest.TestCase):
    def test_from_preset_builds_a_cosine_cache_and_serves_lookups(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-cosine-") as tmp:
            cache = MLCache.from_preset(
                root_dir=Path(tmp) / "state",
                scorer="cosine",
                threshold=0.0,
                persistence=False,
            )

            self.assertEqual(str(cache.scorer.name), "Cosine")
            self.assertEqual(cache.threshold, Threshold(0.0))

            vector = np.ones(8, dtype=np.float32)
            cache.index([_entry("anchor-1", "anchor query", vector)])
            result = cache.lookup_with_decision(_lookup("anchor query", vector))

            self.assertTrue(result.decision.accepted)
            self.assertEqual(result.response, Response("response-for-anchor-1"))

    def test_build_mlcache_factory_function_matches_from_preset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-factory-") as tmp:
            cache = build_mlcache(root_dir=Path(tmp) / "state", scorer="cosine", persistence=False)

            self.assertIsInstance(cache, MLCache)
            self.assertEqual(str(cache.scorer.name), "Cosine")


class FromPresetEnsembleTests(unittest.TestCase):
    def test_from_preset_builds_an_ensemble_with_all_five_scorers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-ensemble-") as tmp:
            cache = MLCache.from_preset(
                root_dir=Path(tmp) / "state",
                scorer="ensemble",
                scorers=["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"],
                threshold=0.05,
                top_k=5,
                persistence=False,
            )

            self.assertIsInstance(cache.scorer, EnsembleScorer)
            member_names = [str(member.name) for member in cache.scorer.members()]
            self.assertEqual(member_names, ["Cosine", "LDA", "PCAWhitenedCosine", "XGBoost", "Tiny MLP"])


class MissingDependencyTests(unittest.TestCase):
    def test_prefit_raises_clean_error_when_ml_dependency_is_missing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-missing-dep-") as tmp:
            cache = MLCache.from_preset(
                root_dir=Path(tmp) / "state",
                scorer="xgboost",
                persistence=False,
            )

            rng = np.random.default_rng(0)
            batch = LabeledPairBatch(
                h0=[tuple(rng.normal(size=8)) for _ in range(4)],
                h1=[tuple(rng.normal(size=8)) for _ in range(4)],
            )

            with mock.patch.dict(sys.modules, {"xgboost": None}):
                with self.assertRaises(ImportError) as ctx:
                    cache.prefit(batch, alpha=0.05, seed=42)

            self.assertEqual(str(ctx.exception), ML_DEPENDENCIES_ERROR)


def _ml_deps_available() -> bool:
    """Return True when optional ML packages (xgboost, sklearn) are importable."""
    for name in ("xgboost", "sklearn"):
        try:
            importlib.import_module(name)
        except ImportError:
            return False
    return True


_SKIP_ML_DEPS = unittest.skipUnless(
    _ml_deps_available(),
    "requires optional ML dependencies; install with pip install -e '.[ml,dev]'",
)


def _write_synthetic_npz(path: Path, *, row_count: int = 8) -> None:
    embeddings = np.eye(row_count, dtype=np.float32)
    np.savez(
        path,
        text=np.asarray([f"query {idx}" for idx in range(row_count)], dtype=object),
        global_cluster=np.asarray([f"anchor {idx}" for idx in range(row_count)], dtype=object),
        label=np.asarray([1 if idx % 2 == 0 else 0 for idx in range(row_count)], dtype=np.int32),
        emb=embeddings,
    )


_EXPECTED_ARTIFACTS = (
    "summary_metrics.json",
    "per_request_decisions.csv",
    "runtime_config.json",
    "schema_report.json",
    "calibration_report.json",
)


class RunCacheScriptSmokeTests(unittest.TestCase):
    def test_run_cache_cosine_smoke_on_tiny_synthetic_npz(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-run-cache-cosine-") as tmp:
            tmp_path = Path(tmp)
            npz_path = tmp_path / "tiny_h1h0.npz"
            _write_synthetic_npz(npz_path)
            output_dir = tmp_path / "experiment"

            exit_code = run_cache.main(
                [
                    "--npz",
                    str(npz_path),
                    "--output-dir",
                    str(output_dir),
                    "--label-field",
                    "label",
                    "--query-field",
                    "text",
                    "--anchor-field",
                    "global_cluster",
                    "--query-embedding-field",
                    "emb",
                    "--scorer",
                    "cosine",
                    "--target-fpr",
                    "0.05",
                    "--top-k",
                    "3",
                    "--no-file-persistence",
                ]
            )

            self.assertEqual(exit_code, 0)
            for name in _EXPECTED_ARTIFACTS:
                self.assertTrue((output_dir / name).exists(), f"missing artifact: {name}")

    @_SKIP_ML_DEPS
    def test_run_cache_ensemble_smoke_on_tiny_synthetic_npz(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mlcache-run-cache-") as tmp:
            tmp_path = Path(tmp)
            npz_path = tmp_path / "tiny_h1h0.npz"
            _write_synthetic_npz(npz_path)
            output_dir = tmp_path / "experiment"

            exit_code = run_cache.main(
                [
                    "--npz",
                    str(npz_path),
                    "--output-dir",
                    str(output_dir),
                    "--label-field",
                    "label",
                    "--query-field",
                    "text",
                    "--anchor-field",
                    "global_cluster",
                    "--query-embedding-field",
                    "emb",
                    "--scorer",
                    "ensemble",
                    "--scorers",
                    "cosine,lda,pca_whitened_cosine,xgboost,mlp",
                    "--target-fpr",
                    "0.05",
                    "--top-k",
                    "5",
                    "--no-file-persistence",
                ]
            )

            self.assertEqual(exit_code, 0)
            for name in _EXPECTED_ARTIFACTS:
                self.assertTrue((output_dir / name).exists(), f"missing artifact: {name}")


if __name__ == "__main__":
    unittest.main()
