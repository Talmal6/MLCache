# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**semantic-desider** is a correctness-controlled online semantic cache library (`mlcache` package). It wraps any LLM with a semantic cache layer that learns when cached responses are safe to reuse, calibrates a false-positive-rate (FPR) budget via Neyman-Pearson thresholding, and adapts online as traffic arrives.

The active Python environment is `.conda/` in the repo root. Use it as the interpreter: `.conda/bin/python`.

## Commands

```bash
# Install (choose one)
make setup          # base: numpy + dev
make setup-ml       # + scikit-learn, xgboost
make setup-embeddings  # + sentence-transformers
make setup-all      # everything

# Tests (integration tests are excluded by default)
python -m pytest -q
python -m pytest tests/test_semantic_cache_system.py -q    # single file
python -m pytest -m integration -q                          # opt-in to integration tests

# Smoke tests
make smoke                    # requires embeddings installed
make smoke-local              # use cached model only

# Key experiment scripts
python scripts/run_cache.py --npz data/h1h0_final.npz --output-dir runs/out --scorer ensemble ...
python scripts/run_mock_cached_llm_smoke.py --scorer ensemble --scorers cosine,lda ...
python scripts/diagnose_semantic_cache_lifecycle.py --scorer ensemble --scorers cosine,lda ...
```

## Architecture

### Layered design (bottom-up)

```
EmbeddingProvider          → raw text → float vector
PairFeatureBuilder         → (query_emb, candidate_emb) → PairFeatures
                              (only impl: NormalizedHadamardFeatureBuilder)
SemanticScorer             → PairFeatures → Score; calibrate() → Threshold
                              impls: CosineScorer, LDAScorer, PCAWhitenedCosineScorer,
                                     XGBoostScorer, TinyMLPScorer, EnsembleScorer
TrainableSemanticCacheOracle  → owns scorer fitting, NP thresholding, top-k retrieval,
                                 shadow judging, atomic scorer swap on refit
SemanticCacheGateway       → wraps oracle + KVStore; produces CacheGatewayResult
MLCacheRuntime             → production boundary: gateway + oracle + stores + observability
MLCache                    → ergonomic facade; from_preset() is the normal constructor
SemanticCacheSystem        → full self-running lifecycle: serving + online learning + freezing
```

### Key data flow

1. **Lookup**: `CacheLookup(query, embedding)` → `MLCacheRuntime.lookup_with_decision` → oracle retrieves top-k from `VectorStore`, builds `PairFeatures` per candidate, scores with active `SemanticScorer`, compares to `Threshold`, returns `OracleDecision` (HIT / MISS / ABSTAIN) + `CacheGatewayResult`.

2. **Online learning** (shadow mode): On each request the oracle also judges top-k candidates via `ShadowTopKCollector` → `SemanticReuseJudge` labels pairs as REUSABLE / NOT_REUSABLE → `SplitJudgeTrainingStore` accumulates H0/H1 pairs in train and calibration buckets → `ConservativeRefitPolicy` gates refit until bucket thresholds are met → background thread refits scorer and recalibrates threshold atomically.

3. **Calibration**: `NPThresholdCalibrator` / `scorer.calibrate()` applies Neyman-Pearson: finds the smallest threshold τ such that empirical FPR ≤ `target_false_accept_rate`. Wilson upper bound (`wilson_upper_bound()`) is used for the activation gate safety check.

4. **Freezing** (`SemanticCacheSystem`): `WindowedOnlineStoppingController` monitors online TPR/FPR; once metrics stabilize, it calls `system.freeze()` which sets `oracle.auto_refit = False` — serving continues, training stops.

### Module map (`src/mlcache/`)

| Package | Responsibility |
|---|---|
| `semantic_types` | Shared `NewType`s and dataclasses (`CacheEntry`, `CacheLookup`, `OracleDecision`, `LabeledPairBatch`, …) |
| `embeddings` | `EmbeddingProvider` ABC, `HashingEmbeddingProvider` (offline, no deps), `SentenceTransformersEmbeddingProvider` |
| `features/` | `PairFeatureBuilder` ABC + `NormalizedHadamardFeatureBuilder` |
| `scorers/` | `SemanticScorer` ABC, all concrete scorers, `EnsembleScorer`, `ScorerRegistry` |
| `calibration/` | `NPThresholdCalibrator`, `ThresholdProvider`, `wilson_upper_bound`, query-level calibration |
| `feedback/` | `JudgeTrainingStore`, `SplitJudgeTrainingStore`, `ShadowTopKCollector`, `SemanticReuseJudge`, NPZ adapters |
| `oracle/` | `SemanticCacheOracle` ABC, `TrainableSemanticCacheOracle` (shadow + refit logic) |
| `policies/` | `CachePolicy`, `RefitPolicy`, `ConservativeRefitPolicy`, query-level policies |
| `cache/` | `KVStore`, `InMemoryKVStore`, `FileKVStore`, `SemanticCacheGateway` |
| `retrieval/` | `VectorStore`, `InMemoryVectorStore`, `FileVectorStore` |
| `runtime/` | `MLCacheRuntime`, `MLCacheRuntimeConfig`, `build_local_mlcache_runtime`, `build_mlcache_runtime` |
| `online/` | `OnlineStoppingController`, `WindowedOnlineStoppingController`, `OnlineUpdater` |
| `observability/` | `AuditLogger`, `MetricsSink`, `DiagnosticsReporter` |
| `builder.py` | `MLCache` facade + `build_mlcache()` / `build_scorer()` factory |
| `llm_wrapper.py` | `LLMClient` protocol, `MockLLM`, `CachedLLM`, `LLMJudge` |
| `system.py` | `SemanticCacheSystem` — the full self-running online cache |
| `api/` | FastAPI server (`serve_mlcache_gateway.py`) |

### Legacy `src/` modules

Files directly under `src/` (not inside `src/mlcache/`) — `audit.py`, `cache.py`, `embeddings.py`, `features.py`, `oracle.py`, `policy.py`, `runtime.py`, `scorers/`, etc. — are earlier prototypes. The canonical code lives in `src/mlcache/`.

### Scorer presets

`MLCache.from_preset(scorer=...)` accepts: `"cosine"`, `"lda"`, `"pca_whitened_cosine"`, `"xgboost"`, `"mlp"`, `"ensemble"`. Trainable scorers (`lda`, `xgboost`, `mlp`, `ensemble`) require `pip install -e '.[ml]'` and need `cache.prefit(LabeledPairBatch(...))` or `cache.prefit_and_calibrate(judged_pairs, target_fpr=...)` before serving. `cosine` works with base install only.

### H0 / H1 terminology

- **H0** = NOT_REUSABLE pairs (false-positive candidates) — used to calibrate threshold
- **H1** = REUSABLE pairs (true-positive candidates) — used to measure recall

The NPZ datasets in `data/` and `experiments/` follow this convention (`label` field: 1 = H1/reusable, 0 = H0/not-reusable).

### Test markers

`pytest.ini_options` excludes `integration` tests by default. Tests that need a real embedding model or network access are marked `@pytest.mark.integration`. Run them explicitly with `pytest -m integration`.
