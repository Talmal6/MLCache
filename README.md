# semantic-desider

Correctness-controlled online semantic cache contracts and runtime boundaries.

## Constructing a cache

`MLCache.from_preset` is the normal way to build a working cache: it wires the
feature builder, scorer(s), threshold, and persistence in one call instead of
composing `build_local_mlcache_runtime` by hand.

```python
from mlcache import MLCache

cache = MLCache.from_preset(
    root_dir="cache_state",
    scorer="ensemble",
    scorers=["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"],
    threshold=0.05,
)
```

`scorer` accepts any of `"cosine"`, `"lda"`, `"pca_whitened_cosine"`,
`"xgboost"`, `"mlp"`, or `"ensemble"` (which combines the `scorers` list into a
weighted `EnsembleScorer`). A factory-function form is also available:
`build_mlcache(...)` takes the same arguments and returns the same `MLCache`.

Trainable scorers (`lda`, `xgboost`, `mlp`, and `ensemble`) need
`cache.prefit(LabeledPairBatch(h0=..., h1=...))` before serving lookups; if the
optional ML dependencies aren't installed, `prefit` raises a clear
`ImportError` telling you to run `pip install -e '.[ml,dev]'`.

## Replaying an H1/H0 dataset

```bash
make setup-all

python scripts/run_cache.py \
  --npz path/to/h1h0_final.npz \
  --output-dir experiments/h1h0_ensemble \
  --label-field label \
  --query-field text \
  --anchor-field global_cluster \
  --query-embedding-field emb \
  --scorer ensemble \
  --scorers cosine,lda,pca_whitened_cosine,xgboost,mlp
```

This builds the cache with `MLCache.from_preset`, prefits trainable scorers on
the dataset's H0/H1 examples, indexes the anchors, replays the query stream,
and writes `summary_metrics.json`, `per_request_decisions.csv`,
`runtime_config.json`, and `schema_report.json` to the output directory.

## Mock LLM wrapper

`CachedLLM` wraps any `LLMClient` (here, the offline `MockLLM`) with a
cache-first orchestration layer: check the cache, fall back to the LLM on a
miss, and write the LLM's answer back. It does **not** implement semantic
matching itself — every lookup, vector search, scoring, thresholding, and
hit/miss decision stays the cache runtime's responsibility. The wrapper only
decides whether to consult the cache, when to call the LLM, and whether to
persist the result.

```python
from mlcache import MLCache, CachedLLM, MockLLM

cache = MLCache.from_preset(
    root_dir="cache_state",
    scorer="ensemble",
    scorers=["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"],
    threshold=0.05,
    top_k=5,
)
llm = MockLLM(response_template="mock response for: {prompt}")
cached_llm = CachedLLM(llm=llm, cache=cache, namespace="mock-llm")

response = cached_llm.generate("Explain Byzantine broadcast in one paragraph.")
print(response.source)  # "llm" first time, "cache" second time
print(response.text)
```

`CachedLLM` converts prompts to vectors with a deterministic, offline
`HashingEmbeddingProvider` (no model download, no network access), so the
whole flow runs without any API key. Try it end to end with:

```bash
python scripts/run_mock_cached_llm_smoke.py \
  --prompt "Explain semantic caching." \
  --root-dir runs/mock_cached_llm \
  --scorer ensemble \
  --scorers cosine,lda,pca_whitened_cosine,xgboost,mlp
```

It builds the cache, calls `generate` twice with the same prompt, and prints
both results — the first served by the mock LLM (`source="llm"`, written into
the cache) and the second served straight from the cache (`source="cache"`).

Once the mock LLM flow is stable, we can add an OpenAI-compatible provider in
a separate step.

## Quickstart with real embeddings

```bash
python3.11 -m venv .venv
source .venv/bin/activate

make setup-embeddings
make smoke
```

Direct pip usage:

```bash
python -m pip install -e ".[embeddings,dev]"
mlcache-smoke
```

The first run may download the embedding model. To force local cached model usage:

```bash
mlcache-smoke --local-files-only
```

If the model is not cached and `--local-files-only` is used, the command fails with a clear error.