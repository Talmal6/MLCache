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
  --scorers cosine,lda,pca_whitened_cosine,xgboost,mlp \
  --target-fpr 0.05
```

This builds the cache with `MLCache.from_preset`, adapts the dataset's
ground-truth H0/H1 rows into `JudgedPairExample`s, and hands them to
`cache.prefit_and_calibrate(judged_pairs, target_fpr=...)` — which owns the
whole cold-start sequence (fit any trainable scorer, score H0 pairs, and pick
a Neyman-Pearson threshold for the requested `--target-fpr` budget) — before
indexing the anchors and replaying the query stream. The script never scores
pairs or sets a threshold by hand; `--target-fpr` is the only calibration knob.
It writes `summary_metrics.json`, `per_request_decisions.csv`,
`runtime_config.json`, `schema_report.json`, and `calibration_report.json`
(the dict returned by `prefit_and_calibrate`) to the output directory.

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

## `SemanticCacheSystem`: a self-running online semantic cache

`MLCache` and `CachedLLM` are building blocks; experiments that compose them
directly tend to re-implement pieces of the calibration/training lifecycle by
hand (scoring pairs, calling `scorer.calibrate(...)`, setting thresholds).
`SemanticCacheSystem` is the alternative — it owns the *entire* lifecycle
end to end:

```python
from mlcache import MockLLM, SemanticCacheSystem

system = SemanticCacheSystem(
    llm=MockLLM(response_template="mock response for: {prompt}"),
    stream=prompts,            # any iterable of prompt strings
    scorer="ensemble",
    target_fpr=0.05,
)
report = system.run()
```

It serves requests (embed -> retrieve -> score -> HIT/MISS -> LLM fallback ->
write-through), and learns in the background (shadow-judges top-k candidates,
accumulates judged H0/H1 pairs, retrains and recalibrates atomically once
enough pairs accumulate, and activates the new scorer+threshold as a single
versioned unit). Until a calibrated policy exists the oracle abstains, so an
untrained scorer never serves a semantic HIT — every cold-start request is a
genuine LLM call (`source="llm"`), written through to the cache, while the
judge keeps labelling candidates for learning. Once calibrated, real cache
hits (`source="cache"`) start flowing.

`system.policy` returns an `ActivePolicy` — an immutable snapshot of the
scorer, threshold, and their version numbers, plus `calibrated`, `trained`,
and `frozen` flags — built from the oracle's atomically-swapped state so you
never observe a scorer from one generation paired with a threshold from
another. `system.handle(prompt)` returns a `SystemResponse` (`text`, `source`,
`cache_key`, `score`, `threshold`, and the `policy` that served it).
`system.report()` summarizes counts, hit rate, and policy state for the whole
run.

`freeze(reason=...)` stops further training/calibration while serving
continues on the frozen policy — and the system can also freeze itself
automatically once its convergence-stopping controller decides online metrics
have stabilized (`freeze_reason="online_metrics_converged"` in the report).

### Diagnosing the online lifecycle

`scripts/diagnose_semantic_cache_lifecycle.py` replays a deterministic,
fully-offline prompt stream through `SemanticCacheSystem` and prints/exports
batch-by-batch evidence of the lifecycle described above — in particular, that

```text
top-k retrieved candidates
  -> judged in shadow mode (every candidate, not just the served one)
  -> stored as H0/H1 pairs in the judge training store
  -> used for training/calibration
  -> not limited to the served/accepted candidate
```

Each logged batch breaks the judge training store down into pairs for the
*served* candidate vs. pairs for *retrieved-but-rejected* candidates (the
latter identified via MISS decisions, where no candidate was served — so every
pair stored under a MISS is, by construction, a rejected one), so you can
watch the rejected share grow alongside the served share as traffic replays:

```bash
python scripts/diagnose_semantic_cache_lifecycle.py \
  --scorer cosine \
  --top-k 5 \
  --batch-size 20 \
  --target-fpr 0.25 \
  --requests 300 \
  --output-dir runs/diagnose_lifecycle_cosine

python scripts/diagnose_semantic_cache_lifecycle.py \
  --scorer ensemble \
  --scorers cosine,lda \
  --top-k 5 \
  --batch-size 20 \
  --target-fpr 0.25 \
  --requests 500 \
  --output-dir runs/diagnose_lifecycle_ensemble
```

The ensemble command above uses the stable two-member default (`cosine,lda`).  To
exercise the full five-member stack, pass `--scorers cosine,lda,pca_whitened_cosine,xgboost,mlp`
and install all optional dependencies (`pip install -e '.[all]'`); the script
waits up to `--fit-wait-secs` (default 10 s) for the first background fit to
finish and then replays a short warm-up burst so the cache can demonstrate hits
with the now-active policy.  Both runs exit 0 only if the lifecycle completes
(`calibrated=True`, finite threshold, hits > 0).

It writes a full timeline (per-batch report + store breakdown) to
`<output-dir>/lifecycle_report.json`. `runs/` is gitignored — these are local
diagnostic artifacts, not committed experiment results.

### Invariant test: shadow judging isn't limited to served candidates

`tests/test_semantic_cache_system.py::JudgedPairAccumulationTests::test_shadow_pairs_from_rejected_candidates_reach_the_training_store_and_drive_calibration`
protects the property the diagnostic script makes visible: shadow top-k
judging labels *every* retrieved candidate — including ones the active policy
rejected — and those judged pairs land in the very same store that feeds
training/calibration (`_refit_rows_from_training_store` reads exactly those
buckets). It asserts that judged pairs from MISS decisions (where no candidate
was served, so every stored pair is necessarily a retrieved-but-rejected
candidate) are present with both H0 and H1 labels, that more pairs are stored
than the system ever served from cache, and that the system still reaches a
trained, calibrated policy — proof that online learning draws on
retrieved-but-rejected candidates, not only the served/accepted one.

### `LLMJudge`: turning an `LLMClient` into a `SemanticReuseJudge`

`SemanticCacheSystem` needs a `SemanticReuseJudge` to label query/candidate
pairs as reusable or not. If you don't supply one, it synthesizes `LLMJudge`
from the same `LLMClient` you passed in — prompting it to reply
`REUSABLE`/`NOT_REUSABLE` for a given pair and parsing the reply into a
`JudgeDecision` (anything else, e.g. `MockLLM`'s echoed text, becomes
`UNCERTAIN`). `LLMClient` and `SemanticReuseJudge` stay distinct interfaces —
one generates responses, the other labels pairs — even when the same model
backs both:

```python
from mlcache import LLMJudge, MockLLM

judge = LLMJudge(MockLLM(response_template="..."), name="my-judge")
```

### `MLCache.prefit_and_calibrate`: cold-start calibration in one call

When you already have a labeled offline dataset (e.g. an H1/H0 NPZ split) and
want to cold-start a cache without a live judge, adapt the dataset's rows into
`JudgedPairExample`s and hand them to `cache.prefit_and_calibrate(...)`. It
owns the whole "fit trainable scorers on H0/H1 pairs -> score H0 -> pick a
Neyman-Pearson threshold for the target false-accept rate -> activate" sequence,
so the caller never scores pairs or calls `scorer.calibrate(...)` directly:

```python
report = cache.prefit_and_calibrate(judged_pairs, target_fpr=0.05)
# {"scorer": ..., "threshold": ..., "target_fpr": ..., "n_h0": ..., "n_h1": ...}
```

See `scripts/run_cold_start_tpr_fpr_experiment.py` for a full example that
adapts `H1H0NPZRecord`s into `JudgedPairExample`s and compares two systems'
cold-start TPR/FPR after calibrating both to the same FPR budget this way.

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