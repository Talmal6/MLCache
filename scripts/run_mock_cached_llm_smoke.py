"""Smoke-test the cache-first mock LLM wrapper end to end.

Builds an `MLCache` via `MLCache.from_preset`, wraps a `MockLLM` in a
`CachedLLM`, and calls `generate` twice with the same prompt: the first call
should be served by the mock LLM (cache miss + write-through), and the second
should be served straight from the cache (semantic hit), with the cache
runtime — not the wrapper — making every matching/scoring/threshold decision.

No network access and no API key are required; the LLM and the embeddings are
both fully offline and deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache import (  # noqa: E402
    CachedLLM,
    CachedLLMResponse,
    HashingEmbeddingProvider,
    LabeledPairBatch,
    MLCache,
    MockLLM,
    SCORER_PRESET_NAMES,
)

EMBEDDING_DIMENSIONS = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain semantic caching.")
    parser.add_argument("--root-dir", default="runs/mock_cached_llm")
    parser.add_argument("--namespace", default="mock-llm")
    parser.add_argument("--scorer", default="ensemble", choices=SCORER_PRESET_NAMES)
    parser.add_argument("--scorers", default="cosine,lda,pca_whitened_cosine,xgboost,mlp")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-file-persistence", action="store_true")
    return parser.parse_args(argv)


def _synthetic_prefit_batch(*, dimensions: int, seed: int = 0, count: int = 32) -> LabeledPairBatch:
    """Build a small labeled pair-feature batch so trainable scorers can fit.

    The wrapper has no labeled traffic to learn from yet, so we hand the
    scorer(s) a synthetic batch of separable pair-feature vectors purely to
    satisfy "fit() must be called before score()" — this has no bearing on the
    cache's actual hit/miss decisions, which remain the oracle's job.
    """

    rng = np.random.default_rng(seed)
    h1 = rng.normal(loc=1.0, scale=0.25, size=(count, dimensions))
    h0 = rng.normal(loc=-1.0, scale=0.25, size=(count, dimensions))
    return LabeledPairBatch(
        h0=[tuple(float(v) for v in row) for row in h0],
        h1=[tuple(float(v) for v in row) for row in h1],
    )


def build_cache(args: argparse.Namespace) -> MLCache:
    scorers = [name.strip() for name in args.scorers.split(",") if name.strip()]
    cache = MLCache.from_preset(
        root_dir=args.root_dir,
        scorer=args.scorer,
        scorers=scorers if args.scorer == "ensemble" else None,
        threshold=args.threshold,
        top_k=args.top_k,
        persistence=not args.no_file_persistence,
    )
    cache.prefit(_synthetic_prefit_batch(dimensions=EMBEDDING_DIMENSIONS), alpha=0.05, seed=42)
    return cache


def _print_result(label: str, result: CachedLLMResponse) -> None:
    print(f"\n{label}:")
    print(f"  source:    {result.source}")
    print(f"  text:      {result.text}")
    print(f"  cache_key: {result.cache_key}")
    print(f"  score:     {result.score}")
    print(f"  threshold: {result.threshold}")
    print(f"  metadata:  {result.metadata}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache = build_cache(args)
    llm = MockLLM(response_template="mock response for: {prompt}")
    cached_llm = CachedLLM(
        llm=llm,
        cache=cache,
        namespace=args.namespace,
        embeddings=HashingEmbeddingProvider(dimensions=EMBEDDING_DIMENSIONS),
    )

    first = cached_llm.generate(args.prompt)
    second = cached_llm.generate(args.prompt)

    _print_result("First call (expected source=llm)", first)
    _print_result("Second call (expected source=cache)", second)

    return {"first_call": first, "second_call": second}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run(args)

    print(f"\nfirst_call.source={results['first_call'].source}")
    print(f"second_call.source={results['second_call'].source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
