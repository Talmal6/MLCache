"""Smoke CLI for exercising a local MLCache runtime with real embeddings."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from mlcache.calibration import ThresholdScope
from mlcache.embeddings import SentenceTransformersEmbeddingProvider
from mlcache.features import NormalizedHadamardFeatureBuilder
from mlcache.runtime import build_local_mlcache_runtime
from mlcache.scorers import CosineScorer
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, CacheMetadata, Query, Response, Threshold


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHED_QUERY = "What is semantic caching?"
DEFAULT_INCOMING_QUERY = "Explain semantic caching."
DEFAULT_RESPONSE = "Semantic caching reuses a previous response when a new query is semantically close enough."
DEFAULT_THRESHOLD = 0.75
DEFAULT_CACHE_KEY = "demo-cache-key"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "smoke":
        tokens = tokens[1:]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--cached-query", default=DEFAULT_CACHED_QUERY)
    parser.add_argument("--incoming-query", default=DEFAULT_INCOMING_QUERY)
    parser.add_argument("--persist", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(tokens)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    provider = SentenceTransformersEmbeddingProvider(
        model_name=args.model,
        normalize=True,
        local_files_only=bool(args.local_files_only),
    )

    if args.persist is None:
        with tempfile.TemporaryDirectory(prefix="mlcache-smoke-") as temp_dir:
            return _run_smoke_with_root(args=args, provider=provider, root_dir=Path(temp_dir))

    root_dir = Path(args.persist)
    root_dir.mkdir(parents=True, exist_ok=True)
    return _run_smoke_with_root(args=args, provider=provider, root_dir=root_dir)


def _run_smoke_with_root(
    *,
    args: argparse.Namespace,
    provider: SentenceTransformersEmbeddingProvider,
    root_dir: Path,
) -> dict[str, Any]:
    runtime = build_local_mlcache_runtime(
        root_dir=root_dir,
        feature_builder=NormalizedHadamardFeatureBuilder(),
        scorer=CosineScorer(),
    )

    cached_embedding = provider.embed(args.cached_query)
    incoming_embedding = provider.embed(args.incoming_query)

    runtime.oracle.threshold_provider.set_threshold(
        Threshold(float(args.threshold)),
        scorer=runtime.oracle.scorer.name,
        scope=ThresholdScope.GLOBAL,
        context={"source": "mlcache_smoke"},
    )

    runtime.put(
        CacheEntry(
            cache_key=CacheKey(DEFAULT_CACHE_KEY),
            query=Query(args.cached_query),
            response=Response(DEFAULT_RESPONSE),
            embedding=cached_embedding,
            metadata=CacheMetadata(model=args.model),
        )
    )

    result = runtime.lookup_with_decision(
        CacheLookup(
            query=Query(args.incoming_query),
            embedding=incoming_embedding,
            metadata=CacheMetadata(model=args.model),
        )
    )

    return {
        "response": None if result.response is None else str(result.response),
        "status": result.decision.status.value,
        "accepted": bool(result.decision.accepted),
        "score": None if result.decision.score is None else float(result.decision.score),
        "threshold": None if result.decision.threshold is None else float(result.decision.threshold),
        "cache_key": None if result.decision.cache_key is None else str(result.decision.cache_key),
        "reason": result.decision.reason,
        "cached_query": args.cached_query,
        "incoming_query": args.incoming_query,
        "embedding_model": args.model,
        "embedding_dim": len(cached_embedding),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output = run_smoke(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print(f"mlcache-smoke failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(output, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())