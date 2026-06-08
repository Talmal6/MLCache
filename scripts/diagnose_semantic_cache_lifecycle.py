"""Diagnose the `SemanticCacheSystem` online lifecycle end to end, with logs and artifacts.

Builds a `SemanticCacheSystem` over a deterministic, offline judge/embedding
pair (paraphrases of the same `#topic` cluster score as REUSABLE, cross-topic
pairs as NOT_REUSABLE) and replays a synthetic prompt stream through it,
printing the lifecycle's progression batch by batch:

  cold start (every request MISS, judge keeps labelling in the shadows)
  -> judged H0/H1 pairs accumulate in the training store
  -> trainable scorer fits, threshold calibrates, policy activates atomically
  -> semantic HITs start being served on the active policy
  -> (optionally) the stopping controller converges and freezes the policy

The diagnostic of particular interest -- the one this script makes visible --
is that shadow top-k judging labels *every* retrieved candidate, including
ones the active policy rejected (MISS decisions, where no candidate was
served), and that those judged pairs land in the very same store that feeds
training/calibration. Each batch's log line breaks the training store down
into "served-candidate" vs "rejected-candidate" pairs so you can watch the
rejected share grow alongside the served share.

No network access or API key is required -- the LLM, embeddings, and judge are
all fully offline and deterministic (seeded via `hashlib.sha256`, never
Python's randomized `hash()`), so a given `--seed`/topology reaches the same
calibrated, finite-threshold state on every machine and process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache import (  # noqa: E402
    JudgeDecision,
    JudgeLabel,
    JudgeRequest,
    JudgeResult,
    MockLLM,
    SemanticCacheSystem,
    SemanticReuseJudge,
)
from mlcache.embeddings import EmbeddingProvider  # noqa: E402
from mlcache.persistence import json_safe  # noqa: E402

_TOPIC_PATTERN = re.compile(r"#(\d+)")

_TEMPLATES: Sequence[str] = (
    "What is the capital of country #{n}?",
    "Tell me the capital city of nation #{n}.",
    "Name the capital of country #{n}, please.",
    "Could you say which city is the capital of country #{n}?",
)


def _stable_seed(text: str) -> int:
    """Process- and machine-independent seed (Python's `hash()` is randomized per-process)."""

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _topic_prompts(count: int, *, topics: int) -> list[str]:
    return [_TEMPLATES[i % len(_TEMPLATES)].format(n=i % topics) for i in range(count)]


class TopicEmbeddingProvider(EmbeddingProvider):
    """Deterministic stub embedding: paraphrases of the same `#topic` cluster tightly."""

    def __init__(self, *, dimensions: int = 32, jitter_scale: float = 0.05) -> None:
        self._dimensions = dimensions
        self._jitter_scale = jitter_scale

    def _topic_centroid(self, topic: int) -> np.ndarray:
        vector = np.random.default_rng(1000 + topic).normal(size=self._dimensions)
        return vector / np.linalg.norm(vector)

    def embed(self, text: str) -> tuple[float, ...]:
        match = _TOPIC_PATTERN.search(text)
        topic = int(match.group(1)) if match else 0
        centroid = self._topic_centroid(topic)
        jitter = np.random.default_rng(_stable_seed(text)).normal(scale=self._jitter_scale, size=self._dimensions)
        vector = centroid + jitter
        vector = vector / np.linalg.norm(vector)
        return tuple(float(value) for value in vector)


class DeterministicTopicJudge(SemanticReuseJudge):
    """Labels a pair REUSABLE iff the query and candidate share the same `#topic`."""

    def __init__(self, *, name: str = "deterministic-topic-judge") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def judge(self, request: JudgeRequest) -> JudgeResult:
        query_topic = self._topic(str(request.query))
        candidate_topic = self._topic(str(request.candidate_query) if request.candidate_query is not None else "")
        if query_topic is None or candidate_topic is None:
            label = JudgeLabel.NOT_REUSABLE
        else:
            label = JudgeLabel.REUSABLE if query_topic == candidate_topic else JudgeLabel.NOT_REUSABLE
        return JudgeResult(request=request, decision=JudgeDecision(label=label, metadata={"judge": self._name}))

    @staticmethod
    def _topic(text: str) -> str | None:
        match = _TOPIC_PATTERN.search(text)
        return match.group(1) if match else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=360)
    parser.add_argument("--topics", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-h0", type=int, default=15)
    parser.add_argument("--min-h1", type=int, default=8)
    parser.add_argument("--target-fpr", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--scorer", default="ensemble")
    parser.add_argument("--scorers", default="cosine,lda")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "diagnose_lifecycle"))
    return parser.parse_args(argv)


def build_system(args: argparse.Namespace, *, root_dir: str) -> SemanticCacheSystem:
    scorers = [name.strip() for name in args.scorers.split(",") if name.strip()]
    return SemanticCacheSystem(
        llm=MockLLM(response_template="answer for: {prompt}"),
        stream=None,
        scorer=args.scorer,
        scorers=scorers if args.scorer == "ensemble" else None,
        judge=DeterministicTopicJudge(),
        embedding_provider=TopicEmbeddingProvider(),
        target_fpr=args.target_fpr,
        top_k=args.top_k,
        root_dir=root_dir,
        batch_size=args.batch_size,
        min_h0=args.min_h0,
        min_h1=args.min_h1,
        persistence=False,
        parallelism=args.parallelism,
    )


def _store_breakdown(system: SemanticCacheSystem) -> dict[str, Any]:
    """Split the judge training store into served-candidate vs rejected-candidate pairs.

    A judged pair's `served_decision_accepted` metadata records whether *some*
    candidate was served for that request -- not whether *this* candidate was
    the one served. But on a MISS (`served_decision_accepted=False`), no
    candidate was served at all, so every pair stored under a MISS is, by
    construction, a retrieved-but-rejected candidate. That's the cleanest
    available signal that shadow judging covers candidates beyond whichever
    single one (if any) was served.
    """

    store = system.cache.runtime.judge_training_store
    if store is None:
        return {"available": False}

    examples = store.h0_train() + store.h0_calibration() + store.h1_train() + store.h1_calibration()
    rejected = [example for example in examples if example.metadata.get("served_decision_accepted") is False]
    served = [example for example in examples if example.metadata.get("served_decision_accepted") is True]

    def _label_counts(rows: Sequence[Any]) -> dict[str, int]:
        return {
            "h0": sum(1 for example in rows if example.decision.label == JudgeLabel.NOT_REUSABLE),
            "h1": sum(1 for example in rows if example.decision.label == JudgeLabel.REUSABLE),
        }

    return {
        "available": True,
        "total_pairs": len(examples),
        "train_h0": len(store.h0_train()),
        "train_h1": len(store.h1_train()),
        "calibration_h0": len(store.h0_calibration()),
        "calibration_h1": len(store.h1_calibration()),
        "rejected_candidate_pairs": len(rejected),
        "rejected_candidate_labels": _label_counts(rejected),
        "served_candidate_pairs": len(served),
        "served_candidate_labels": _label_counts(served),
    }


def _log_batch(batch_index: int, *, report: dict[str, Any], breakdown: dict[str, Any]) -> None:
    print(
        f"[batch {batch_index:>3}] requests={report['requests']:<4} "
        f"hits={report['cache_hits']:<4} hit_rate={report['hit_rate']:.2f}  "
        f"trained={str(report['trained']):<5} calibrated={str(report['calibrated']):<5} "
        f"threshold={report['threshold']!r} (v{report['threshold_version']})  "
        f"frozen={report['frozen']}"
    )
    if breakdown["available"]:
        print(
            f"             store: total={breakdown['total_pairs']:<4} "
            f"served_candidate={breakdown['served_candidate_pairs']:<4}{breakdown['served_candidate_labels']}  "
            f"rejected_candidate={breakdown['rejected_candidate_pairs']:<4}{breakdown['rejected_candidate_labels']}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _topic_prompts(args.requests, topics=args.topics)

    timeline: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="diagnose-lifecycle-") as tmp:
        system = build_system(args, root_dir=tmp)

        print(f"Replaying {len(prompts)} requests across {args.topics} topics "
              f"(batch_size={args.batch_size}, top_k={args.top_k}, target_fpr={args.target_fpr})\n")

        first_hit_at: int | None = None
        first_calibrated_at: int | None = None
        for index, prompt in enumerate(prompts, start=1):
            response = system.handle(prompt)
            if first_hit_at is None and response.source == "cache":
                first_hit_at = index
            if first_calibrated_at is None and system.policy.calibrated:
                first_calibrated_at = index

            if index % args.batch_size == 0 or index == len(prompts):
                system._maybe_check_stopping()
                report = system.report()
                breakdown = _store_breakdown(system)
                _log_batch(index, report=report, breakdown=breakdown)
                timeline.append({"request_index": index, "report": report, "store": breakdown})

        final_report = system.report()
        final_breakdown = _store_breakdown(system)

    print()
    print(f"first_cache_hit_at_request   = {first_hit_at}")
    print(f"first_calibrated_at_request  = {first_calibrated_at}")
    print(f"final report                 = {final_report}")
    if final_breakdown["available"]:
        print(
            f"final store breakdown        = total={final_breakdown['total_pairs']}, "
            f"served_candidate={final_breakdown['served_candidate_pairs']} "
            f"({final_breakdown['served_candidate_labels']}), "
            f"rejected_candidate={final_breakdown['rejected_candidate_pairs']} "
            f"({final_breakdown['rejected_candidate_labels']})"
        )
        if final_breakdown["rejected_candidate_pairs"] > 0:
            print(
                "-> shadow judging stored pairs for retrieved-but-rejected candidates "
                "(MISS decisions with no served candidate) -- online learning is not "
                "limited to served/accepted pairs."
            )

    summary = {
        "args": vars(args),
        "first_cache_hit_at_request": first_hit_at,
        "first_calibrated_at_request": first_calibrated_at,
        "final_report": final_report,
        "final_store_breakdown": final_breakdown,
        "timeline": timeline,
    }

    artifact_path = output_dir / "lifecycle_report.json"
    artifact_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"\nWrote lifecycle report to {artifact_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
