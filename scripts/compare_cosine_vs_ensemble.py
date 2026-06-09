"""Deterministic offline cosine-vs-ensemble comparison for SemanticCacheSystem.

Runs the same synthetic prompt stream through two independent systems —
scorer="cosine" and scorer="ensemble" — and writes a comparison report.

The *full* ensemble is:  cosine,lda,pca_whitened_cosine,xgboost,mlp
A debug/smoke subset is: cosine,lda  (faster, for development only)

No network or API keys required.  All randomness uses hashlib.sha256.

Full comparison (primary):
  python scripts/compare_cosine_vs_ensemble.py \\
    --requests 1000 --topics 8 --top-k 5 --batch-size 20 \\
    --target-fpr 0.10 \\
    --ensemble-scorers cosine,lda,pca_whitened_cosine,xgboost,mlp \\
    --fit-wait-secs 60 \\
    --output-dir runs/compare_cosine_vs_ensemble_full_1000

Debug/smoke (not the real comparison):
  python scripts/compare_cosine_vs_ensemble.py \\
    --requests 300 --topics 6 --target-fpr 0.25 \\
    --ensemble-scorers cosine,lda \\
    --output-dir runs/compare_cosine_vs_ensemble_debug
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache import (  # noqa: E402
    JudgeDecision, JudgeLabel, JudgeRequest, JudgeResult,
    MockLLM, SemanticCacheSystem, SemanticReuseJudge,
)
from mlcache.embeddings import EmbeddingProvider  # noqa: E402
from mlcache.persistence import json_safe  # noqa: E402
from data.dataset_extractor import WildChatDatasetExtractor  # noqa: E402

# ── ensemble presets ──────────────────────────────────────────────────────

FULL_ENSEMBLE = ["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"]
DEBUG_ENSEMBLE = ["cosine", "lda"]  # fast subset; does not prove final policy quality

# ── synthetic data ─────────────────────────────────────────────────────────

_PAT = re.compile(r"#(\d+)")
_TEMPLATES: Sequence[str] = (
    "What is the capital of country #{n}?",
    "Tell me the capital city of nation #{n}.",
    "Name the capital of country #{n}, please.",
    "Could you say which city is the capital of country #{n}?",
)


dataset_extractor = WildChatDatasetExtractor()  

def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**32)


def topic_of(text: str) -> str | None:
    m = _PAT.search(text)
    return m.group(1) if m else None

#def label_of(text: str) -> str | None:



def topic_prompts(n: int, *, topics: int) -> list[str]:
    # Inner loop = topic, outer loop = template so each topic gets multiple templates
    # before repeating.  (i%4,i%8) alignment produces H1=0 when topics%len(_TEMPLATES)==0.)
    return [
        _TEMPLATES[(i // topics) % len(_TEMPLATES)].format(n=i % topics)
        for i in range(n)
    ]







class DatasetJudge(SemanticReuseJudge):
    """REUSABLE iff query topic == candidate topic."""

    @property
    def name(self) -> str:
        return "topic-judge"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        request_label = dataset_extractor.get_label(request.query)
        candidate_label = dataset_extractor.get_label(request.candidate_query)
        request_cluster_id = dataset_extractor.get_cluster_id(request.query)
        candidate_cluster_id = dataset_extractor.get_cluster_id(request.candidate_query)

        if request_label == JudgeLabel.REUSEABLE and candidate_label == JudgeLabel.REUSABLE and request_cluster_id == candidate_cluster_id:
            return JudgeResult(JudgeDecision.REUSABLE, reason="topics match and are reusable")
        else:
            return JudgeResult(JudgeDecision.NOT_REUSABLE, reason="topics do not match or are not reusable")


class DatasetEmbeddingProvider(EmbeddingProvider):
    """Provides the single embedding from the dataset for a query, ignoring candidate_query."""

    def get_embeddings(self, query: str) -> list[np.ndarray]:
        return dataset_extractor.get_embeddings(query)

# ── run one system ─────────────────────────────────────────────────────────

def run_one(
    *,
    scorer: str,
    scorers: list[str],
    prompts: list[str],
    topics: int,
    target_fpr: float,
    top_k: int,
    batch_size: int,
    min_h0: int,
    min_h1: int,
    root_dir: str,
    fit_wait_secs: float,
    dim: int,
    jitter: float,
) -> dict[str, Any]:
    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer: {prompt}"),
        scorer=scorer,
        scorers=scorers if scorer == "ensemble" else None,
        judge=DatasetJudge(),
        embedding_provider=DatasetEmbeddingProvider(),
        target_fpr=target_fpr,
        top_k=top_k,
        root_dir=root_dir,
        batch_size=batch_size,
        min_h0=min_h0,
        min_h1=min_h1,
        persistence=False,
        parallelism=4,
    )

    trace: list[dict[str, Any]] = []
    key_topic: dict[str, str] = {}
    first_hit: int | None = None

    for i, prompt in enumerate(prompts, 1):
        pt = topic_of(prompt)
        resp = system.handle(prompt)
        pol = resp.policy

        if resp.source == "llm" and resp.cache_key and pt is not None:
            key_topic[resp.cache_key] = pt
        if first_hit is None and resp.source == "cache":
            first_hit = i

        trace.append({
            "index": i, "topic": pt, "source": resp.source,
            "score": resp.score, "threshold": resp.threshold,
            "calibrated": pol.calibrated, "trained": pol.trained,
        })
        if i % batch_size == 0 or i == len(prompts):
            system._maybe_check_stopping()

    # wait for background fit
    t0 = time.monotonic()
    while time.monotonic() - t0 < fit_wait_secs and not system.policy.calibrated:
        time.sleep(0.2)

    # warm-up burst if calibration arrived after the stream
    if system.policy.calibrated and first_hit is None:
        for j, p in enumerate(topic_prompts(30 * topics, topics=topics), len(prompts) + 1):
            resp = system.handle(p)
            if first_hit is None and resp.source == "cache":
                first_hit = j
            pt2 = topic_of(p)
            if resp.source == "llm" and resp.cache_key and pt2 is not None:
                key_topic[resp.cache_key] = pt2
            trace.append({
                "index": j, "topic": pt2, "source": resp.source,
                "score": resp.score, "threshold": resp.threshold,
                "calibrated": resp.policy.calibrated, "trained": resp.policy.trained,
            })
        system._maybe_check_stopping()

    rep = system.report()
    pol = system.policy
    return {
        "n_requests": rep["requests"],
        "cache_hits": rep["cache_hits"],
        "llm_calls": rep["llm_calls"],
        "hit_rate": rep["hit_rate"],
        "trained": pol.trained,
        "calibrated": pol.calibrated,
        "threshold": pol.threshold,
        "finite_threshold": pol.threshold is not None and math.isfinite(pol.threshold),
        "scorer_version": pol.scorer_version,
        "threshold_version": pol.threshold_version,
        "first_hit_at": first_hit,
        "_trace": trace,
    }


# ── winner logic ───────────────────────────────────────────────────────────

def determine_winner(cosine: dict, ensemble: dict) -> dict[str, str]:
    """Honest comparison — does not assume ensemble wins."""
    cc, ec = cosine["calibrated"], ensemble["calibrated"]
    ch, eh = cosine["hit_rate"], ensemble["hit_rate"]
    if not cc and not ec:
        return {"by_hit_rate": "undetermined", "reason": "neither system calibrated"}
    if cc and not ec:
        return {"by_hit_rate": "cosine", "reason": "cosine calibrated; ensemble did not"}
    if ec and not cc:
        return {"by_hit_rate": "ensemble", "reason": "ensemble calibrated; cosine did not"}
    if abs(ch - eh) < 0.005:
        return {"by_hit_rate": "tie",
                "reason": f"both calibrated; hit_rate within 0.5% (cos={ch:.3f} ens={eh:.3f})"}
    w = "cosine" if ch > eh else "ensemble"
    return {"by_hit_rate": w,
            "reason": f"both calibrated; higher hit_rate wins (cos={ch:.3f} ens={eh:.3f})"}


def build_delta(c: dict, e: dict) -> dict[str, Any]:
    def fd(k: str) -> float | None:
        a, b = c.get(k), e.get(k)
        return None if (a is None or b is None) else round(float(b) - float(a), 6)
    def id_(k: str) -> int | None:
        a, b = c.get(k), e.get(k)
        return None if (a is None or b is None) else int(b) - int(a)
    return {"hit_rate": fd("hit_rate"), "llm_calls": id_("llm_calls"),
            "threshold": fd("threshold")}


def build_validity(cosine: dict, ensemble: dict) -> tuple[bool, str]:
    """comparison_valid=True only when both systems fully calibrated and ensemble trained."""
    gates = {
        "cosine.calibrated": cosine.get("calibrated"),
        "cosine.finite_threshold": cosine.get("finite_threshold"),
        "ensemble.calibrated": ensemble.get("calibrated"),
        "ensemble.finite_threshold": ensemble.get("finite_threshold"),
        "ensemble.trained": ensemble.get("trained"),
    }
    failed = [k for k, v in gates.items() if not v]
    if not failed:
        return True, "all validity gates passed"
    return False, (
        f"gates not passed: {', '.join(failed)}. "
        "Try --requests 1000+ or a looser --target-fpr (e.g. 0.25)."
    )


# ── output ─────────────────────────────────────────────────────────────────

def write_trace(path: Path, trace: list[dict]) -> None:
    fields = ["index", "topic", "source", "score", "threshold", "calibrated", "trained"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(trace)


def write_report(path: Path, *, config: dict, cosine: dict, ensemble: dict,
                 delta: dict, winner: dict,
                 comparison_valid: bool, comparison_valid_reason: str) -> None:
    def fv(v: Any, pct: bool = False) -> str:
        if v is None: return "N/A"
        if isinstance(v, bool): return str(v)
        if isinstance(v, float): return f"{v*100:.2f}%" if pct else f"{v:.4f}"
        return str(v)

    scorers = config.get("ensemble_scorers", [])
    is_full = scorers == FULL_ENSEMBLE
    ensemble_label = "full (5-member)" if is_full else f"debug ({len(scorers)}-member)"
    rows = [
        ("hit_rate", True), ("llm_calls", False), ("threshold", False),
        ("trained", False), ("calibrated", False), ("finite_threshold", False),
    ]
    valid_badge = "VALID" if comparison_valid else "INVALID"
    lines = [
        "# Cosine vs Ensemble Comparison\n",
        f"> **Ensemble mode:** {ensemble_label} — members: `{', '.join(scorers)}`\n",
        f"> **Comparison valid:** {valid_badge} — {comparison_valid_reason}\n",
        "## Config\n",
        "| key | value |\n|---|---:|",
        *[f"| {k} | {v} |" for k, v in config.items()],
        "\n## Results\n",
        "| metric | cosine | ensemble | delta |\n|---|---:|---:|---:|",
        *[f"| {k} | {fv(cosine.get(k),p)} | {fv(ensemble.get(k),p)} | {fv(delta.get(k),p)} |"
          for k, p in rows],
        f"\n**Winner:** `{winner['by_hit_rate']}`  \n**Reason:** {winner['reason']}\n",
        "\n## Validity\n",
        "A comparison is valid only when both systems calibrated, both have finite thresholds,",
        "and the ensemble scorer was actually trained (not just a cold fallback).\n",
        f"- comparison_valid: {comparison_valid}",
        f"- reason: {comparison_valid_reason}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── orchestrator ───────────────────────────────────────────────────────────

def run_comparison(
    *,
    n_requests: int = 1000,
    topics: int = 8,
    top_k: int = 5,
    batch_size: int = 20,
    min_h0: int = 15,
    min_h1: int = 8,
    target_fpr: float = 0.10,
    ensemble_scorers: list[str] | None = None,
    fit_wait_secs: float = 60.0,
    output_dir: str | Path = "runs/compare_cosine_vs_ensemble",
    dim: int = 32,
    jitter: float = 0.05,
) -> dict[str, Any]:
    if ensemble_scorers is None:
        ensemble_scorers = FULL_ENSEMBLE
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompts = topic_prompts(n_requests, topics=topics)
    config = dict(n_requests=n_requests, topics=topics, top_k=top_k,
                  batch_size=batch_size, min_h0=min_h0, min_h1=min_h1,
                  target_fpr=target_fpr, ensemble_scorers=ensemble_scorers,
                  fit_wait_secs=fit_wait_secs)

    common = dict(prompts=prompts, topics=topics, target_fpr=target_fpr,
                  top_k=top_k, batch_size=batch_size, min_h0=min_h0,
                  min_h1=min_h1, fit_wait_secs=fit_wait_secs, dim=dim, jitter=jitter)

    print(f"[cosine]   {n_requests} requests, topics={topics}")
    with tempfile.TemporaryDirectory(prefix="cmp-cos-") as d:
        cosine = run_one(scorer="cosine", scorers=[], root_dir=d, **common)

    print(f"[ensemble] {n_requests} requests, scorers={ensemble_scorers}")
    with tempfile.TemporaryDirectory(prefix="cmp-ens-") as d:
        ensemble = run_one(scorer="ensemble", scorers=ensemble_scorers, root_dir=d, **common)

    winner = determine_winner(cosine, ensemble)
    delta = build_delta(cosine, ensemble)
    valid, valid_reason = build_validity(cosine, ensemble)

    pub_c = {k: v for k, v in cosine.items() if not k.startswith("_")}
    pub_e = {k: v for k, v in ensemble.items() if not k.startswith("_")}
    summary = {
        "config": config,
        "cosine": pub_c,
        "ensemble": pub_e,
        "delta": delta,
        "winner": winner,
        "comparison_valid": valid,
        "comparison_valid_reason": valid_reason,
    }

    write_trace(out / "cosine_trace.csv", cosine["_trace"])
    write_trace(out / "ensemble_trace.csv", ensemble["_trace"])
    (out / "comparison_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    write_report(out / "comparison_report.md",
                 config=config, cosine=pub_c, ensemble=pub_e,
                 delta=delta, winner=winner,
                 comparison_valid=valid, comparison_valid_reason=valid_reason)

    print(f"\nhit_rate:  cosine={cosine['hit_rate']:.3f}  ensemble={ensemble['hit_rate']:.3f}")
    print(f"calibrated: cosine={cosine['calibrated']}  ensemble={ensemble['calibrated']}")
    print(f"valid: {valid} ({valid_reason})")
    print(f"winner: {winner['by_hit_rate']}  ({winner['reason']})")
    print(f"artifacts -> {out.resolve()}")
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--topics", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--min-h0", type=int, default=15)
    p.add_argument("--min-h1", type=int, default=8)
    p.add_argument("--target-fpr", type=float, default=0.10)
    p.add_argument("--ensemble-scorers", default=",".join(FULL_ENSEMBLE))
    p.add_argument("--fit-wait-secs", type=float, default=60.0)
    p.add_argument("--output-dir", default="runs/compare_cosine_vs_ensemble")
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--jitter-scale", type=float, default=0.05)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    run_comparison(
        n_requests=a.requests, topics=a.topics, top_k=a.top_k,
        batch_size=a.batch_size, min_h0=a.min_h0, min_h1=a.min_h1,
        target_fpr=a.target_fpr,
        ensemble_scorers=[s.strip() for s in a.ensemble_scorers.split(",") if s.strip()],
        fit_wait_secs=a.fit_wait_secs, output_dir=a.output_dir,
        dim=a.embedding_dim, jitter=a.jitter_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
