"""Integration test: StrictLLMJudge → OpenAICompatibleLLM → live vLLM server.

Skipped automatically when no vLLM server is reachable.  Run explicitly with:

    pytest tests/test_llm_judge_vllm_integration.py -m integration -v

What is being verified
----------------------
1. ``OpenAICompatibleLLM`` can reach the live server and get a response.
2. ``StrictLLMJudge`` correctly labels H1 pairs (same-cluster, label==1) as
   REUSABLE and H0 pairs (different cluster or label==0) as NOT_REUSABLE,
   using real query texts from ``data/h1h0_final.npz``.
3. Overall agreement with ground-truth labels meets a lenient floor (>=60 %)
   so the test is robust to occasional UNCERTAIN outputs from the model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache.feedback.types import JudgeLabel, JudgeRequest
from mlcache.llm_providers.openai_compatible import OpenAICompatibleLLM
from mlcache.llm_providers.prompts import StrictLLMJudge

# ── constants ────────────────────────────────────────────────────────────────

DEFAULT_VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B-AWQ")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "local-vllm-token")
NPZ_PATH = ROOT / "data" / "h1h0_final.npz"

N_H1_PAIRS = 5   # same-cluster pairs  → expected REUSABLE
N_H0_PAIRS = 5   # cross-cluster pairs → expected NOT_REUSABLE
MIN_AGREEMENT = 0.60   # fraction of pairs the judge must get right


# ── helpers ──────────────────────────────────────────────────────────────────

def _vllm_reachable(base_url: str, api_key: str) -> bool:
    req = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


class _Pair(NamedTuple):
    query: str
    candidate_query: str
    candidate_response: str   # realistic cached response for candidate_query
    expected: JudgeLabel      # REUSABLE or NOT_REUSABLE


def _sample_raw_pairs(rng: np.random.Generator) -> list[tuple[str, str, JudgeLabel]]:
    """Return (query, candidate_query, expected_label) triples from the NPZ."""
    data = np.load(NPZ_PATH, allow_pickle=True)
    texts: np.ndarray = data["text"]
    labels: np.ndarray = data["label"]
    clusters: np.ndarray = data["global_cluster"]

    pairs: list[tuple[str, str, JudgeLabel]] = []

    # H1 — both label==1, same non-noise cluster
    h1_mask = (labels == 1) & (clusters >= 0)
    h1_indices = np.where(h1_mask)[0]
    h1_clusters, h1_counts = np.unique(clusters[h1_indices], return_counts=True)
    eligible = h1_clusters[h1_counts >= 2]
    rng.shuffle(eligible)
    for cid in eligible:
        if len(pairs) >= N_H1_PAIRS:
            break
        members = h1_indices[clusters[h1_indices] == cid]
        chosen = rng.choice(members, size=2, replace=False)
        pairs.append((str(texts[chosen[0]]), str(texts[chosen[1]]), JudgeLabel.REUSABLE))

    # H0 — texts from distinct clusters
    h0_candidates = np.where((labels == 1) & (clusters >= 0))[0]
    rng.shuffle(h0_candidates)
    seen_clusters: set[int] = set()
    pool: list[int] = []
    for idx in h0_candidates:
        cid = int(clusters[idx])
        if cid not in seen_clusters:
            seen_clusters.add(cid)
            pool.append(idx)
        if len(pool) >= N_H0_PAIRS * 4:
            break
    i = 0
    while len(pairs) < N_H1_PAIRS + N_H0_PAIRS and i + 1 < len(pool):
        a, b = pool[i], pool[i + 1]
        if clusters[a] != clusters[b]:
            pairs.append((str(texts[a]), str(texts[b]), JudgeLabel.NOT_REUSABLE))
        i += 2

    return pairs


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vllm_judge() -> StrictLLMJudge:
    if not _vllm_reachable(DEFAULT_VLLM_URL, DEFAULT_VLLM_API_KEY):
        pytest.skip("No live vLLM server reachable — skipping integration test.")
    llm = OpenAICompatibleLLM(
        base_url=DEFAULT_VLLM_URL,
        api_key=DEFAULT_VLLM_API_KEY,
        model=DEFAULT_VLLM_MODEL,
    )
    return StrictLLMJudge(llm)


@pytest.fixture(scope="module")
def sampled_pairs(vllm_judge: StrictLLMJudge) -> list[_Pair]:
    """Sample NPZ pairs and generate a real cached response for each candidate query."""
    if not NPZ_PATH.exists():
        pytest.skip(f"NPZ dataset not found at {NPZ_PATH}")
    rng = np.random.default_rng(42)
    raw = _sample_raw_pairs(rng)
    if not raw:
        pytest.skip("Could not sample pairs from NPZ.")

    pairs: list[_Pair] = []
    for query, candidate_query, expected in raw:
        response = vllm_judge._llm.generate(
            candidate_query[:1000],  # cap to avoid very long prompts
            max_tokens=200,
        )
        pairs.append(_Pair(
            query=query,
            candidate_query=candidate_query,
            candidate_response=response.text,
            expected=expected,
        ))
    return pairs


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_vllm_server_responds(vllm_judge: StrictLLMJudge) -> None:
    """Sanity-check: the judge can complete one trivial request."""
    req = JudgeRequest(
        query="What is the capital of France?",
        candidate_query="What is the capital city of France?",
        candidate_response="The capital of France is Paris.",
    )
    result = vllm_judge.judge(req)
    assert result.decision.label in {JudgeLabel.REUSABLE, JudgeLabel.NOT_REUSABLE, JudgeLabel.UNCERTAIN}
    assert result.decision.rationale is not None


@pytest.mark.integration
def test_obvious_reusable_pair(vllm_judge: StrictLLMJudge) -> None:
    """Identical semantic content → must be REUSABLE."""
    req = JudgeRequest(
        query="What is the boiling point of water?",
        candidate_query="At what temperature does water boil?",
        candidate_response="Water boils at 100°C (212°F) at standard atmospheric pressure.",
    )
    result = vllm_judge.judge(req)
    assert result.decision.label == JudgeLabel.REUSABLE, (
        f"Expected REUSABLE, got {result.decision.label!r}. "
        f"Rationale: {result.decision.rationale!r}"
    )


@pytest.mark.integration
def test_obvious_not_reusable_pair(vllm_judge: StrictLLMJudge) -> None:
    """Clearly different intent → must be NOT_REUSABLE."""
    req = JudgeRequest(
        query="What is the boiling point of water?",
        candidate_query="Who wrote Hamlet?",
        candidate_response="Hamlet was written by William Shakespeare.",
    )
    result = vllm_judge.judge(req)
    assert result.decision.label == JudgeLabel.NOT_REUSABLE, (
        f"Expected NOT_REUSABLE, got {result.decision.label!r}. "
        f"Rationale: {result.decision.rationale!r}"
    )


@pytest.mark.integration
def test_judge_agreement_on_npz_pairs(
    vllm_judge: StrictLLMJudge,
    sampled_pairs: list[_Pair],
) -> None:
    """Judge >= 60 % of sampled NPZ pairs correctly (UNCERTAIN counts as wrong)."""
    if not sampled_pairs:
        pytest.skip("No pairs could be sampled from the NPZ.")

    results: list[dict] = []
    for pair in sampled_pairs:
        req = JudgeRequest(
            query=pair.query,
            candidate_query=pair.candidate_query,
            candidate_response=pair.candidate_response,
        )
        result = vllm_judge.judge(req)
        correct = result.decision.label == pair.expected
        results.append({
            "expected": pair.expected,
            "got": result.decision.label,
            "correct": correct,
            "rationale": result.decision.rationale,
        })

    n_correct = sum(r["correct"] for r in results)
    n_total = len(results)
    agreement = n_correct / n_total

    # Print a per-pair breakdown for debugging on failure
    for i, r in enumerate(results):
        status = "OK" if r["correct"] else "WRONG"
        print(f"  [{status}] expected={r['expected']} got={r['got']} | {r['rationale']!r:.80}")

    assert agreement >= MIN_AGREEMENT, (
        f"Judge agreement {n_correct}/{n_total} ({agreement:.0%}) "
        f"is below the {MIN_AGREEMENT:.0%} floor.\n"
        + "\n".join(
            f"  expected={r['expected']} got={r['got']} | {r['rationale']!r:.80}"
            for r in results if not r["correct"]
        )
    )
