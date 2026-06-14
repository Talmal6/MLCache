"""Re-grade an online_replay result checkpoint with the LLM-judge ground truth.

A checkpoint produced BEFORE the `reuse_label` switch graded served-hit FP/TP with
the dataset H0/H1 rule. This rewrites every audit row's `oracle_label` /
`is_false_positive` from the persistent LLM-judge cache (the verdicts the system
actually trained/served on) and recomputes exactly the summary fields that depend
on those labels -- identical to what a fresh run on the new code would report,
since serving decisions are unchanged and only the grading differs.

Label-INDEPENDENT fields (shadow_*, thresholds, counts, calibration_h0 / active
score distributions) are left untouched. The original is backed up first.

Run:
  .conda/bin/python experiments/rescore_audit_with_llm_judge.py \
    --checkpoint experiments/llm_judge_replay/ensemble/result_checkpoint.json \
    --judge-cache experiments/llm_judge_replay/judge_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

_JUDGE_TO_HYP = {"REUSABLE": "H1", "NOT_REUSABLE": "H0", "UNCERTAIN": "UNCERTAIN"}


def _safe_rate(x: int, n: int):
    return (x / n) if n else None


def _dist(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size), "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "min": float(np.min(arr)), "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)), "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def load_judge_cache(path: Path) -> dict[tuple[int, int], str]:
    jc: dict[tuple[int, int], str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                jc[(int(r["q"]), int(r["c"]))] = str(r["label"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return jc


def llm_oracle_label(row: dict, jc: dict[tuple[int, int], str]) -> str:
    anchor = row.get("anchor_row_id")
    if anchor == "" or anchor is None:
        return "UNKNOWN"  # anchor recovery failed -> excluded, as before
    verdict = jc.get((int(row["query_row_id"]), int(anchor)))
    if verdict is None:
        return "UNCERTAIN"  # judge never evaluated this served pair
    return _JUDGE_TO_HYP.get(verdict, "UNCERTAIN")


def rescore(checkpoint: Path, judge_cache: Path) -> int:
    jc = load_judge_cache(judge_cache)
    d = json.loads(checkpoint.read_text())
    audit = d["audit_rows"]
    s = d["summary"]

    before = {k: s.get(k) for k in
              ("active_true_positives", "active_false_positives", "active_fp_rate")}

    # 1) re-label every audit row from the LLM-judge verdict
    miss = 0
    for a in audit:
        lab = llm_oracle_label(a, jc)
        if lab == "UNCERTAIN" and a.get("anchor_row_id") not in ("", None):
            miss += 1
        a["oracle_label"] = lab
        a["is_false_positive"] = lab == "H0"

    # 2) recompute the label-dependent summary fields (mirrors run_one_system)
    active = [a for a in audit if a["oracle_label"] in ("H0", "H1")]
    active_tp = sum(1 for a in active if a["oracle_label"] == "H1")
    active_fp = sum(1 for a in active if a["oracle_label"] == "H0")
    post = [a for a in active if a["post_activation"]]
    post_tp = sum(1 for a in post if a["oracle_label"] == "H1")
    post_fp = sum(1 for a in post if a["oracle_label"] == "H0")

    s["active_hits"] = len(audit)
    s["active_decisive_hits"] = len(active)
    s["active_true_positives"] = active_tp
    s["active_false_positives"] = active_fp
    s["active_uncertain"] = len(audit) - len(active)
    s["active_fp_rate"] = _safe_rate(active_fp, active_tp + active_fp)
    s["post_activation_active_hits"] = len(post)
    s["post_activation_true_positives"] = post_tp
    s["post_activation_false_positives"] = post_fp
    s["post_activation_fp_rate"] = _safe_rate(post_fp, post_tp + post_fp)

    # 3) calibration_vs_active: only the FP-score distribution depends on labels
    cva = s.get("calibration_vs_active")
    if isinstance(cva, dict):
        fp_scores = np.asarray(
            [a["score"] for a in audit if a["is_false_positive"] and a.get("score") is not None],
            dtype=np.float64,
        )
        if fp_scores.size:
            cva["active_false_positive_hits"] = _dist(fp_scores)
        else:
            cva.pop("active_false_positive_hits", None)

    # 4) provenance marker
    s["grading_oracle"] = "llm_judge"
    s["grading_note"] = ("served-hit FP/TP re-graded from the LLM-judge cache; "
                         "serving decisions unchanged. Previously: dataset_h0h1_rule.")

    backup = checkpoint.with_suffix(".dataset_rule_backup.json")
    if not backup.exists():
        shutil.copy2(checkpoint, backup)
    checkpoint.write_text(json.dumps(d))

    print(f"backed up original -> {backup}")
    print(f"served hits: {len(audit)}  judge-cache misses on served pairs: {miss}")
    print(f"  active_true_positives : {before['active_true_positives']:>6} -> {active_tp}")
    print(f"  active_false_positives: {before['active_false_positives']:>6} -> {active_fp}")
    bf = before["active_fp_rate"]; af = s["active_fp_rate"]
    print(f"  active_fp_rate        : {bf:.4f} -> {af:.4f}")
    prec_b = before["active_true_positives"] / max(
        1, before["active_true_positives"] + before["active_false_positives"])
    prec_a = active_tp / max(1, active_tp + active_fp)
    print(f"  served precision      : {prec_b:.4f} -> {prec_a:.4f}")
    print(f"  active_uncertain      : {s['active_uncertain']} (excluded from FP/TP)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--judge-cache", required=True)
    args = p.parse_args()
    return rescore(Path(args.checkpoint), Path(args.judge_cache))


if __name__ == "__main__":
    raise SystemExit(main())
