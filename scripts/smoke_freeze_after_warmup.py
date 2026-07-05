"""Smoke test for run_one_system + --freeze-after-warmup + latency stats.

Drives the REAL experiment code path (experiments/online_replay_h1h0_final.run_one_system)
with a mock LLM judge over small synthetic clustered data, so it needs no vLLM
server and runs in seconds. Verifies:

  * the full warm-up -> measured-replay lifecycle runs without error,
  * the ensemble actually trains during warm-up (latency 'fits' >= 1),
  * latency instrumentation is populated (inference calls > 0, avg_ms is a float),
  * with --freeze-after-warmup the scorer is pinned: scorer_version is CONSTANT
    across every measured request (no online promotion), and the summary records
    freeze_after_warmup=True,
  * without the flag the summary records freeze_after_warmup=False.
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_cosine_vs_ensemble import DatasetRow, encode_prompt  # noqa: E402
from mlcache.llm_wrapper import LLMResponse  # noqa: E402
import online_replay_h1h0_final as exp  # noqa: E402


# ── mock LLM judge: REUSABLE iff same group AND both rows are label==1 ───────

_TAG = re.compile(r"grp(\d+) lbl(\d+)")


class MockJudgeLLM:
    """Implements the LLMClient protocol; labels pairs from tags in row text."""

    model = "mock-judge"
    base_url = "mock://local"
    timeout_seconds = 0.0
    max_retries = 0

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        tags = _TAG.findall(prompt)  # [(qgrp, qlbl), (cgrp, clbl)] in template order
        if len(tags) >= 2:
            (qg, ql), (cg, cl) = tags[0], tags[1]
            reusable = (qg == cg) and (ql == "1") and (cl == "1")
            text = "REUSABLE" if reusable else "NOT_REUSABLE"
        else:
            text = "NOT_REUSABLE"
        return LLMResponse(text=text, raw=None, metadata={"provider": "mock"})


def build_rows(*, n_groups: int = 6, per_group: int = 12, dim: int = 32, seed: int = 7):
    """Synthesize clustered rows: each group has a base vector; half the rows in
    a group are label==1 and half label==0, so retrieval surfaces both H1
    (same-group 1&1) and H0 (same-group 1&0) pairs for the judge.
    """
    rng = np.random.default_rng(seed)
    rows: list[DatasetRow] = []
    rid = 0
    for g in range(n_groups):
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        for j in range(per_group):
            emb = base + 0.15 * rng.normal(size=dim)
            label = 1 if j % 2 == 0 else 0
            text = f"grp{g} lbl{label} sample text for row {rid}"
            rows.append(DatasetRow(
                row_id=rid, text=text, label=label, group=f"g{g}",
                embedding=emb.astype(np.float64), prompt=encode_prompt(rid, text),
            ))
            rid += 1
    return rows


def run(*, freeze: bool, logger) -> dict:
    rows = build_rows()
    rows_by_id = {r.row_id: r for r in rows}
    rng = np.random.default_rng(123)

    def sample(n: int) -> list[DatasetRow]:
        idx = rng.integers(0, len(rows), size=n)
        return [rows[int(i)] for i in idx]

    warmup_stream = sample(400)
    measured_stream = sample(160)

    with tempfile.TemporaryDirectory() as td:
        return exp.run_one_system(
            method="ensemble", scorer="ensemble",
            scorers=["cosine", "lda", "pca_whitened_cosine", "xgboost", "mlp"],
            stream=measured_stream, rows_by_id=rows_by_id,
            # batch_size drives calibration_every_n = max(1, batch_size//10) in
            # system.py; too-small a value routes EVERY judged pair to the
            # calibration bucket and starves the train bucket. Use 40 so the
            # train/calibration split (1-in-4 to calibration) actually fills both.
            target_fpr=0.2, top_k=5, batch_size=40,
            gate_min_train_h0=4, gate_min_train_h1=2,
            gate_min_calib_h0=4, gate_min_calib_h1=2,
            parallelism=1, persistence=False,
            state_dir=Path(td) / "state", logger=logger, progress_every=0,
            warmup_stream=warmup_stream,
            use_wilson=False,
            shared_cache=False,
            freeze_after_warmup=freeze,
            llm_client=MockJudgeLLM(),
            judge_cache=None,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("smoke")

    failures: list[str] = []

    print("\n========== FROZEN run (--freeze-after-warmup) ==========")
    frozen = run(freeze=True, logger=logger)
    fs = frozen["summary"]
    lat = fs["latency"]
    versions = {r["scorer_version"] for r in frozen["minimal_rows"]}

    print("\n--- frozen summary ---")
    print(f"  trained={fs['trained']} calibrated={fs['calibrated']} "
          f"threshold={fs['threshold']}")
    print(f"  freeze_after_warmup={fs['freeze_after_warmup']}")
    print(f"  scorer_versions across measured stream={sorted(versions)}")
    print(f"  latency.inference: calls={lat['inference']['calls']} "
          f"avg_ms={lat['inference']['avg_ms']}")
    print(f"  latency.train: fits={lat['train']['fits']} "
          f"avg_seconds={lat['train']['avg_seconds']}")
    print(f"  latency.calibrate: calls={lat['calibrate']['calls']}")

    if not fs["freeze_after_warmup"]:
        failures.append("frozen run: summary.freeze_after_warmup should be True")
    if not fs["trained"]:
        failures.append("frozen run: ensemble never trained during warm-up")
    if len(versions) != 1:
        failures.append(
            f"frozen run: scorer_version changed during measured phase "
            f"({sorted(versions)}) -- retraining was NOT actually frozen")
    if lat["inference"]["calls"] <= 0 or lat["inference"]["avg_ms"] is None:
        failures.append("frozen run: latency inference stats not populated")
    if lat["train"]["fits"] < 1:
        failures.append("frozen run: expected >= 1 fit during warm-up")

    print("\n========== UNFROZEN run (online retraining) ==========")
    live = run(freeze=False, logger=logger)
    ls = live["summary"]
    print("\n--- unfrozen summary ---")
    print(f"  trained={ls['trained']} freeze_after_warmup={ls['freeze_after_warmup']} "
          f"latency.fits={ls['latency']['train']['fits']}")
    if ls["freeze_after_warmup"]:
        failures.append("unfrozen run: summary.freeze_after_warmup should be False")

    print("\n================= RESULT =================")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
