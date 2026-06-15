"""Submit online_replay_h1h0_final.py experiments as a SLURM array.

Each job runs all seeds for one (target_fpr, selection, scorers, mode)
combination sequentially. Jobs run in parallel across the SLURM array.

Typical usage
-------------
# Dry-run: inspect the task matrix without writing files
python submit_replay_array.py --dry-run

# Write files + submit
python submit_replay_array.py --submit

# Smoke suite: 1 seed, fpr=0.05, 300 requests
python submit_replay_array.py --suite smoke --submit

# Sweep multiple FPRs and seeds, submit and wait
python submit_replay_array.py \\
    --target-fprs 0.01 0.03 0.05 0.10 \\
    --seeds 0 1 2 3 4 \\
    --submit --wait

# Re-aggregate after a completed run (no re-run)
python submit_replay_array.py \\
    --exp-root runs/replay_20250611_120000 \\
    --skip-submit --aggregate-only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_SCRIPT = REPO_ROOT / "experiments" / "online_replay_h1h0_final.py"
SBATCH_FILE = REPO_ROOT / "replay_experiment_array.sbatch"
COMPLETE_SENTINEL = "_COMPLETE"
DEFAULT_NPZ = REPO_ROOT / "data" / "h1h0_final.npz"
FULL_ENSEMBLE = "cosine,lda,pca_whitened_cosine,xgboost,mlp"

# Student t critical values (two-sided 95%, df=1..30; 1.96 for df>30)
T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

AGGREGATE_METRICS = [
    "hit_rate",
    "shadow_empirical_fpr", "shadow_tpr", "shadow_precision",
    "active_fpr", "active_fdr", "active_tpr", "active_precision",
    "active_fpr_budget_gap", "active_fpr_budget_utilization",
    "post_activation_fpr", "post_activation_fdr",
    "active_true_positives", "active_false_positives",
    "train_h0", "train_h1", "calibration_h0", "calibration_h1",
    "activation_request_index", "first_calibrated_at_request",
]


# ── task dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReplayTask:
    task_id: int
    seed: int
    target_fpr: float
    selection: str
    scorers: str               # comma-separated, e.g. "cosine,lda,..."
    mode: str                  # "compare" or "retrieved_anchor_diagnostics"
    max_requests: int
    cache_size: int
    warmup_requests: int
    progress_every: int
    top_k: int
    batch_size: int
    min_train_h0: int
    min_train_h1: int
    min_calib_h0: int
    min_calib_h1: int
    npz_path: Path
    output_dir: Path
    extra_flags: tuple[str, ...] = field(default_factory=tuple)


def is_complete(task: ReplayTask) -> bool:
    return (task.output_dir / COMPLETE_SENTINEL).exists()


def task_to_cli_args(task: ReplayTask, python_bin: str) -> list[str]:
    args = [
        python_bin, str(EXPERIMENT_SCRIPT),
        "--npz", str(task.npz_path),
        "--output-dir", str(task.output_dir),
        "--mode", task.mode,
        "--seed", str(task.seed),
        "--target-fpr", str(task.target_fpr),
        "--selection", task.selection,
        "--scorers", task.scorers,
        "--max-requests", str(task.max_requests),
        "--cache-size", str(task.cache_size),
        "--warmup-requests", str(task.warmup_requests),
        "--progress-every", str(task.progress_every),
        "--top-k", str(task.top_k),
        "--batch-size", str(task.batch_size),
        "--min-train-h0", str(task.min_train_h0),
        "--min-train-h1", str(task.min_train_h1),
        "--min-calib-h0", str(task.min_calib_h0),
        "--min-calib-h1", str(task.min_calib_h1),
    ]
    args.extend(task.extra_flags)
    return args


def task_to_command_line(python_bin: str, task: ReplayTask) -> str:
    return " ".join(shlex.quote(a) for a in task_to_cli_args(task, python_bin))


# ── task suite construction ─────────────────────────────────────────────────

def _task_group_key(task: ReplayTask) -> tuple:
    return (
        task.target_fpr, task.selection, task.scorers, task.mode,
        task.max_requests, task.cache_size, task.warmup_requests,
        task.progress_every, task.top_k, task.batch_size,
        task.min_train_h0, task.min_train_h1,
        task.min_calib_h0, task.min_calib_h1,
        str(task.npz_path), task.extra_flags,
    )


def build_replay_suite(
    exp_root: Path,
    seeds: Sequence[int],
    target_fprs: Sequence[float],
    selections: Sequence[str],
    scorers_list: Sequence[str],
    modes: Sequence[str],
    *,
    max_requests: int,
    cache_size: int,
    warmup_requests: int,
    progress_every: int,
    top_k: int,
    batch_size: int,
    min_train_h0: int,
    min_train_h1: int,
    min_calib_h0: int,
    min_calib_h1: int,
    npz_path: Path,
    extra_flags: tuple[str, ...],
) -> list[ReplayTask]:
    tasks: list[ReplayTask] = []
    task_id = 1
    for mode in modes:
        for scorers in scorers_list:
            for fpr in target_fprs:
                for selection in selections:
                    for seed in seeds:
                        fpr_tag = f"{fpr:.4g}".replace(".", "p")
                        sc_tag = scorers.split(",")[0] if "," in scorers else scorers
                        tag = f"mode_{mode}__fpr_{fpr_tag}__sel_{selection}__sc_{sc_tag}__seed_{seed}"
                        out_dir = exp_root / "tasks" / tag
                        tasks.append(ReplayTask(
                            task_id=task_id,
                            seed=seed,
                            target_fpr=float(fpr),
                            selection=selection,
                            scorers=scorers,
                            mode=mode,
                            max_requests=int(max_requests),
                            cache_size=int(cache_size),
                            warmup_requests=int(warmup_requests),
                            progress_every=int(progress_every),
                            top_k=int(top_k),
                            batch_size=int(batch_size),
                            min_train_h0=int(min_train_h0),
                            min_train_h1=int(min_train_h1),
                            min_calib_h0=int(min_calib_h0),
                            min_calib_h1=int(min_calib_h1),
                            npz_path=npz_path,
                            output_dir=out_dir,
                            extra_flags=extra_flags,
                        ))
                        task_id += 1
    return tasks


# ── job grouping (seeds sequential, combos parallel) ───────────────────────

@dataclass(frozen=True)
class ReplayJob:
    job_id: int
    tasks: tuple[ReplayTask, ...]
    output_dir: Path


def group_tasks_into_jobs(tasks: Sequence[ReplayTask], exp_root: Path) -> list[ReplayJob]:
    grouped: dict[tuple, list[ReplayTask]] = {}
    for task in tasks:
        grouped.setdefault(_task_group_key(task), []).append(task)

    jobs: list[ReplayJob] = []
    for job_id, group_tasks in enumerate(grouped.values(), start=1):
        ordered = tuple(sorted(group_tasks, key=lambda t: int(t.seed)))
        jobs.append(ReplayJob(
            job_id=job_id,
            tasks=ordered,
            output_dir=exp_root / "jobs" / f"job_{job_id:06d}",
        ))
    return jobs


def _job_seed_text(job: ReplayJob) -> str:
    return ",".join(str(t.seed) for t in job.tasks)


# ── job script writing ──────────────────────────────────────────────────────

def _vllm_guard_block(
    *,
    base_url: str,
    api_key: str,
    model: str,
    port: int,
    startup_timeout: int,
    python_bin: str,
) -> list[str]:
    """Return bash lines that ensure the vLLM server is up before the experiment runs.

    If the server at `base_url` is already reachable, nothing is done.
    Otherwise the server is started in the background on `port`, the script
    waits up to `startup_timeout` seconds for it to become ready, and an EXIT
    trap shuts it down when the job finishes.
    """
    q_base_url = shlex.quote(base_url)
    q_api_key = shlex.quote(api_key)
    q_model = shlex.quote(model)
    q_python = shlex.quote(python_bin)
    # Derive the conda lib dir from the python binary so the correct libstdc++ is used.
    conda_lib = shlex.quote(str(Path(python_bin).parent.parent / "lib"))
    return [
        "# ── vLLM server guard ────────────────────────────────────────────────",
        f"export VLLM_BASE_URL={q_base_url}",
        f"export VLLM_MODEL={q_model}",
        f"export VLLM_API_KEY={q_api_key}",
        # Ensure the conda env's libstdc++ is preferred over the system one.
        f"export LD_LIBRARY_PATH={conda_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}",
        "",
        "_vllm_ready() {",
        f'    curl -sf --max-time 5 "${{VLLM_BASE_URL}}/models" \\',
        f'        -H "Authorization: Bearer ${{VLLM_API_KEY}}" > /dev/null 2>&1',
        "}",
        "",
        "if _vllm_ready; then",
        '    echo "vLLM server already up at ${VLLM_BASE_URL}"',
        "else",
        f'    echo "vLLM server not reachable at {base_url}; starting (model={model}, port={port})..."',
        f"    {q_python} -m vllm.entrypoints.openai.api_server \\",
        f"        --model {q_model} \\",
        f"        --port {port} \\",
        f"        --api-key {q_api_key} &",
        "    _VLLM_PID=$!",
        f'    trap "echo \\"Stopping vLLM (pid=${{_VLLM_PID}})\\"; kill \\"${{_VLLM_PID}}\\" 2>/dev/null" EXIT',
        f"    _vllm_elapsed=0",
        "    until _vllm_ready; do",
        f"        if [ $_vllm_elapsed -ge {startup_timeout} ]; then",
        f'            echo "ERROR: vLLM did not start within {startup_timeout}s" >&2',
        "            exit 1",
        "        fi",
        "        sleep 10",
        "        _vllm_elapsed=$((_vllm_elapsed + 10))",
        '        echo "  waiting for vLLM... ${_vllm_elapsed}s elapsed"',
        "    done",
        '    echo "vLLM server ready after ${_vllm_elapsed}s"',
        "fi",
        "# ─────────────────────────────────────────────────────────────────────",
        "",
    ]


def write_job_script(job: ReplayJob, python_bin: str, vllm_cfg: dict | None = None) -> Path:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    script = job.output_dir / "job.sh"
    first = job.tasks[0]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo " + shlex.quote(
            f"Running job_id={job.job_id} "
            f"mode={first.mode} fpr={first.target_fpr:.4g} "
            f"sel={first.selection} scorers={first.scorers} "
            f"seeds={_job_seed_text(job)}"
        ),
        "",
    ]
    if vllm_cfg is not None:
        lines += _vllm_guard_block(python_bin=python_bin, **vllm_cfg)
    for task in job.tasks:
        lines.append("echo " + shlex.quote(f"  Starting seed={task.seed} task_id={task.task_id}"))
        lines.append("mkdir -p " + shlex.quote(str(task.output_dir)))
        lines.append(task_to_command_line(python_bin, task))
        lines.append("touch " + shlex.quote(str(task.output_dir / COMPLETE_SENTINEL)))
        lines.append("echo " + shlex.quote(f"  Finished seed={task.seed} task_id={task.task_id}"))
        lines.append("")
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o755)
    return script


# ── file writing ────────────────────────────────────────────────────────────

def _default_exp_root(args: argparse.Namespace) -> Path:
    """Return a stable directory for one exact experiment configuration."""
    config = {
        "suite": args.suite,
        "seeds": args.seeds,
        "target_fprs": args.target_fprs,
        "selections": args.selections,
        "scorers_list": args.scorers_list,
        "modes": args.modes,
        "npz": str(Path(args.npz).resolve()),
        "max_requests": args.max_requests,
        "cache_size": args.cache_size,
        "warmup_requests": args.warmup_requests,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "min_train_h0": args.min_train_h0,
        "min_train_h1": args.min_train_h1,
        "min_calib_h0": args.min_calib_h0,
        "min_calib_h1": args.min_calib_h1,
        "shared_cache": args.shared_cache,
        "use_wilson": args.use_wilson,
        "persist": args.persist,
        "vllm_model": args.vllm_model,
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return REPO_ROOT / "runs" / f"replay_{args.suite}_{digest}"


def write_files(
    exp_root: Path,
    jobs: list[ReplayJob],
    python_bin: str,
    args: argparse.Namespace,
    vllm_cfg: dict | None = None,
) -> Path:
    (exp_root / "slurm").mkdir(parents=True, exist_ok=True)
    (exp_root / "jobs").mkdir(parents=True, exist_ok=True)

    commands_file = exp_root / "commands.txt"
    with commands_file.open("w", encoding="utf-8") as cmd_f:
        for job in jobs:
            script = write_job_script(job, python_bin, vllm_cfg=vllm_cfg)
            cmd_f.write("bash " + shlex.quote(str(script)) + "\n")

    return commands_file


# ── SLURM submission ────────────────────────────────────────────────────────

def _env_setup(conda_env: str, python_bin: str | None = None) -> str:
    # When an explicit python binary path is given, derive the conda env root
    # from it (.conda/bin/python -> .conda/) and activate by path — works for
    # local envs that have no registered name.
    if python_bin and Path(python_bin).is_file():
        env_path = shlex.quote(str(Path(python_bin).parent.parent))
    else:
        env_path = shlex.quote(conda_env)
    return (
        f'set -euo pipefail; CONDA_BASE="$(conda info --base)"; '
        f'source "${{CONDA_BASE}}/etc/profile.d/conda.sh"; conda activate {env_path}'
    )


def _resolve_python(conda_env: str, python_bin: str | None) -> str:
    if python_bin:
        try:
            proc = subprocess.run(
                [python_bin, "-c", "import sys; print(sys.executable)"],
                text=True, capture_output=True, check=True,
            )
            return proc.stdout.strip().splitlines()[-1].strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Could not verify python_bin {python_bin!r}: {exc}") from exc
    try:
        proc = subprocess.run(
            ["conda", "run", "-n", conda_env, "which", "python"],
            text=True, capture_output=True, check=True,
        )
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        if not lines:
            raise RuntimeError(f"Conda env {conda_env!r} resolved no python executable.")
        return lines[-1]
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not resolve python for conda env {conda_env!r}. "
            f"Try --python-bin to pass an explicit path.\n{getattr(exc, 'stderr', '')}"
        ) from exc


def _sbatch_command(
    exp_root: Path, commands_file: Path, n_jobs: int, args: argparse.Namespace, python_bin: str | None = None
) -> list[str]:
    array_spec = f"1-{n_jobs}"
    if int(args.max_parallel) > 0:
        array_spec += f"%{int(args.max_parallel)}"
    cmd = [
        "sbatch",
        f"--job-name={args.job_name}",
        f"--array={array_spec}",
        f"--time={args.time}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        f"--output={exp_root}/slurm/%x_%A_%a.out",
        f"--error={exp_root}/slurm/%x_%A_%a.err",
        "--export="
        f"ALL,COMMANDS_FILE={commands_file},"
        f"EXP_ROOT={exp_root},"
        f"ENV_SETUP={_env_setup(args.conda_env, python_bin)}",
    ]
    for attr, opt in (("partition", "partition"), ("account", "account"), ("qos", "qos")):
        value = getattr(args, attr, "")
        if value:
            cmd.append(f"--{opt}={value}")
    if getattr(args, "gpus", ""):
        cmd.append(f"--gres=gpu:{args.gpus}")
    cmd.append(str(SBATCH_FILE))
    return cmd


def _parse_sbatch_job_id(output: str) -> str | None:
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return match.group(1) if match else None


def submit_sbatch(sbatch: Sequence[str]) -> str:
    proc = subprocess.run(sbatch, text=True, capture_output=True, check=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    job_id = _parse_sbatch_job_id(proc.stdout)
    if job_id is None:
        raise RuntimeError(f"Could not parse SLURM job id from sbatch output: {proc.stdout.strip()!r}")
    return job_id


def wait_for_slurm_job(job_id: str, *, poll_interval: float, timeout: float) -> None:
    poll = max(float(poll_interval), 1.0)
    started = time.monotonic()
    print(f"Waiting for SLURM job {job_id} …", flush=True)
    while True:
        proc = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%i %T"],
            text=True, capture_output=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"squeue failed for job {job_id}: {proc.stderr.strip()}")
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        if not lines:
            print(f"SLURM job {job_id} is no longer in the queue.", flush=True)
            return
        states = sorted({l.split(maxsplit=1)[1] if " " in l else "UNKNOWN" for l in lines})
        elapsed = time.monotonic() - started
        print(
            f"[wait] job={job_id} elapsed={elapsed/60:.1f}m "
            f"queue_entries={len(lines)} states={','.join(states)}",
            flush=True,
        )
        if timeout > 0 and elapsed >= timeout:
            raise RuntimeError(f"Timed out waiting for SLURM job {job_id} after {timeout:.0f}s.")
        time.sleep(poll)


# ── aggregation ─────────────────────────────────────────────────────────────

def _float_or_none(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(out) else out


def _t_mult_95(n: int) -> float:
    if n <= 1:
        return 0.0
    df = n - 1
    return T_CRITICAL_975.get(df, 1.96)


def _mean_ci95(values: Iterable[Any]) -> dict[str, Any]:
    xs = [v for v in (_float_or_none(v) for v in values) if v is not None]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": "", "std": "", "ci95_low": "", "ci95_high": "", "ci95_half_width": ""}
    mean = statistics.fmean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    hw = _t_mult_95(n) * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n, "mean": float(mean), "std": float(std),
        "ci95_low": float(mean - hw), "ci95_high": float(mean + hw),
        "ci95_half_width": float(hw),
    }


def collect_results(tasks: Sequence[ReplayTask]) -> tuple[list[dict], list[ReplayTask], list[ReplayTask]]:
    """Read summary_metrics.json from each completed task. Returns (rows, incomplete, missing)."""
    rows: list[dict] = []
    incomplete: list[ReplayTask] = []
    missing: list[ReplayTask] = []

    for task in tasks:
        if not is_complete(task):
            incomplete.append(task)
        summary_path = task.output_dir / "summary_metrics.json"
        if not summary_path.exists():
            missing.append(task)
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"WARNING: could not read {summary_path}: {exc}", file=sys.stderr)
            missing.append(task)
            continue

        for method_key in ("cosine", "ensemble"):
            method_data = data.get(method_key)
            if not isinstance(method_data, dict):
                continue
            rows.append({
                "_task_id": task.task_id,
                "_seed": task.seed,
                "_target_fpr": task.target_fpr,
                "_selection": task.selection,
                "_scorers": task.scorers,
                "_mode": task.mode,
                "_method": method_key,
                **{k: method_data.get(k) for k in AGGREGATE_METRICS},
            })
    return rows, incomplete, missing


def _aggregate_rows(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["_mode"], row["_scorers"], row["_target_fpr"], row["_selection"], row["_method"])
        grouped[key].append(row)

    out: list[dict] = []
    for (mode, scorers, fpr, selection, method), items in sorted(grouped.items()):
        agg: dict[str, Any] = {
            "mode": mode, "scorers": scorers, "target_fpr": fpr,
            "selection": selection, "method": method,
            "n_seeds": len(items),
        }
        for metric in AGGREGATE_METRICS:
            stats = _mean_ci95(row.get(metric) for row in items)
            for stat_name, stat_val in stats.items():
                agg[f"{metric}_{stat_name}"] = stat_val
        out.append(agg)
    return out


def write_aggregate_tables(exp_root: Path, tasks: Sequence[ReplayTask], *, allow_partial: bool = False) -> dict:
    rows, incomplete, missing = collect_results(tasks)
    if incomplete and not allow_partial:
        raise RuntimeError(
            f"{len(incomplete)} task(s) not yet complete. "
            "Re-run with --allow-partial-aggregate or wait for jobs to finish."
        )
    if not rows:
        raise RuntimeError("No summary_metrics.json rows found to aggregate.")

    agg_rows = _aggregate_rows(rows)
    table_dir = exp_root / "final_tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["mode", "scorers", "target_fpr", "selection", "method", "n_seeds"]
        + [f"{m}_{s}" for m in AGGREGATE_METRICS
           for s in ("n", "mean", "std", "ci95_low", "ci95_high", "ci95_half_width")]
    )
    all_csv = table_dir / "aggregate.csv"
    with all_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg_rows)

    metadata = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "trial_rows": len(rows),
        "aggregate_rows": len(agg_rows),
        "incomplete_tasks": len(incomplete),
        "missing_summary_tasks": len(missing),
    }
    (table_dir / "aggregation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {"rows": len(rows), "agg_rows": len(agg_rows), "path": all_csv}


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and submit online_replay_h1h0_final.py as a SLURM array.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = p.add_argument

    # Experiment sweep axes
    add("--seeds", nargs="+", type=int, default=[42], help="Random seeds to sweep.")
    add("--target-fprs", nargs="+", type=float, default=[0.05], help="Target FPR budgets.")
    add("--selections", nargs="+", default=["uniform"],
        choices=["uniform", "cluster_reuse", "mixed"], help="Stream selection modes.")
    add("--scorers-list", nargs="+", default=[FULL_ENSEMBLE],
        help="Scorer combos (comma-separated each). Can be given multiple times.")
    add("--modes", nargs="+", default=["compare"],
        choices=["compare", "retrieved_anchor_diagnostics"],
        help="Experiment modes to run.")

    # Per-task experiment parameters
    add("--npz", default=str(DEFAULT_NPZ), help="Path to the h1h0 NPZ dataset.")
    add("--max-requests", type=int, default=40000)
    add("--cache-size", type=int, default=40000)
    add("--warmup-requests", type=int, default=4000,
        help="Warm-up requests before the measured stream. 0=disabled.")
    add("--progress-every", type=int, default=100)
    add("--top-k", type=int, default=5)
    add("--batch-size", type=int, default=20)
    add("--min-train-h0", type=int, default=40)
    add("--min-train-h1", type=int, default=10)
    add("--min-calib-h0", type=int, default=20)
    add("--min-calib-h1", type=int, default=5)
    add("--shared-cache", action="store_true",
        help="Pass --shared-cache to experiment (fair A/B retrieval pool).")
    add("--use-wilson", action="store_true",
        help="Pass --use-wilson to experiment (Wilson upper-bound activation gate).")
    add("--persist", action="store_true",
        help="Pass --persist to experiment (file-backed state, slower).")

    # Suite presets
    add("--suite", choices=["smoke", "paper"], default="paper",
        help="smoke: 1 seed, fpr=0.05, 300 requests. paper: default axes.")

    # Output
    add("--exp-root", default=None,
        help="Experiment root directory. Default: stable runs/replay_<suite>_<config-hash>.")

    # Python / conda
    add("--conda-env", default=os.environ.get("CONDA_ENV_NAME", "mlcache"),
        help="Conda environment name (used when --python-bin is not set).")
    add("--python-bin", default=str(REPO_ROOT / ".conda" / "bin" / "python"),
        help="Explicit python executable. Overrides --conda-env.")

    # SLURM
    add("--partition", default="", help="SLURM partition.")
    add("--time", default="48:00:00", help="SLURM time limit.")
    add("--cpus-per-task", default="4", help="SLURM CPUs per task.")
    add("--mem", default="16G", help="SLURM memory per task.")
    add("--gpus", default="rtx_4090:1", help="SLURM GPU request (e.g. rtx_4090:1).")
    add("--account", default="", help="SLURM account.")
    add("--qos", default="", help="SLURM QoS.")
    add("--max-parallel", type=int, default=0,
        help="Max simultaneous array jobs; 0 = uncapped.")
    add("--job-name", default="mlcache_llm_replay", help="SLURM job name.")

    # vLLM server (optional; embed health-check + auto-start in job.sh when set)
    add("--vllm-base-url", default="http://localhost:8000/v1",
        help="vLLM server base URL (e.g. http://localhost:8000/v1). "
             "When non-empty, each job.sh checks liveness and starts the server if needed.")
    add("--vllm-model", default="Qwen/Qwen3-8B-AWQ", help="vLLM model name.")
    add("--vllm-api-key", default="local-vllm-token", help="vLLM API key.")
    add("--vllm-port", type=int, default=8000,
        help="Port to use when starting the vLLM server.")
    add("--vllm-startup-timeout", type=int, default=600,
        help="Seconds to wait for vLLM to become ready before aborting.")
    add("--llm-timeout-seconds", type=float, default=1800.0,
        help="Per-request timeout passed to the replay's local vLLM client.")
    add("--llm-max-retries", type=int, default=0,
        help="OpenAI client retries after a failed vLLM request.")

    # Control flow
    add("--dry-run", action="store_true",
        help="Print task matrix and sample command; do not write files or submit.")
    add("--submit", action="store_true", help="Submit with sbatch after writing files.")
    add("--no-wait", dest="wait", action="store_false",
        help="Do not wait for the submitted array to finish.")
    add("--poll-interval", type=float, default=60.0,
        help="Seconds between SLURM status polls while waiting.")
    add("--wait-timeout", type=float, default=0.0,
        help="Max wait seconds; 0 = no timeout.")
    add("--force", action="store_true",
        help="Re-run tasks even when _COMPLETE exists.")
    add("--skip-aggregate", action="store_true",
        help="Do not build aggregate tables after jobs finish.")
    add("--allow-partial-aggregate", action="store_true",
        help="Build aggregate tables even if some tasks are still incomplete.")
    add("--aggregate-only", action="store_true",
        help="Skip submission; only build aggregate tables from an existing --exp-root.")

    return p.parse_args(argv)


def _apply_suite(args: argparse.Namespace) -> None:
    if args.suite == "smoke":
        args.seeds = [0]
        args.target_fprs = [0.05]
        args.max_requests = 300
        args.cache_size = 300
        args.warmup_requests = 0


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_suite(args)

    exp_root = Path(args.exp_root) if args.exp_root else _default_exp_root(args)

    extra_flags: list[str] = []
    if args.shared_cache:
        extra_flags.append("--shared-cache")
    if args.use_wilson:
        extra_flags.append("--use-wilson")
    if args.persist:
        extra_flags.append("--persist")
    extra_flags.extend([
        "--llm-timeout-seconds", str(args.llm_timeout_seconds),
        "--llm-max-retries", str(args.llm_max_retries),
    ])

    all_tasks = build_replay_suite(
        exp_root,
        seeds=args.seeds,
        target_fprs=args.target_fprs,
        selections=args.selections,
        scorers_list=args.scorers_list,
        modes=args.modes,
        max_requests=args.max_requests,
        cache_size=args.cache_size,
        warmup_requests=args.warmup_requests,
        progress_every=args.progress_every,
        top_k=args.top_k,
        batch_size=args.batch_size,
        min_train_h0=args.min_train_h0,
        min_train_h1=args.min_train_h1,
        min_calib_h0=args.min_calib_h0,
        min_calib_h1=args.min_calib_h1,
        npz_path=Path(args.npz),
        extra_flags=tuple(extra_flags),
    )

    if args.aggregate_only:
        try:
            result = write_aggregate_tables(exp_root, all_tasks, allow_partial=True)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Aggregated {result['rows']} trial rows -> {result['agg_rows']} aggregate rows.")
        print(f"Table: {result['path']}")
        return 0

    remaining = all_tasks if args.force else [t for t in all_tasks if not is_complete(t)]
    jobs = group_tasks_into_jobs(remaining, exp_root)

    print(f"Total tasks: {len(all_tasks)}  |  Remaining: {len(remaining)}  |  Jobs: {len(jobs)}")

    if args.dry_run:
        print("\njob_id\tmode\ttarget_fpr\tselection\tscorers\tseed_count\tseeds")
        for job in jobs:
            first = job.tasks[0]
            print(f"{job.job_id}\t{first.mode}\t{first.target_fpr:.4g}\t"
                  f"{first.selection}\t{first.scorers}\t{len(job.tasks)}\t{_job_seed_text(job)}")
        if jobs:
            python_bin = args.python_bin or "python"
            print(f"\nSample command (job 1, seed {jobs[0].tasks[0].seed}):")
            print(task_to_command_line(python_bin, jobs[0].tasks[0]))
        return 0

    python_bin = _resolve_python(args.conda_env, args.python_bin)
    print(f"Python: {python_bin}")

    if not remaining:
        print("Nothing to do — all tasks already complete.")
        if not args.skip_aggregate:
            try:
                result = write_aggregate_tables(
                    exp_root, all_tasks,
                    allow_partial=args.allow_partial_aggregate,
                )
                print(f"Aggregated {result['rows']} rows -> {result['path']}")
            except RuntimeError as exc:
                print(f"Aggregation skipped: {exc}", file=sys.stderr)
        return 0

    vllm_cfg: dict | None = None
    if getattr(args, "vllm_base_url", ""):
        vllm_cfg = {
            "base_url": args.vllm_base_url,
            "model": args.vllm_model,
            "api_key": args.vllm_api_key,
            "port": args.vllm_port,
            "startup_timeout": args.vllm_startup_timeout,
        }
        print(f"vLLM guard enabled: base_url={vllm_cfg['base_url']}  model={vllm_cfg['model']}")

    commands_file = write_files(exp_root, jobs, python_bin, args, vllm_cfg=vllm_cfg)
    print(f"Experiment root: {exp_root}")
    print(f"Commands file:   {commands_file}")

    sbatch = _sbatch_command(exp_root, commands_file, len(jobs), args, python_bin)
    print("Sbatch command:")
    print("  " + " ".join(shlex.quote(p) for p in sbatch))

    if not args.submit:
        print("(Pass --submit to submit.)")
        return 0

    job_id = submit_sbatch(sbatch)
    print(f"Submitted SLURM job {job_id}.")

    if not args.wait:
        print(
            f"Re-run with --exp-root {shlex.quote(str(exp_root))} "
            "after completion to aggregate results."
        )
        return 0

    wait_for_slurm_job(job_id, poll_interval=args.poll_interval, timeout=args.wait_timeout)

    if args.skip_aggregate:
        return 0

    try:
        result = write_aggregate_tables(
            exp_root, all_tasks,
            allow_partial=args.allow_partial_aggregate,
        )
        print(f"Aggregated {result['rows']} trial rows -> {result['agg_rows']} aggregate rows.")
        print(f"Table: {result['path']}")
    except RuntimeError as exc:
        print(f"ERROR during aggregation: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
