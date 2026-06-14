#!/usr/bin/env bash
# SLURM array worker for submit_replay_array.py.
# Each array index runs one job script (bash <job.sh>) containing the
# python commands for all seeds in that job.
#
# Environment variables injected by the sbatch command:
#   COMMANDS_FILE  - path to commands.txt (one "bash <job.sh>" line per job)
#   ENV_SETUP      - shell snippet that activates the conda/python environment
#   EXP_ROOT       - experiment root directory (for diagnostics)

#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

eval "$ENV_SETUP"

if [ -z "${COMMANDS_FILE:-}" ]; then
    echo "ERROR: COMMANDS_FILE is not set" >&2
    exit 1
fi

command_line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")

if [ -z "$command_line" ]; then
    echo "ERROR: no command at SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $COMMANDS_FILE" >&2
    exit 1
fi

echo "=== SLURM array task $SLURM_ARRAY_TASK_ID starting ==="
echo "Job: $command_line"
echo "Host: $(hostname)"
echo "Python: $(which python)"
date --iso-8601=seconds

eval "$command_line"
status=$?

echo "=== SLURM array task $SLURM_ARRAY_TASK_ID finished (exit=$status) ==="
date --iso-8601=seconds
exit $status
