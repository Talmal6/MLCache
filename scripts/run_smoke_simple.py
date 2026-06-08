"""Simple launcher for the MLCache smoke runner.

Usage:
    python scripts/run_smoke_simple.py [--local-files-only] [--model MODEL] [--threshold 0.75]

This file simply forwards arguments to the `mlcache.cli` main entrypoint.
"""

from __future__ import annotations

import sys

from mlcache.cli import main


def _argv_or_none() -> list[str] | None:
    # sys.argv[0] is this script; pass the rest through or None for defaults
    return sys.argv[1:] if len(sys.argv) > 1 else None


if __name__ == "__main__":
    raise SystemExit(main(_argv_or_none()))
