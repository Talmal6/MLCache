#!/usr/bin/env python
"""Run the MLCache gateway (FastAPI + uvicorn) in front of `SemanticCacheSystem`.

Usage:

    python scripts/serve_mlcache_gateway.py

Configuration is read from environment variables (`VLLM_*`, `MLCACHE_*`,
`MLCACHE_API_PORT`) -- see docs/real_llm_api.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcache.api.server import create_app  # noqa: E402

app = create_app()


def main() -> None:
    import uvicorn

    port = int(os.environ.get("MLCACHE_API_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
