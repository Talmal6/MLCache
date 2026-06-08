from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlcache import cli


ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingProvider:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", normalize: bool = True, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.local_files_only = local_files_only

    def embed(self, text: str) -> tuple[float, ...]:
        del text
        return (1.0, 0.0)


def test_cli_smoke_formats_json_with_fake_embeddings(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "SentenceTransformersEmbeddingProvider", FakeEmbeddingProvider)

    exit_code = cli.main(
        [
            "--persist",
            str(tmp_path / "state"),
            "--threshold",
            "0.75",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)

    expected_fields = {
        "response",
        "status",
        "accepted",
        "score",
        "threshold",
        "cache_key",
        "reason",
        "cached_query",
        "incoming_query",
        "embedding_model",
        "embedding_dim",
    }
    assert expected_fields.issubset(payload)
    assert payload["accepted"] is True
    assert payload["status"] == "hit"
    assert payload["cache_key"] == "demo-cache-key"
    assert payload["embedding_dim"] == 2
    assert payload["score"] is not None


@pytest.mark.integration
def test_cli_smoke_real_embeddings_subprocess() -> None:
    if os.getenv("MLCACHE_RUN_EMBEDDING_INTEGRATION") != "1":
        pytest.skip("set MLCACHE_RUN_EMBEDDING_INTEGRATION=1 to run the embedding integration smoke test")

    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else src_path + os.pathsep + existing_pythonpath

    result = subprocess.run(
        [sys.executable, "-m", "mlcache"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["embedding_dim"] > 0
    assert payload["score"] is not None
    assert payload["status"] in {"hit", "miss", "abstain"}