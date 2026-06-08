# semantic-desider

Correctness-controlled online semantic cache contracts and runtime boundaries.

## Quickstart with real embeddings

```bash
python3.11 -m venv .venv
source .venv/bin/activate

make setup-embeddings
make smoke
```

Direct pip usage:

```bash
python -m pip install -e ".[embeddings,dev]"
mlcache-smoke
```

The first run may download the embedding model. To force local cached model usage:

```bash
mlcache-smoke --local-files-only
```

If the model is not cached and `--local-files-only` is used, the command fails with a clear error.