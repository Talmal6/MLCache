setup:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

setup-ml:
	python -m pip install -U pip
	python -m pip install -e ".[ml,dev]"

setup-embeddings:
	python -m pip install -U pip
	python -m pip install -e ".[embeddings,dev]"

setup-all:
	python -m pip install -U pip
	python -m pip install -e ".[ml,embeddings,dev]"

test:
	python -m pytest -q

smoke:
	mlcache-smoke

smoke-local:
	mlcache-smoke --local-files-only

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info