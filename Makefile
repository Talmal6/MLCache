POSTGRES_ADMIN_URL ?= postgresql://mlcache:mlcache@127.0.0.1:55432/postgres
MYSQL_ADMIN_URL ?= mysql://root:mlcache@127.0.0.1:33306/mysql

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

setup-postgres-faiss:
	python -m pip install -U pip
	python -m pip install -e ".[postgres,faiss,dev]"

setup-mysql-faiss:
	python -m pip install -U pip
	python -m pip install -e ".[mysql,faiss,dev]"

setup-sqlite-faiss:
	python -m pip install -U pip
	python -m pip install -e ".[sqlite,faiss,dev]"

test:
	python -m pytest -q

test-postgres-faiss:
	python scripts/run_postgres_faiss_integration.py --admin-database-url "$(POSTGRES_ADMIN_URL)"
	MLCACHE_TEST_ADMIN_DATABASE_URL="$(POSTGRES_ADMIN_URL)" python -m pytest -m integration

test-mysql-faiss:
	python scripts/run_mysql_faiss_integration.py --admin-database-url "$(MYSQL_ADMIN_URL)"
	MLCACHE_TEST_MYSQL_ADMIN_DATABASE_URL="$(MYSQL_ADMIN_URL)" python -m pytest tests/test_mysql_faiss_integration.py tests/test_sql_storage_contracts.py -m integration

test-sqlite-faiss:
	python scripts/run_sqlite_faiss_integration.py
	python -m pytest tests/test_sqlite_faiss_integration.py tests/test_sqlite_storage_contract.py

smoke:
	mlcache-smoke

smoke-local:
	mlcache-smoke --local-files-only

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
