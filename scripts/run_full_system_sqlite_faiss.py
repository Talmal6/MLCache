"""Run the full SemanticCacheSystem end to end against SQLite + FAISS.

This is the "whole system" path: an online `SemanticCacheSystem` (embed ->
retrieve -> score -> HIT/MISS -> LLM fallback -> write-through, learning in the
background) wired to the durable SQLite storage backend as its source of truth
and FAISS as its vector index. Unlike the in-memory runtime, SQLite+FAISS makes
every write a `PENDING_INDEX` row plus an outbox event, so this driver also runs
the `FaissOutboxIndexer` to move entries `PENDING_INDEX -> ACTIVE` and add their
vectors to FAISS -- the same outbox flow the Postgres/MySQL integration uses.

Everything is fully offline and deterministic (a topic-clustered embedding stub
and a judge that labels same-topic paraphrases REUSABLE), so the system reaches
a calibrated policy and serves real cache hits without any network or API key.

    python scripts/run_full_system_sqlite_faiss.py

It prints the lifecycle batch by batch, a final report, and a snapshot of the
SQLite tables and FAISS index, then writes a JSON report under
runs/full_system_sqlite_faiss/. Exits 0 only if the system calibrated, served
cache hits, and SQLite holds ACTIVE entries indexed into FAISS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_semantic_cache_lifecycle import (  # noqa: E402
    DeterministicTopicJudge,
    TopicEmbeddingProvider,
    _topic_prompts,
)

# Must match TopicEmbeddingProvider(dimensions=...) below; FAISS needs the dim.
_EMBEDDING_DIM = 32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=420)
    parser.add_argument("--topics", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--min-h0", type=int, default=15)
    parser.add_argument("--min-h1", type=int, default=8)
    parser.add_argument("--target-fpr", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--scorer", default="cosine")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--fit-wait-secs", type=float, default=10.0)
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "full_system_sqlite_faiss"))
    return parser.parse_args(argv)


def _configure_backend_env(*, database_url: str, faiss_index_path: Path) -> None:
    os.environ.update(
        {
            "MLCACHE_STORAGE_BACKEND": "sqlite",
            "MLCACHE_VECTOR_BACKEND": "faiss",
            "MLCACHE_DATABASE_URL": database_url,
            "MLCACHE_SQLITE_DATABASE_URL": database_url,
            "MLCACHE_FAISS_DIM": str(_EMBEDDING_DIM),
            "MLCACHE_FAISS_INDEX_PATH": str(faiss_index_path),
            "MLCACHE_FAISS_METRIC": "cosine",
            "MLCACHE_FAISS_SHARD_ID": "full-system",
            "MLCACHE_TOP_K": "5",
        }
    )


def _sqlite_snapshot(database_url: str) -> dict[str, Any]:
    from mlcache.persistence.sqlite import _connect_sqlite_url

    with closing(_connect_sqlite_url(database_url)) as conn:
        def scalar(sql: str) -> int:
            return int(conn.execute(sql).fetchone()["n"])

        status_rows = conn.execute(
            "SELECT status, count(*) AS n FROM cache_entries GROUP BY status ORDER BY status"
        ).fetchall()
        label_rows = conn.execute(
            "SELECT label, count(*) AS n FROM feedback_pairs GROUP BY label ORDER BY label"
        ).fetchall()
        return {
            "cache_entries_total": scalar("SELECT count(*) AS n FROM cache_entries"),
            "cache_entries_by_status": {str(r["status"]): int(r["n"]) for r in status_rows},
            "cache_payloads_total": scalar("SELECT count(*) AS n FROM cache_payloads"),
            "query_events_total": scalar("SELECT count(*) AS n FROM query_events"),
            "query_events_accepted": scalar("SELECT count(*) AS n FROM query_events WHERE accepted = 1"),
            "query_candidates_total": scalar("SELECT count(*) AS n FROM query_candidates"),
            "feedback_pairs_total": scalar("SELECT count(*) AS n FROM feedback_pairs"),
            "feedback_pairs_by_label": {f"label_{int(r['label'])}": int(r["n"]) for r in label_rows},
            "threshold_versions_total": scalar("SELECT count(*) AS n FROM threshold_versions"),
            "threshold_versions_active": scalar("SELECT count(*) AS n FROM threshold_versions WHERE active = 1"),
            "outbox_pending": scalar("SELECT count(*) AS n FROM outbox_events WHERE processed_at IS NULL"),
            "outbox_processed": scalar("SELECT count(*) AS n FROM outbox_events WHERE processed_at IS NOT NULL"),
            "faiss_snapshots_total": scalar("SELECT count(*) AS n FROM faiss_snapshots"),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from mlcache import FaissOutboxIndexer, MockLLM, SemanticCacheSystem, SQLiteCacheRepository
    from mlcache.persistence import json_safe

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "state"
    database_url = f"sqlite:///{(output_dir / 'mlcache.db').as_posix()}"
    faiss_index_path = output_dir / "faiss.index"

    # Fresh run: clear any prior database/index so the lifecycle starts cold.
    for stale in (output_dir / "mlcache.db", output_dir / "mlcache.db-wal", output_dir / "mlcache.db-shm", faiss_index_path):
        if stale.exists():
            stale.unlink()

    _configure_backend_env(database_url=database_url, faiss_index_path=faiss_index_path)

    # Apply the durable schema once (PENDING_INDEX/ACTIVE entries, payloads,
    # outbox, threshold/feedback/query-audit tables) before the system opens it.
    SQLiteCacheRepository(database_url, shard_id="full-system").apply_migrations()

    system = SemanticCacheSystem(
        llm=MockLLM(response_template="answer for: {prompt}"),
        stream=None,
        scorer=args.scorer,
        judge=DeterministicTopicJudge(),
        embedding_provider=TopicEmbeddingProvider(dimensions=_EMBEDDING_DIM),
        target_fpr=args.target_fpr,
        top_k=args.top_k,
        root_dir=str(state_dir),
        batch_size=args.batch_size,
        min_h0=args.min_h0,
        min_h1=args.min_h1,
        persistence=True,
        parallelism=args.parallelism,
    )

    vector_store = system.cache.runtime.vector_store
    repository = vector_store.repository
    if repository is None:
        raise RuntimeError("expected a SQLite-backed FAISS vector store; storage env not applied")
    indexer = FaissOutboxIndexer(repository=repository, vector_store=vector_store)

    print(
        f"Full system on SQLite+FAISS: replaying {args.requests} requests across {args.topics} topics "
        f"(scorer={args.scorer}, batch_size={args.batch_size}, top_k={args.top_k}, target_fpr={args.target_fpr})"
    )
    print(f"  db = {database_url}")
    print(f"  faiss index = {faiss_index_path}\n")

    prompts = _topic_prompts(args.requests, topics=args.topics)
    timeline: list[dict[str, Any]] = []
    first_hit_at: int | None = None
    first_calibrated_at: int | None = None

    for index, prompt in enumerate(prompts, start=1):
        response = system.handle(prompt)
        # Drain the outbox so freshly written entries become ACTIVE and indexed
        # into FAISS, making them eligible to serve subsequent lookups.
        indexer.process_once()
        if first_hit_at is None and response.source == "cache":
            first_hit_at = index
        if first_calibrated_at is None and system.policy.calibrated:
            first_calibrated_at = index

        if index % args.batch_size == 0 or index == len(prompts):
            system._maybe_check_stopping()
            report = system.report()
            _log_batch(index, report=report)
            timeline.append({"request_index": index, "report": report})

    # The first background fit can lag the stream (lazy library imports); wait
    # briefly for it, then replay a warm-up burst so the now-active policy can
    # demonstrate hits against the entries already indexed in FAISS.
    deadline = time.monotonic() + args.fit_wait_secs
    while time.monotonic() < deadline and not system.policy.calibrated:
        time.sleep(0.1)

    if system.policy.calibrated and first_hit_at is None:
        print("\n[post-stream calibration detected -- replaying warm-up requests to verify hits]")
        warmup_prompts = _topic_prompts(50 * args.topics, topics=args.topics)
        for w_index, prompt in enumerate(warmup_prompts, start=args.requests + 1):
            response = system.handle(prompt)
            indexer.process_once()
            if first_hit_at is None and response.source == "cache":
                first_hit_at = w_index
        system._maybe_check_stopping()
        warmup_report = system.report()
        _log_batch(w_index, report=warmup_report)
        timeline.append({"request_index": w_index, "report": warmup_report, "note": "post-stream warm-up"})

    final_report = system.report()
    snapshot = vector_store.save_snapshot(faiss_index_path)
    faiss_size = vector_store.size()
    db_snapshot = _sqlite_snapshot(database_url)

    print()
    print(f"first_cache_hit_at_request   = {first_hit_at}")
    print(f"first_calibrated_at_request  = {first_calibrated_at}")
    print(f"final report                 = {final_report}")
    print(f"faiss index vectors          = {faiss_size}")
    print(f"faiss snapshot               = {snapshot.get('snapshot_uri')}")
    print("sqlite tables:")
    for key, value in db_snapshot.items():
        print(f"  {key:<28} = {value}")

    summary = {
        "args": vars(args),
        "database_url": database_url,
        "first_cache_hit_at_request": first_hit_at,
        "first_calibrated_at_request": first_calibrated_at,
        "final_report": final_report,
        "faiss_index_vectors": faiss_size,
        "faiss_snapshot": snapshot,
        "sqlite_snapshot": db_snapshot,
        "timeline": timeline,
    }
    report_path = output_dir / "full_system_report.json"
    report_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"\nWrote report to {report_path}")

    _assert_system_healthy(final_report, faiss_size=faiss_size, db_snapshot=db_snapshot)
    return summary


def _log_batch(index: int, *, report: dict[str, Any]) -> None:
    print(
        f"[req {index:>4}] hits={report['cache_hits']:<4} hit_rate={report['hit_rate']:.2f}  "
        f"trained={str(report['trained']):<5} calibrated={str(report['calibrated']):<5} "
        f"threshold={report['threshold']!r} (v{report['threshold_version']})  frozen={report['frozen']}"
    )


def _assert_system_healthy(report: dict[str, Any], *, faiss_size: int, db_snapshot: dict[str, Any]) -> None:
    failures: list[str] = []
    if not report.get("calibrated"):
        failures.append("calibrated must be True")
    threshold = report.get("threshold")
    finite = isinstance(threshold, (int, float)) and threshold == threshold and threshold not in (float("inf"), float("-inf"))
    if not finite:
        failures.append(f"finite threshold required (got {threshold!r})")
    if report.get("cache_hits", 0) <= 0:
        failures.append("cache hits > 0 required")
    if db_snapshot["cache_entries_by_status"].get("ACTIVE", 0) <= 0:
        failures.append("SQLite must hold ACTIVE cache_entries")
    if faiss_size <= 0:
        failures.append("FAISS index must contain vectors")
    if db_snapshot["query_events_total"] <= 0:
        failures.append("SQLite must hold query_events audit rows")

    print()
    if failures:
        print("[FAIL] full-system SQLite+FAISS run did not reach a healthy state:")
        for msg in failures:
            print(f"  FAIL: {msg}")
        sys.exit(1)
    print("[PASS] full system ran on SQLite + FAISS: calibrated, served cache hits, "
          "ACTIVE entries indexed into FAISS, audit rows persisted.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
