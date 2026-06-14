CREATE TABLE IF NOT EXISTS faiss_id_seq (
    id INTEGER PRIMARY KEY AUTOINCREMENT
);

CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key TEXT PRIMARY KEY,
    faiss_id INTEGER NOT NULL UNIQUE,
    namespace TEXT,
    tenant_id TEXT,
    model_name TEXT,
    query_text TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    metadata TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING_INDEX','ACTIVE','TOMBSTONED','EXPIRED')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now')),
    expires_at TEXT,
    indexed_at TEXT,
    deleted_at TEXT,
    last_access_at TEXT,
    hit_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cache_entries_serving
    ON cache_entries (status, namespace, tenant_id, model_name, expires_at);

CREATE TABLE IF NOT EXISTS cache_payloads (
    cache_key TEXT PRIMARY KEY,
    content_type TEXT NOT NULL DEFAULT 'text/plain',
    response_text TEXT,
    payload_json TEXT,
    payload_bytes BLOB,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now')),
    FOREIGN KEY (cache_key) REFERENCES cache_entries(cache_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS threshold_versions (
    threshold_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scorer_name TEXT NOT NULL,
    scope TEXT NOT NULL,
    region_id TEXT,
    cluster_id TEXT,
    threshold REAL NOT NULL,
    calibrated INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now'))
);

CREATE INDEX IF NOT EXISTS idx_threshold_versions_active
    ON threshold_versions (scorer_name, scope, region_id, cluster_id, active, created_at, threshold_version_id);

CREATE TABLE IF NOT EXISTS feedback_pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    candidate_cache_key TEXT,
    label INTEGER NOT NULL,
    features_blob BLOB,
    split_name TEXT,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_pairs_label_split
    ON feedback_pairs (label, split_name, created_at, pair_id);

CREATE TABLE IF NOT EXISTS query_events (
    query_id TEXT PRIMARY KEY,
    namespace TEXT,
    tenant_id TEXT,
    model_name TEXT,
    query_text TEXT NOT NULL,
    query_embedding BLOB,
    top_k INTEGER NOT NULL DEFAULT 0,
    decision_status TEXT NOT NULL,
    accepted INTEGER NOT NULL DEFAULT 0,
    accepted_cache_key TEXT,
    score REAL,
    threshold REAL,
    scorer_name TEXT,
    threshold_version_id INTEGER,
    reason TEXT,
    evidence TEXT NOT NULL,
    latency_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now')),
    FOREIGN KEY (threshold_version_id) REFERENCES threshold_versions(threshold_version_id)
);

CREATE INDEX IF NOT EXISTS idx_query_events_created_at ON query_events (created_at);

CREATE TABLE IF NOT EXISTS query_candidates (
    query_id TEXT NOT NULL,
    candidate_rank INTEGER NOT NULL,
    cache_key TEXT,
    faiss_id INTEGER,
    vector_score REAL,
    scorer_score REAL,
    label INTEGER,
    candidate_metadata TEXT NOT NULL,
    PRIMARY KEY (query_id, candidate_rank),
    FOREIGN KEY (query_id) REFERENCES query_events(query_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS outbox_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL CHECK (event_type IN ('UPSERT_ENTRY','DELETE_ENTRY','REBUILD_SHARD')),
    cache_key TEXT,
    faiss_id INTEGER,
    shard_id TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now')),
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_outbox_events_pending ON outbox_events (processed_at, event_id);

CREATE TABLE IF NOT EXISTS faiss_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    index_type TEXT NOT NULL,
    snapshot_uri TEXT NOT NULL,
    checksum TEXT NOT NULL,
    watermark_event_id INTEGER NOT NULL,
    metadata TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now'))
);

CREATE INDEX IF NOT EXISTS idx_faiss_snapshots_active
    ON faiss_snapshots (shard_id, active, generation, snapshot_id);
