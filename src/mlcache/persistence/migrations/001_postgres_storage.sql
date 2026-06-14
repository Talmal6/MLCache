CREATE SEQUENCE IF NOT EXISTS cache_entries_faiss_id_seq AS BIGINT;

CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key UUID PRIMARY KEY,
    faiss_id BIGINT UNIQUE NOT NULL DEFAULT nextval('cache_entries_faiss_id_seq'),
    namespace TEXT,
    tenant_id TEXT,
    model_name TEXT,
    query_text TEXT NOT NULL,
    embedding_dim INT NOT NULL,
    embedding_blob BYTEA NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL CHECK (status IN ('PENDING_INDEX','ACTIVE','TOMBSTONED','EXPIRED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    indexed_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    last_access_at TIMESTAMPTZ,
    hit_count BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cache_payloads (
    cache_key UUID PRIMARY KEY REFERENCES cache_entries(cache_key) ON DELETE CASCADE,
    content_type TEXT NOT NULL DEFAULT 'text/plain',
    response_text TEXT,
    payload_json JSONB,
    payload_bytes BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS threshold_versions (
    threshold_version_id BIGSERIAL PRIMARY KEY,
    scorer_name TEXT NOT NULL,
    scope TEXT NOT NULL,
    region_id TEXT,
    cluster_id TEXT,
    threshold DOUBLE PRECISION NOT NULL,
    calibrated BOOLEAN NOT NULL DEFAULT true,
    active BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback_pairs (
    pair_id BIGSERIAL PRIMARY KEY,
    query_id UUID,
    candidate_cache_key UUID,
    label SMALLINT NOT NULL,
    features_blob BYTEA,
    split_name TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_events (
    query_id UUID PRIMARY KEY,
    namespace TEXT,
    tenant_id TEXT,
    model_name TEXT,
    query_text TEXT NOT NULL,
    query_embedding BYTEA,
    top_k INT NOT NULL DEFAULT 0,
    decision_status TEXT NOT NULL,
    accepted BOOLEAN NOT NULL DEFAULT false,
    accepted_cache_key UUID,
    score DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    scorer_name TEXT,
    threshold_version_id BIGINT REFERENCES threshold_versions(threshold_version_id),
    reason TEXT,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_candidates (
    query_id UUID NOT NULL REFERENCES query_events(query_id) ON DELETE CASCADE,
    candidate_rank INT NOT NULL,
    cache_key UUID,
    faiss_id BIGINT,
    vector_score DOUBLE PRECISION,
    scorer_score DOUBLE PRECISION,
    label SMALLINT,
    candidate_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (query_id, candidate_rank)
);

CREATE TABLE IF NOT EXISTS outbox_events (
    event_id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (event_type IN ('UPSERT_ENTRY','DELETE_ENTRY','REBUILD_SHARD')),
    cache_key UUID,
    faiss_id BIGINT,
    shard_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS faiss_snapshots (
    snapshot_id BIGSERIAL PRIMARY KEY,
    shard_id TEXT NOT NULL,
    generation BIGINT NOT NULL,
    index_type TEXT NOT NULL,
    snapshot_uri TEXT NOT NULL,
    checksum TEXT NOT NULL,
    watermark_event_id BIGINT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    active BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cache_entries_faiss_id ON cache_entries(faiss_id);
CREATE INDEX IF NOT EXISTS idx_cache_entries_serving
    ON cache_entries(status, namespace, tenant_id, model_name, expires_at);
CREATE INDEX IF NOT EXISTS idx_threshold_versions_active
    ON threshold_versions(scorer_name, scope, region_id, cluster_id, active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_pairs_label_split
    ON feedback_pairs(label, split_name, created_at);
CREATE INDEX IF NOT EXISTS idx_query_events_created_at ON query_events(created_at);
CREATE INDEX IF NOT EXISTS idx_outbox_events_pending
    ON outbox_events(processed_at, event_id)
    WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_faiss_snapshots_active
    ON faiss_snapshots(shard_id, active, generation DESC);
