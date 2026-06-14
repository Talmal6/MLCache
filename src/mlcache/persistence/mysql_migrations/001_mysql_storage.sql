CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key CHAR(36) PRIMARY KEY,
    faiss_id BIGINT NOT NULL AUTO_INCREMENT,
    namespace VARCHAR(255),
    tenant_id VARCHAR(255),
    model_name VARCHAR(255),
    query_text TEXT NOT NULL,
    embedding_dim INT NOT NULL,
    embedding_blob LONGBLOB NOT NULL,
    metadata JSON NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('PENDING_INDEX','ACTIVE','TOMBSTONED','EXPIRED')),
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    expires_at DATETIME(6),
    indexed_at DATETIME(6),
    deleted_at DATETIME(6),
    last_access_at DATETIME(6),
    hit_count BIGINT NOT NULL DEFAULT 0,
    UNIQUE KEY uq_cache_entries_faiss_id (faiss_id),
    KEY idx_cache_entries_serving (status, namespace, tenant_id, model_name, expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS cache_payloads (
    cache_key CHAR(36) PRIMARY KEY,
    content_type VARCHAR(255) NOT NULL DEFAULT 'text/plain',
    response_text LONGTEXT,
    payload_json JSON,
    payload_bytes LONGBLOB,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    CONSTRAINT fk_cache_payloads_entry
        FOREIGN KEY (cache_key) REFERENCES cache_entries(cache_key) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS threshold_versions (
    threshold_version_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    scorer_name VARCHAR(255) NOT NULL,
    scope VARCHAR(255) NOT NULL,
    region_id VARCHAR(255),
    cluster_id VARCHAR(255),
    threshold DOUBLE NOT NULL,
    calibrated BOOLEAN NOT NULL DEFAULT true,
    active BOOLEAN NOT NULL DEFAULT true,
    metadata JSON NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY idx_threshold_versions_active (
        scorer_name, scope, region_id, cluster_id, active, created_at, threshold_version_id
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS feedback_pairs (
    pair_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    query_id CHAR(36),
    candidate_cache_key CHAR(36),
    label SMALLINT NOT NULL,
    features_blob LONGBLOB,
    split_name VARCHAR(255),
    metadata JSON NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY idx_feedback_pairs_label_split (label, split_name, created_at, pair_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS query_events (
    query_id CHAR(36) PRIMARY KEY,
    namespace VARCHAR(255),
    tenant_id VARCHAR(255),
    model_name VARCHAR(255),
    query_text TEXT NOT NULL,
    query_embedding LONGBLOB,
    top_k INT NOT NULL DEFAULT 0,
    decision_status VARCHAR(255) NOT NULL,
    accepted BOOLEAN NOT NULL DEFAULT false,
    accepted_cache_key CHAR(36),
    score DOUBLE,
    threshold DOUBLE,
    scorer_name VARCHAR(255),
    threshold_version_id BIGINT,
    reason TEXT,
    evidence JSON NOT NULL,
    latency_ms INT,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY idx_query_events_created_at (created_at),
    CONSTRAINT fk_query_events_threshold
        FOREIGN KEY (threshold_version_id) REFERENCES threshold_versions(threshold_version_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS query_candidates (
    query_id CHAR(36) NOT NULL,
    candidate_rank INT NOT NULL,
    cache_key CHAR(36),
    faiss_id BIGINT,
    vector_score DOUBLE,
    scorer_score DOUBLE,
    label SMALLINT,
    candidate_metadata JSON NOT NULL,
    PRIMARY KEY (query_id, candidate_rank),
    CONSTRAINT fk_query_candidates_event
        FOREIGN KEY (query_id) REFERENCES query_events(query_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS outbox_events (
    event_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    event_type VARCHAR(32) NOT NULL CHECK (event_type IN ('UPSERT_ENTRY','DELETE_ENTRY','REBUILD_SHARD')),
    cache_key CHAR(36),
    faiss_id BIGINT,
    shard_id VARCHAR(255),
    payload JSON NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    processed_at DATETIME(6),
    KEY idx_outbox_events_pending (processed_at, event_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS faiss_snapshots (
    snapshot_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    shard_id VARCHAR(255) NOT NULL,
    generation BIGINT NOT NULL,
    index_type VARCHAR(255) NOT NULL,
    snapshot_uri TEXT NOT NULL,
    checksum VARCHAR(255) NOT NULL,
    watermark_event_id BIGINT NOT NULL,
    metadata JSON NOT NULL,
    active BOOLEAN NOT NULL DEFAULT false,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY idx_faiss_snapshots_active (shard_id, active, generation, snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
