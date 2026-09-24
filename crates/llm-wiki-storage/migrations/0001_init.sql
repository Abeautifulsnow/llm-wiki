-- 0001_init: core state schema (PRD §18).
-- All statements run in one transaction by the migrator.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta (key, value) VALUES ('registry_revision', '0');

CREATE TABLE IF NOT EXISTS sources (
    source_id          TEXT PRIMARY KEY,
    locator_key        TEXT NOT NULL UNIQUE,
    rel_path           TEXT NOT NULL,
    content_hash       TEXT NOT NULL,
    size               INTEGER NOT NULL,
    status             TEXT NOT NULL DEFAULT 'active',
    first_seen_build_id TEXT,
    last_seen_build_id  TEXT
);

CREATE TABLE IF NOT EXISTS source_sections (
    section_id          TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL REFERENCES sources(source_id),
    heading_path_json   TEXT NOT NULL,
    heading_path_key    TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    range_start         INTEGER NOT NULL,
    range_end           INTEGER NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    created_build_id    TEXT,
    retired_build_id    TEXT
);

CREATE INDEX IF NOT EXISTS idx_source_sections_source
    ON source_sections(source_id);

CREATE TABLE IF NOT EXISTS knowledge_registry (
    id                TEXT PRIMARY KEY,
    node_kind         TEXT NOT NULL CHECK (node_kind IN ('entity', 'concept', 'topic', 'claim')),
    canonical_key     TEXT NOT NULL,
    canonical_name    TEXT,
    status            TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'merged', 'rejected', 'retired')),
    merged_into       TEXT REFERENCES knowledge_registry(id),
    created_build_id  TEXT,
    retired_build_id  TEXT,
    created_revision  INTEGER NOT NULL,
    UNIQUE (node_kind, canonical_key)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_registry_status
    ON knowledge_registry(status);

CREATE TABLE IF NOT EXISTS builds (
    build_id              TEXT PRIMARY KEY,
    started_at            TEXT NOT NULL,
    finished_at           TEXT,
    status                TEXT NOT NULL,
    source_snapshot_hash  TEXT,
    build_fingerprint     TEXT,
    model                 TEXT,
    prompt_version        TEXT,
    compiler_version      TEXT,
    parser_version        TEXT,
    schema_version        TEXT,
    config_hash           TEXT,
    registry_revision     INTEGER
);
