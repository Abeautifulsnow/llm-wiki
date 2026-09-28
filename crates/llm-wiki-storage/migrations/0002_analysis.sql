-- 0002_analysis: stage-one analysis persistence (PRD §11, §18).
-- Knowledge node details live on the registry row (entity_type/description/
-- aliases); claims are registry-anchored (node_kind='claim') so pages can
-- reference them by KnowledgeNodeId later (PRD §12.1.1).

ALTER TABLE knowledge_registry ADD COLUMN entity_type TEXT;
ALTER TABLE knowledge_registry ADD COLUMN description TEXT;
ALTER TABLE knowledge_registry ADD COLUMN aliases_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE knowledge_registry ADD COLUMN updated_build_id TEXT;

CREATE TABLE IF NOT EXISTS document_analyses (
    analysis_id          TEXT PRIMARY KEY,
    source_id            TEXT NOT NULL REFERENCES sources(source_id),
    build_id             TEXT,
    model                TEXT,
    prompt_version       TEXT,
    unit_count           INTEGER NOT NULL DEFAULT 0,
    llm_request_count    INTEGER NOT NULL DEFAULT 0,
    claim_count          INTEGER NOT NULL DEFAULT 0,
    rejected_claim_count INTEGER NOT NULL DEFAULT 0,
    rejected_relation_count INTEGER NOT NULL DEFAULT 0,
    status               TEXT NOT NULL DEFAULT 'completed',
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_analyses_source
    ON document_analyses(source_id);

CREATE TABLE IF NOT EXISTS claims (
    claim_id       TEXT PRIMARY KEY,
    node_id        TEXT NOT NULL REFERENCES knowledge_registry(id),
    source_id      TEXT NOT NULL REFERENCES sources(source_id),
    section_id     TEXT REFERENCES source_sections(section_id),
    analysis_id    TEXT NOT NULL REFERENCES document_analyses(analysis_id),
    statement      TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    confidence     REAL,
    status         TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'retired')),
    created_build_id TEXT,
    retired_build_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_id);
CREATE INDEX IF NOT EXISTS idx_claims_section ON claims(section_id);

CREATE TABLE IF NOT EXISTS citations (
    citation_id       TEXT PRIMARY KEY,
    owner_kind        TEXT NOT NULL CHECK (owner_kind IN ('claim', 'relation')),
    owner_id          TEXT NOT NULL,
    source_id         TEXT NOT NULL REFERENCES sources(source_id),
    section_id        TEXT REFERENCES source_sections(section_id),
    range_start       INTEGER NOT NULL,
    range_end         INTEGER NOT NULL,
    source_hash       TEXT NOT NULL,
    evidence_digest   TEXT NOT NULL,
    heading_path_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_citations_owner ON citations(owner_kind, owner_id);

CREATE TABLE IF NOT EXISTS relations (
    relation_id    TEXT PRIMARY KEY,
    analysis_id    TEXT NOT NULL REFERENCES document_analyses(analysis_id),
    source_node_id TEXT NOT NULL REFERENCES knowledge_registry(id),
    relation_type  TEXT NOT NULL,
    target_node_id TEXT NOT NULL REFERENCES knowledge_registry(id),
    section_id     TEXT REFERENCES source_sections(section_id),
    created_build_id TEXT,
    retired_build_id TEXT,
    status         TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'retired'))
);

CREATE INDEX IF NOT EXISTS idx_relations_analysis ON relations(analysis_id);

CREATE TABLE IF NOT EXISTS rejected_claims (
    rejected_id   TEXT PRIMARY KEY,
    analysis_id   TEXT NOT NULL REFERENCES document_analyses(analysis_id),
    source_id     TEXT NOT NULL REFERENCES sources(source_id),
    candidate_json TEXT NOT NULL,
    claimed_section_id TEXT,
    reason        TEXT NOT NULL,
    build_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_rejected_claims_analysis
    ON rejected_claims(analysis_id);
