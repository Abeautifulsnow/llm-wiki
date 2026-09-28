-- 0005_cache: LLM response cache (PRD §28) and per-build page identity
-- (PRD §45), both required by the §37.3 rebuild-determinism gate.
--
-- This migration rebuilds wiki_pages/page_citations/page_links following the
-- documented SQLite table-rebuild procedure: the migration runner disables
-- foreign-key enforcement for the duration (PRAGMA foreign_keys is a no-op
-- inside a transaction) and re-enables it with a foreign_key_check after.

-- Every cache row records the full key material it was derived from — model,
-- prompt/schema/parser versions, effective config hash and the source
-- snapshot hash — so cross-version reuse is impossible (PRD §28).
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key            TEXT PRIMARY KEY,
    task_type            TEXT NOT NULL,
    model                TEXT NOT NULL,
    prompt_version       TEXT NOT NULL,
    schema_version       TEXT NOT NULL,
    parser_version       TEXT NOT NULL,
    config_hash          TEXT NOT NULL,
    source_snapshot_hash TEXT NOT NULL,
    response             TEXT NOT NULL,
    created_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_cache_task ON llm_cache(task_type);

-- Move the old generation tables aside; their rows are copied into the
-- rebuilt schemas below.
ALTER TABLE page_citations RENAME TO page_citations_old;
ALTER TABLE page_links RENAME TO page_links_old;

-- PRD §45: WikiPageId is created by the planner and PERSISTED across rebuilds
-- (§28 plan-identity cache), so the same page id legitimately reappears in
-- the next generation. Page rows are therefore scoped per build:
-- PRIMARY KEY (build_id, page_id) replaces the global page_id primary key.
-- knowledge_refs_json (the nodes a page was compiled from) joins the rebuilt
-- table so the §36 orphan check and search need no plan replay.
CREATE TABLE wiki_pages_new (
    page_id    TEXT NOT NULL,
    build_id   TEXT NOT NULL,
    slug       TEXT NOT NULL,
    title      TEXT NOT NULL,
    category   TEXT NOT NULL,
    language   TEXT NOT NULL DEFAULT 'und',
    body_hash  TEXT NOT NULL,
    content    TEXT NOT NULL,
    knowledge_refs_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    PRIMARY KEY (build_id, page_id),
    UNIQUE (build_id, slug)
);

INSERT INTO wiki_pages_new
    (page_id, build_id, slug, title, category, language, body_hash, content, knowledge_refs_json, created_at)
SELECT page_id, build_id, slug, title, category, language, body_hash, content, '[]', created_at
FROM wiki_pages;

DROP TABLE wiki_pages;
ALTER TABLE wiki_pages_new RENAME TO wiki_pages;
CREATE INDEX IF NOT EXISTS idx_wiki_pages_build ON wiki_pages(build_id);

-- Citation/link rows gain build_id (scoped to their generation) and composite
-- foreign keys so several generations can carry the same persisted page id
-- and rows cascade with their page.
CREATE TABLE page_citations (
    citation_id     TEXT PRIMARY KEY,
    build_id        TEXT NOT NULL,
    page_id         TEXT NOT NULL,
    claim_node_id   TEXT REFERENCES knowledge_registry(id),
    source_id       TEXT REFERENCES sources(source_id),
    section_id      TEXT REFERENCES source_sections(section_id),
    range_start     INTEGER NOT NULL,
    range_end       INTEGER NOT NULL,
    source_hash     TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    heading_path_json TEXT NOT NULL,
    FOREIGN KEY (build_id, page_id)
        REFERENCES wiki_pages(build_id, page_id) ON DELETE CASCADE
);

INSERT INTO page_citations
    (citation_id, build_id, page_id, claim_node_id, source_id, section_id,
     range_start, range_end, source_hash, evidence_digest, heading_path_json)
SELECT c.citation_id, p.build_id, c.page_id, c.claim_node_id, c.source_id, c.section_id,
       c.range_start, c.range_end, c.source_hash, c.evidence_digest, c.heading_path_json
FROM page_citations_old AS c
JOIN wiki_pages AS p ON p.page_id = c.page_id;

DROP TABLE page_citations_old;
CREATE INDEX IF NOT EXISTS idx_page_citations_page ON page_citations(page_id);
CREATE INDEX IF NOT EXISTS idx_page_citations_build ON page_citations(build_id);

CREATE TABLE page_links (
    link_id      TEXT PRIMARY KEY,
    build_id     TEXT NOT NULL,
    from_page_id TEXT NOT NULL,
    to_page_id   TEXT NOT NULL,
    target_title TEXT NOT NULL,
    FOREIGN KEY (build_id, from_page_id)
        REFERENCES wiki_pages(build_id, page_id) ON DELETE CASCADE,
    FOREIGN KEY (build_id, to_page_id)
        REFERENCES wiki_pages(build_id, page_id)
);

INSERT INTO page_links (link_id, build_id, from_page_id, to_page_id, target_title)
SELECT l.link_id, p.build_id, l.from_page_id, l.to_page_id, l.target_title
FROM page_links_old AS l
JOIN wiki_pages AS p ON p.page_id = l.from_page_id;

DROP TABLE page_links_old;
CREATE INDEX IF NOT EXISTS idx_page_links_from ON page_links(from_page_id);
CREATE INDEX IF NOT EXISTS idx_page_links_build ON page_links(build_id);
