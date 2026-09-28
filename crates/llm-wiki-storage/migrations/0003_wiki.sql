-- 0003_wiki: compiled generation persistence (PRD §15/§16/§18).
-- Pages, their machine-parseable citation mapping and resolved WikiLinks for
-- one build. The atomic-writer slice (§35) adds generations/current.json on
-- the filesystem side; the database rows are the machine state that backs
-- lint and search.

CREATE TABLE IF NOT EXISTS wiki_pages (
    page_id    TEXT PRIMARY KEY,
    build_id   TEXT NOT NULL,
    slug       TEXT NOT NULL,
    title      TEXT NOT NULL,
    category   TEXT NOT NULL,
    language   TEXT NOT NULL DEFAULT 'und',
    body_hash  TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (build_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_wiki_pages_build ON wiki_pages(build_id);

CREATE TABLE IF NOT EXISTS page_citations (
    citation_id     TEXT PRIMARY KEY,
    page_id         TEXT NOT NULL REFERENCES wiki_pages(page_id) ON DELETE CASCADE,
    claim_node_id   TEXT REFERENCES knowledge_registry(id),
    source_id       TEXT REFERENCES sources(source_id),
    section_id      TEXT REFERENCES source_sections(section_id),
    range_start     INTEGER NOT NULL,
    range_end       INTEGER NOT NULL,
    source_hash     TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    heading_path_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_page_citations_page ON page_citations(page_id);

CREATE TABLE IF NOT EXISTS page_links (
    link_id      TEXT PRIMARY KEY,
    from_page_id TEXT NOT NULL REFERENCES wiki_pages(page_id) ON DELETE CASCADE,
    to_page_id   TEXT NOT NULL REFERENCES wiki_pages(page_id),
    target_title TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_page_links_from ON page_links(from_page_id);
