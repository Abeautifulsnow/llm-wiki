-- 0008_search: full-text search staging rows (PRD §20, §5.5).
--
-- wiki_page_text holds ONE ROW PER PAGE SECTION of a published generation:
-- (page_id, build_id, slug, title, aliases_json, heading_path_json, body).
-- The rebuild (`search_index::rebuild_search_index`) deletes all rows and
-- re-inserts the ACTIVE generation's sections, tokenized by the shared
-- TextAnalyzer, INSIDE the publish/activate transaction so the index and
-- `active_build_id` flip atomically (PRD §35 step 6).
--
-- The `wiki_fts` FTS5 virtual table is deliberately NOT created here: a
-- SQLite build without FTS5 must still open the state db (scan/status/
-- doctor keep working and report the degradation). The table is created
-- lazily with `CREATE VIRTUAL TABLE IF NOT EXISTS` at index time; its
-- availability is probed with `probe_fts5` and surfaced by `doctor`.

CREATE TABLE IF NOT EXISTS wiki_page_text (
    text_id           INTEGER PRIMARY KEY,
    page_id           TEXT NOT NULL,
    build_id          TEXT NOT NULL,
    slug              TEXT NOT NULL,
    title             TEXT NOT NULL,
    aliases_json      TEXT NOT NULL DEFAULT '[]',
    heading_path_json TEXT NOT NULL DEFAULT '[]',
    body              TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_wiki_page_text_build ON wiki_page_text(build_id);
CREATE INDEX IF NOT EXISTS idx_wiki_page_text_page ON wiki_page_text(page_id);
