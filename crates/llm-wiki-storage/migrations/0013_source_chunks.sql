-- 0013_source_chunks: raw-source retrieval staging rows (EPIC A PR1).
--
-- source_chunk_text holds ONE ROW PER SOURCE SEGMENT of a build: a section
-- segmented at block boundaries, carrying absolute offsets into the
-- normalized source text. It is the source-side mirror of wiki_page_text
-- (migration 0008): the rebuild (`chunks::rebuild_source_fts`) deletes all
-- `source_fts` rows and re-inserts the ACTIVE build's tokenized rows INSIDE
-- the publish/activate transaction so the index and `active_build_id` flip
-- atomically (PRD §35 step 6). Staging rows for several builds coexist,
-- keyed by (source_id, build_id) — `replace_source_chunks` swaps one pair
-- atomically.
--
-- The `source_fts` FTS5 virtual table is deliberately NOT created here: a
-- SQLite build without FTS5 must still open the state db (scan/status/
-- doctor keep working and report the degradation). The table is created
-- lazily with `CREATE VIRTUAL TABLE IF NOT EXISTS` at index time; its
-- availability is probed with `probe_fts5` and surfaced by `doctor`.
--
-- `canonical_url` (EPIC B) and `product_version` (EPIC D) are forward-
-- compatibility columns only — no producer or logic in this PR.

CREATE TABLE IF NOT EXISTS source_chunk_text (
    chunk_id          INTEGER PRIMARY KEY,
    source_id         TEXT NOT NULL,
    build_id          TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    title             TEXT NOT NULL DEFAULT '',
    heading_path_json TEXT NOT NULL DEFAULT '[]',
    locale            TEXT,
    ordinal           INTEGER NOT NULL,
    range_start       INTEGER NOT NULL,
    range_end         INTEGER NOT NULL,
    body              TEXT NOT NULL DEFAULT '',
    canonical_url     TEXT,
    product_version   TEXT
);

CREATE INDEX IF NOT EXISTS idx_source_chunk_text_build ON source_chunk_text(build_id);
CREATE INDEX IF NOT EXISTS idx_source_chunk_text_source ON source_chunk_text(source_id);
