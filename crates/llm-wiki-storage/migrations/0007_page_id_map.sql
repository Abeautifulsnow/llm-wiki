-- 0007_page_id_map: stable page-identity relations (PRD §45, §19.2).
-- The explicit global re-plan (`llm-wiki replan`, V0.2) diffs the fresh plan
-- against the current generation and records here how every WikiPageId
-- continues, merges, splits or retires. This is the substrate for link,
-- cache and old-URL migration consumers (V0.2+); the explicit command writes:
--
--   keep   predecessor_page_id = successor_page_id (unchanged/modified page;
--          identity preserved, page possibly recompiled)
--   merge  one row per predecessor (dominant predecessor first by ref
--          overlap, then slug); the dominant row has
--          predecessor_page_id = successor_page_id
--   split  one row per (old page, fresh successor) pair
--   retire predecessor only (successor_page_id NULL); pages absorbed by a
--          merge/split are recorded there instead
--
-- Page rows persist across builds (page_id is the wiki_pages primary key),
-- so both sides can reference wiki_pages even when the predecessor belongs
-- to a superseded generation. Successor rows are inserted after
-- persist_generation of the new build.
--
-- NOTE: the page columns intentionally carry NO foreign key. Since migration
-- 0005, wiki_pages is keyed (build_id, page_id) — a carried page repeats the
-- same page_id across builds, so page_id is not UNIQUE and a single-column
-- FK to it is a foreign-key mismatch in SQLite. Identity continuity is the
-- replan flow's responsibility (rows are written only from validated diff
-- results against persisted generations); the build FK below stays.

CREATE TABLE IF NOT EXISTS page_id_map (
    mapping_id          TEXT PRIMARY KEY,
    predecessor_page_id TEXT,
    successor_page_id   TEXT,
    kind                TEXT NOT NULL CHECK (kind IN ('merge', 'split', 'retire', 'keep')),
    build_id            TEXT NOT NULL REFERENCES builds(build_id),
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_page_id_map_predecessor ON page_id_map(predecessor_page_id);
CREATE INDEX IF NOT EXISTS idx_page_id_map_successor ON page_id_map(successor_page_id);
CREATE INDEX IF NOT EXISTS idx_page_id_map_build ON page_id_map(build_id);
