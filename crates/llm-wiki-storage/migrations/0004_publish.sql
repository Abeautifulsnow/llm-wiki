-- 0004_publish: atomic publish support (PRD §35).
-- `wiki_state` is a small key-value store for cross-build runtime state; the
-- `active_build_id` key names the generation the database considers current
-- and is switched in the SAME transaction that marks a build COMPLETED.

CREATE TABLE IF NOT EXISTS wiki_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
