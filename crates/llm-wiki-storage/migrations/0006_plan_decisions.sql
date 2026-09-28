-- 0006_plan_decisions: incremental-build decision audit (PRD §19.2).
-- Every incremental judgment of a build is recorded here: the mapping outcome
-- ('local-update' | 'replan-required' | 'fast-path'), the trigger that forced
-- a REPLAN_REQUIRED ('fingerprint-changed' | 'structural-change' |
-- 'unmappable-node' | 'page-emptied' | NULL), the affected page count and a
-- human-readable note. This is the audit substrate `replan --dry-run`
-- (V0.2, next slice) reads to explain why a workspace needs a replan.
--
-- `trigger` is a SQLite keyword and is therefore quoted in every statement
-- that touches the column.

CREATE TABLE IF NOT EXISTS plan_decisions (
    decision_id    TEXT PRIMARY KEY,
    build_id       TEXT NOT NULL REFERENCES builds(build_id),
    source_id      TEXT REFERENCES sources(source_id),
    outcome        TEXT NOT NULL,
    "trigger"      TEXT,
    affected_pages INTEGER NOT NULL DEFAULT 0,
    notes          TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plan_decisions_build ON plan_decisions(build_id);
