-- 0010_insights: verified insight write-back (audit FIX-020, Karpathy loop).
-- One row per query-derived synthesis that passed citation verification
-- (`llm-wiki ask --write-back`). Insights are a curated layer SEPARATE from
-- the generated wiki: the wiki stays purely source-derived (§36
-- hand-edited-file would flag anything else), while insights record what the
-- system itself synthesized — with full provenance (query, generation,
-- cited claims). Consumers of the future (semantic lint, plan-time hints)
-- read this table; nothing in the build pipeline writes to it.
--
-- citations_json holds the EXPANDED citation records (claim node id, source
-- path, heading path, range, evidence digest) captured at synthesis time —
-- self-contained even if the cited sources later change (stale-citation
-- style drift is detectable by re-checking the digests against `sources`).

CREATE TABLE IF NOT EXISTS wiki_insights (
    insight_id     TEXT PRIMARY KEY,
    build_id       TEXT NOT NULL REFERENCES builds(build_id),
    query          TEXT NOT NULL,
    answer         TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wiki_insights_build ON wiki_insights(build_id);
