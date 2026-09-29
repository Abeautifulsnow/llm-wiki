-- 0009_graph: minimal Wiki Graph (PRD §17).
--
-- Nodes are the three §17 node types only:
--   wiki_page — one per page of the ACTIVE generation (label = page title)
--   entity / concept — every ACTIVE knowledge_registry node of that kind
--     (label = canonical_name, falling back to canonical_key). Claims and
--     topics are provenance-anchored facts, not §17 graph nodes.
--
-- Node ids are the composite namespace "{kind}:{typed_id}" (page:wp_…,
-- entity:kn_…) so the three id spaces cannot collide inside the single
-- TEXT primary key the PRD specifies.
--
-- Edges:
--   links_to — one per page_links row of the ACTIVE build (§16 WikiLinks)
--   every active relations.relation_type carried through VERBATIM between
--     graph-present endpoints (depends_on / uses / related_to / …; edges
--     whose endpoints are not §17 node types are counted as skipped, never
--     silently mixed in). `defined_in` has no analysis-side producer yet —
--     it is deliberately absent rather than fabricated.
--
-- Rebuild lifecycle mirrors the FTS index (migration 0008): rows are
-- deleted and re-inserted for the ACTIVE generation INSIDE the publish
-- activate transaction (§35 step 6) by `graph::rebuild_graph`, and
-- verified against the active generation on every publish recovery.

CREATE TABLE IF NOT EXISTS graph_nodes (
    id        TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    label     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id            TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY(source_id) REFERENCES graph_nodes(id),
    FOREIGN KEY(target_id) REFERENCES graph_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_relation ON graph_edges(relation_type);
