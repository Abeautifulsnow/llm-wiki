//! Minimal Wiki Graph (PRD §17): `graph_nodes` / `graph_edges` over the
//! ACTIVE generation, rebuilt inside the §35 publish commit point the same
//! way the FTS index is (see `search_index`).
//!
//! Node vocabulary (§17 — exactly three types):
//! - `wiki_page` — one per page of the ACTIVE generation, label = page title.
//! - `entity` / `concept` — every ACTIVE `knowledge_registry` node of that
//!   kind, label = `canonical_name` falling back to `canonical_key`. Claims
//!   and topics are provenance-anchored facts, not §17 graph nodes.
//!
//! Node ids form the composite namespace `{kind}:{typed_id}` (`page:wp_…`,
//! `entity:kn_…`) so the three id spaces cannot collide inside the single
//! TEXT primary key the PRD specifies.
//!
//! Edges: `links_to` from the ACTIVE build's `page_links`, plus every ACTIVE
//! analysis `relations.relation_type` carried through VERBATIM between
//! graph-present endpoints. Endpoints outside the §17 vocabulary (claims,
//! topics) are COUNTED as skipped in [`GraphStats`] — filtered, never
//! silently mixed in. `defined_in` has no analysis-side producer yet and is
//! deliberately absent rather than fabricated.
//!
//! The rebuild is NOT config-gated: `config.search.graph` gates the
//! query-side consumption only (`llm-wiki-search::SqliteGraphExploration` →
//! CLI `search`), while the graph itself always matches the active
//! generation so recovery can verify it the way it verifies the FTS index.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{params, Connection, Transaction};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, WikiPageId};

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

pub const NODE_TYPE_PAGE: &str = "wiki_page";
pub const NODE_TYPE_ENTITY: &str = "entity";
pub const NODE_TYPE_CONCEPT: &str = "concept";
pub const RELATION_LINKS_TO: &str = "links_to";

/// §22 graph-expansion defaults: hybrid retrieval expands at most one hop
/// and at most ten nodes ("防止上下文爆炸"). The query-side expansion uses
/// the same cap until the Context Builder (§24) owns its own budget.
pub const EXPAND_MAX_NODES: usize = 10;

/// The graph node id of a wiki page: the composite `page:{page_id}`
/// namespace (module docs).
pub fn page_node_id(page_id: &WikiPageId) -> String {
    format!("page:{}", page_id.as_str())
}

/// Population counters — `skipped_edges` makes relation filtering audible
/// instead of silent (edges whose endpoints are not §17 node types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphStats {
    pub nodes: usize,
    pub edges: usize,
    pub skipped_edges: usize,
}

/// One 1-hop neighbor of a graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNeighbor {
    /// Graph node id (`page:wp_…`, `entity:kn_…`).
    pub node_id: String,
    pub node_type: String,
    pub label: String,
    /// Edge relation type, carried through verbatim (`links_to`, `uses`, …).
    pub relation: String,
    pub direction: NeighborDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborDirection {
    /// The expanded node holds the edge's `source_id`.
    Outgoing,
    /// The expanded node holds the edge's `target_id`.
    Incoming,
}

/// Rebuilds the graph over the ACTIVE generation: clears both tables and
/// re-inserts pages, `links_to` edges, active entity/concept registry nodes
/// and active relation edges. MUST run inside the §35 activate transaction
/// (`activate_build_with_search_index`) so the graph and the active pointer
/// flip atomically — a rebuild failure aborts the publish. Idempotent.
pub fn rebuild_graph(tx: &Transaction, build_id: &BuildId) -> Result<GraphStats> {
    tx.execute("DELETE FROM graph_edges", []).map_err(db)?;
    tx.execute("DELETE FROM graph_nodes", []).map_err(db)?;

    let mut nodes = 0usize;
    let mut edges = 0usize;

    nodes += insert_page_nodes(tx, build_id)?;
    let registry_nodes = insert_registry_nodes(tx)?;
    nodes += registry_nodes.len();
    edges += insert_links_to_edges(tx, build_id)?;
    let skipped_edges = insert_relation_edges(tx, &registry_nodes, &mut edges)?;

    Ok(GraphStats {
        nodes,
        edges,
        skipped_edges,
    })
}

/// Phase 1: one `wiki_page` node per page of the activated generation,
/// label = page title. Returns the number of nodes inserted.
fn insert_page_nodes(tx: &Transaction, build_id: &BuildId) -> Result<usize> {
    let mut stmt = tx
        .prepare("SELECT page_id, title FROM wiki_pages WHERE build_id = ?1 ORDER BY page_id")
        .map_err(|e| WikiError::Storage(format!("prepare graph pages: {e}")))?;
    let pages: Vec<(String, String)> = stmt
        .query_map(params![build_id.as_str()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| WikiError::Storage(format!("graph pages: {e}")))?
        .collect::<std::result::Result<_, _>>()
        .map_err(db)?;
    drop(stmt);
    let mut insert_node = tx
        .prepare("INSERT INTO graph_nodes (id, node_type, label) VALUES (?1, ?2, ?3)")
        .map_err(|e| WikiError::Storage(format!("prepare graph node insert: {e}")))?;
    let mut inserted = 0usize;
    for (page_id, title) in pages {
        insert_node
            .execute(params![format!("page:{page_id}"), NODE_TYPE_PAGE, title])
            .map_err(db)?;
        inserted += 1;
    }
    Ok(inserted)
}

/// Phase 2: one node per ACTIVE `entity`/`concept` registry entry (§17 node
/// vocabulary; claims/topics never enter). Returns the map from RAW registry
/// id to graph node id — a registry id carries no kind, the graph node does,
/// and this map is what relation endpoints are resolved against.
fn insert_registry_nodes(tx: &Transaction) -> Result<BTreeMap<String, String>> {
    let mut stmt = tx
        .prepare(
            "SELECT id, node_kind, canonical_name, canonical_key FROM knowledge_registry
             WHERE status = 'active' AND node_kind IN ('entity', 'concept')
             ORDER BY id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph registry nodes: {e}")))?;
    let registry_rows: Vec<(String, String, Option<String>, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("graph registry nodes: {e}")))?
        .collect::<std::result::Result<_, _>>()
        .map_err(db)?;
    drop(stmt);
    let mut insert_node = tx
        .prepare("INSERT INTO graph_nodes (id, node_type, label) VALUES (?1, ?2, ?3)")
        .map_err(|e| WikiError::Storage(format!("prepare graph node insert: {e}")))?;
    let mut registry_nodes = BTreeMap::new();
    for (id, kind, canonical_name, canonical_key) in registry_rows {
        let label = canonical_name
            .filter(|name| !name.trim().is_empty())
            .unwrap_or(canonical_key);
        let node_id = format!("{kind}:{id}");
        insert_node
            .execute(params![node_id, kind, label])
            .map_err(db)?;
        registry_nodes.insert(id, node_id);
    }
    Ok(registry_nodes)
}

/// Phase 3: one `links_to` edge per WikiLink of the activated build.
/// `persist_generation` guarantees links reference sibling pages of their own
/// build, so both endpoints are already graph nodes. Returns edges inserted.
fn insert_links_to_edges(tx: &Transaction, build_id: &BuildId) -> Result<usize> {
    let mut stmt = tx
        .prepare(
            "SELECT link_id, from_page_id, to_page_id FROM page_links
             WHERE build_id = ?1 ORDER BY link_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph links: {e}")))?;
    let links: Vec<(String, String, String)> = stmt
        .query_map(params![build_id.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("graph links: {e}")))?
        .collect::<std::result::Result<_, _>>()
        .map_err(db)?;
    drop(stmt);
    let mut insert_edge = tx
        .prepare(
            "INSERT INTO graph_edges (id, source_id, relation_type, target_id)
             VALUES (?1, ?2, ?3, ?4)",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph edge insert: {e}")))?;
    let mut inserted = 0usize;
    for (link_id, from_page, to_page) in links {
        insert_edge
            .execute(params![
                format!("link:{link_id}"),
                format!("page:{from_page}"),
                RELATION_LINKS_TO,
                format!("page:{to_page}"),
            ])
            .map_err(db)?;
        inserted += 1;
    }
    Ok(inserted)
}

/// Phase 4: active analysis relations, carried through verbatim BETWEEN
/// graph-present endpoints. Endpoints outside the §17 vocabulary (claims,
/// topics, non-active nodes) are skipped and COUNTED — auditable in
/// `GraphStats.skipped_edges`, never silently mixed in. Returns the skip
/// count; inserted edges accumulate into the caller's total.
fn insert_relation_edges(
    tx: &Transaction,
    registry_nodes: &BTreeMap<String, String>,
    edges: &mut usize,
) -> Result<usize> {
    let mut stmt = tx
        .prepare(
            "SELECT relation_id, source_node_id, relation_type, target_node_id FROM relations
             WHERE status = 'active' ORDER BY relation_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph relations: {e}")))?;
    let relations: Vec<(String, String, String, String)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("graph relations: {e}")))?
        .collect::<std::result::Result<_, _>>()
        .map_err(db)?;
    drop(stmt);
    let mut insert_edge = tx
        .prepare(
            "INSERT INTO graph_edges (id, source_id, relation_type, target_id)
             VALUES (?1, ?2, ?3, ?4)",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph edge insert: {e}")))?;
    let mut skipped = 0usize;
    for (relation_id, source, relation_type, target) in relations {
        match (registry_nodes.get(&source), registry_nodes.get(&target)) {
            (Some(source_node), Some(target_node)) => {
                insert_edge
                    .execute(params![
                        format!("rel:{relation_id}"),
                        source_node,
                        relation_type,
                        target_node,
                    ])
                    .map_err(db)?;
                *edges += 1;
            }
            _ => skipped += 1,
        }
    }
    Ok(skipped)
}

// ---------------------------------------------------------------------------
// Recovery-side verification
// ---------------------------------------------------------------------------

/// Verifies that the graph covers exactly the ACTIVE generation's pages and
/// rebuilds on drift (idempotent). Mirrors
/// `search_index::ensure_search_index_matches_active`; called at the end of
/// publish recovery. `Ok(None)` means the graph already matched.
pub fn ensure_graph_matches_active(conn: &mut Connection) -> Result<Option<GraphStats>> {
    let Some(active) = crate::state::get_active_build_id(conn)? else {
        if graph_has_rows(conn)? {
            let tx = conn
                .transaction()
                .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
            tx.execute("DELETE FROM graph_edges", []).map_err(db)?;
            tx.execute("DELETE FROM graph_nodes", []).map_err(db)?;
            tx.commit()
                .map_err(|e| WikiError::Storage(format!("commit clear graph: {e}")))?;
            tracing::warn!("cleared the wiki graph: no generation is active");
        }
        return Ok(None);
    };
    if graph_page_ids(conn)? == active_page_ids(conn, &active)? {
        return Ok(None);
    }
    tracing::warn!(
        build = %active,
        "wiki graph does not match the active generation; rebuilding"
    );
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let stats = rebuild_graph(&tx, &active)?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit graph rebuild: {e}")))?;
    Ok(Some(stats))
}

fn graph_has_rows(conn: &Connection) -> Result<bool> {
    Ok(conn
        .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(db)?
        > 0)
}

fn graph_page_ids(conn: &Connection) -> Result<BTreeSet<String>> {
    let mut stmt = conn
        .prepare("SELECT id FROM graph_nodes WHERE node_type = 'wiki_page'")
        .map_err(|e| WikiError::Storage(format!("prepare graph page ids: {e}")))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| WikiError::Storage(format!("graph page ids: {e}")))?;
    let mut set = BTreeSet::new();
    for row in rows {
        let node_id = row.map_err(db)?;
        // Strip the `page:` namespace prefix to compare raw page ids;
        // strip_prefix (not slicing) so a malformed id is skipped, not a panic.
        if let Some(page_id) = node_id.strip_prefix("page:") {
            set.insert(page_id.to_owned());
        }
    }
    Ok(set)
}

fn active_page_ids(conn: &Connection, build_id: &BuildId) -> Result<BTreeSet<String>> {
    let mut stmt = conn
        .prepare("SELECT page_id FROM wiki_pages WHERE build_id = ?1")
        .map_err(|e| WikiError::Storage(format!("prepare active page ids: {e}")))?;
    let rows = stmt
        .query_map(params![build_id.as_str()], |row| row.get::<_, String>(0))
        .map_err(|e| WikiError::Storage(format!("active page ids: {e}")))?;
    let mut set = BTreeSet::new();
    for row in rows {
        set.insert(row.map_err(db)?);
    }
    Ok(set)
}

// ---------------------------------------------------------------------------
// Query-side expansion (§22 limits)
// ---------------------------------------------------------------------------

/// One-hop neighbors of ANY graph node (§22: depth ≤ 1, at most
/// [`EXPAND_MAX_NODES`] nodes, "防止上下文爆炸"). Both directions are
/// returned; the result is deterministic (ordered by relation, label, id)
/// and capped AFTER the merge so the cap is a true total.
pub fn graph_expand(conn: &Connection, node_id: &str, limit: usize) -> Result<Vec<GraphNeighbor>> {
    let mut neighbors = Vec::new();
    for (direction, edge_col, node_join_col) in [
        (NeighborDirection::Outgoing, "source_id", "target_id"),
        (NeighborDirection::Incoming, "target_id", "source_id"),
    ] {
        // The two interpolated identifiers are COMPILE-TIME CONSTANTS from
        // the array above, never external input — user-controlled values only
        // ever travel as bound parameters (?1/?2), per the §42 hard rule.
        let sql = format!(
            "SELECT e.relation_type, n.id, n.node_type, n.label
             FROM graph_edges e JOIN graph_nodes n ON n.id = e.{node_join_col}
             WHERE e.{edge_col} = ?1
             ORDER BY e.relation_type, n.label, n.id
             LIMIT ?2"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| WikiError::Storage(format!("prepare graph expand: {e}")))?;
        let rows = stmt
            .query_map(params![node_id, limit as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|e| WikiError::Storage(format!("graph expand: {e}")))?;
        for row in rows {
            let (relation, nid, node_type, label) = row.map_err(db)?;
            neighbors.push((
                (relation.clone(), label.clone(), nid.clone()),
                GraphNeighbor {
                    node_id: nid,
                    node_type,
                    label,
                    relation,
                    direction,
                },
            ));
        }
    }
    // The per-direction queries share the ORDER BY, so sorting the merged
    // pairs by the same key keeps a global deterministic order before the
    // total cap truncates.
    neighbors.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(neighbors
        .into_iter()
        .take(limit)
        .map(|(_, neighbor)| neighbor)
        .collect())
}

/// One-hop neighbors of a wiki page (the common query-side entry point).
pub fn graph_expand_from_page(
    conn: &Connection,
    page_id: &WikiPageId,
    limit: usize,
) -> Result<Vec<GraphNeighbor>> {
    graph_expand(conn, &page_node_id(page_id), limit)
}

// ---------------------------------------------------------------------------
// Tests (PRD §54: same-file unit tests, storage uses open_in_memory)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::NodeDraft;
    use crate::sources::upsert_source;
    use crate::wiki::{persist_generation, PageLinkRecord, WikiPageRecord};
    use crate::{
        activate_build_with_search_index, default_tokenizer, get_active_build_id, start_build,
        BuildDraft, NodeKind,
    };
    use llm_wiki_core::hash::sha256_hex;
    use llm_wiki_core::ids::{KnowledgeNodeId, SourceLocatorKey};

    fn page(slug: &str, title: &str, links: Vec<WikiPageId>) -> WikiPageRecord {
        WikiPageRecord {
            page_id: WikiPageId::generate(),
            slug: slug.to_owned(),
            title: title.to_owned(),
            category: "concepts".into(),
            language: "en".into(),
            body_hash: sha256_hex(slug.as_bytes()),
            content: format!("# {title}\n\nbody\n"),
            knowledge_refs: Vec::new(),
            citations: Vec::new(),
            links: links
                .into_iter()
                .map(|to_page_id| PageLinkRecord {
                    to_page_id,
                    target_title: String::new(),
                })
                .collect(),
        }
    }

    /// Publishes three pages (overview →streaming, →sso) plus an
    /// entity→concept relation and one relation touching a claim (outside
    /// the §17 vocabulary, must be skipped). Returns the connection, the
    /// overview page id, the sso page id and the registry node ids.
    fn published_conn() -> (Connection, WikiPageId, WikiPageId, Vec<KnowledgeNodeId>) {
        let mut conn = crate::open_in_memory().unwrap();
        let (source_id, _) = upsert_source(
            &mut conn,
            &SourceLocatorKey::compute("ws", "plugin/arch.md"),
            "plugin/arch.md",
            "hash-1",
            10,
            None,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
             VALUES ('an_graph_test', ?1, 'completed', '2026-01-01')",
            params![source_id.as_str()],
        )
        .unwrap();

        let drafts = vec![
            NodeDraft {
                kind: NodeKind::Entity,
                canonical_key: "sso".into(),
                canonical_name: "SSO".into(),
                entity_type: None,
                description: None,
            },
            NodeDraft {
                kind: NodeKind::Concept,
                canonical_key: "single sign-on".into(),
                canonical_name: "Single Sign-On".into(),
                entity_type: None,
                description: None,
            },
            NodeDraft {
                kind: NodeKind::Claim,
                canonical_key: "claim-key".into(),
                canonical_name: "claim".into(),
                entity_type: None,
                description: None,
            },
        ];
        let ids = crate::get_or_create_batch(&mut conn, &drafts, None).unwrap();
        conn.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
             VALUES ('rel_vocabulary', 'an_graph_test', ?1, 'related_to', ?2, 'active')",
            params![ids[0].as_str(), ids[1].as_str()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
             VALUES ('rel_claim', 'an_graph_test', ?1, 'related_to', ?2, 'active')",
            params![ids[0].as_str(), ids[2].as_str()],
        )
        .unwrap();

        let p1 = page("streaming", "Streaming Processing", Vec::new());
        let p2 = page("sso", "Identity & Access", Vec::new());
        let (id1, id2) = (p1.page_id.clone(), p2.page_id.clone());
        let overview = page("overview", "Overview", vec![id1.clone(), id2.clone()]);
        let overview_id = overview.page_id.clone();

        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        persist_generation(&mut conn, &build, &[p1, p2, overview]).unwrap();
        activate_build_with_search_index(&mut conn, &build, default_tokenizer()).unwrap();
        (conn, overview_id, id2, ids)
    }

    fn counts(conn: &Connection) -> (usize, usize) {
        let nodes: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
            .unwrap();
        let edges: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0))
            .unwrap();
        (nodes as usize, edges as usize)
    }

    #[test]
    fn rebuild_populates_pages_links_and_vocabulary_nodes() {
        let (conn, _, _, _) = published_conn();
        // 3 pages + entity + concept = 5 nodes (the claim never enters).
        let (nodes, edges) = counts(&conn);
        assert_eq!(nodes, 5);
        // overview→streaming, overview→sso, entity→concept = 3 edges.
        assert_eq!(edges, 3);

        // Page nodes carry the page title as label; registry nodes the
        // canonical name.
        let entity: (String, String) = conn
            .query_row(
                "SELECT id, label FROM graph_nodes WHERE node_type = 'entity'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert!(entity.0.starts_with("entity:kn_"));
        assert_eq!(entity.1, "SSO");
    }

    #[test]
    fn rebuild_is_idempotent() {
        let (mut conn, _, _, _) = published_conn();
        let build = get_active_build_id(&conn).unwrap().unwrap();
        let mut run = || -> GraphStats {
            let tx = conn.transaction().unwrap();
            let stats = rebuild_graph(&tx, &build).unwrap();
            tx.commit().unwrap();
            stats
        };
        let first = run();
        let second = run();
        assert_eq!(first, second);
        assert_eq!(first.skipped_edges, 1, "the claim relation is counted");
    }

    #[test]
    fn non_active_registry_nodes_and_their_relations_stay_out() {
        let (mut conn, _, _, ids) = published_conn();
        // Retire the concept: a rebuild must drop its node and skip its
        // relation as an out-of-vocabulary endpoint (counted, not mixed in).
        conn.execute(
            "UPDATE knowledge_registry SET status = 'retired' WHERE id = ?1",
            params![ids[1].as_str()],
        )
        .unwrap();
        let build = get_active_build_id(&conn).unwrap().unwrap();
        let stats = {
            let tx = conn.transaction().unwrap();
            let stats = rebuild_graph(&tx, &build).unwrap();
            tx.commit().unwrap();
            stats
        };
        // 3 pages + the surviving entity = 4 nodes.
        assert_eq!(stats.nodes, 4, "the retired concept leaves the graph");
        // Only the two links_to edges remain — both relations skipped.
        assert_eq!(stats.edges, 2);
        assert_eq!(stats.skipped_edges, 2);
    }

    #[test]
    fn expand_with_zero_limit_returns_no_neighbors() {
        let (conn, overview, _, _) = published_conn();
        assert!(graph_expand_from_page(&conn, &overview, 0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn expand_returns_both_directions_deterministically_capped() {
        let (conn, overview, sso_page, ids) = published_conn();
        // overview →streaming, →sso (outgoing links only).
        let out = graph_expand_from_page(&conn, &overview, EXPAND_MAX_NODES).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out
            .iter()
            .all(|n| n.direction == NeighborDirection::Outgoing && n.relation == "links_to"));
        assert_eq!(out[0].label, "Identity & Access", "sorted by label");
        assert_eq!(out[1].label, "Streaming Processing");

        // sso has exactly one incoming link (the claim relation does not
        // touch page nodes — different graph component).
        let back = graph_expand_from_page(&conn, &sso_page, EXPAND_MAX_NODES).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].direction, NeighborDirection::Incoming);
        assert_eq!(back[0].label, "Overview");

        // Expansion is generic over node types: the entity reaches its
        // concept via the carried relation.
        let entity_node = format!("entity:{}", ids[0]);
        let concept = graph_expand(&conn, &entity_node, EXPAND_MAX_NODES).unwrap();
        assert_eq!(concept.len(), 1);
        assert_eq!(concept[0].node_type, NODE_TYPE_CONCEPT);
        assert_eq!(concept[0].label, "Single Sign-On");
        assert_eq!(concept[0].relation, "related_to");

        // The cap is a true total across both directions.
        let capped = graph_expand_from_page(&conn, &overview, 1).unwrap();
        assert_eq!(capped.len(), 1);
        assert_eq!(capped[0].label, "Identity & Access");
    }

    #[test]
    fn ensure_matches_rebuilds_on_drift_and_clears_without_active_build() {
        let (mut conn, _, _, _) = published_conn();
        // In sync: no-op.
        assert_eq!(ensure_graph_matches_active(&mut conn).unwrap(), None);

        // Drift: wipe a page node (its edges first — FK) → rebuilt to full
        // coverage.
        conn.execute("DELETE FROM graph_edges", []).unwrap();
        conn.execute(
            "DELETE FROM graph_nodes WHERE id = (
                 SELECT MIN(id) FROM graph_nodes WHERE node_type = 'wiki_page')",
            [],
        )
        .unwrap();
        let stats = ensure_graph_matches_active(&mut conn).unwrap();
        assert!(stats.is_some(), "drift triggers a rebuild");
        assert_eq!(counts(&conn).0, 5);

        // Nothing active → graph cleared so consumers see the truth.
        conn.execute(
            "DELETE FROM wiki_state WHERE key = ?1",
            params![crate::state::ACTIVE_BUILD_KEY],
        )
        .unwrap();
        assert_eq!(ensure_graph_matches_active(&mut conn).unwrap(), None);
        assert!(!graph_has_rows(&conn).unwrap());
    }
}
