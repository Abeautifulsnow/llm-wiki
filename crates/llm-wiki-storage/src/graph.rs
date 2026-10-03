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
//! Edges: `links_to` from the ACTIVE build's `page_links`, `contains` from
//! each ACTIVE page to the entity/concept nodes in its persisted
//! `knowledge_refs` (the page ↔ semantic bridge, audit FIX-009 — this is
//! what lets a page search hit expand into the semantic component), plus
//! every ACTIVE analysis `relations.relation_type` carried through VERBATIM
//! between graph-present endpoints. Endpoints outside the §17 vocabulary
//! (claims, topics) are COUNTED as skipped in [`GraphStats`] for relations —
//! filtered, never silently mixed in; `contains` skips claim/topic refs
//! silently because pages citing claims are the NORM, not a filter event.
//! `defined_in` has no analysis-side producer yet and is deliberately absent
//! rather than fabricated.
//!
//! The rebuild is NOT config-gated: `config.search.graph` gates the
//! query-side consumption only (`llm-wiki-search::SqliteGraphExploration` →
//! CLI `search`), while the graph itself always matches the active
//! generation so recovery can verify it the way it verifies the FTS index.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{params, Connection, Transaction};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, WikiPageId};

use crate::wiki::parse_knowledge_refs;

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

pub const NODE_TYPE_PAGE: &str = "wiki_page";
pub const NODE_TYPE_ENTITY: &str = "entity";
pub const NODE_TYPE_CONCEPT: &str = "concept";
pub const RELATION_LINKS_TO: &str = "links_to";
/// The page ↔ semantic bridge edge (audit FIX-009): page → entity/concept.
pub const RELATION_CONTAINS: &str = "contains";

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
    edges += insert_contains_edges(tx, build_id, &registry_nodes)?;
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

/// The ACTIVE `entity`/`concept` registry entries: (raw id, kind, label).
fn registry_rows(conn: &Connection) -> Result<Vec<(String, String, String)>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, node_kind, canonical_name, canonical_key FROM knowledge_registry
             WHERE status = 'active' AND node_kind IN ('entity', 'concept')
             ORDER BY id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph registry nodes: {e}")))?;
    let rows: Vec<(String, String, Option<String>, String)> = stmt
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
    Ok(rows
        .into_iter()
        .map(|(id, kind, canonical_name, canonical_key)| {
            let label = canonical_name
                .filter(|name| !name.trim().is_empty())
                .unwrap_or(canonical_key);
            (id, kind, label)
        })
        .collect())
}

/// Phase 2: one node per ACTIVE `entity`/`concept` registry entry (§17 node
/// vocabulary; claims/topics never enter). Returns the map from RAW registry
/// id to graph node id — a registry id carries no kind, the graph node does,
/// and this map is what relation endpoints are resolved against.
fn insert_registry_nodes(tx: &Transaction) -> Result<BTreeMap<String, String>> {
    let mut insert_node = tx
        .prepare("INSERT INTO graph_nodes (id, node_type, label) VALUES (?1, ?2, ?3)")
        .map_err(|e| WikiError::Storage(format!("prepare graph node insert: {e}")))?;
    let mut registry_nodes = BTreeMap::new();
    for (id, kind, label) in registry_rows(tx)? {
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

/// Phase 3.5: the page ↔ semantic bridge (audit FIX-009) — one `contains`
/// edge from each ACTIVE page to every entity/concept in its persisted
/// `knowledge_refs`. Claim/topic refs are the NORM on pages (every citation
/// is a claim), so out-of-vocabulary refs are skipped SILENTLY here — they
/// are §17 filtering by design, not relation-skip events worth counting.
/// Returns the edges inserted.
fn insert_contains_edges(
    tx: &Transaction,
    build_id: &BuildId,
    registry_nodes: &BTreeMap<String, String>,
) -> Result<usize> {
    let mut stmt = tx
        .prepare(
            "SELECT page_id, knowledge_refs_json FROM wiki_pages
             WHERE build_id = ?1 ORDER BY page_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph page refs: {e}")))?;
    let rows: Vec<(String, String)> = stmt
        .query_map(params![build_id.as_str()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| WikiError::Storage(format!("graph page refs: {e}")))?
        .collect::<std::result::Result<_, _>>()
        .map_err(db)?;
    drop(stmt);
    let mut insert_edge = tx
        .prepare(
            "INSERT INTO graph_edges (id, source_id, relation_type, target_id)
             VALUES (?1, ?2, ?3, ?4)",
        )
        .map_err(|e| WikiError::Storage(format!("prepare graph contains insert: {e}")))?;
    let mut inserted = 0usize;
    for (page_id, refs_json) in rows {
        for node_id in parse_knowledge_refs(&refs_json)? {
            // Only §17-vocabulary endpoints bridge; the page id namespace
            // keeps edge ids unique per (page, node) pair.
            let Some(target) = registry_nodes.get(node_id.as_str()) else {
                continue;
            };
            insert_edge
                .execute(params![
                    format!("contains:{page_id}:{node_id}"),
                    format!("page:{page_id}"),
                    RELATION_CONTAINS,
                    target,
                ])
                .map_err(db)?;
            inserted += 1;
        }
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

/// One page's graph-relevant signature: title, link pairs, knowledge refs.
/// Pages whose signature is unchanged between generations keep their node
/// and edges untouched (audit FIX-011) — body-only edits never touch the graph.
struct PageGraphSignature {
    title: String,
    links: BTreeSet<(String, String)>,
    refs: BTreeSet<String>,
}

/// `page_id → signature` for one build.
fn page_graph_signatures(
    conn: &Connection,
    build_id: &BuildId,
) -> Result<BTreeMap<String, PageGraphSignature>> {
    let mut sigs: BTreeMap<String, PageGraphSignature> = BTreeMap::new();
    {
        let mut stmt = conn
            .prepare("SELECT page_id, title FROM wiki_pages WHERE build_id = ?1")
            .map_err(|e| WikiError::Storage(format!("prepare sig pages: {e}")))?;
        let rows = stmt
            .query_map(params![build_id.as_str()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| WikiError::Storage(format!("sig pages: {e}")))?;
        for row in rows {
            let (page_id, title) = row.map_err(db)?;
            sigs.insert(
                page_id,
                PageGraphSignature {
                    title,
                    links: BTreeSet::new(),
                    refs: BTreeSet::new(),
                },
            );
        }
    }
    {
        let mut stmt = conn
            .prepare("SELECT from_page_id, to_page_id FROM page_links WHERE build_id = ?1")
            .map_err(|e| WikiError::Storage(format!("prepare sig links: {e}")))?;
        let rows = stmt
            .query_map(params![build_id.as_str()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| WikiError::Storage(format!("sig links: {e}")))?;
        for row in rows {
            let (from, to) = row.map_err(db)?;
            if let Some(sig) = sigs.get_mut(&from) {
                sig.links.insert((from.clone(), to));
            }
        }
    }
    {
        let mut stmt = conn
            .prepare("SELECT page_id, knowledge_refs_json FROM wiki_pages WHERE build_id = ?1")
            .map_err(|e| WikiError::Storage(format!("prepare sig refs: {e}")))?;
        let rows = stmt
            .query_map(params![build_id.as_str()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| WikiError::Storage(format!("sig refs: {e}")))?;
        for row in rows {
            let (page_id, refs_json) = row.map_err(db)?;
            let refs: BTreeSet<String> = parse_knowledge_refs(&refs_json)?
                .into_iter()
                .map(|node_id| node_id.as_str().to_owned())
                .collect();
            if let Some(sig) = sigs.get_mut(&page_id) {
                sig.refs = refs;
            }
        }
    }
    Ok(sigs)
}

/// Incremental graph update (audit FIX-011): diff `build_id` against the
/// previous ACTIVE generation and touch only what changed —
/// - pages whose graph signature (title, links, refs) changed are rebuilt
///   node-and-edges; removed pages lose node and all touching edges; added
///   pages gain theirs. Unchanged pages (including their inbound links from
///   carried siblings) are left alone;
/// - the registry node set and the active relation set are applied as
///   deltas, keyed by their stable edge ids (`rel:{relation_id}`).
///
/// `prev_build = None` (first publish) falls back to the full rebuild. Same
/// transaction contract as the rebuild (§35 activate transaction).
pub fn update_graph(
    tx: &Transaction,
    prev_build: Option<&BuildId>,
    build_id: &BuildId,
) -> Result<GraphStats> {
    let Some(prev_build) = prev_build else {
        return rebuild_graph(tx, build_id);
    };
    let prev_sigs = page_graph_signatures(tx, prev_build)?;
    let new_sigs = page_graph_signatures(tx, build_id)?;

    let mut statements = tx
        .prepare("DELETE FROM graph_edges WHERE source_id = ?1 OR target_id = ?1")
        .map_err(db)?;
    let mut delete_node = tx
        .prepare("DELETE FROM graph_nodes WHERE id = ?1")
        .map_err(db)?;
    let mut insert_node = tx
        .prepare("INSERT INTO graph_nodes (id, node_type, label) VALUES (?1, ?2, ?3)")
        .map_err(db)?;

    // Page nodes and edges for the touch set (edges before nodes — FK).
    let mut touched: Vec<(String, &PageGraphSignature)> = Vec::new();
    for (page_id, sig) in &new_sigs {
        let changed = match prev_sigs.get(page_id) {
            Some(prev_sig) => {
                prev_sig.title != sig.title
                    || prev_sig.links != sig.links
                    || prev_sig.refs != sig.refs
            }
            None => true, // added
        };
        if changed {
            let node_id = format!("page:{page_id}");
            statements.execute(params![node_id.clone()]).map_err(db)?;
            delete_node.execute(params![node_id]).map_err(db)?;
            touched.push((page_id.clone(), sig));
        }
    }
    for page_id in prev_sigs.keys() {
        if !new_sigs.contains_key(page_id) {
            let node_id = format!("page:{page_id}");
            statements.execute(params![node_id.clone()]).map_err(db)?;
            delete_node.execute(params![node_id]).map_err(db)?;
        }
    }
    for (page_id, sig) in &touched {
        insert_node
            .execute(params![
                format!("page:{page_id}"),
                NODE_TYPE_PAGE,
                sig.title
            ])
            .map_err(db)?;
    }
    drop(statements);
    drop(delete_node);
    drop(insert_node);

    // Outgoing links_to for touched pages, from the NEW build's rows. A
    // changed page's incoming links from unchanged siblings are rebuilt too:
    // the (from → to) pair query covers both directions of the touch set.
    {
        let mut select_links = tx
            .prepare(
                "SELECT link_id, from_page_id, to_page_id FROM page_links
                 WHERE build_id = ?1 AND (from_page_id = ?2 OR to_page_id = ?2)",
            )
            .map_err(db)?;
        let mut insert_edge = tx
            .prepare(
                "INSERT INTO graph_edges (id, source_id, relation_type, target_id)
                 VALUES (?1, ?2, ?3, ?4)",
            )
            .map_err(db)?;
        for (page_id, _) in &touched {
            let rows = select_links
                .query_map(params![build_id.as_str(), page_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(db)?;
            for row in rows {
                let (link_id, from, to) = row.map_err(db)?;
                insert_edge
                    .execute(params![
                        format!("link:{link_id}"),
                        format!("page:{from}"),
                        RELATION_LINKS_TO,
                        format!("page:{to}")
                    ])
                    .map_err(db)?;
            }
        }
    }

    // Registry node delta: graph currently holds the PREVIOUS active set.
    let mut registry_nodes: BTreeMap<String, String> = BTreeMap::new();
    {
        let mut existing: BTreeMap<String, String> = BTreeMap::new(); // raw id → node id
        {
            let mut select = tx
                .prepare("SELECT id FROM graph_nodes WHERE node_type IN ('entity', 'concept')")
                .map_err(db)?;
            let rows = select
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(db)?;
            for row in rows {
                let node_id: String = row.map_err(db)?;
                let raw = node_id
                    .split_once(':')
                    .map(|(_, raw)| raw.to_owned())
                    .unwrap_or_else(|| node_id.clone());
                existing.insert(raw, node_id);
            }
        }
        let active_rows = registry_rows(tx)?;
        let mut insert_node = tx
            .prepare("INSERT INTO graph_nodes (id, node_type, label) VALUES (?1, ?2, ?3)")
            .map_err(db)?;
        for (raw, kind, label) in &active_rows {
            let node_id = format!("{kind}:{raw}");
            if !existing.contains_key(raw) {
                insert_node
                    .execute(params![node_id.clone(), kind, label])
                    .map_err(db)?;
            }
            registry_nodes.insert(raw.clone(), node_id);
        }
        drop(insert_node);
        // Retired registry nodes leave the graph with their edges.
        let mut delete_edges = tx
            .prepare("DELETE FROM graph_edges WHERE source_id = ?1 OR target_id = ?1")
            .map_err(db)?;
        let mut delete_node = tx
            .prepare("DELETE FROM graph_nodes WHERE id = ?1")
            .map_err(db)?;
        for (raw, node_id) in &existing {
            if !registry_nodes.contains_key(raw) {
                delete_edges.execute(params![node_id]).map_err(db)?;
                delete_node.execute(params![node_id]).map_err(db)?;
            }
        }
    }

    // `contains` edges for touched pages (claims/topics skip silently —
    // pages citing claims are the norm, not a filter event).
    {
        let mut insert_edge = tx
            .prepare(
                "INSERT INTO graph_edges (id, source_id, relation_type, target_id)
                 VALUES (?1, ?2, ?3, ?4)",
            )
            .map_err(db)?;
        for (page_id, sig) in &touched {
            for raw in &sig.refs {
                let Some(target) = registry_nodes.get(raw) else {
                    continue;
                };
                insert_edge
                    .execute(params![
                        format!("contains:{page_id}:{raw}"),
                        format!("page:{page_id}"),
                        RELATION_CONTAINS,
                        target,
                    ])
                    .map_err(db)?;
            }
        }
    }

    // Relation delta keyed by the stable `rel:{relation_id}` edge id.
    let mut skipped_edges = 0usize;
    {
        let mut desired: BTreeMap<String, (String, String, String)> = BTreeMap::new();
        {
            let mut select = tx
                .prepare(
                    "SELECT relation_id, source_node_id, relation_type, target_node_id
                     FROM relations WHERE status = 'active' ORDER BY relation_id",
                )
                .map_err(db)?;
            let rows = select
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })
                .map_err(db)?;
            for row in rows {
                let (relation_id, source, relation_type, target) = row.map_err(db)?;
                match (registry_nodes.get(&source), registry_nodes.get(&target)) {
                    (Some(source_node), Some(target_node)) => {
                        desired.insert(
                            format!("rel:{relation_id}"),
                            (source_node.clone(), relation_type, target_node.clone()),
                        );
                    }
                    _ => skipped_edges += 1,
                }
            }
        }
        let mut existing_rel: BTreeSet<String> = BTreeSet::new();
        {
            let mut select = tx
                .prepare("SELECT id FROM graph_edges WHERE id LIKE 'rel:%'")
                .map_err(db)?;
            let rows = select
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(db)?;
            for row in rows {
                existing_rel.insert(row.map_err(db)?);
            }
        }
        let mut insert_edge = tx
            .prepare(
                "INSERT INTO graph_edges (id, source_id, relation_type, target_id)
                 VALUES (?1, ?2, ?3, ?4)",
            )
            .map_err(db)?;
        for (edge_id, (source, relation_type, target)) in &desired {
            if !existing_rel.contains(edge_id) {
                insert_edge
                    .execute(params![edge_id, source, relation_type, target])
                    .map_err(db)?;
            }
        }
        let mut delete_edge = tx
            .prepare("DELETE FROM graph_edges WHERE id = ?1")
            .map_err(db)?;
        let desired_ids: BTreeSet<String> = desired.keys().cloned().collect();
        for edge_id in existing_rel.difference(&desired_ids).collect::<Vec<_>>() {
            delete_edge.execute(params![edge_id]).map_err(db)?;
        }
    }

    // Final totals (the update's answer must equal a full rebuild's).
    let nodes: usize = tx
        .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(db)? as usize;
    let edges: usize = tx
        .query_row("SELECT COUNT(*) FROM graph_edges", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(db)? as usize;
    Ok(GraphStats {
        nodes,
        edges,
        skipped_edges,
    })
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
    use crate::open_in_memory;
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
    fn pages_bridge_into_the_semantic_component_via_contains() {
        let (conn, overview, _, ids) = published_conn();
        // Give the overview page a knowledge ref on the entity (and one on
        // the claim, which must stay OUT of the §17 graph). The fixture
        // pages carry no refs, so the rebuild adds exactly one bridge edge.
        let refs_json = format!("[\"{}\", \"{}\"]", ids[0], ids[2]);
        conn.execute(
            "UPDATE wiki_pages SET knowledge_refs_json = ?1 WHERE slug = 'overview'",
            params![refs_json],
        )
        .unwrap();
        let build = get_active_build_id(&conn).unwrap().unwrap();
        {
            let tx = conn.unchecked_transaction().unwrap();
            rebuild_graph(&tx, &build).unwrap();
            tx.commit().unwrap();
        }

        // A page search hit now expands across the bridge (audit FIX-009):
        // the two links_to neighbors + the contained entity.
        let out = graph_expand_from_page(&conn, &overview, EXPAND_MAX_NODES).unwrap();
        assert_eq!(out.len(), 3, "links_to + one contains edge: {out:?}");
        let contains: Vec<_> = out
            .iter()
            .filter(|n| n.relation == RELATION_CONTAINS)
            .collect();
        assert_eq!(contains.len(), 1);
        assert_eq!(contains[0].node_id, format!("entity:{}", ids[0]));
        assert_eq!(contains[0].node_type, NODE_TYPE_ENTITY);
        assert_eq!(contains[0].direction, NeighborDirection::Outgoing);

        // The bridge is traversable from the semantic side too: the entity
        // reaches the page as an incoming `contains` edge (plus its concept
        // relation).
        let entity_node = format!("entity:{}", ids[0]);
        let semantic = graph_expand(&conn, &entity_node, EXPAND_MAX_NODES).unwrap();
        assert!(semantic
            .iter()
            .any(|n| n.node_type == NODE_TYPE_PAGE && n.relation == RELATION_CONTAINS));

        // The claim ref produced NO edge: claims never enter the §17 graph.
        let edges: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM graph_edges WHERE relation_type = 'contains'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(edges, 1);
    }

    /// Audit FIX-011 acceptance: the incremental update path must land in
    /// EXACTLY the same graph state as a full rebuild of the same generation.
    /// The scenario covers every diff class: a carried page (untouched), a
    /// page whose title and refs change, an added page with links, a removed
    /// page, a registry node retired between generations, and a new active
    /// relation.
    #[test]
    fn incremental_graph_update_equals_full_rebuild() {
        // Registry fixture per connection: entity "sso" + concept
        // "single sign-on" (+ a claim that must never enter the graph).
        let setup = |conn: &mut Connection| -> (BuildId, BuildId, Vec<KnowledgeNodeId>) {
            let (source_id, _) = upsert_source(
                conn,
                &SourceLocatorKey::compute("ws", "eq/arch.md"),
                "eq/arch.md",
                "hash-eq",
                10,
                None,
            )
            .unwrap();
            conn.execute(
                "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
                 VALUES ('an_eq', ?1, 'completed', '2026-01-01')",
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
                    canonical_key: "eq-claim".into(),
                    canonical_name: "claim".into(),
                    entity_type: None,
                    description: None,
                },
            ];
            let ids = crate::get_or_create_batch(conn, &drafts, None).unwrap();
            // Generation A: a relation entity→concept, one claim relation
            // (skipped), two pages: p1 refs the entity, p2 links to p1 and
            // refs the claim.
            conn.execute(
                "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
                 VALUES ('rel_eq_1', 'an_eq', ?1, 'related_to', ?2, 'active')",
                params![ids[0].as_str(), ids[1].as_str()],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
                 VALUES ('rel_eq_2', 'an_eq', ?1, 'related_to', ?2, 'active')",
                params![ids[0].as_str(), ids[2].as_str()],
            )
            .unwrap();
            (
                start_build(conn, &BuildDraft::default()).unwrap(),
                start_build(conn, &BuildDraft::default()).unwrap(),
                ids,
            )
        };

        // Pages per connection (page ULIDs are connection-local; the carried
        // page keeps its id across the two builds of the SAME connection).
        let make_pages = |ids: &[KnowledgeNodeId]| {
            let page_with_refs = |page_id: Option<WikiPageId>,
                                  slug: &str,
                                  title: &str,
                                  links: &[WikiPageRecord],
                                  refs: &[usize]| WikiPageRecord {
                page_id: page_id.unwrap_or_else(WikiPageId::generate),
                slug: slug.to_owned(),
                title: title.to_owned(),
                category: "concepts".into(),
                language: "en".into(),
                body_hash: sha256_hex(slug.as_bytes()),
                content: format!("# {title}\n\nbody\n"),
                knowledge_refs: refs.iter().map(|&i| ids[i].clone()).collect(),
                citations: Vec::new(),
                links: links
                    .iter()
                    .map(|prev| PageLinkRecord {
                        to_page_id: prev.page_id.clone(),
                        target_title: prev.title.clone(),
                    })
                    .collect(),
            };
            let p1 = page_with_refs(None, "eq-one", "Eq One", &[], &[0]);
            let p2 = page_with_refs(None, "eq-two", "Eq Two", std::slice::from_ref(&p1), &[2]);
            let a_pages = vec![p1.clone(), p2.clone()];
            // Generation B: p1 carried verbatim (same id); p2 retitled + refs
            // the concept instead of the claim; p3 added linking to p1; p2's
            // old claim ref disappears with the recompile.
            let p1b = page_with_refs(Some(p1.page_id.clone()), "eq-one", "Eq One", &[], &[0]);
            let p2b = page_with_refs(None, "eq-two", "Eq Two Renamed", &[], &[1]);
            let p3 = page_with_refs(
                None,
                "eq-three",
                "Eq Three",
                std::slice::from_ref(&p1b),
                &[1],
            );
            let b_pages = vec![p1b, p2b, p3];
            (a_pages, b_pages)
        };

        // Connection 1: rebuild(A) then the INCREMENTAL update A→B. A new
        // active relation concept→entity lands in B via a second analysis row.
        let mut incremental = open_in_memory().unwrap();
        let (build_a, build_b, inc_ids) = setup(&mut incremental);
        incremental
            .execute(
                "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
                 VALUES ('rel_eq_3', 'an_eq', ?1, 'uses', ?2, 'active')",
                params![inc_ids[1].as_str(), inc_ids[0].as_str()],
            )
            .unwrap();
        let (a_pages, b_pages) = make_pages(&inc_ids);
        persist_generation(&mut incremental, &build_a, &a_pages).unwrap();
        persist_generation(&mut incremental, &build_b, &b_pages).unwrap();
        {
            let tx = incremental.transaction().unwrap();
            rebuild_graph(&tx, &build_a).unwrap();
            update_graph(&tx, Some(&build_a), &build_b).unwrap();
            tx.commit().unwrap();
        }
        // Connection 2: same fixture, full rebuild of B only.
        let mut full = open_in_memory().unwrap();
        let (build_b2, _, full_ids) = setup(&mut full);
        full.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
             VALUES ('rel_eq_3', 'an_eq', ?1, 'uses', ?2, 'active')",
            params![full_ids[1].as_str(), full_ids[0].as_str()],
        )
        .unwrap();
        let (_, full_b_pages) = make_pages(&full_ids);
        persist_generation(&mut full, &build_b2, &full_b_pages).unwrap();
        {
            let tx = full.transaction().unwrap();
            rebuild_graph(&tx, &build_b2).unwrap();
            tx.commit().unwrap();
        }

        // Registry ULIDs differ per connection, so node ids are canonicalized:
        // registry nodes get a `{type}::{label}` alias, page ids are stable
        // (the records are shared). Edges are compared by canonical endpoints
        // and relation type — the id column is connection-local (`link:…`
        // row ids are regenerated per build).
        type NodeRow = (String, String, String);
        let canonical = |conn: &Connection| -> (Vec<NodeRow>, Vec<NodeRow>) {
            let mut alias: std::collections::BTreeMap<String, String> =
                std::collections::BTreeMap::new();
            // Page nodes are connection-local ULIDs too: alias by slug.
            {
                let mut stmt = conn
                    .prepare("SELECT page_id, slug FROM wiki_pages")
                    .unwrap();
                let rows = stmt
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })
                    .unwrap();
                for row in rows {
                    let (page_id, slug) = row.unwrap();
                    alias.insert(format!("page:{page_id}"), format!("page::{slug}"));
                }
            }
            let mut nodes = Vec::new();
            {
                let mut stmt = conn
                    .prepare("SELECT id, node_type, label FROM graph_nodes ORDER BY id")
                    .unwrap();
                let rows = stmt
                    .query_map([], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    })
                    .unwrap();
                for row in rows {
                    let (id, node_type, label) = row.unwrap();
                    let key = if node_type == "wiki_page" {
                        alias[&id].clone()
                    } else {
                        format!("{node_type}::{label}")
                    };
                    alias.insert(id, key.clone());
                    nodes.push((key, node_type, label));
                }
            }
            let mut edges = Vec::new();
            {
                let mut stmt = conn
                    .prepare(
                        "SELECT source_id, relation_type, target_id FROM graph_edges
                             ORDER BY source_id, relation_type, target_id",
                    )
                    .unwrap();
                let rows = stmt
                    .query_map([], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    })
                    .unwrap();
                for row in rows {
                    let (source, relation_type, target) = row.unwrap();
                    edges.push((
                        alias[&source].clone(),
                        relation_type,
                        alias[&target].clone(),
                    ));
                }
            }
            (nodes, edges)
        };
        let (mut left_nodes, mut left_edges) = canonical(&incremental);
        let (mut right_nodes, mut right_edges) = canonical(&full);
        // Iteration order follows each connection's ULIDs — compare as sets
        // by sorting on the canonical keys.
        left_nodes.sort();
        right_nodes.sort();
        left_edges.sort();
        right_edges.sort();
        assert_eq!(
            left_nodes, right_nodes,
            "node set must equal a full rebuild"
        );
        assert_eq!(
            left_edges, right_edges,
            "edge set must equal a full rebuild"
        );
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
