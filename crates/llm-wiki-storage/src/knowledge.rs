//! Planner/compilation input loading (PRD §14): the `KnowledgeBase` view of
//! active registry nodes, active relations and claim citation anchors.
//!
//! Security note: every query takes parameters — no string assembly of
//! external input (PRD §42).

use std::collections::BTreeMap;

use rusqlite::Connection;

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{KnowledgeNodeId, SectionId, SourceId};
use llm_wiki_core::model::SourceRange;
use llm_wiki_core::plan::{KnowledgeBase, PlanAnchor, PlanNode, PlanRelation};

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// Loads every active knowledge node, active relation and active claim anchor
/// into the canonical planning view. Ordering is fully deterministic (sorted
/// ids), so derived cluster/cache keys are stable (PRD §14).
pub fn load_knowledge_base(conn: &Connection) -> Result<KnowledgeBase> {
    let mut nodes = load_registry_nodes(conn)?;
    attach_claim_anchors(conn, &mut nodes)?;
    let relations = load_active_relations(conn, &nodes)?;
    Ok(KnowledgeBase { nodes, relations })
}

/// Active registry nodes of every plannable kind, ordered by id.
fn load_registry_nodes(conn: &Connection) -> Result<BTreeMap<KnowledgeNodeId, PlanNode>> {
    let mut nodes: BTreeMap<KnowledgeNodeId, PlanNode> = BTreeMap::new();
    let mut stmt = conn
        .prepare(
            "SELECT id, node_kind, canonical_key, canonical_name, entity_type, description
             FROM knowledge_registry
             WHERE status = 'active' AND node_kind IN ('entity', 'concept', 'topic', 'claim')
             ORDER BY id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare nodes: {e}")))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| WikiError::Storage(format!("query nodes: {e}")))?;
    while let Some(row) = rows.next().map_err(db)? {
        let id: String = row.get("id").map_err(db)?;
        let kind: String = row.get("node_kind").map_err(db)?;
        let canonical_key: String = row.get("canonical_key").map_err(db)?;
        let canonical_name: Option<String> = row.get("canonical_name").map_err(db)?;
        let entity_type: Option<String> = row.get("entity_type").map_err(db)?;
        let description: Option<String> = row.get("description").map_err(db)?;
        let node_id = KnowledgeNodeId::from_validated(id);
        nodes.insert(
            node_id.clone(),
            PlanNode {
                id: node_id,
                kind,
                name: canonical_name.unwrap_or(canonical_key),
                entity_type,
                description,
                statement: None,
                anchors: Vec::new(),
            },
        );
    }
    Ok(nodes)
}

/// Attaches active claim statements + one anchor per citation row (PRD §12.3).
fn attach_claim_anchors(
    conn: &Connection,
    nodes: &mut BTreeMap<KnowledgeNodeId, PlanNode>,
) -> Result<()> {
    let mut stmt = conn
        .prepare(
            "SELECT cl.node_id, cl.statement, ct.source_id, s.rel_path, cl.section_id,
                    ct.heading_path_json, ct.range_start, ct.range_end, ct.source_hash, ct.evidence_digest
             FROM claims cl
             JOIN citations ct ON ct.owner_kind = 'claim' AND ct.owner_id = cl.claim_id
             JOIN sources s ON s.source_id = ct.source_id
             WHERE cl.status = 'active'
             ORDER BY cl.node_id, ct.citation_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare claim anchors: {e}")))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| WikiError::Storage(format!("query claim anchors: {e}")))?;
    while let Some(row) = rows.next().map_err(db)? {
        let node_id: String = row.get("node_id").map_err(db)?;
        let statement: String = row.get("statement").map_err(db)?;
        let source_id: String = row.get("source_id").map_err(db)?;
        let rel_path: String = row.get("rel_path").map_err(db)?;
        let section_id: Option<String> = row.get("section_id").map_err(db)?;
        let heading_path_json: String = row.get("heading_path_json").map_err(db)?;
        let range_start: i64 = row.get("range_start").map_err(db)?;
        let range_end: i64 = row.get("range_end").map_err(db)?;
        let source_hash: String = row.get("source_hash").map_err(db)?;
        let evidence_digest: String = row.get("evidence_digest").map_err(db)?;
        let Some(node) = nodes.get_mut(&KnowledgeNodeId::from_validated(node_id)) else {
            continue;
        };
        if node.statement.is_none() {
            node.statement = Some(statement);
        }
        let heading_path: Vec<String> = serde_json::from_str(&heading_path_json)
            .map_err(|e| WikiError::Storage(format!("heading path json: {e}")))?;
        node.anchors.push(PlanAnchor {
            source_id: SourceId::from_validated(source_id),
            rel_path,
            section_id: section_id.map(SectionId::from_validated),
            heading_path,
            range: SourceRange::new(range_start.max(0) as usize, range_end.max(0) as usize),
            evidence_digest,
            source_hash,
        });
    }
    Ok(())
}

/// Active relations between active nodes; endpoints that are merged or
/// retired must not silently extend a cluster.
fn load_active_relations(
    conn: &Connection,
    nodes: &BTreeMap<KnowledgeNodeId, PlanNode>,
) -> Result<Vec<PlanRelation>> {
    let mut relations = Vec::new();
    let mut stmt = conn
        .prepare(
            "SELECT source_node_id, relation_type, target_node_id
             FROM relations
             WHERE status = 'active'
             ORDER BY source_node_id, target_node_id, relation_type",
        )
        .map_err(|e| WikiError::Storage(format!("prepare relations: {e}")))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| WikiError::Storage(format!("query relations: {e}")))?;
    while let Some(row) = rows.next().map_err(db)? {
        let source: String = row.get("source_node_id").map_err(db)?;
        let relation_type: String = row.get("relation_type").map_err(db)?;
        let target: String = row.get("target_node_id").map_err(db)?;
        let source = KnowledgeNodeId::from_validated(source);
        let target = KnowledgeNodeId::from_validated(target);
        if nodes.contains_key(&source) && nodes.contains_key(&target) {
            relations.push(PlanRelation {
                source,
                relation_type,
                target,
            });
        }
    }
    Ok(relations)
}

/// Convenience lookup used by tests and the CLI: registry revision + base.
pub fn load_plan_input(conn: &Connection) -> Result<(KnowledgeBase, u64)> {
    let base = load_knowledge_base(conn)?;
    let revision = crate::registry::current_revision(conn)?;
    Ok((base, revision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;
    use crate::registry::NodeDraft;
    use crate::sources::upsert_source;
    use crate::{get_or_create_batch, NodeKind};
    use llm_wiki_core::ids::SourceLocatorKey;
    use rusqlite::params;

    #[test]
    fn knowledge_base_loads_nodes_relations_and_anchors() {
        let mut conn = open_in_memory().unwrap();
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
            "INSERT INTO source_sections (section_id, source_id, heading_path_json, heading_path_key, content_fingerprint, range_start, range_end, status)
             VALUES ('sec_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, '[\"Doc\",\"Intro\"]', 'Doc\u{1f}Intro', 'fp', 0, 10, 'active')",
            params![source_id.as_str()],
        )
        .unwrap();

        let drafts = vec![
            NodeDraft {
                kind: NodeKind::Entity,
                canonical_key: "runtime".into(),
                canonical_name: "Plugin Runtime".into(),
                entity_type: Some("component".into()),
                description: Some("hosts plugins".into()),
            },
            NodeDraft {
                kind: NodeKind::Concept,
                canonical_key: "delivery".into(),
                canonical_name: "At-Least-Once Delivery".into(),
                entity_type: None,
                description: Some("guarantee".into()),
            },
            NodeDraft {
                kind: NodeKind::Claim,
                canonical_key: "claim-key".to_owned(),
                canonical_name: "claim".into(),
                entity_type: None,
                description: None,
            },
        ];
        let ids = get_or_create_batch(&mut conn, &drafts, None).unwrap();

        conn.execute(
            "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
             VALUES ('an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'completed', '2026-01-01')",
            params![source_id.as_str()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO claims (claim_id, node_id, source_id, section_id, analysis_id, statement, evidence_digest, status)
             VALUES ('cl_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, ?2, 'sec_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'Retry happens three times.', 'digest-1', 'active')",
            params![ids[2].as_str(), source_id.as_str()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO citations (citation_id, owner_kind, owner_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json)
             VALUES ('cit_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'claim', 'cl_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'sec_01ARZ3NDEKTSV4RRFFQ69G5FAV', 4, 20, 'hash-1', 'digest-1', '[\"Doc\",\"Intro\"]')",
            params![source_id.as_str()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
             VALUES ('rel_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'implements', ?2, 'active')",
            params![ids[0].as_str(), ids[1].as_str()],
        )
        .unwrap();

        let base = load_knowledge_base(&conn).unwrap();
        assert_eq!(base.nodes.len(), 3);
        assert_eq!(base.relations.len(), 1);

        let claim = base.get(&ids[2]).unwrap();
        assert_eq!(claim.kind, "claim");
        assert_eq!(
            claim.statement.as_deref(),
            Some("Retry happens three times.")
        );
        assert_eq!(claim.anchors.len(), 1);
        assert_eq!(claim.anchors[0].rel_path, "plugin/arch.md");
        assert_eq!(claim.anchors[0].heading_path, vec!["Doc", "Intro"]);

        let (reloaded, revision) = load_plan_input(&conn).unwrap();
        assert_eq!(reloaded.nodes.len(), 3);
        assert_eq!(revision, 3, "one revision bump per created node");
    }
}
