//! Stage-one analysis persistence (PRD §11, §18): document_analyses, claims,
//! citations, relations and rejected_claims. All writes for one analysis
//! happen in a single transaction; knowledge node identity is assigned by the
//! registry beforehand (claims are `kind='claim'` registry nodes, PRD §12.1.1).

use rusqlite::{params, Connection, Transaction};
use serde::{Deserialize, Serialize};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, ClaimRowId, KnowledgeNodeId, SectionId, SourceId};
use llm_wiki_core::model::SourceRange;

use crate::registry::bump_revision;

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// One knowledge node this source currently supports, with the section that
/// anchors it (claims: the claim's section; relations: the relation's
/// section). The incremental pipeline (PRD §19.2) uses these to map new
/// nodes onto the pages that previously owned the source's sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceNodeSection {
    pub node_id: KnowledgeNodeId,
    pub section_id: Option<SectionId>,
}

/// Distinct knowledge nodes (+ anchoring sections) this source currently
/// supports through its ACTIVE claims and relations, ordered by node id.
pub fn list_source_active_node_sections(
    conn: &Connection,
    source_id: &SourceId,
) -> Result<Vec<SourceNodeSection>> {
    let mut out: std::collections::BTreeMap<KnowledgeNodeId, Option<SectionId>> =
        std::collections::BTreeMap::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT node_id, section_id FROM claims
                 WHERE source_id = ?1 AND status = 'active' ORDER BY node_id",
            )
            .map_err(|e| WikiError::Storage(format!("prepare source claim nodes: {e}")))?;
        let rows = stmt
            .query_map(params![source_id.as_str()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .map_err(|e| WikiError::Storage(format!("source claim nodes: {e}")))?;
        for row in rows {
            let (node_id, section_id) = row.map_err(db)?;
            out.entry(KnowledgeNodeId::from_validated(node_id))
                .or_insert(section_id.map(SectionId::from_validated));
        }
    }
    {
        let mut stmt = conn
            .prepare(
                "SELECT r.source_node_id, r.section_id FROM relations r
                 JOIN document_analyses a ON a.analysis_id = r.analysis_id
                 WHERE a.source_id = ?1 AND r.status = 'active'
                 UNION
                 SELECT r.target_node_id, r.section_id FROM relations r
                 JOIN document_analyses a ON a.analysis_id = r.analysis_id
                 WHERE a.source_id = ?1 AND r.status = 'active'",
            )
            .map_err(|e| WikiError::Storage(format!("prepare source relation nodes: {e}")))?;
        let rows = stmt
            .query_map(params![source_id.as_str()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .map_err(|e| WikiError::Storage(format!("source relation nodes: {e}")))?;
        for row in rows {
            let (node_id, section_id) = row.map_err(db)?;
            out.entry(KnowledgeNodeId::from_validated(node_id))
                .or_insert(section_id.map(SectionId::from_validated));
        }
    }
    Ok(out
        .into_iter()
        .map(|(node_id, section_id)| SourceNodeSection {
            node_id,
            section_id,
        })
        .collect())
}

/// What [`retire_source_knowledge`] retired for one removed source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetiredSourceKnowledge {
    pub retired_claims: u64,
    pub retired_relations: u64,
    /// Distinct nodes the source supported (its claims + relation endpoints).
    pub affected_nodes: Vec<KnowledgeNodeId>,
    /// Registry nodes that lost ALL support and were retired (no ghost
    /// knowledge may survive a deletion, PRD §19.3).
    pub retired_registry_nodes: Vec<KnowledgeNodeId>,
}

/// Length note: ~102 lines — one §19.3 transaction (retire claims + relations + unsupported nodes) with exactly one revision bump; the parts must stay in one tx body.
/// Retires the active claims and relations of a REMOVED source without
/// replacement (PRD §19.3: deletion must leave no ghost knowledge). Registry
/// nodes that lose all supporting claims AND touching relations are retired
/// too (one revision bump per transaction). Read
/// [`list_source_active_node_sections`] BEFORE calling this if the affected
/// nodes' section anchors are still needed.
pub fn retire_source_knowledge(
    conn: &mut Connection,
    source_id: &SourceId,
    build_id: Option<&str>,
) -> Result<RetiredSourceKnowledge> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;

    let mut affected: std::collections::BTreeSet<KnowledgeNodeId> =
        std::collections::BTreeSet::new();
    {
        let mut stmt = tx
            .prepare(
                "SELECT DISTINCT node_id FROM claims
                 WHERE source_id = ?1 AND status = 'active'",
            )
            .map_err(db)?;
        let rows = stmt
            .query_map(params![source_id.as_str()], |row| row.get::<_, String>(0))
            .map_err(db)?;
        for row in rows {
            affected.insert(KnowledgeNodeId::from_validated(row.map_err(db)?));
        }
    }
    for column in ["r.source_node_id", "r.target_node_id"] {
        // Column names are compile-time constants, never user input.
        let sql = format!(
            "SELECT DISTINCT {column} FROM relations r
             JOIN document_analyses a ON a.analysis_id = r.analysis_id
             WHERE a.source_id = ?1 AND r.status = 'active'"
        );
        let mut stmt = tx.prepare(&sql).map_err(db)?;
        let rows = stmt
            .query_map(params![source_id.as_str()], |row| row.get::<_, String>(0))
            .map_err(db)?;
        for row in rows {
            affected.insert(KnowledgeNodeId::from_validated(row.map_err(db)?));
        }
    }

    let retired_claims = tx
        .execute(
            "UPDATE claims SET status = 'retired', retired_build_id = ?2
             WHERE source_id = ?1 AND status = 'active'",
            params![source_id.as_str(), build_id],
        )
        .map_err(db)?;
    let retired_relations = tx
        .execute(
            "UPDATE relations SET status = 'retired', retired_build_id = ?2
             WHERE status = 'active' AND analysis_id IN
             (SELECT analysis_id FROM document_analyses WHERE source_id = ?1)",
            params![source_id.as_str(), build_id],
        )
        .map_err(db)?;

    // Nodes left without any active claim and without any touching relation
    // are knowledge only this source supported: retire them (PRD §19.3.2).
    let mut retired_registry_nodes = Vec::new();
    for node_id in &affected {
        let active_claims: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM claims WHERE node_id = ?1 AND status = 'active'",
                params![node_id.as_str()],
                |row| row.get(0),
            )
            .map_err(db)?;
        let active_relations: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM relations
                 WHERE status = 'active' AND (source_node_id = ?1 OR target_node_id = ?1)",
                params![node_id.as_str()],
                |row| row.get(0),
            )
            .map_err(db)?;
        if active_claims == 0 && active_relations == 0 {
            let changed = tx
                .execute(
                    "UPDATE knowledge_registry SET status = 'retired', retired_build_id = ?2
                     WHERE id = ?1 AND status = 'active'",
                    params![node_id.as_str(), build_id],
                )
                .map_err(db)?;
            if changed > 0 {
                retired_registry_nodes.push(node_id.clone());
            }
        }
    }
    if !retired_registry_nodes.is_empty() {
        bump_revision(&tx)?;
    }

    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit retire source knowledge: {e}")))?;
    Ok(RetiredSourceKnowledge {
        retired_claims: retired_claims as u64,
        retired_relations: retired_relations as u64,
        affected_nodes: affected.into_iter().collect(),
        retired_registry_nodes,
    })
}

#[derive(Debug, Clone)]
pub struct AnalysisRecord {
    pub source_id: SourceId,
    pub build_id: Option<BuildId>,
    pub model: Option<String>,
    pub prompt_version: Option<String>,
    pub unit_count: u32,
    pub llm_request_count: u32,
    pub status: String,
}

/// One evidence range of a claim; each becomes a citation row (PRD §12.3).
#[derive(Debug, Clone)]
pub struct EvidenceRange {
    pub range: SourceRange,
    pub evidence_digest: String,
}

/// A verified claim to persist; `node_id` is the registry-assigned claim node.
#[derive(Debug, Clone)]
pub struct PersistedClaim {
    pub node_id: KnowledgeNodeId,
    pub section_id: Option<SectionId>,
    pub statement: String,
    pub confidence: Option<f32>,
    pub evidence_ranges: Vec<EvidenceRange>,
}

#[derive(Debug, Clone)]
pub struct PersistedRelation {
    pub source_node_id: KnowledgeNodeId,
    pub relation_type: String,
    pub target_node_id: KnowledgeNodeId,
    pub section_id: Option<SectionId>,
    /// Optional evidence quote → one citation row for the relation.
    pub evidence: Option<EvidenceRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedRejectedClaim {
    pub candidate_json: String,
    pub claimed_section_id: Option<String>,
    pub reason: String,
}

/// Everything produced by analyzing one source; persisted atomically.
#[derive(Debug, Clone)]
pub struct AnalysisPersistence {
    pub record: AnalysisRecord,
    /// Retires the source's previous active claims/relations before writing
    /// the new ones (re-analysis replaces; no ghost knowledge, PRD §19.3).
    pub replace_source: bool,
    pub claims: Vec<PersistedClaim>,
    pub relations: Vec<PersistedRelation>,
    pub rejected_claims: Vec<PersistedRejectedClaim>,
    /// heading paths per section id, for citation rows.
    pub heading_paths: Vec<(SectionId, Vec<String>)>,
    pub source_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedAnalysis {
    pub analysis_id: llm_wiki_core::ids::AnalysisId,
    pub claim_count: usize,
    pub rejected_claim_count: usize,
    pub relation_count: usize,
    pub citation_count: usize,
}

/// Persists one analysis inside a single transaction.
/// Serializes a section's heading path for citation rows (empty when the
/// section id has no recorded path).
fn heading_path_json_of(
    persistence: &AnalysisPersistence,
    section_id: &Option<SectionId>,
) -> Result<String> {
    let path = section_id
        .as_ref()
        .and_then(|sid| {
            persistence
                .heading_paths
                .iter()
                .find(|(candidate, _)| candidate == sid)
                .map(|(_, path)| path.clone())
        })
        .unwrap_or_default();
    serde_json::to_string(&path)
        .map_err(|e| WikiError::Storage(format!("serialize heading path: {e}")))
}

/// Replace-source step 1: collect the claim nodes AND relation endpoints this
/// source supported BEFORE the replacement, then retire the source's active
/// claims and relations. The nodes that end up with no active claim and no
/// touching relation are ghost knowledge (PRD §19.3) and are retired by
/// [`sweep_ghost_nodes`] AFTER the new claims and relations exist.
fn replace_previous_knowledge(
    tx: &Transaction<'_>,
    persistence: &AnalysisPersistence,
) -> Result<Vec<KnowledgeNodeId>> {
    let retired_build = persistence.record.build_id.as_ref().map(|b| b.as_str());
    let mut replaced_nodes: Vec<KnowledgeNodeId> = Vec::new();
    if !persistence.replace_source {
        return Ok(replaced_nodes);
    }
    {
        let mut stmt = tx
            .prepare(
                "SELECT DISTINCT node_id FROM claims
                 WHERE source_id = ?1 AND status = 'active'",
            )
            .map_err(db)?;
        let rows = stmt
            .query_map(params![persistence.record.source_id.as_str()], |row| {
                row.get::<_, String>(0)
            })
            .map_err(db)?;
        for row in rows {
            replaced_nodes.push(KnowledgeNodeId::from_validated(row.map_err(db)?));
        }
    }
    for column in ["r.source_node_id", "r.target_node_id"] {
        // Column names are compile-time constants, never user input.
        let sql = format!(
            "SELECT DISTINCT {column} FROM relations r
             JOIN document_analyses a ON a.analysis_id = r.analysis_id
             WHERE a.source_id = ?1 AND r.status = 'active'"
        );
        let mut stmt = tx.prepare(&sql).map_err(db)?;
        let rows = stmt
            .query_map(params![persistence.record.source_id.as_str()], |row| {
                row.get::<_, String>(0)
            })
            .map_err(db)?;
        for row in rows {
            let node_id = KnowledgeNodeId::from_validated(row.map_err(db)?);
            if !replaced_nodes.contains(&node_id) {
                replaced_nodes.push(node_id);
            }
        }
    }
    tx.execute(
        "UPDATE claims SET status = 'retired', retired_build_id = ?2 WHERE source_id = ?1 AND status = 'active'",
        params![persistence.record.source_id.as_str(), retired_build],
    )
    .map_err(db)?;
    tx.execute(
        "UPDATE relations SET status = 'retired', retired_build_id = ?2
         WHERE status = 'active' AND analysis_id IN
         (SELECT analysis_id FROM document_analyses WHERE source_id = ?1)",
        params![persistence.record.source_id.as_str(), retired_build],
    )
    .map_err(db)?;
    Ok(replaced_nodes)
}

/// Inserts the analysis's active claims with their evidence citations.
/// Returns the number of citation rows written.
fn insert_claim_rows(
    tx: &Transaction<'_>,
    persistence: &AnalysisPersistence,
    analysis_id: &llm_wiki_core::ids::AnalysisId,
) -> Result<usize> {
    let retired_build = persistence.record.build_id.as_ref().map(|b| b.as_str());
    let heading_path_json =
        |section_id: &Option<SectionId>| heading_path_json_of(persistence, section_id);
    let mut citation_count = 0usize;
    for claim in &persistence.claims {
        let claim_row_id = ClaimRowId::generate();
        tx.execute(
            "INSERT INTO claims (claim_id, node_id, source_id, section_id, analysis_id, statement, evidence_digest, confidence, status, created_build_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'active', ?9)",
            params![
                claim_row_id.as_str(),
                claim.node_id.as_str(),
                persistence.record.source_id.as_str(),
                claim.section_id.as_ref().map(|s| s.as_str()),
                analysis_id.as_str(),
                claim.statement,
                claim
                    .evidence_ranges
                    .first()
                    .map(|e| e.evidence_digest.clone())
                    .unwrap_or_default(),
                claim.confidence,
                retired_build
            ],
        )
        .map_err(db)?;

        for evidence in &claim.evidence_ranges {
            tx.execute(
                "INSERT INTO citations (citation_id, owner_kind, owner_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json)
                 VALUES (?1, 'claim', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    llm_wiki_core::ids::CitationId::generate().as_str(),
                    claim_row_id.as_str(),
                    persistence.record.source_id.as_str(),
                    claim.section_id.as_ref().map(|s| s.as_str()),
                    evidence.range.start as i64,
                    evidence.range.end as i64,
                    persistence.source_hash,
                    evidence.evidence_digest,
                    heading_path_json(&claim.section_id)?
                ],
            )
            .map_err(db)?;
            citation_count += 1;
        }
    }
    Ok(citation_count)
}

/// Inserts the analysis's active relations with their evidence citations.
/// Returns the number of citation rows written.
fn insert_relation_rows(
    tx: &Transaction<'_>,
    persistence: &AnalysisPersistence,
    analysis_id: &llm_wiki_core::ids::AnalysisId,
) -> Result<usize> {
    let retired_build = persistence.record.build_id.as_ref().map(|b| b.as_str());
    let heading_path_json =
        |section_id: &Option<SectionId>| heading_path_json_of(persistence, section_id);
    let mut citation_count = 0usize;
    for relation in &persistence.relations {
        let relation_row_id = llm_wiki_core::ids::RelationRowId::generate();
        tx.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, section_id, created_build_id, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'active')",
            params![
                relation_row_id.as_str(),
                analysis_id.as_str(),
                relation.source_node_id.as_str(),
                relation.relation_type,
                relation.target_node_id.as_str(),
                relation.section_id.as_ref().map(|s| s.as_str()),
                retired_build
            ],
        )
        .map_err(db)?;

        if let Some(evidence) = &relation.evidence {
            tx.execute(
                "INSERT INTO citations (citation_id, owner_kind, owner_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json)
                 VALUES (?1, 'relation', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    llm_wiki_core::ids::CitationId::generate().as_str(),
                    relation_row_id.as_str(),
                    persistence.record.source_id.as_str(),
                    relation.section_id.as_ref().map(|s| s.as_str()),
                    evidence.range.start as i64,
                    evidence.range.end as i64,
                    persistence.source_hash,
                    evidence.evidence_digest,
                    heading_path_json(&relation.section_id)?
                ],
            )
            .map_err(db)?;
            citation_count += 1;
        }
    }
    Ok(citation_count)
}

/// Inserts the auditable rejected-candidate rows (PRD §11: unverifiable
/// candidates never become knowledge).
fn insert_rejected_rows(
    tx: &Transaction<'_>,
    persistence: &AnalysisPersistence,
    analysis_id: &llm_wiki_core::ids::AnalysisId,
) -> Result<()> {
    let retired_build = persistence.record.build_id.as_ref().map(|b| b.as_str());
    for rejected in &persistence.rejected_claims {
        tx.execute(
            "INSERT INTO rejected_claims (rejected_id, analysis_id, source_id, candidate_json, claimed_section_id, reason, build_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                llm_wiki_core::ids::RejectedClaimId::generate().as_str(),
                analysis_id.as_str(),
                persistence.record.source_id.as_str(),
                rejected.candidate_json,
                rejected.claimed_section_id,
                rejected.reason,
                retired_build
            ],
        )
        .map_err(db)?;
    }
    Ok(())
}

/// Ghost-knowledge sweep (PRD §19.3): with the NEW claims and relations
/// already inserted, retire replaced knowledge nodes (claims AND relation
/// endpoints) that neither carry an active claim anymore nor touch an
/// active relation. One revision bump for the whole batch. Re-analysis of
/// unchanged content re-inserts the same identities (the registry
/// re-activates a retired node under its stable id on resolution), so
/// identical rebuilds retire nothing and the registry revision stays
/// stable (§37.3 determinism).
fn sweep_ghost_nodes(
    tx: &Transaction<'_>,
    persistence: &AnalysisPersistence,
    replaced_nodes: &[KnowledgeNodeId],
) -> Result<()> {
    if !persistence.replace_source || replaced_nodes.is_empty() {
        return Ok(());
    }
    let retired_build = persistence.record.build_id.as_ref().map(|b| b.as_str());
    let mut retired_registry_nodes = 0u64;
    for node_id in replaced_nodes {
        let active_claims: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM claims WHERE node_id = ?1 AND status = 'active'",
                params![node_id.as_str()],
                |row| row.get(0),
            )
            .map_err(db)?;
        let active_relations: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM relations
                 WHERE status = 'active' AND (source_node_id = ?1 OR target_node_id = ?1)",
                params![node_id.as_str()],
                |row| row.get(0),
            )
            .map_err(db)?;
        if active_claims == 0 && active_relations == 0 {
            retired_registry_nodes += tx
                .execute(
                    "UPDATE knowledge_registry SET status = 'retired', retired_build_id = ?2
                     WHERE id = ?1 AND status = 'active'",
                    params![node_id.as_str(), retired_build],
                )
                .map_err(db)? as u64;
        }
    }
    if retired_registry_nodes > 0 {
        bump_revision(tx)?;
        tracing::info!(
            source = %persistence.record.source_id,
            nodes = retired_registry_nodes,
            "retired unsupported knowledge nodes after re-analysis"
        );
    }
    Ok(())
}

pub fn persist_analysis(
    conn: &mut Connection,
    persistence: &AnalysisPersistence,
) -> Result<PersistedAnalysis> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;

    let analysis_id = llm_wiki_core::ids::AnalysisId::generate();
    tx.execute(
        "INSERT INTO document_analyses
         (analysis_id, source_id, build_id, model, prompt_version, unit_count, llm_request_count, status, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            analysis_id.as_str(),
            persistence.record.source_id.as_str(),
            persistence.record.build_id.as_ref().map(|b| b.as_str()),
            persistence.record.model,
            persistence.record.prompt_version,
            persistence.record.unit_count,
            persistence.record.llm_request_count,
            persistence.record.status,
            chrono::Utc::now().to_rfc3339()
        ],
    )
    .map_err(db)?;

    let replaced_nodes = replace_previous_knowledge(&tx, persistence)?;
    let claim_citations = insert_claim_rows(&tx, persistence, &analysis_id)?;
    let relation_citations = insert_relation_rows(&tx, persistence, &analysis_id)?;
    insert_rejected_rows(&tx, persistence, &analysis_id)?;

    tx.execute(
        "UPDATE document_analyses SET claim_count = ?2, rejected_claim_count = ?3, rejected_relation_count = ?4 WHERE analysis_id = ?1",
        params![
            analysis_id.as_str(),
            persistence.claims.len() as i64,
            persistence.rejected_claims.len() as i64,
            0i64
        ],
    )
    .map_err(db)?;

    sweep_ghost_nodes(&tx, persistence, &replaced_nodes)?;

    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit analysis: {e}")))?;
    Ok(PersistedAnalysis {
        analysis_id,
        claim_count: persistence.claims.len(),
        rejected_claim_count: persistence.rejected_claims.len(),
        relation_count: persistence.relations.len(),
        citation_count: claim_citations + relation_citations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;
    use crate::registry::NodeDraft;
    use crate::sources::upsert_source;
    use crate::{get_or_create_batch, NodeKind};
    use llm_wiki_core::hash::sha256_hex;
    use llm_wiki_core::ids::SourceLocatorKey;

    fn setup() -> (Connection, SourceId, SectionId) {
        let mut conn = open_in_memory().unwrap();
        let (source_id, _) = upsert_source(
            &mut conn,
            &SourceLocatorKey::compute("ws", "a.md"),
            "a.md",
            "hash-1",
            10,
            None,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO source_sections (section_id, source_id, heading_path_json, heading_path_key, content_fingerprint, range_start, range_end, status)
             VALUES ('sec_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, '[\"Doc\",\"Intro\"]', 'Doc\u{1f}Intro', 'fp', 0, 100, 'active')",
            params![source_id.as_str()],
        )
        .unwrap();
        let section_id = SectionId::parse("sec_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        (conn, source_id, section_id)
    }

    #[test]
    fn analysis_persists_atomically_with_citations_and_replacement() {
        let (mut conn, source_id, section_id) = setup();

        let claim_node = {
            let drafts = vec![NodeDraft {
                kind: NodeKind::Claim,
                canonical_key: "claim-key-1".to_owned(),
                canonical_name: "claim".to_owned(),
                entity_type: None,
                description: None,
            }];
            let mut ids = get_or_create_batch(&mut conn, &drafts, None).unwrap();
            ids.remove(0)
        };

        let persistence = AnalysisPersistence {
            record: AnalysisRecord {
                source_id: source_id.clone(),
                build_id: None,
                model: Some("fake".into()),
                prompt_version: Some("document-analysis@1".into()),
                unit_count: 1,
                llm_request_count: 1,
                status: "completed".into(),
            },
            replace_source: true,
            claims: vec![PersistedClaim {
                node_id: claim_node,
                section_id: Some(section_id.clone()),
                statement: "Retry happens three times.".into(),
                confidence: Some(0.9),
                evidence_ranges: vec![EvidenceRange {
                    range: SourceRange::new(10, 40),
                    evidence_digest: sha256_hex(b"retry evidence"),
                }],
            }],
            relations: vec![],
            rejected_claims: vec![PersistedRejectedClaim {
                candidate_json: "{}".into(),
                claimed_section_id: Some("sec_missing".into()),
                reason: "SECTION_NOT_FOUND".into(),
            }],
            heading_paths: vec![(section_id, vec!["Doc".into(), "Intro".into()])],
            source_hash: "hash-1".into(),
        };

        let first = persist_analysis(&mut conn, &persistence).unwrap();
        assert_eq!(first.claim_count, 1);
        assert_eq!(first.citation_count, 1);
        assert_eq!(first.rejected_claim_count, 1);

        let counts: (i64, i64, i64) = conn
            .query_row(
                "SELECT (SELECT COUNT(*) FROM claims WHERE status='active'),
                        (SELECT COUNT(*) FROM citations),
                        (SELECT COUNT(*) FROM rejected_claims)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 1, 1));

        // Re-analysis with replace_source retires the old claim but keeps
        // audit rows (citations/rejected are history).
        let second = persist_analysis(&mut conn, &persistence).unwrap();
        assert_ne!(first.analysis_id, second.analysis_id);
        let active: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM claims WHERE status='active'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let retired: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM claims WHERE status='retired'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!((active, retired), (1, 1));
    }

    #[test]
    fn source_node_sections_cover_claims_and_relation_endpoints() {
        let (mut conn, source_id, section_id) = setup();
        let drafts = vec![
            NodeDraft {
                kind: NodeKind::Entity,
                canonical_key: "runtime".into(),
                canonical_name: "Runtime".into(),
                entity_type: None,
                description: None,
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
             VALUES ('cl_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, ?2, ?3, 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', 's', 'd', 'active')",
            params![ids[1].as_str(), source_id.as_str(), section_id.as_str()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, section_id, status)
             VALUES ('rel_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'uses', ?2, ?3, 'active')",
            params![ids[0].as_str(), ids[1].as_str(), section_id.as_str()],
        )
        .unwrap();

        let sections = list_source_active_node_sections(&conn, &source_id).unwrap();
        assert_eq!(sections.len(), 2, "entity + claim node");
        assert!(sections
            .iter()
            .all(|ns| ns.section_id == Some(section_id.clone())));
    }

    #[test]
    fn retire_source_knowledge_leaves_no_ghost_claims_or_unsupported_nodes() {
        let (mut conn, source_id, section_id) = setup();
        // A second source that independently supports a shared node.
        let (other_id, _) = upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "b.md"),
            "b.md",
            "hash-2",
            10,
            None,
        )
        .unwrap();

        let drafts = vec![
            NodeDraft {
                kind: NodeKind::Claim,
                canonical_key: "only-from-a".into(),
                canonical_name: "only".into(),
                entity_type: None,
                description: None,
            },
            NodeDraft {
                kind: NodeKind::Claim,
                canonical_key: "shared".into(),
                canonical_name: "shared".into(),
                entity_type: None,
                description: None,
            },
        ];
        let ids = get_or_create_batch(&mut conn, &drafts, None).unwrap();
        let only_node = ids[0].clone();
        let shared_node = ids[1].clone();

        conn.execute(
            "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
             VALUES ('an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'completed', '2026-01-01')",
            params![source_id.as_str()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
             VALUES ('an_01BX5ZZKBKACTAV9WEVGEMMVRZ', ?1, 'completed', '2026-01-01')",
            params![other_id.as_str()],
        )
        .unwrap();
        for (claim_id, node, src, analysis) in [
            (
                "cl_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                &only_node,
                &source_id,
                "an_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            ),
            (
                "cl_01BX5ZZKBKACTAV9WEVGEMMVRZ",
                &shared_node,
                &source_id,
                "an_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            ),
            (
                "cl_01CZZZZZZZZZZZZZZZZZZZZZZZ",
                &shared_node,
                &other_id,
                "an_01BX5ZZKBKACTAV9WEVGEMMVRZ",
            ),
        ] {
            conn.execute(
                "INSERT INTO claims (claim_id, node_id, source_id, section_id, analysis_id, statement, evidence_digest, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, 's', 'd', 'active')",
                params![claim_id, node.as_str(), src.as_str(), section_id.as_str(), analysis],
            )
            .unwrap();
        }

        let retired = retire_source_knowledge(&mut conn, &source_id, Some("bld_del")).unwrap();
        assert_eq!(retired.retired_claims, 2);
        assert_eq!(retired.retired_relations, 0);
        // Order-insensitive: node ids minted in the same millisecond sort
        // randomly, and affected_nodes is a set (the §19 mapping treats it as
        // one).
        let mut affected = retired.affected_nodes.clone();
        let mut expected_affected = vec![only_node.clone(), shared_node.clone()];
        affected.sort();
        expected_affected.sort();
        assert_eq!(affected, expected_affected);
        // The node only this source supported is retired from the registry;
        // the shared node stays (the other source still claims it).
        assert_eq!(retired.retired_registry_nodes, vec![only_node.clone()]);

        let active_from_deleted: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM claims WHERE source_id = ?1 AND status = 'active'",
                params![source_id.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(active_from_deleted, 0, "no ghost claims (PRD §19.3)");
        let registry_status = |id: &KnowledgeNodeId| -> String {
            conn.query_row(
                "SELECT status FROM knowledge_registry WHERE id = ?1",
                params![id.as_str()],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(registry_status(&only_node), "retired");
        assert_eq!(registry_status(&shared_node), "active");
        // One revision bump for the registry retirement batch.
        assert_eq!(crate::registry::current_revision(&conn).unwrap(), 3);

        // Retiring an already-clean source is a no-op that does not bump.
        let again = retire_source_knowledge(&mut conn, &source_id, None).unwrap();
        assert_eq!(again.retired_claims, 0);
        assert!(again.retired_registry_nodes.is_empty());
        assert_eq!(crate::registry::current_revision(&conn).unwrap(), 3);
    }

    #[test]
    fn reanalysis_retires_claim_nodes_that_lost_all_support() {
        let (mut conn, source_id, section_id) = setup();

        fn analysis_with(
            source_id: &SourceId,
            section_id: &SectionId,
            node_id: KnowledgeNodeId,
        ) -> AnalysisPersistence {
            AnalysisPersistence {
                record: AnalysisRecord {
                    source_id: source_id.clone(),
                    build_id: None,
                    model: Some("fake".into()),
                    prompt_version: Some("document-analysis@1".into()),
                    unit_count: 1,
                    llm_request_count: 1,
                    status: "completed".into(),
                },
                replace_source: true,
                claims: vec![PersistedClaim {
                    node_id,
                    section_id: Some(section_id.clone()),
                    statement: "statement".into(),
                    confidence: Some(0.9),
                    evidence_ranges: vec![EvidenceRange {
                        range: SourceRange::new(0, 10),
                        evidence_digest: "digest".into(),
                    }],
                }],
                relations: vec![],
                rejected_claims: vec![],
                heading_paths: vec![(section_id.clone(), vec!["Doc".into()])],
                source_hash: "hash-1".into(),
            }
        }
        fn node_of(conn: &mut Connection, key: &str) -> KnowledgeNodeId {
            let drafts = vec![NodeDraft {
                kind: NodeKind::Claim,
                canonical_key: key.to_owned(),
                canonical_name: key.to_owned(),
                entity_type: None,
                description: None,
            }];
            let mut ids = get_or_create_batch(conn, &drafts, None).unwrap();
            ids.remove(0)
        }
        fn registry_status(conn: &Connection, id: &KnowledgeNodeId) -> String {
            conn.query_row(
                "SELECT status FROM knowledge_registry WHERE id = ?1",
                params![id.as_str()],
                |r| r.get(0),
            )
            .unwrap()
        }

        let old_node = node_of(&mut conn, "old-claim-text");
        persist_analysis(
            &mut conn,
            &analysis_with(&source_id, &section_id, old_node.clone()),
        )
        .unwrap();
        assert_eq!(registry_status(&conn, &old_node), "active");

        // Re-analysis replaces the source's knowledge with a DIFFERENT claim:
        // the old claim node has no support left and must retire (PRD §19.3).
        let new_node = node_of(&mut conn, "new-claim-text");
        persist_analysis(
            &mut conn,
            &analysis_with(&source_id, &section_id, new_node.clone()),
        )
        .unwrap();
        assert_eq!(registry_status(&conn, &old_node), "retired");
        assert_eq!(registry_status(&conn, &new_node), "active");

        let active_claims: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM claims WHERE status = 'active'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(active_claims, 1, "only the new claim stays active");

        // Re-analyzing UNCHANGED content re-inserts the same identity and
        // must NOT retire or bump (§37.3 determinism: stable revision).
        let revision_before = crate::registry::current_revision(&conn).unwrap();
        persist_analysis(
            &mut conn,
            &analysis_with(&source_id, &section_id, new_node.clone()),
        )
        .unwrap();
        assert_eq!(registry_status(&conn, &new_node), "active");
        assert_eq!(
            crate::registry::current_revision(&conn).unwrap(),
            revision_before
        );
    }

    #[test]
    fn reanalysis_sweep_covers_relation_endpoints_without_touching_shared_nodes() {
        // A relation-only entity whose relation vanishes during replacement
        // must retire (ghost knowledge, PRD §19.3) — the sweep is not limited
        // to claim nodes. A node another source still supports stays.
        let (mut conn, source_id, section_id) = setup();
        let (other_id, _) = upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "b.md"),
            "b.md",
            "hash-2",
            10,
            None,
        )
        .unwrap();

        let drafts = vec![
            NodeDraft {
                kind: NodeKind::Entity,
                canonical_key: "solo-entity".into(),
                canonical_name: "Solo".into(),
                entity_type: None,
                description: None,
            },
            NodeDraft {
                kind: NodeKind::Entity,
                canonical_key: "shared-entity".into(),
                canonical_name: "Shared".into(),
                entity_type: None,
                description: None,
            },
        ];
        let ids = get_or_create_batch(&mut conn, &drafts, None).unwrap();
        let (solo, shared) = (ids[0].clone(), ids[1].clone());

        let registry_status = |conn: &Connection, id: &KnowledgeNodeId| -> String {
            conn.query_row(
                "SELECT status FROM knowledge_registry WHERE id = ?1",
                params![id.as_str()],
                |r| r.get(0),
            )
            .unwrap()
        };
        fn analysis_row(conn: &mut Connection, source_id: &SourceId, analysis_id: &str) {
            conn.execute(
                "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
                 VALUES (?1, ?2, 'completed', '2026-01-01')",
                params![analysis_id, source_id.as_str()],
            )
            .unwrap();
        }

        // The source under test relates solo ↔ shared; another source keeps a
        // second relation touching `shared` alive.
        analysis_row(&mut conn, &source_id, "an_01ARZ3NDEKTSV4RRFFQ69G5FAV");
        analysis_row(&mut conn, &other_id, "an_01BX5ZZKBKACTAV9WEVGEMMVRZ");
        conn.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, section_id, status)
             VALUES ('rel_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'uses', ?2, ?3, 'active')",
            params![solo.as_str(), shared.as_str(), section_id.as_str()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, section_id, status)
             VALUES ('rel_01BX5ZZKBKACTAV9WEVGEMMVRZ', 'an_01BX5ZZKBKACTAV9WEVGEMMVRZ', ?1, 'uses', ?2, ?3, 'active')",
            params![shared.as_str(), shared.as_str(), section_id.as_str()],
        )
        .unwrap();

        // Unchanged re-analysis: same relation re-inserted → nothing retires,
        // the revision is stable (§37.3 determinism).
        let revision_before = crate::registry::current_revision(&conn).unwrap();
        persist_analysis(
            &mut conn,
            &AnalysisPersistence {
                record: AnalysisRecord {
                    source_id: source_id.clone(),
                    build_id: None,
                    model: None,
                    prompt_version: None,
                    unit_count: 1,
                    llm_request_count: 0,
                    status: "completed".into(),
                },
                replace_source: true,
                claims: vec![],
                relations: vec![PersistedRelation {
                    source_node_id: solo.clone(),
                    relation_type: "uses".into(),
                    target_node_id: shared.clone(),
                    section_id: Some(section_id.clone()),
                    evidence: None,
                }],
                rejected_claims: vec![],
                heading_paths: vec![],
                source_hash: "hash-1".into(),
            },
        )
        .unwrap();
        assert_eq!(registry_status(&conn, &solo), "active");
        assert_eq!(
            crate::registry::current_revision(&conn).unwrap(),
            revision_before
        );

        // Re-analysis WITHOUT the relation: `solo` loses all support and
        // retires; `shared` stays (the other source still relates it).
        persist_analysis(
            &mut conn,
            &AnalysisPersistence {
                record: AnalysisRecord {
                    source_id: source_id.clone(),
                    build_id: None,
                    model: None,
                    prompt_version: None,
                    unit_count: 1,
                    llm_request_count: 0,
                    status: "completed".into(),
                },
                replace_source: true,
                claims: vec![],
                relations: vec![],
                rejected_claims: vec![],
                heading_paths: vec![],
                source_hash: "hash-2".into(),
            },
        )
        .unwrap();
        assert_eq!(registry_status(&conn, &solo), "retired");
        assert_eq!(registry_status(&conn, &shared), "active");
    }
}
