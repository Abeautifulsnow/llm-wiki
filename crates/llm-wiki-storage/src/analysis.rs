//! Stage-one analysis persistence (PRD §11, §18): document_analyses, claims,
//! citations, relations and rejected_claims. All writes for one analysis
//! happen in a single transaction; knowledge node identity is assigned by the
//! registry beforehand (claims are `kind='claim'` registry nodes, PRD §12.1.1).

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, ClaimRowId, KnowledgeNodeId, SectionId, SourceId};
use llm_wiki_core::model::SourceRange;

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
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

    let retired_build = persistence.record.build_id.as_ref().map(|b| b.as_str());
    if persistence.replace_source {
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
    }

    let heading_path_json = |section_id: &Option<SectionId>| -> Result<String> {
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
    };

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

    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit analysis: {e}")))?;
    Ok(PersistedAnalysis {
        analysis_id,
        claim_count: persistence.claims.len(),
        rejected_claim_count: persistence.rejected_claims.len(),
        relation_count: persistence.relations.len(),
        citation_count,
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
}
