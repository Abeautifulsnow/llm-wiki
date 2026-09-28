//! Persists a verified [`AnalysisOutcome`]: resolves entity/concept/claim
//! node identities through the Knowledge Registry (PRD §12.1.1) and writes
//! the analysis atomically (PRD §11: unverifiable candidates become
//! rejected_claims records, never knowledge).

use std::collections::HashMap;

use llm_wiki_core::error::Result;
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_storage::registry::{get_or_create_batch, NodeDraft};
use llm_wiki_storage::{
    canonical_key, persist_analysis, AnalysisPersistence, AnalysisRecord, EvidenceRange,
    PersistedAnalysis, PersistedClaim, PersistedRejectedClaim, PersistedRelation,
};
use rusqlite::Connection;

use crate::analysis::AnalysisOutcome;
use crate::analysis::AnalyzedDocument;

#[derive(Debug, Clone, Default)]
pub struct PersistOptions {
    pub build_id: Option<llm_wiki_core::ids::BuildId>,
    pub model: Option<String>,
    /// `name@version` of the analysis prompt — recorded on the analysis row.
    pub prompt_version: Option<String>,
    /// Re-analysis replaces the source's previous active knowledge.
    pub replace_source: bool,
}

/// Outcome of the full persistence step.
pub struct PersistReport {
    pub persisted: PersistedAnalysis,
    pub registry_nodes_reused: usize,
    pub registry_nodes_created: usize,
    pub registry_revision: u64,
}

/// Deduplicated registry draft set, keyed by (kind, canonical_key).
#[derive(Default)]
struct DraftSet {
    drafts: Vec<NodeDraft>,
    index: HashMap<(String, String), usize>,
}

impl DraftSet {
    fn push(
        &mut self,
        kind: llm_wiki_storage::NodeKind,
        canonical_key: String,
        canonical_name: String,
        entity_type: Option<String>,
        description: Option<String>,
    ) {
        let key = (kind.as_str().to_owned(), canonical_key.clone());
        if self.index.contains_key(&key) {
            return; // first occurrence wins for detail fields
        }
        self.index.insert(key, self.drafts.len());
        self.drafts.push(NodeDraft {
            kind,
            canonical_key,
            canonical_name,
            entity_type,
            description,
        });
    }
}

pub fn persist_outcome(
    conn: &mut Connection,
    doc: &AnalyzedDocument,
    outcome: &AnalysisOutcome,
    options: &PersistOptions,
) -> Result<PersistReport> {
    let build_tag = options.build_id.as_ref().map(|b| b.as_str());

    // ---- Collect registry drafts: entities, concepts, claims and relation
    // endpoints, deduplicated by (kind, canonical_key). ----
    let mut drafts = DraftSet::default();
    let mut entity_keys = Vec::new();
    for entity in &outcome.analysis.entities {
        let key = canonical_key(&entity.name);
        drafts.push(
            llm_wiki_storage::NodeKind::Entity,
            key.clone(),
            entity.name.clone(),
            Some(entity.entity_type.clone()).filter(|t| !t.is_empty()),
            entity.description.clone(),
        );
        entity_keys.push(key);
    }
    for concept in &outcome.analysis.concepts {
        let key = canonical_key(&concept.name);
        drafts.push(
            llm_wiki_storage::NodeKind::Concept,
            key.clone(),
            concept.name.clone(),
            None,
            concept.description.clone(),
        );
    }
    let mut claim_keys = Vec::new();
    for claim in &outcome.analysis.claims {
        let key = sha256_hex(crate::analysis::claim_identity_key(&claim.text).as_bytes());
        drafts.push(
            llm_wiki_storage::NodeKind::Claim,
            key.clone(),
            short_label(&claim.text),
            None,
            None,
        );
        claim_keys.push(key);
    }
    for verified in &outcome.verified_relations {
        for name in [
            &verified.relation.source_name,
            &verified.relation.target_name,
        ] {
            let key = canonical_key(name);
            drafts.push(
                llm_wiki_storage::NodeKind::Entity,
                key.clone(),
                name.clone(),
                Some("other".to_owned()),
                None,
            );
        }
    }

    // ---- Resolve identities in ONE registry transaction. ----
    let before = llm_wiki_storage::current_revision(conn)?;
    let ids = get_or_create_batch(conn, &drafts.drafts, build_tag)?;
    let after = llm_wiki_storage::current_revision(conn)?;
    let created = (after - before) as usize;
    let reused = drafts.drafts.len().saturating_sub(created);

    let node_id = |kind: &str, key: &str| -> llm_wiki_core::ids::KnowledgeNodeId {
        ids[drafts.index[&(kind.to_owned(), key.to_owned())]].clone()
    };

    let claims = outcome
        .analysis
        .claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| PersistedClaim {
            node_id: node_id("claim", &claim_keys[idx]),
            section_id: Some(claim.source_section_id.clone()),
            statement: claim.text.clone(),
            confidence: claim.confidence,
            evidence_ranges: claim
                .evidence_ranges
                .iter()
                .map(|range| EvidenceRange {
                    range: *range,
                    evidence_digest: claim.evidence_digest.clone(),
                })
                .collect(),
        })
        .collect();

    let relations = outcome
        .verified_relations
        .iter()
        .map(|verified| PersistedRelation {
            source_node_id: node_id("entity", &canonical_key(&verified.relation.source_name)),
            relation_type: verified.relation.relation_type.clone(),
            target_node_id: node_id("entity", &canonical_key(&verified.relation.target_name)),
            section_id: Some(verified.relation.source_section_id.clone()),
            evidence: Some(EvidenceRange {
                range: verified.evidence_range,
                evidence_digest: verified.evidence_digest.clone(),
            }),
        })
        .collect();

    let rejected_claims = outcome
        .rejected_claims
        .iter()
        .map(|rejected| {
            let candidate_json = serde_json::json!({
                "text": rejected.candidate_text,
                "source_section_id": rejected.source_section_id,
            });
            PersistedRejectedClaim {
                candidate_json: candidate_json.to_string(),
                claimed_section_id: rejected.source_section_id.clone(),
                reason: rejected.reason.clone(),
            }
        })
        .collect();

    let persistence = AnalysisPersistence {
        record: AnalysisRecord {
            source_id: doc.source_id.clone(),
            build_id: options.build_id.clone(),
            model: options.model.clone(),
            prompt_version: options.prompt_version.clone(),
            unit_count: outcome.unit_count,
            llm_request_count: outcome.llm_request_count,
            status: "completed".to_owned(),
        },
        replace_source: options.replace_source,
        claims,
        relations,
        rejected_claims,
        heading_paths: doc
            .sections
            .iter()
            .map(|section| (section.section_id.clone(), section.heading_path.clone()))
            .collect(),
        source_hash: doc.content_hash.clone(),
    };

    let persisted = persist_analysis(conn, &persistence)?;
    Ok(PersistReport {
        persisted,
        registry_nodes_reused: reused,
        registry_nodes_created: created,
        registry_revision: after,
    })
}

fn short_label(text: &str) -> String {
    text.chars().take(60).collect()
}
