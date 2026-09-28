//! Document analysis domain model (PRD §11).
//!
//! The LLM never produces the final wiki; stage one produces a structured
//! [`DocumentAnalysis`] of auditable atomic claims. [`ClaimCandidate`] carries
//! the *verified* evidence shape from §11.1 (ranges + digest) — values that
//! application code fills in after locating the evidence text; the raw LLM
//! JSON shape is deliberately different and lives in the compiler crate.
//! Hallucinated section ids and unverifiable evidence never reach this model.

use serde::{Deserialize, Serialize};

use crate::ids::{BuildId, SectionId};
use crate::model::SourceRange;

/// Stage-one analysis output (PRD §11).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentAnalysis {
    pub summary: String,
    pub entities: Vec<EntityCandidate>,
    pub concepts: Vec<ConceptCandidate>,
    pub claims: Vec<ClaimCandidate>,
    pub relations: Vec<RelationCandidate>,
    pub topics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCandidate {
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptCandidate {
    pub name: String,
    pub description: Option<String>,
}

/// An auditable atomic fact (PRD §11.1). `evidence_ranges` are absolute
/// offsets into the analyzed document text and always fall inside the cited
/// section's range; `evidence_digest` is `sha256(evidence_text)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimCandidate {
    pub text: String,
    pub source_section_id: SectionId,
    pub evidence_ranges: Vec<SourceRange>,
    pub evidence_digest: String,
    /// Sorting / human-review priority only — never a substitute for
    /// evidence (PRD §11.1).
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationCandidate {
    pub source_name: String,
    pub relation_type: String,
    pub target_name: String,
    pub source_section_id: SectionId,
    /// Verbatim quote from the cited section supporting the relation.
    pub evidence: String,
}

/// Auditable record for a claim whose evidence could not be verified after
/// one repair (PRD §11.1: 原始候选、source、reason、build ID).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedClaim {
    pub candidate_text: String,
    pub source_section_id: Option<String>,
    pub reason: String,
    pub build_id: Option<BuildId>,
}

/// Machine-readable validation issue (PRD §11.1/§28: the repair request must
/// carry the validator's reasons verbatim).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Stable code, e.g. `SECTION_NOT_FOUND`, `RANGE_OUT_OF_SECTION`,
    /// `EVIDENCE_NOT_IN_SECTION`, `EMPTY_STATEMENT`.
    pub code: String,
    pub message: String,
}

impl ValidationIssue {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_owned(),
            message: message.into(),
        }
    }
}

/// Auditable rejection of a relation candidate whose section reference or
/// evidence could not be verified (kept for parity with RejectedClaim —
/// the PRD's "no silent drop" rule applies to all outputs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedRelation {
    pub source_name: String,
    pub target_name: String,
    pub relation_type: String,
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_analysis_serializes_roundtrip() {
        let analysis = DocumentAnalysis {
            summary: "s".into(),
            entities: vec![EntityCandidate {
                name: "Plugin Runtime".into(),
                entity_type: "component".into(),
                description: None,
            }],
            concepts: vec![ConceptCandidate {
                name: "at-least-once delivery".into(),
                description: Some("delivery guarantee".into()),
            }],
            claims: vec![ClaimCandidate {
                text: "The runtime retries resolved→active up to three times.".into(),
                source_section_id: SectionId::parse("sec_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap(),
                evidence_ranges: vec![SourceRange::new(0, 10)],
                evidence_digest: "abc".into(),
                confidence: Some(0.9),
            }],
            relations: vec![],
            topics: vec!["plugin lifecycle".into()],
        };
        let json = serde_json::to_string(&analysis).unwrap();
        let back: DocumentAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(back.claims.len(), 1);
        assert_eq!(back.entities[0].name, "Plugin Runtime");
    }
}
