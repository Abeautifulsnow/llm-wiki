//! Stage-one document analyzer (PRD §10, §11, §28).
//!
//! Pipeline per analysis unit:
//! 1. LLM request (JSON mode, temperature 0) → **stage 1** shape validation
//!    via `structured::parse_json`;
//! 2. on a stage-1 failure → **one** repair request carrying the
//!    machine-readable reasons (PRD §11: 仅 repair 一次); a second shape
//!    failure fails the unit;
//! 3. **stage 2** referential: section ids must exist in the unit manifest,
//!    evidence quotes must locate inside the cited section — evidence digests
//!    are computed by application code, never trusted from the model;
//! 4. **stage 3** semantic: atomic non-empty statements, sane relations;
//! 5. unverifiable claims become auditable [`RejectedClaim`] records; a unit
//!    exceeding `max_rejected_claim_ratio` fails with `EvidenceValidationError`
//!    and never yields knowledge (PRD §11.1).

use std::sync::Arc;

use serde::Deserialize;

use llm_wiki_core::analysis::{
    ClaimCandidate, ConceptCandidate, DocumentAnalysis, EntityCandidate, RejectedClaim,
    RejectedRelation, RelationCandidate, ValidationIssue,
};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::{BuildId, SectionId, SourceId};
use llm_wiki_core::model::SourceRange;
use llm_wiki_llm::structured;
use llm_wiki_llm::{LlmProvider, LlmRequest};
use llm_wiki_markdown::{estimate_tokens, split_section};

use crate::prompt::PromptDocument;

/// One section handed to analysis; `range` is absolute in the analyzed text.
#[derive(Debug, Clone)]
pub struct AnalysisSection {
    pub section_id: SectionId,
    pub heading_path: Vec<String>,
    pub range: SourceRange,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct AnalyzedDocument {
    pub source_id: SourceId,
    pub rel_path: String,
    pub content_hash: String,
    pub language: String,
    pub sections: Vec<AnalysisSection>,
}

/// A relation with its verified evidence anchor.
#[derive(Debug, Clone)]
pub struct VerifiedRelation {
    pub relation: RelationCandidate,
    pub evidence_range: SourceRange,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, Default)]
pub struct AnalysisOutcome {
    pub analysis: DocumentAnalysis,
    /// Evidence anchors parallel to `analysis.relations` (by index).
    pub verified_relations: Vec<VerifiedRelation>,
    pub rejected_claims: Vec<RejectedClaim>,
    pub rejected_relations: Vec<RejectedRelation>,
    pub unit_count: u32,
    pub llm_request_count: u32,
}

pub struct DocumentAnalyzer {
    provider: Arc<dyn LlmProvider>,
    prompt: PromptDocument,
    section_target_tokens: u32,
    max_rejected_claim_ratio: f32,
}

impl DocumentAnalyzer {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        prompt: PromptDocument,
        section_target_tokens: u32,
        max_rejected_claim_ratio: f32,
    ) -> Self {
        Self {
            provider,
            prompt,
            section_target_tokens,
            max_rejected_claim_ratio,
        }
    }

    /// Analyzes one document. A unit exceeding the rejected-claim ratio fails
    /// the whole analysis (the build must not publish, PRD §11.1).
    pub async fn analyze_document(
        &self,
        doc: &AnalyzedDocument,
        build_id: Option<&BuildId>,
    ) -> Result<AnalysisOutcome> {
        let units = build_units(&doc.sections, self.section_target_tokens);
        let mut outcome = AnalysisOutcome {
            unit_count: units.len() as u32,
            ..Default::default()
        };

        for unit in &units {
            let unit_outcome = self.analyze_unit(unit, build_id).await?;
            outcome.llm_request_count += unit_outcome.llm_request_count;
            outcome.rejected_claims.extend(unit_outcome.rejected_claims);
            outcome
                .rejected_relations
                .extend(unit_outcome.rejected_relations);
            if !unit_outcome.summary.is_empty() {
                outcome
                    .analysis
                    .summary
                    .push_str(unit_outcome.summary.trim());
                outcome.analysis.summary.push('\n');
            }
            outcome.analysis.topics.extend(unit_outcome.topics);
            outcome.analysis.entities.extend(unit_outcome.entities);
            outcome.analysis.concepts.extend(unit_outcome.concepts);
            outcome.analysis.claims.extend(unit_outcome.claims);
            for relation in unit_outcome.verified_relations {
                outcome.analysis.relations.push(relation.relation.clone());
                outcome.verified_relations.push(relation);
            }
        }
        Ok(outcome)
    }

    async fn analyze_unit(
        &self,
        unit: &AnalysisUnit,
        build_id: Option<&BuildId>,
    ) -> Result<UnitOutcome> {
        let sections_json = serde_json::to_string(&unit.manifest())
            .map_err(|e| WikiError::SchemaValidation(format!("manifest serialize: {e}")))?;
        // Render everything EXCEPT the repair slot: the {{REPAIR_NOTES}}
        // placeholder stays in the template so a repair request can inject
        // the validator reasons there (PRD §11: machine-readable reasons).
        let template = self.prompt.render(&[
            ("LANGUAGE", "the document's own"),
            ("SECTIONS", &sections_json),
        ]);
        let base_request = LlmRequest {
            task_tag: "document-analysis".to_owned(),
            system: None,
            prompt: template.replace("{{REPAIR_NOTES}}", ""),
            temperature: 0.0,
            max_output_tokens: 4096,
            json_mode: true,
        };

        // ---- Stage 1 (+ one repair on shape failure, PRD §11) ----
        let mut llm_request_count = 1u32;
        let mut response = self
            .provider
            .generate(base_request.clone())
            .await
            .map_err(WikiError::from)?;
        let raw: RawAnalysis = match structured::parse_json(&response.text) {
            Ok(raw) => raw,
            Err(stage1) => {
                tracing::warn!(reason = %stage1, "analysis stage-1 failed, repairing once");
                let repair = repair_request(&base_request, &template, &[stage1.machine_reason()]);
                llm_request_count += 1;
                response = self
                    .provider
                    .generate(repair)
                    .await
                    .map_err(WikiError::from)?;
                structured::parse_json(&response.text).map_err(|stage1_repair| {
                    WikiError::SchemaValidation(format!(
                        "schema validation failed after repair: {}",
                        stage1_repair.machine_reason()
                    ))
                })?
            }
        };

        // ---- Stage 2 (referential) + stage 3 (semantic) ----
        let claims = self.verify_claims(&raw.claims, unit, build_id);
        let (verified_relations, rejected_relations) = self.verify_relations(&raw.relations, unit);

        let rejected_claims = claims.rejected;
        let total_claims = claims.verified.len() + rejected_claims.len();
        if total_claims > 0 {
            let ratio = rejected_claims.len() as f32 / total_claims as f32;
            if ratio > self.max_rejected_claim_ratio {
                return Err(WikiError::EvidenceValidation(format!(
                    "analysis unit rejected {}/{} claims ({:.0}% > {:.0}% threshold); the unit fails and must not produce knowledge",
                    rejected_claims.len(),
                    total_claims,
                    ratio * 100.0,
                    self.max_rejected_claim_ratio * 100.0
                )));
            }
        }

        Ok(UnitOutcome {
            summary: raw.summary,
            topics: raw.topics,
            entities: raw
                .entities
                .into_iter()
                .map(|e| EntityCandidate {
                    name: e.name,
                    entity_type: e.entity_type,
                    description: e.description,
                })
                .collect(),
            concepts: raw
                .concepts
                .into_iter()
                .map(|c| ConceptCandidate {
                    name: c.name,
                    description: c.description,
                })
                .collect(),
            claims: claims.verified,
            verified_relations,
            rejected_relations,
            rejected_claims,
            llm_request_count,
        })
    }

    fn verify_claims(
        &self,
        candidates: &[RawClaim],
        unit: &AnalysisUnit,
        build_id: Option<&BuildId>,
    ) -> ClaimVerification {
        let mut verified = Vec::new();
        let mut rejected = Vec::new();
        for candidate in candidates {
            let issues = validate_claim(candidate, unit);
            let located = if issues.is_empty() {
                locate_evidence(
                    &candidate.evidence_text,
                    candidate.evidence_start,
                    &candidate.section_id,
                    unit,
                )
            } else {
                Err(issues)
            };
            match located {
                Ok((section_id, range, digest)) => {
                    verified.push(ClaimCandidate {
                        text: candidate.text.clone(),
                        source_section_id: section_id,
                        evidence_ranges: vec![range],
                        evidence_digest: digest,
                        confidence: candidate.confidence,
                    });
                }
                Err(issues) => {
                    tracing::debug!(issues = ?issues, "claim rejected");
                    rejected.push(RejectedClaim {
                        candidate_text: candidate.text.clone(),
                        source_section_id: Some(candidate.section_id.clone()),
                        reason: render_issues(&issues),
                        build_id: build_id.cloned(),
                    });
                }
            }
        }
        ClaimVerification { verified, rejected }
    }

    fn verify_relations(
        &self,
        candidates: &[RawRelation],
        unit: &AnalysisUnit,
    ) -> (Vec<VerifiedRelation>, Vec<RejectedRelation>) {
        let mut verified = Vec::new();
        let mut rejected = Vec::new();
        for candidate in candidates {
            let mut issues = Vec::new();
            if candidate.source.trim().is_empty() || candidate.target.trim().is_empty() {
                issues.push(ValidationIssue::new(
                    "EMPTY_ENDPOINT",
                    "relation endpoint is empty",
                ));
            }
            if candidate.source.trim() == candidate.target.trim()
                && !candidate.source.trim().is_empty()
            {
                issues.push(ValidationIssue::new(
                    "SAME_ENDPOINTS",
                    "relation source equals target",
                ));
            }

            let located =
                locate_evidence(&candidate.evidence_text, None, &candidate.section_id, unit);
            match located {
                Ok((section_id, range, digest)) if issues.is_empty() => {
                    verified.push(VerifiedRelation {
                        relation: RelationCandidate {
                            source_name: candidate.source.clone(),
                            relation_type: candidate.relation_type.clone(),
                            target_name: candidate.target.clone(),
                            source_section_id: section_id,
                            evidence: candidate.evidence_text.clone(),
                        },
                        evidence_range: range,
                        evidence_digest: digest,
                    });
                }
                Ok(_) => {
                    // Evidence verified but semantic issues present.
                    tracing::debug!(issues = ?issues, "relation rejected");
                    rejected.push(RejectedRelation {
                        source_name: candidate.source.clone(),
                        target_name: candidate.target.clone(),
                        relation_type: candidate.relation_type.clone(),
                        reason: render_issues(&issues),
                    });
                }
                Err(evidence_issues) => {
                    tracing::debug!(issues = ?evidence_issues, "relation rejected");
                    rejected.push(RejectedRelation {
                        source_name: candidate.source.clone(),
                        target_name: candidate.target.clone(),
                        relation_type: candidate.relation_type.clone(),
                        reason: render_issues(&evidence_issues),
                    });
                }
            }
        }
        (verified, rejected)
    }
}

fn render_issues(issues: &[ValidationIssue]) -> String {
    issues
        .iter()
        .map(|issue| format!("{}: {}", issue.code, issue.message))
        .collect::<Vec<_>>()
        .join("; ")
}

struct ClaimVerification {
    verified: Vec<ClaimCandidate>,
    rejected: Vec<RejectedClaim>,
}

struct UnitOutcome {
    summary: String,
    topics: Vec<String>,
    entities: Vec<EntityCandidate>,
    concepts: Vec<ConceptCandidate>,
    claims: Vec<ClaimCandidate>,
    verified_relations: Vec<VerifiedRelation>,
    rejected_relations: Vec<RejectedRelation>,
    rejected_claims: Vec<RejectedClaim>,
    llm_request_count: u32,
}

// ---------------------------------------------------------------------------
// LLM-facing raw shapes (deliberately different from the verified core model:
// evidence is a verbatim quote + hint offset; digests are computed by us).
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawAnalysis {
    #[serde(default)]
    summary: String,
    #[serde(default)]
    topics: Vec<String>,
    #[serde(default)]
    entities: Vec<RawEntity>,
    #[serde(default)]
    concepts: Vec<RawConcept>,
    #[serde(default)]
    claims: Vec<RawClaim>,
    #[serde(default)]
    relations: Vec<RawRelation>,
}

#[derive(Debug, Deserialize)]
struct RawEntity {
    name: String,
    #[serde(default)]
    entity_type: String,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawConcept {
    name: String,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawClaim {
    text: String,
    section_id: String,
    #[serde(default)]
    evidence_text: String,
    #[serde(default)]
    evidence_start: Option<i64>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawRelation {
    source: String,
    relation_type: String,
    target: String,
    section_id: String,
    #[serde(default)]
    evidence_text: String,
}

fn repair_request(base: &LlmRequest, template: &str, reasons: &[String]) -> LlmRequest {
    let mut repair = base.clone();
    let notes = format!(
        "## Previous attempt rejected\nYour previous reply failed validation:\n{}\n\nFix every issue and resend the COMPLETE JSON object.",
        reasons
            .iter()
            .map(|reason| format!("- {reason}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    repair.prompt = template.replace("{{REPAIR_NOTES}}", &notes);
    repair
}

// ---------------------------------------------------------------------------
// Unit packing (PRD §10): never truncate; oversized sections split into
// segments at block boundaries; each segment is its own unit entry and keeps
// the SAME SectionId.
// ---------------------------------------------------------------------------

pub(crate) struct UnitEntry {
    pub section_id: SectionId,
    pub heading_path: Vec<String>,
    pub content: String,
    /// Absolute offset of `content` inside the analyzed document text, so
    /// located evidence can be converted to absolute citation ranges.
    pub range_start: usize,
}

pub(crate) struct AnalysisUnit {
    pub entries: Vec<UnitEntry>,
}

impl AnalysisUnit {
    fn manifest(&self) -> Vec<serde_json::Value> {
        self.entries
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "section_id": entry.section_id.as_str(),
                    "heading_path": entry.heading_path,
                    "content": entry.content,
                })
            })
            .collect()
    }
}

pub(crate) fn build_units(sections: &[AnalysisSection], target_tokens: u32) -> Vec<AnalysisUnit> {
    let target = (target_tokens as u64).max(64);
    let mut units: Vec<AnalysisUnit> = Vec::new();
    let mut current: Vec<UnitEntry> = Vec::new();
    let mut current_tokens = 0u64;

    fn flush(current: &mut Vec<UnitEntry>, units: &mut Vec<AnalysisUnit>) {
        if !current.is_empty() {
            units.push(AnalysisUnit {
                entries: std::mem::take(current),
            });
        }
    }

    for section in sections {
        if estimate_tokens(&section.content) > target {
            // Oversized section: split at block boundaries (PRD §10).
            flush(&mut current, &mut units);
            current_tokens = 0;
            let pseudo = llm_wiki_markdown::SectionOutput {
                heading: None,
                heading_level: 0,
                heading_path: section.heading_path.clone(),
                content: section.content.clone(),
                source_range: section.range,
            };
            for segment in split_section(&pseudo, target) {
                units.push(AnalysisUnit {
                    entries: vec![UnitEntry {
                        section_id: section.section_id.clone(),
                        heading_path: section.heading_path.clone(),
                        content: segment.content.clone(),
                        range_start: segment.source_range.start,
                    }],
                });
            }
            continue;
        }

        let tokens = estimate_tokens(&section.content);
        if !current.is_empty() && current_tokens + tokens > target {
            flush(&mut current, &mut units);
            current_tokens = 0;
        }
        current_tokens += tokens;
        current.push(UnitEntry {
            section_id: section.section_id.clone(),
            heading_path: section.heading_path.clone(),
            content: section.content.clone(),
            range_start: section.range.start,
        });
    }
    flush(&mut current, &mut units);
    units
}

// ---------------------------------------------------------------------------
// Stage 2 + 3 validation helpers
// ---------------------------------------------------------------------------

fn validate_claim(candidate: &RawClaim, unit: &AnalysisUnit) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if candidate.text.trim().is_empty() {
        issues.push(ValidationIssue::new(
            "EMPTY_STATEMENT",
            "claim text is empty",
        ));
    }
    if candidate.evidence_text.trim().is_empty() {
        issues.push(ValidationIssue::new(
            "EMPTY_EVIDENCE",
            "claim has no evidence quote",
        ));
    }
    if unit
        .entries
        .iter()
        .all(|entry| entry.section_id.as_str() != candidate.section_id)
    {
        issues.push(ValidationIssue::new(
            "SECTION_NOT_FOUND",
            format!(
                "section id '{}' does not exist in this source",
                candidate.section_id
            ),
        ));
    }
    issues
}

/// Locates a verbatim evidence quote inside the cited section (whitespace-run
/// tolerant) and computes its absolute range + digest. Returns
/// `(section id, absolute range, sha256 of quoted text)`, or the validation
/// issues on failure.
fn locate_evidence(
    evidence_text: &str,
    hint_start: Option<i64>,
    section_id: &str,
    unit: &AnalysisUnit,
) -> std::result::Result<(SectionId, SourceRange, String), Vec<ValidationIssue>> {
    let entry = unit
        .entries
        .iter()
        .find(|entry| entry.section_id.as_str() == section_id)
        .ok_or_else(|| {
            vec![ValidationIssue::new(
                "SECTION_NOT_FOUND",
                format!("section id '{section_id}' does not exist in this source"),
            )]
        })?;

    let evidence_tokens = tokenize(evidence_text);
    if evidence_tokens.is_empty() {
        return Err(vec![ValidationIssue::new(
            "EMPTY_EVIDENCE",
            "claim has no evidence quote",
        )]);
    }
    let content_tokens = tokenize(&entry.content);
    let first = locate_key(&evidence_tokens[0].text);

    let mut matches: Vec<(usize, usize)> = Vec::new(); // byte start, byte end
    for (start_idx, token) in content_tokens.iter().enumerate() {
        if locate_key(&token.text) == first
            && start_idx + evidence_tokens.len() <= content_tokens.len()
        {
            let window = &content_tokens[start_idx..start_idx + evidence_tokens.len()];
            if window
                .iter()
                .zip(evidence_tokens.iter())
                .all(|(content, evidence)| locate_key(&content.text) == locate_key(&evidence.text))
            {
                matches.push((window[0].start, window[window.len() - 1].end));
            }
        }
    }

    let picked = match matches.len() {
        0 => {
            return Err(vec![ValidationIssue::new(
                "EVIDENCE_NOT_IN_SECTION",
                format!("evidence quote not found in section '{section_id}'"),
            )])
        }
        1 => matches[0],
        _ => match hint_start {
            Some(hint) => matches
                .iter()
                .copied()
                .min_by_key(|(start, _)| (*start as i64 - hint).abs())
                .unwrap(),
            None => matches[0],
        },
    };

    let quoted = &entry.content[picked.0..picked.1];
    Ok((
        entry.section_id.clone(),
        SourceRange::new(entry.range_start + picked.0, entry.range_start + picked.1),
        sha256_hex(quoted.as_bytes()),
    ))
}

/// Comparison key for locating a quote token: punctuation at token edges is
/// ignored (models routinely drop a trailing period), so "times." locates
/// "times". The citation range still covers the original bytes.
fn locate_key(text: &str) -> &str {
    text.trim_matches(|ch: char| !ch.is_alphanumeric())
}

/// Splits text into NFC whitespace-separated tokens with byte offsets.
/// Matching stays case-sensitive: quotes must match the document verbatim
/// modulo whitespace and edge punctuation.
fn tokenize(text: &str) -> Vec<Token> {
    use unicode_normalization::UnicodeNormalization;
    let normalized: String = text.nfc().collect();
    let mut tokens = Vec::new();
    let mut start: Option<usize> = None;
    for (idx, ch) in normalized.char_indices() {
        if ch.is_whitespace() {
            if let Some(begin) = start.take() {
                tokens.push(Token {
                    text: normalized[begin..idx].to_owned(),
                    start: begin,
                    end: idx,
                });
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(begin) = start {
        tokens.push(Token {
            text: normalized[begin..].to_owned(),
            start: begin,
            end: normalized.len(),
        });
    }
    tokens
}

struct Token {
    text: String,
    start: usize,
    end: usize,
}

/// Stable registry canonical key for a claim statement: fingerprint of the
/// normalized text, so identical statements across builds share one node.
pub fn claim_identity_key(statement: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let normalized: String = statement.nfc().collect();
    let collapsed = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    content_fingerprint_of(&collapsed)
}

fn content_fingerprint_of(text: &str) -> String {
    sha256_hex(text.as_bytes())
}
