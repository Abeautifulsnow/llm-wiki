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
use llm_wiki_llm::{LlmProvider, LlmRequest, LlmResponse};
use llm_wiki_markdown::{estimate_tokens, split_section};

use crate::cache::{generate_cached, remember_validated, repair_request, StageCache};
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
    /// Per-unit output ceiling (config `[llm] max_output_tokens`): thinking
    /// models spend chain-of-thought from this same budget, so 4096 can be
    /// exhausted before any visible JSON is emitted.
    max_output_tokens: u32,
    /// §28 stage cache; only validated unit responses are stored.
    cache: Option<Arc<dyn StageCache>>,
}

impl DocumentAnalyzer {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        prompt: PromptDocument,
        section_target_tokens: u32,
        max_rejected_claim_ratio: f32,
        max_output_tokens: u32,
    ) -> Self {
        Self {
            provider,
            prompt,
            section_target_tokens,
            max_rejected_claim_ratio,
            max_output_tokens,
            cache: None,
        }
    }

    /// Wires the §28 cache: lookups short-circuit identical units, writes
    /// happen only after the unit's schema+evidence validation succeeded.
    pub fn with_cache(mut self, cache: Arc<dyn StageCache>) -> Self {
        self.cache = Some(cache);
        self
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
            max_output_tokens: self.max_output_tokens,
            json_mode: true,
        };

        // ---- Stage 1 (+ one repair on shape failure, PRD §11) ----
        // Hits from the §28 cache consume no LLM request (§37.3).
        let mut llm_request_count = 0u32;
        let (first_response, added) =
            generate_cached(&self.provider, self.cache.as_ref(), base_request.clone()).await?;
        llm_request_count += added;
        // Held until the unit fully validates; only then enters the cache.
        let stage1 = structured::parse_json::<RawAnalysis>(&first_response.text);
        let (raw, validated): (RawAnalysis, (LlmRequest, LlmResponse)) = match stage1 {
            Ok(raw) => (raw, (base_request.clone(), first_response)),
            Err(stage1) => {
                tracing::warn!(reason = %stage1, "analysis stage-1 failed, repairing once");
                let repair = repair_request(&base_request, &template, &[stage1.machine_reason()]);
                let (repair_response, added) =
                    generate_cached(&self.provider, self.cache.as_ref(), repair.clone()).await?;
                llm_request_count += added;
                let parsed = structured::parse_json::<RawAnalysis>(&repair_response.text).map_err(
                    |stage1_repair| {
                        WikiError::SchemaValidation(format!(
                            "schema validation failed after repair: {}",
                            stage1_repair.machine_reason()
                        ))
                    },
                )?;
                (parsed, (repair, repair_response))
            }
        };

        // ---- Stage 2 (referential) + stage 3 (semantic) ----
        let mut raw = raw;
        let mut validated = validated;
        let mut claims = self.verify_claims(&raw.claims, unit, build_id);
        let (mut verified_relations, mut rejected_relations) =
            self.verify_relations(&raw.relations, unit);

        // Over-threshold units get ONE second-chance repair (T1 finding #6):
        // the rejection reasons are fed back verbatim so the model can re-copy
        // evidence quotes; the re-verified response replaces the first. The
        // first response is still NOT cached (PRD §28) — only the repaired
        // response, if the unit now passes.
        let mut rejected_claims = claims.rejected;
        let mut total_claims = claims.verified.len() + rejected_claims.len();
        if total_claims > 0
            && rejected_claims.len() as f32 / total_claims as f32 > self.max_rejected_claim_ratio
        {
            tracing::warn!(
                rejected = rejected_claims.len(),
                total = total_claims,
                "analysis unit over rejected-claim threshold; repairing once with per-claim reasons"
            );
            let feedback = rejected_feedback(&rejected_claims);
            let repair = repair_request(
                &base_request,
                &template,
                &[format!(
                    "evidence validation rejected {} of {} claims:
{}
Re-emit the COMPLETE JSON analysis. For EVERY claim, copy evidence_text character-for-character from the section content — do not rephrase, do not normalize spacing or punctuation; a quote that cannot be located verbatim is rejected again. Extract fewer claims rather than weakly-evidenced ones.",
                    total_claims - claims.verified.len(),
                    total_claims,
                    feedback
                )],
            );
            let (repair_response, added) =
                generate_cached(&self.provider, self.cache.as_ref(), repair.clone()).await?;
            llm_request_count += added;
            let parsed = structured::parse_json::<RawAnalysis>(&repair_response.text).map_err(
                |repair_stage1| {
                    WikiError::SchemaValidation(format!(
                        "evidence repair failed schema validation: {}",
                        repair_stage1.machine_reason()
                    ))
                },
            )?;
            let reverified = self.verify_claims(&parsed.claims, unit, build_id);
            let (re_verified_relations, re_rejected_relations) =
                self.verify_relations(&parsed.relations, unit);
            claims = ClaimVerification {
                verified: reverified.verified,
                rejected: reverified.rejected,
            };
            rejected_relations = re_rejected_relations;
            verified_relations = re_verified_relations;
            rejected_claims = claims.rejected;
            total_claims = claims.verified.len() + rejected_claims.len();
            raw = parsed;
            validated = (repair, repair_response);
            if total_claims > 0
                && rejected_claims.len() as f32 / total_claims as f32
                    > self.max_rejected_claim_ratio
            {
                // Still over threshold: the unit fails, nothing is cached.
                return Err(WikiError::EvidenceValidation(format!(
                    "analysis unit rejected {}/{} claims ({:.0}% > {:.0}% threshold) even after evidence repair; the unit fails and must not produce knowledge",
                    rejected_claims.len(),
                    total_claims,
                    rejected_claims.len() as f32 / total_claims as f32 * 100.0,
                    self.max_rejected_claim_ratio * 100.0
                )));
            }
        } else if total_claims > 0 {
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

        // The unit passed schema + evidence validation: cache it (PRD §28).
        remember_validated(self.cache.as_ref(), &validated.0, &validated.1);

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
            let mut issues = validate_claim(candidate, unit);
            // Cross-section fallback (T1 Run 11): SECTION_NOT_FOUND from
            // segmentation must not mask an evidence quote that locates in
            // another section of this unit — drop the issue and let
            // locate_evidence relocate by content.
            if issues.len() == 1 && issues[0].code == "SECTION_NOT_FOUND" {
                issues.clear();
            }
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

/// Truncated per-claim feedback for the second-chance evidence repair:
/// candidate text (so the model knows WHICH claim) plus the machine reason.
fn rejected_feedback(rejected: &[RejectedClaim]) -> String {
    rejected
        .iter()
        .take(12)
        .map(|r| {
            let text = r.candidate_text.chars().take(120).collect::<String>();
            format!("- claim \"{}\": {}", text.replace('"', "'"), r.reason)
        })
        .collect::<Vec<_>>()
        .join(
            "
",
        )
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

/// Locates an evidence quote inside the cited section and computes its
/// absolute range + digest. Matching is fold-insensitive (T1 finding #8:
/// thinking models transcribe with case/width/punctuation/whitespace drift,
/// and the observed rejected claims were *transcription drift*, not
/// hallucination): both texts are NFKC-normalized, punctuation-stripped,
/// lowercased into a character stream that IGNORES whitespace and punctuation
/// entirely — so hyphenation ("long-running" vs "long running") and CJK/ASCII
/// punctuation differences cannot break the anchor. The returned range covers
/// the ORIGINAL bytes, keeping provenance (range + digest) verifiable.
/// Returns `(section id, absolute range, sha256 of quoted text)`, or the
/// validation issues on failure.
fn locate_evidence(
    evidence_text: &str,
    _hint_start: Option<i64>,
    section_id: &str,
    unit: &AnalysisUnit,
) -> std::result::Result<(SectionId, SourceRange, String), Vec<ValidationIssue>> {
    let needle = fold_with_offsets(evidence_text);

    // Cited-section first (T1 Run 11: SECTION_NOT_FOUND outnumbered
    // EVIDENCE_NOT_IN_SECTION 28:1 — segmentation splits one SOURCE into
    // several units, and thinking models routinely attribute a claim to a
    // section_id from a NEIGHBORING unit. When the quoted text actually
    // locates in one of THIS unit's sections, honor the evidence over the
    // model's bookkeeping and use the section where it was found.)
    let entry = match unit
        .entries
        .iter()
        .find(|e| e.section_id.as_str() == section_id)
    {
        Some(entry) => entry,
        None => {
            // Try every section of the unit; first section where the quote
            // locates wins (deterministic: entries are in document order).
            for candidate in &unit.entries {
                if locate_in_entry(&needle, candidate).is_some() {
                    let (range, digest) = locate_in_entry(&needle, candidate).unwrap();
                    return Ok((
                        candidate.section_id.clone(),
                        SourceRange::new(
                            candidate.range_start + range.0,
                            candidate.range_start + range.1,
                        ),
                        digest,
                    ));
                }
            }
            return Err(vec![ValidationIssue::new(
                "SECTION_NOT_FOUND",
                format!("section id '{section_id}' does not exist in this source"),
            )]);
        }
    };

    if let Some((range, digest)) = locate_in_entry(&needle, entry) {
        return Ok((
            entry.section_id.clone(),
            SourceRange::new(entry.range_start + range.0, entry.range_start + range.1),
            digest,
        ));
    }
    Err(vec![ValidationIssue::new(
        "EVIDENCE_NOT_IN_SECTION",
        format!("evidence quote not found in section '{section_id}'"),
    )])
}

/// Locates the folded needle in one entry; returns
/// `((content byte start, content byte end), digest of original bytes)`.
fn locate_in_entry(needle: &FoldedText, entry: &UnitEntry) -> Option<((usize, usize), String)> {
    let haystack = fold_with_offsets(&entry.content);
    let hay: Vec<char> = haystack.text.chars().collect();
    let needle_chars: Vec<char> = needle.text.chars().collect();
    if needle_chars.is_empty() || needle_chars.len() > hay.len() {
        return None;
    }
    let mut from = 0usize;
    while from + needle_chars.len() <= hay.len() {
        if hay[from..from + needle_chars.len()] == needle_chars[..] {
            let a = from;
            let b = a + needle_chars.len();
            let picked = (haystack.offsets[a].0, haystack.offsets[b - 1].1);
            let quoted = &entry.content[picked.0..picked.1];
            return Some((picked, sha256_hex(quoted.as_bytes())));
        }
        from += 1;
    }
    None
}

/// A fold-insensitive view of a text: NFKC, keep only alphanumeric
/// characters, lowercase — with, for every folded char, the ORIGINAL byte
/// range it came from (ranges repeat across multi-char case expansions, so
/// mapping a folded span back to original bytes is a lookup, never arithmetic).
struct FoldedText {
    text: String,
    /// (original start, original end) per folded char.
    offsets: Vec<(usize, usize)>,
}

fn fold_with_offsets(text: &str) -> FoldedText {
    use unicode_normalization::UnicodeNormalization;
    let normalized: String = text.nfc().collect();
    let mut text = String::with_capacity(normalized.len());
    let mut offsets = Vec::with_capacity(normalized.len());
    for (idx, ch) in normalized.char_indices() {
        if !ch.is_alphanumeric() {
            continue;
        }
        let end = idx + ch.len_utf8();
        for low in ch.to_lowercase() {
            text.push(low);
            offsets.push((idx, end));
        }
    }
    FoldedText { text, offsets }
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

#[cfg(test)]
mod locate_tests {
    use super::*;

    fn unit_with(content: &str) -> AnalysisUnit {
        AnalysisUnit {
            entries: vec![UnitEntry {
                section_id: SectionId::parse("sec_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap(),
                heading_path: vec!["Doc".into()],
                content: content.to_owned(),
                range_start: 0,
            }],
        }
    }

    fn section() -> SectionId {
        SectionId::parse("sec_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap()
    }

    /// T1 finding #8: thinking models transcribe with case/width/punctuation
    /// drift; the locator must still anchor the evidence (range over ORIGINAL
    /// bytes — digest verifiable).
    #[test]
    fn locator_folds_case_width_and_punctuation_drift() {
        // Case drift.
        let unit = unit_with("The plugin runtime retries up to THREE times.");
        let (_, range, _) = locate_evidence("up to three TIMES.", None, section().as_str(), &unit)
            .map_err(|e| e.clone())
            .unwrap();
        assert_eq!(
            &unit.entries[0].content[range.start..range.end],
            "up to THREE times" // trailing "." dropped: fold keeps alphanumeric bytes only
        );

        // Full-width punctuation and CJK comma drift.
        let unit = unit_with("检查点默认每 30 秒持久化一次。");
        let (located, cjk_range, _) = locate_evidence(
            "检查点默认每 30 秒持久化一次，",
            None,
            section().as_str(),
            &unit,
        )
        .map_err(|e| e.clone())
        .unwrap();
        assert_eq!(located, section());
        assert_eq!(
            &unit.entries[0].content[cjk_range.start..cjk_range.end],
            "检查点默认每 30 秒持久化一次"
        );

        // Internal punctuation difference (hyphen vs space) — token-count
        // drift the previous whitespace tokenizer could not bridge.
        let unit = unit_with("a long-running task cannot be killed");
        let (_, hyphen_range, _) = locate_evidence(
            "a long running task cannot be killed",
            None,
            section().as_str(),
            &unit,
        )
        .map_err(|e| e.clone())
        .unwrap();
        assert_eq!(
            &unit.entries[0].content[hyphen_range.start..hyphen_range.end],
            "a long-running task cannot be killed"
        );
    }

    /// T1 Run 11: SECTION_NOT_FOUND outnumbered EVIDENCE_NOT_IN_SECTION
    /// 28:1 — segmentation splits one source into several units and models
    /// attribute claims to section_ids from neighboring units. The locator
    /// must honor the EVIDENCE over the model's bookkeeping.
    #[test]
    fn wrong_section_id_falls_back_to_evidence_location() {
        let wrong_id = SectionId::generate();
        let unit = AnalysisUnit {
            entries: vec![UnitEntry {
                section_id: SectionId::parse("sec_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap(),
                heading_path: vec!["Doc".into()],
                content: "The plugin runtime retries up to three times.".to_owned(),
                range_start: 0,
            }],
        };
        // Cited section id does not exist in this unit, but the quote does.
        let (located, range, _) =
            locate_evidence("up to three times", None, wrong_id.as_str(), &unit)
                .map_err(|e| e.clone())
                .unwrap();
        assert_eq!(located, section());
        assert_eq!(
            &unit.entries[0].content[range.start..range.end],
            "up to three times"
        );

        // A quote absent from EVERY section still fails with the cited-id
        // error (truthful about what went wrong).
        let err = locate_evidence("the runtime never retries", None, wrong_id.as_str(), &unit)
            .unwrap_err();
        assert_eq!(err[0].code, "SECTION_NOT_FOUND");
    }

    #[test]
    fn locator_still_rejects_genuinely_absent_quotes() {
        let unit = unit_with("The plugin runtime retries up to three times.");
        let err = locate_evidence("the runtime never retries", None, section().as_str(), &unit)
            .unwrap_err();
        assert_eq!(err[0].code, "EVIDENCE_NOT_IN_SECTION");
    }
}
