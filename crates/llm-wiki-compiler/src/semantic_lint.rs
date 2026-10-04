//! Semantic lint (PRD §36 extension, audit FIX-019): LLM-judged quality
//! checks the structural lint cannot express — contradictions between claims
//! a page cites, superseded facts, weak synthesis, and corpus-level coverage
//! gaps (the entry point of the Karpathy compounding loop).
//!
//! Design contract:
//! - **Advisory only**: findings never fail the command or the exit code;
//!   lint is a read-only product capability and the judgments are
//!   model-derived (false positives are possible). They are reported
//!   separately from the deterministic `LintReport`.
//! - **Fail-open per page**: a page whose review cannot be produced (budget
//!   overshoot, shape failure, validation failure after the single repair)
//!   lands in `skipped_pages` — never silently dropped, never turned into a
//!   fabricated finding.
//! - **Validated responses only enter the §28 cache** (PRD §28): kinds are
//!   enum-checked, every finding's `claim_ids` must be claims the page
//!   actually cites, and excerpts must appear in the body. Invalid entries
//!   are dropped individually with a warning — an advisory diagnostic must
//!   never fabricate issues.
//! - **Deterministic ordering**: findings sort by (kind, page, message); the
//!   corpus gap scan runs after all page reviews.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::BuildId;
use llm_wiki_core::plan::{estimate_tokens, KnowledgeBase};
use llm_wiki_llm::structured;
use llm_wiki_llm::LlmProvider;
use llm_wiki_storage::load_generation_view;

use crate::build::prepare_pipeline_env;
use crate::cache::{generate_cached, repair_request, StageCache};
use crate::compile::strip_citations;
use crate::prompt::load_prompt;
use crate::publish::read_current_pointer;

/// One LLM-judged semantic finding (advisory severity by construction).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemanticFinding {
    pub kind: SemanticFindingKind,
    /// Slug of the reviewed page; `"(corpus)"` for coverage gaps.
    pub page_slug: String,
    /// Cited claim ids involved (empty for weak synthesis and gaps).
    pub claim_ids: Vec<String>,
    pub message: String,
}

/// The semantic checks of the V1 lint (PRD §36 extension; the audit's
/// `missing concept` folds into [`SemanticFindingKind::KnowledgeGap`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticFindingKind {
    Contradiction,
    Superseded,
    WeakSynthesis,
    KnowledgeGap,
    SupersededInsight,
    ContradictedInsight,
}

impl SemanticFindingKind {
    pub fn label(&self) -> &'static str {
        match self {
            SemanticFindingKind::Contradiction => "contradictory-claims",
            SemanticFindingKind::Superseded => "superseded-facts",
            SemanticFindingKind::WeakSynthesis => "weak-synthesis",
            SemanticFindingKind::KnowledgeGap => "knowledge-gap",
            SemanticFindingKind::SupersededInsight => "superseded-insight",
            SemanticFindingKind::ContradictedInsight => "contradicted-insight",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "contradiction" => Some(Self::Contradiction),
            "superseded" => Some(Self::Superseded),
            "weak-synthesis" => Some(Self::WeakSynthesis),
            "superseded-insight" => Some(Self::SupersededInsight),
            "contradicted-insight" => Some(Self::ContradictedInsight),
            _ => None,
        }
    }
}

/// Result of one semantic lint run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SemanticReport {
    pub findings: Vec<SemanticFinding>,
    /// Pages (or `"(corpus)"`) whose review could not be produced — budget
    /// overshoot or invalid model responses after the single repair.
    pub skipped_pages: Vec<String>,
    pub pages_reviewed: usize,
}

impl SemanticReport {
    fn finish(mut self) -> Self {
        self.findings.sort();
        self.skipped_pages.sort();
        self.skipped_pages.dedup();
        self
    }
}

/// Raw model response for one page review (shape validation only here).
#[derive(Debug, Deserialize)]
struct RawPageFindings {
    #[serde(default)]
    findings: Vec<RawFinding>,
}

#[derive(Debug, Deserialize)]
struct RawFinding {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    claim_ids: Vec<String>,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    excerpt: String,
}

#[derive(Debug, Deserialize)]
struct RawCorpusGaps {
    #[serde(default)]
    gaps: Vec<RawGap>,
}

#[derive(Debug, Deserialize)]
struct RawGap {
    #[serde(default)]
    topic: String,
    #[serde(default)]
    reason: String,
}

/// Per-page review inputs snapped together from the generation view and the
/// knowledge base.
struct PageReview {
    slug: String,
    title: String,
    claims: Vec<(String, String)>,
    cited: BTreeSet<String>,
    body: String,
}

/// Runs the semantic lint over the currently published generation. Returns
/// `Ok(None)` when nothing is published. Requires an `[llm]` provider; model
/// calls are validated before use and cached in the §28 space (so a repeated
/// `lint --semantic` pays zero model calls).
pub async fn run_semantic_lint(
    workspace_root: &Path,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
) -> Result<Option<SemanticReport>> {
    let state_db = workspace_root.join(".llm-wiki").join("state.db");
    if !state_db.exists() {
        return Ok(None);
    }
    let wiki_dir = lexical_absolute(workspace_root, &config.project.wiki_dir);
    let Some(pointer) = read_current_pointer(&crate::publish::PublishPaths::new(&wiki_dir))? else {
        return Ok(None);
    };
    let build_id = BuildId::parse(&pointer.build_id)?;
    let conn = llm_wiki_storage::open(&state_db)?;
    let pages = load_generation_view(&conn, &build_id)?;
    if pages.is_empty() {
        return Err(WikiError::PublishRecovery(format!(
            "current.json points at build {} but no page rows exist for it; run `llm-wiki doctor`",
            pointer.build_id
        )));
    }
    let base: KnowledgeBase = llm_wiki_storage::load_knowledge_base(&conn)?;

    // §28 cache: same space as the build pipeline (same model, config hash
    // and prompt-version table), so lint and build never cross-contaminate
    // and a repeated lint is free.
    let env = prepare_pipeline_env(&workspace_root.join(".llm-wiki"), config, &provider)?;
    let cache: Arc<dyn StageCache> = env.cache.clone();

    let statements: BTreeMap<String, String> = base
        .nodes
        .iter()
        .filter(|(_, node)| node.kind == "claim")
        .filter_map(|(id, node)| node.statement.clone().map(|s| (id.as_str().to_owned(), s)))
        .collect();

    let prompt = load_prompt("wiki-semantic-lint", None)?;
    let page_stage = prompt.stage_block("page-review")?;
    let max_input = config.analysis.max_input_tokens as u64;
    let max_output = config.llm.max_output_tokens;
    let language = "the sources' language".to_owned();

    let reviews: Vec<PageReview> = pages
        .iter()
        .map(|page| {
            let mut claims: Vec<(String, String)> = Vec::new();
            let mut cited: BTreeSet<String> = BTreeSet::new();
            for citation in &page.citations {
                let id = citation.claim_node_id.as_str().to_owned();
                if cited.insert(id.clone()) {
                    let statement = statements
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| "(statement unavailable)".to_owned());
                    claims.push((id, statement));
                }
            }
            PageReview {
                slug: page.slug.clone(),
                title: page.title.clone(),
                claims,
                cited,
                body: strip_citations(&page.content),
            }
        })
        .collect();

    // ---- Per-page reviews, bounded-parallel: SQLite work stays on this
    // task, only the model calls run inside the JoinSet window (PRD §27). ----
    let mut findings: Vec<SemanticFinding> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut reviewed = 0usize;
    let mut next = 0usize;
    let mut join_set = tokio::task::JoinSet::new();
    let concurrency = config.llm.max_concurrency.max(1) as usize;
    while next < reviews.len() || !join_set.is_empty() {
        while next < reviews.len() && join_set.len() < concurrency {
            let review = &reviews[next];
            // Budget overshoot is an advisory skip, never a truncation (§14).
            let payload = page_payload(&review.title, &review.claims, &review.body);
            if estimate_tokens(&payload) > max_input {
                tracing::warn!(page = %review.slug, "page exceeds max_input_tokens for semantic review; skipped");
                skipped.push(review.slug.clone());
                next += 1;
                continue;
            }
            let stage = page_stage.clone();
            let cache = Arc::clone(&cache);
            let provider = Arc::clone(&provider);
            let language = language.clone();
            let review_owned = PageReview {
                slug: review.slug.clone(),
                title: review.title.clone(),
                claims: review.claims.clone(),
                cited: review.cited.clone(),
                body: review.body.clone(),
            };
            join_set.spawn(async move {
                let (raw, cacheable) = match review_page(
                    &provider,
                    Some(&cache),
                    &stage,
                    &language,
                    max_output,
                    &review_owned.title,
                    &review_owned.claims,
                    &review_owned.body,
                )
                .await
                {
                    Ok(result) => result,
                    Err(err) => return Err((review_owned.slug, err)),
                };
                // Validation first, cache second (PRD §28): the response
                // enters the cache only once the findings were screened.
                let findings = validate_findings(
                    raw,
                    &review_owned.slug,
                    &review_owned.cited,
                    &review_owned.body,
                );
                if let Some((request, response)) = cacheable {
                    crate::cache::remember_validated(Some(&cache), &request, &response);
                }
                Ok::<_, (String, WikiError)>(findings)
            });
            next += 1;
        }
        if let Some(joined) = join_set.join_next().await {
            // A task panic is fatal (propagates); a review error is an
            // advisory page skip.
            match joined {
                Ok(Ok(page_findings)) => {
                    reviewed += 1;
                    findings.extend(page_findings);
                }
                Ok(Err((slug, err))) => {
                    tracing::warn!(page = %slug, error = %err, "semantic review failed; page skipped");
                    skipped.push(slug);
                }
                Err(e) => {
                    return Err(WikiError::Llm(format!("semantic lint task panicked: {e}")));
                }
            }
        }
    }
    drop(join_set);

    // ---- Corpus-level coverage gaps (one request; titles are small). ----
    let gap_stage = prompt.stage_block("corpus-gaps")?;
    let titles = pages
        .iter()
        .map(|page| format!("{} | {}", page.title, page.category))
        .collect::<Vec<_>>()
        .join("\n");
    match review_gaps(
        &provider,
        Some(&cache),
        &gap_stage,
        &language,
        max_output,
        &titles,
    )
    .await
    {
        Ok((gaps, cacheable)) => {
            let mut kept = 0usize;
            for gap in gaps {
                if gap.topic.trim().is_empty() || gap.reason.trim().is_empty() {
                    continue;
                }
                kept += 1;
                findings.push(SemanticFinding {
                    kind: SemanticFindingKind::KnowledgeGap,
                    page_slug: "(corpus)".to_owned(),
                    claim_ids: Vec::new(),
                    message: format!("{} — {}", gap.topic.trim(), gap.reason.trim()),
                });
            }
            let _ = kept;
            // Shape-valid response: cached regardless of how many gaps
            // survived screening (the same input yields the same judgment).
            if let Some((request, response)) = cacheable {
                crate::cache::remember_validated(Some(&cache), &request, &response);
            }
        }
        Err(err) => {
            tracing::warn!(error = %err, "coverage-gap scan failed; corpus gaps not reported");
            skipped.push("(corpus)".to_owned());
        }
    }

    // ---- Insight review (the write-back loop's LLM consumer): each stored
    // insight is judged against the CURRENT claims it cites, so a verified
    // synthesis that the recompiled wiki no longer supports is surfaced as
    // superseded/contradicted. Sequential by design: insights are a curated
    // layer (hand-fuls, not pages). Findings keep the insight identity in
    // the slug (`(insight <id>)`); the §28 cache makes re-runs free. ----
    let insight_stage = prompt.stage_block("insight-review")?;
    let insights = llm_wiki_storage::list_insights(&conn).unwrap_or_default();
    for insight in insights {
        let slug = format!("(insight {})", insight.insight_id.as_str());
        let cited: BTreeSet<String> = insight
            .citations
            .iter()
            .map(|citation| citation.claim_node_id.clone())
            .collect();
        let claims: Vec<(String, String)> = insight
            .citations
            .iter()
            .filter_map(|citation| {
                statements
                    .get(&citation.claim_node_id)
                    .map(|statement| (citation.claim_node_id.clone(), statement.clone()))
            })
            .collect();
        if claims.is_empty() {
            // Every cited claim vanished from the registry — the structural
            // stale-insight lint owns that report; there is nothing left to
            // judge semantically.
            continue;
        }
        let payload = insight_payload(&insight.query, &insight.answer, &claims);
        if estimate_tokens(&payload) > max_input {
            tracing::warn!(insight = %insight.insight_id, "insight exceeds max_input_tokens for semantic review; skipped");
            skipped.push(slug);
            continue;
        }
        match review_insight(
            &provider,
            Some(&cache),
            &insight_stage,
            &language,
            max_output,
            &insight.query,
            &insight.answer,
            &claims,
        )
        .await
        {
            Ok((raw, cacheable)) => {
                findings.extend(validate_findings(raw, &slug, &cited, &insight.answer));
                if let Some((request, response)) = cacheable {
                    crate::cache::remember_validated(Some(&cache), &request, &response);
                }
            }
            Err(err) => {
                tracing::warn!(insight = %insight.insight_id, error = %err, "insight review failed; skipped");
                skipped.push(slug);
            }
        }
    }

    Ok(Some(
        SemanticReport {
            findings,
            skipped_pages: skipped,
            pages_reviewed: reviewed,
        }
        .finish(),
    ))
}

fn page_payload(title: &str, claims: &[(String, String)], body: &str) -> String {
    let mut payload = format!("PAGE {title}\n");
    for (id, statement) in claims {
        payload.push_str(&format!("{id}: {statement}\n"));
    }
    payload.push_str(body);
    payload
}

fn insight_payload(query: &str, answer: &str, claims: &[(String, String)]) -> String {
    let mut payload = format!("QUERY {query}\n");
    for (id, statement) in claims {
        payload.push_str(&format!("{id}: {statement}\n"));
    }
    payload.push_str(answer);
    payload
}

/// One insight review: shape → single repair (mirrors [`review_page`]; the
/// quoted text is the ANSWER, not a page body).
#[allow(clippy::too_many_arguments)]
async fn review_insight(
    provider: &Arc<dyn LlmProvider>,
    cache: Option<&Arc<dyn StageCache>>,
    stage: &crate::prompt::PromptDocument,
    language: &str,
    max_output: u32,
    query: &str,
    answer: &str,
    claims: &[(String, String)],
) -> Result<(
    RawPageFindings,
    Option<(llm_wiki_llm::LlmRequest, llm_wiki_llm::LlmResponse)>,
)> {
    let claims_json = serde_json::json!(claims
        .iter()
        .map(|(id, statement)| serde_json::json!({ "id": id, "statement": statement }))
        .collect::<Vec<_>>())
    .to_string();
    let template = stage.render(&[
        ("LANGUAGE", language),
        ("QUERY", query),
        ("ANSWER", answer),
        ("CLAIMS", &claims_json),
    ]);
    let base_request = llm_wiki_llm::LlmRequest {
        task_tag: stage.name.clone(),
        system: None,
        prompt: template.replace("{{REPAIR_NOTES}}", ""),
        temperature: 0.0,
        max_output_tokens: max_output,
        json_mode: true,
    };
    let (first, added) = generate_cached(provider, cache, base_request.clone()).await?;
    match structured::parse_json::<RawPageFindings>(&first.text) {
        Ok(parsed) => {
            let cacheable = (added > 0).then_some((base_request, first));
            Ok((parsed, cacheable))
        }
        Err(shape) => {
            tracing::warn!(reason = %shape, query = %query, "insight review shape failure, repairing once");
            let repair = repair_request(&base_request, &template, &[shape.machine_reason()]);
            let (repaired, added) = generate_cached(provider, cache, repair.clone()).await?;
            let parsed =
                structured::parse_json::<RawPageFindings>(&repaired.text).map_err(|shape| {
                    WikiError::Llm(format!(
                        "insight review failed after repair: {}",
                        shape.machine_reason()
                    ))
                })?;
            let cacheable = (added > 0).then_some((repair, repaired));
            Ok((parsed, cacheable))
        }
    }
}

/// One page review: shape → single repair. Model errors propagate (the
/// caller turns them into a page skip).
/// Returns the parsed review plus the (request, response) pair to cache —
/// `None` on a cache hit (already stored). The pair is remembered only AFTER
/// the caller validated the findings (PRD §28).
#[allow(clippy::too_many_arguments)]
async fn review_page(
    provider: &Arc<dyn LlmProvider>,
    cache: Option<&Arc<dyn StageCache>>,
    stage: &crate::prompt::PromptDocument,
    language: &str,
    max_output: u32,
    title: &str,
    claims: &[(String, String)],
    body: &str,
) -> Result<(
    RawPageFindings,
    Option<(llm_wiki_llm::LlmRequest, llm_wiki_llm::LlmResponse)>,
)> {
    let claims_json = serde_json::json!(claims
        .iter()
        .map(|(id, statement)| serde_json::json!({ "id": id, "statement": statement }))
        .collect::<Vec<_>>())
    .to_string();
    let template = stage.render(&[
        ("LANGUAGE", language),
        ("PAGE", title),
        ("CLAIMS", &claims_json),
        ("BODY", body),
    ]);
    let base_request = llm_wiki_llm::LlmRequest {
        task_tag: stage.name.clone(),
        system: None,
        prompt: template.replace("{{REPAIR_NOTES}}", ""),
        temperature: 0.0,
        max_output_tokens: max_output,
        json_mode: true,
    };
    let (first, added) = generate_cached(provider, cache, base_request.clone()).await?;
    match structured::parse_json::<RawPageFindings>(&first.text) {
        Ok(parsed) => {
            let cacheable = (added > 0).then_some((base_request, first));
            Ok((parsed, cacheable))
        }
        Err(shape) => {
            tracing::warn!(reason = %shape, page = %title, "semantic review shape failure, repairing once");
            let repair = repair_request(&base_request, &template, &[shape.machine_reason()]);
            let (repaired, added) = generate_cached(provider, cache, repair.clone()).await?;
            let parsed =
                structured::parse_json::<RawPageFindings>(&repaired.text).map_err(|shape| {
                    WikiError::Llm(format!(
                        "semantic review failed after repair: {}",
                        shape.machine_reason()
                    ))
                })?;
            let cacheable = (added > 0).then_some((repair, repaired));
            Ok((parsed, cacheable))
        }
    }
}

/// Referential + enum validation over the model's findings. Invalid entries
/// are dropped individually with a warning: an advisory diagnostic must not
/// fabricate issues, and a fully-invalid response leaves the page simply
/// unreported rather than guessed about.
fn validate_findings(
    raw: RawPageFindings,
    page_slug: &str,
    cited: &BTreeSet<String>,
    body: &str,
) -> Vec<SemanticFinding> {
    let mut findings = Vec::new();
    for finding in raw.findings {
        let Some(kind) = SemanticFindingKind::parse(&finding.kind) else {
            tracing::warn!(kind = %finding.kind, page = %page_slug, "unknown semantic finding kind; dropped");
            continue;
        };
        let mut claim_ids = Vec::new();
        let mut all_known = true;
        for id in &finding.claim_ids {
            if cited.contains(id) {
                claim_ids.push(id.clone());
            } else {
                tracing::warn!(claim = %id, page = %page_slug, "finding cites an unknown claim; entry dropped");
                all_known = false;
            }
        }
        let no_claims_ok =
            kind == SemanticFindingKind::WeakSynthesis && finding.claim_ids.is_empty();
        if !all_known && !no_claims_ok {
            continue;
        }
        if finding.reason.trim().is_empty() {
            continue;
        }
        // Excerpts must be verbatim body quotes when provided; a hallucinated
        // excerpt downgrades the finding to its reason only.
        let excerpt = finding.excerpt.trim();
        let message = if excerpt.is_empty() {
            finding.reason.trim().to_owned()
        } else if body.contains(excerpt) {
            format!("{} — \"{excerpt}\"", finding.reason.trim())
        } else {
            tracing::warn!(page = %page_slug, "finding excerpt is not a verbatim body quote; dropped excerpt");
            finding.reason.trim().to_owned()
        };
        findings.push(SemanticFinding {
            kind,
            page_slug: page_slug.to_owned(),
            claim_ids,
            message,
        });
    }
    findings
}

/// Corpus-level coverage gaps: one request over page titles.
async fn review_gaps(
    provider: &Arc<dyn LlmProvider>,
    cache: Option<&Arc<dyn StageCache>>,
    stage: &crate::prompt::PromptDocument,
    language: &str,
    max_output: u32,
    titles: &str,
) -> Result<(
    Vec<RawGap>,
    Option<(llm_wiki_llm::LlmRequest, llm_wiki_llm::LlmResponse)>,
)> {
    let template = stage.render(&[("LANGUAGE", language), ("TITLES", titles)]);
    let base_request = llm_wiki_llm::LlmRequest {
        task_tag: stage.name.clone(),
        system: None,
        prompt: template.replace("{{REPAIR_NOTES}}", ""),
        temperature: 0.0,
        max_output_tokens: max_output,
        json_mode: true,
    };
    let (first, added) = generate_cached(provider, cache, base_request.clone()).await?;
    match structured::parse_json::<RawCorpusGaps>(&first.text) {
        Ok(parsed) => {
            let cacheable = (added > 0).then_some((base_request, first));
            Ok((parsed.gaps, cacheable))
        }
        Err(shape) => {
            tracing::warn!(reason = %shape, "coverage-gap scan shape failure, repairing once");
            let repair = repair_request(&base_request, &template, &[shape.machine_reason()]);
            let (repaired, added) = generate_cached(provider, cache, repair.clone()).await?;
            let parsed =
                structured::parse_json::<RawCorpusGaps>(&repaired.text).map_err(|shape| {
                    WikiError::Llm(format!(
                        "coverage-gap scan failed after repair: {}",
                        shape.machine_reason()
                    ))
                })?;
            let cacheable = (added > 0).then_some((repair, repaired));
            Ok((parsed.gaps, cacheable))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn cited(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    #[test]
    fn findings_validation_drops_unknown_claims_and_kinds() {
        let body = "The runtime retries transitions. A retry budget caps the cost.";
        let raw = RawPageFindings {
            findings: vec![
                RawFinding {
                    kind: "contradiction".into(),
                    claim_ids: vec!["kn_A".into()],
                    reason: "conflicting retry guarantees".into(),
                    excerpt: "retries transitions".into(),
                },
                // Unknown claim → dropped entirely.
                RawFinding {
                    kind: "contradiction".into(),
                    claim_ids: vec!["kn_NOPE".into()],
                    reason: "hallucinated reference".into(),
                    excerpt: String::new(),
                },
                // Unknown kind → dropped.
                RawFinding {
                    kind: "vibes".into(),
                    claim_ids: vec!["kn_A".into()],
                    reason: "mystery".into(),
                    excerpt: String::new(),
                },
                // Empty reason → dropped.
                RawFinding {
                    kind: "superseded".into(),
                    claim_ids: vec!["kn_A".into()],
                    reason: String::new(),
                    excerpt: String::new(),
                },
            ],
        };
        let findings = validate_findings(raw, "page", &cited(&["kn_A"]), body);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert_eq!(findings[0].kind, SemanticFindingKind::Contradiction);
        assert_eq!(findings[0].claim_ids, vec!["kn_A".to_owned()]);
        // The verbatim excerpt rides into the message.
        assert!(findings[0].message.contains("retries transitions"));
    }

    #[test]
    fn non_verbatim_excerpts_are_downgraded_not_kept() {
        let body = "Grounded prose only.";
        let raw = RawPageFindings {
            findings: vec![RawFinding {
                kind: "weak-synthesis".into(),
                claim_ids: vec![],
                reason: "listy body".into(),
                excerpt: "hallucinated quote".into(),
            }],
        };
        let findings = validate_findings(raw, "page", &cited(&["kn_A"]), body);
        assert_eq!(findings.len(), 1);
        assert!(!findings[0].message.contains("hallucinated"));
    }

    #[test]
    fn weak_synthesis_needs_no_claims() {
        let raw = RawPageFindings {
            findings: vec![RawFinding {
                kind: "weak-synthesis".into(),
                claim_ids: vec![],
                reason: "concatenated statements".into(),
                excerpt: String::new(),
            }],
        };
        let findings = validate_findings(raw, "page", &cited(&[]), "body");
        assert_eq!(findings.len(), 1);
        assert!(findings[0].claim_ids.is_empty());
    }
}
