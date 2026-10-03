//! `llm-wiki ask` (audit FIX-020): retrieval-grounded query answering with
//! verified write-back — the Karpathy loop's Query → Reason/Synthesize →
//! Verified Write-back leg.
//!
//! Pipeline: FTS retrieval over the ACTIVE generation (top sections) →
//! context = snippets + the stored claims those pages cite (real claim ids,
//! statements, sources) → LLM synthesis under the compiler's citation
//! contract (`<!-- llm-wiki:cite claim="…" -->`) → **verification** (PRD §11
//! single repair, then fail-closed): every cited id must be a claim of the
//! context, the answer must cite something when claims were provided, and
//! comments must be well-formed. Only after validation do the citations
//! EXPAND from stored anchors (PRD §16, never model-supplied provenance).
//!
//! Write-back (`--write-back`) persists the verified insight with full
//! provenance (query, generation, expanded citations) into `wiki_insights` —
//! a curated layer SEPARATE from the generated wiki, which stays purely
//! source-derived (§36 hand-edited-file stays meaningful). The synthesis
//! shares the §28 cache (task tag `wiki-ask`): the same question over the
//! same retrieval costs zero model calls on repeat.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::BuildId;
use llm_wiki_core::plan::{estimate_tokens, KnowledgeBase, PlanAnchor};
use llm_wiki_llm::structured;
use llm_wiki_llm::LlmProvider;
use llm_wiki_storage::{
    insert_insight, load_knowledge_base, search_index, InsightCitation, InsightRecord,
    PageCitationRecord,
};

use crate::build::prepare_pipeline_env;
use crate::cache::{generate_cached, repair_request, StageCache};
use crate::compile::{expand_citations, scan_citations, strip_citations};
use crate::publish::read_current_pointer;

/// The result of one verified ask.
#[derive(Debug, Clone)]
pub struct AskReport {
    pub query: String,
    /// The answer with citations EXPANDED from stored anchors (PRD §16).
    pub answer: String,
    pub citations: Vec<PageCitationRecord>,
    /// Distinct source paths, first-use order.
    pub sources: Vec<String>,
    pub llm_request_count: u32,
    /// Set when the verified insight was written back.
    pub insight_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawAnswer {
    #[serde(default)]
    answer: String,
}

/// Runs one grounded ask. Returns `Ok(None)` when nothing is published.
/// `write_back = false` verifies and prints without persisting (the dry mode).
pub async fn run_ask(
    workspace_root: &Path,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    query: &str,
    write_back: bool,
) -> Result<Option<AskReport>> {
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
    let pages = load_generation_view_pages(&conn, &build_id)?;
    let base: KnowledgeBase = load_knowledge_base(&conn)?;

    // ---- Retrieval: FTS over the active generation, top sections. ----
    let fts_query = fts_expression(query);
    let hits = search_index(&conn, &fts_query, RETRIEVAL_HITS)?;
    if hits.is_empty() {
        return Err(WikiError::Index(format!(
            "no wiki section matches {query:?}; the published generation may not cover this topic"
        )));
    }
    let hit_slugs: BTreeSet<&str> = hits.iter().map(|hit| hit.slug.as_str()).collect();

    // ---- Context claims: the claims cited by the hit pages (deduped), with
    // their stored statements and source paths. Only these ids are citable. ----
    let mut claims: Vec<(String, String, String)> = Vec::new();
    let mut allowed: BTreeSet<String> = BTreeSet::new();
    let mut anchors_by_claim: BTreeMap<String, Vec<&PlanAnchor>> = BTreeMap::new();
    for page in &pages {
        if !hit_slugs.contains(page.slug.as_str()) {
            continue;
        }
        for citation in &page.citations {
            let id = citation.claim_node_id.as_str().to_owned();
            if !allowed.insert(id.clone()) {
                continue;
            }
            let Some(node) = base.nodes.get(&citation.claim_node_id) else {
                continue;
            };
            let source = node
                .anchors
                .first()
                .map(|anchor| anchor.rel_path.clone())
                .unwrap_or_default();
            claims.push((
                id.clone(),
                node.statement.clone().unwrap_or_default(),
                source,
            ));
            anchors_by_claim.insert(id, node.anchors.iter().collect());
        }
    }

    let context = context_payload(&hits, &claims);

    // Fail closed on budget overshoot (§14: never truncate, never send).
    let env = prepare_pipeline_env(&workspace_root.join(".llm-wiki"), config, &provider)?;
    let cache: Arc<dyn StageCache> = env.cache.clone();
    let max_input = config.analysis.max_input_tokens as u64;
    if estimate_tokens(&context) > max_input {
        return Err(WikiError::Index(format!(
            "ask context is ~{} tokens, above max_input_tokens = {max_input}; narrow the query",
            estimate_tokens(&context)
        )));
    }

    // ---- Synthesis under the citation contract, one repair, fail-closed. ----
    let prompt = crate::prompt::load_prompt("wiki-ask", None)?;
    let stage = prompt.stage_block("synthesis")?;
    let language = "the sources' language".to_owned();
    let (raw, cacheable, requests) = synthesize(
        &provider,
        Some(&cache),
        &stage,
        &language,
        config.llm.max_output_tokens,
        query,
        &context,
    )
    .await?;
    let answer_raw = raw.answer;
    let issues = validate_answer(&answer_raw, &allowed, !claims.is_empty());
    if !issues.is_empty() {
        return Err(WikiError::Compilation(format!(
            "ask synthesis failed validation after repair: {}",
            issues.join("; ")
        )));
    }
    // Validation first, cache second (PRD §28): only the verified synthesis
    // enters the §28 cache.
    if let Some((request, response)) = cacheable {
        crate::cache::remember_validated(Some(&cache), &request, &response);
    }

    // ---- Citation expansion from stored anchors (PRD §16). ----
    let (answer, citations, sources) = expand_citations(&answer_raw, &anchors_by_claim);

    // ---- Verified write-back (the curated insight layer). ----
    let mut insight_id = None;
    if write_back {
        let record = InsightRecord {
            insight_id: llm_wiki_core::ids::InsightId::generate(),
            build_id: build_id.clone(),
            query: query.to_owned(),
            answer: answer.clone(),
            citations: citations
                .iter()
                .map(|citation| {
                    // The human-readable source path comes from the claim's
                    // stored anchor (source_id itself is an opaque ULID).
                    let source = anchors_by_claim
                        .get(citation.claim_node_id.as_str())
                        .and_then(|anchors| anchors.first())
                        .map(|anchor| anchor.rel_path.clone())
                        .unwrap_or_else(|| citation.source_id.as_str().to_owned());
                    InsightCitation {
                        claim_node_id: citation.claim_node_id.as_str().to_owned(),
                        source,
                        heading_path: citation.heading_path.clone(),
                        range: (citation.range.start, citation.range.end),
                        evidence_digest: citation.evidence_digest.clone(),
                    }
                })
                .collect(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        insight_id = Some(record.insight_id.as_str().to_owned());
        let mut writable = llm_wiki_storage::open(&state_db)?;
        insert_insight(&mut writable, &record)?;
    }

    Ok(Some(AskReport {
        query: query.to_owned(),
        answer,
        citations,
        sources,
        llm_request_count: requests,
        insight_id,
    }))
}

/// Loads the active generation's pages (page-level view with citations).
fn load_generation_view_pages(
    conn: &rusqlite::Connection,
    build_id: &BuildId,
) -> Result<Vec<llm_wiki_storage::GenerationPageView>> {
    llm_wiki_storage::load_generation_view(conn, build_id)
}

/// The OR-of-quoted-terms MATCH expression over the shared tokenizer's
/// tokens (mirrors `llm-wiki-search::TextAnalyzer::fts_query`; kept local so
/// the compiler does not depend on the search crate).
fn fts_expression(text: &str) -> String {
    let tokenizer = llm_wiki_storage::default_tokenizer();
    let mut seen = BTreeSet::new();
    let mut terms = Vec::new();
    for token in tokenizer.analyze(text) {
        if seen.insert(token.clone()) {
            terms.push(format!("\"{}\"", token.replace('"', "\"\"")));
        }
    }
    terms.join(" OR ")
}

fn context_payload(
    hits: &[llm_wiki_storage::SearchIndexRow],
    claims: &[(String, String, String)],
) -> String {
    let sections: Vec<serde_json::Value> = hits
        .iter()
        .map(|hit| {
            serde_json::json!({
                "page": hit.title,
                "slug": hit.slug,
                "headings": hit.heading_path,
                "snippet": hit.snippet,
            })
        })
        .collect();
    let claim_values: Vec<serde_json::Value> = claims
        .iter()
        .map(|(id, statement, source)| {
            serde_json::json!({ "id": id, "statement": statement, "source": source })
        })
        .collect();
    serde_json::json!({ "sections": sections, "claims": claim_values }).to_string()
}

/// One synthesis round: shape → single repair (PRD §11). Model errors
/// propagate as hard failures — write-back is fail-closed.
async fn synthesize(
    provider: &Arc<dyn LlmProvider>,
    cache: Option<&Arc<dyn StageCache>>,
    stage: &crate::prompt::PromptDocument,
    language: &str,
    max_output: u32,
    query: &str,
    context: &str,
) -> Result<(
    RawAnswer,
    Option<(llm_wiki_llm::LlmRequest, llm_wiki_llm::LlmResponse)>,
    u32,
)> {
    let template = stage.render(&[
        ("LANGUAGE", language),
        ("QUERY", query),
        ("CONTEXT", context),
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
    let mut requests = added;
    match structured::parse_json::<RawAnswer>(&first.text) {
        Ok(parsed) => {
            let cacheable = (added > 0).then_some((base_request, first));
            Ok((parsed, cacheable, requests))
        }
        Err(shape) => {
            tracing::warn!(reason = %shape, "ask synthesis shape failure, repairing once");
            let repair = repair_request(&base_request, &template, &[shape.machine_reason()]);
            let (repaired, added) = generate_cached(provider, cache, repair.clone()).await?;
            requests += added;
            let parsed = structured::parse_json::<RawAnswer>(&repaired.text).map_err(|shape| {
                WikiError::Compilation(format!(
                    "ask synthesis failed after repair: {}",
                    shape.machine_reason()
                ))
            })?;
            let cacheable = (added > 0).then_some((repair, repaired));
            Ok((parsed, cacheable, requests))
        }
    }
}

/// Referential + structural validation (the compiler's §15 contract applied
/// to answers): every citation must reference a context claim, citations
/// must be present when claim knowledge was provided, comments well-formed,
/// and the stripped answer non-empty. Returns machine-readable issues for
/// the repair loop / final error.
fn validate_answer(
    answer: &str,
    allowed: &BTreeSet<String>,
    claims_available: bool,
) -> Vec<String> {
    let mut issues = Vec::new();
    let cites = scan_citations(answer);
    if cites.is_empty() && claims_available {
        issues.push(
            "UNGROUNDED_ANSWER: the answer cites no claims although claim knowledge was provided"
                .to_owned(),
        );
    }
    for cite in &cites {
        let claim = cite.claim.as_deref().unwrap_or("");
        if claim.is_empty() {
            issues.push("EMPTY_CLAIM_REF: a citation comment has no claim id".to_owned());
        } else if !allowed.contains(claim) {
            issues.push(format!(
                "UNKNOWN_CLAIM_REF: '{claim}' is not a claim of the retrieved context"
            ));
        }
    }
    if answer.matches("<!--").count() != answer.matches("-->").count() {
        issues.push("UNCLOSED_COMMENT: an HTML comment is not closed".to_owned());
    }
    if strip_citations(answer).trim().is_empty() {
        issues.push("EMPTY_ANSWER: the answer body is empty".to_owned());
    }
    issues
}

/// Retrieval width for one ask (§22-style cap; the Context Builder of the
/// full hybrid stack will own its own budget later).
const RETRIEVAL_HITS: usize = 8;
