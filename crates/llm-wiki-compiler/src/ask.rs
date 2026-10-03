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
use llm_wiki_llm::{EmbeddingProvider, LlmProvider};
use llm_wiki_storage::{
    insert_insight, load_knowledge_base, InsightCitation, InsightRecord, PageCitationRecord,
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

/// Hybrid retrieval inputs (§19.3 Vector layer): the embedding provider and
/// model for query-time vector candidates. `None` = lexical-only.
#[derive(Clone)]
pub struct HybridContext<'a> {
    pub provider: &'a Arc<dyn EmbeddingProvider>,
    pub model: String,
}

/// Vector candidates taken per ask (top-K cosine over the embedded
/// sections; the diversity budgets still apply downstream).
const VECTOR_TOP_K: usize = 8;

/// Runs one grounded ask. Returns `Ok(None)` when nothing is published.
/// `write_back = false` verifies and prints without persisting (the dry
/// mode). `hybrid` adds Vector-layer candidates (§19.3 RRF fusion) — without
/// embed coverage it degrades to lexical-only with a warning.
pub async fn run_ask(
    workspace_root: &Path,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    query: &str,
    write_back: bool,
    hybrid: Option<HybridContext<'_>>,
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

    // ---- Retrieval: the Context Builder assembles sections under the
    // budgets (§19.3) — page/source diversity, token ceiling, graph
    // neighborhood. The synthesis budget derives from max_input_tokens. ----
    let context_budget = llm_wiki_search::ContextBudget {
        max_tokens: config.analysis.max_input_tokens as u64,
        ..llm_wiki_search::ContextBudget::default()
    };
    let vector: Vec<llm_wiki_search::VectorCandidate> = match &hybrid {
        None => Vec::new(),
        Some(hybrid) => vector_candidates(&conn, hybrid, query).await?,
    };
    let assembled = llm_wiki_search::build_context(&conn, query, &context_budget, &vector)?;
    let hit_slugs: BTreeSet<&str> = assembled
        .chunks
        .iter()
        .map(|chunk| chunk.slug.as_str())
        .collect();

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

    let context = context_payload(&assembled, &claims);

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

/// Top-K cosine candidates over the embedded sections of the active
/// generation. No coverage → empty (lexical-only), with a warning that says
/// how to fix it. Shared by `ask` and the server's `/v1/context` endpoint.
///
/// NOTE: the returned future is `!Send` (it holds the connection across the
/// embed call); HTTP handlers should compose [`embed_query_vector`] +
/// [`top_cosine_candidates`] instead.
pub async fn vector_candidates(
    conn: &rusqlite::Connection,
    hybrid: &HybridContext<'_>,
    query: &str,
) -> Result<Vec<llm_wiki_search::VectorCandidate>> {
    // Coverage first: with zero stored vectors the query embedding is never
    // spent — degraded lexical-only, never failed (§19.3).
    if !embedding_coverage(conn, &hybrid.model)? {
        tracing::warn!(
            model = %hybrid.model,
            "no embeddings stored for this model; run `llm-wiki embed` for hybrid retrieval"
        );
        return Ok(Vec::new());
    }
    let query_vector = embed_query_vector(hybrid, query).await?;
    top_cosine_candidates(conn, &hybrid.model, query_vector.as_deref())
}

/// Whether any stored section embeddings exist for `model` over the active
/// generation — the cheap pre-check that keeps uncovered workspaces from
/// spending a query embedding.
pub fn embedding_coverage(conn: &rusqlite::Connection, model: &str) -> Result<bool> {
    let sections = llm_wiki_search::active_context_sections(conn)?;
    let hashes: Vec<String> = sections.iter().map(|s| s.text_hash.clone()).collect();
    Ok(!llm_wiki_storage::section_embeddings_by_hash(conn, model, &hashes)?.is_empty())
}

/// The query side of [`vector_candidates`]: one embedding for the query.
/// `None` when the provider returns no vector.
pub async fn embed_query_vector(
    hybrid: &HybridContext<'_>,
    query: &str,
) -> Result<Option<Vec<f32>>> {
    let vectors = hybrid
        .provider
        .embed(&hybrid.model, &[query.to_owned()])
        .await
        .map_err(WikiError::from)?;
    Ok(vectors.first().cloned())
}

/// The storage side of [`vector_candidates`] — synchronous and
/// spawn_blocking-friendly: section hashes + stored embeddings + cosine over
/// the ACTIVE generation's embedded sections (top-K, deterministic).
pub fn top_cosine_candidates(
    conn: &rusqlite::Connection,
    model: &str,
    query_vector: Option<&[f32]>,
) -> Result<Vec<llm_wiki_search::VectorCandidate>> {
    let Some(query_vector) = query_vector else {
        return Ok(Vec::new());
    };
    let sections = llm_wiki_search::active_context_sections(conn)?;
    let hashes: Vec<String> = sections.iter().map(|s| s.text_hash.clone()).collect();
    let stored = llm_wiki_storage::section_embeddings_by_hash(conn, model, &hashes)?;
    if stored.is_empty() {
        tracing::warn!(
            model,
            "no embeddings stored for this model; run `llm-wiki embed` for hybrid retrieval"
        );
        return Ok(Vec::new());
    }
    let mut scored: Vec<(f32, &str)> = sections
        .iter()
        .filter_map(|section| {
            stored
                .get(&section.text_hash)
                .map(|vector| (cosine(query_vector, vector), section.text_hash.as_str()))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(b.1))
    });
    Ok(scored
        .into_iter()
        .take(VECTOR_TOP_K)
        .map(|(score, text_hash)| llm_wiki_search::VectorCandidate {
            text_hash: text_hash.to_owned(),
            score,
        })
        .collect())
}

/// Cosine similarity (vectors of different length score 0 — different
/// embedding dimensions never mix).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

fn context_payload(
    assembled: &llm_wiki_search::AssembledContext,
    claims: &[(String, String, String)],
) -> String {
    let sections: Vec<serde_json::Value> = assembled
        .chunks
        .iter()
        .map(|chunk| {
            serde_json::json!({
                "page": chunk.title,
                "slug": chunk.slug,
                "headings": chunk.heading_path,
                "snippet": chunk.snippet,
                "sources": chunk.sources,
            })
        })
        .collect();
    let related: Vec<serde_json::Value> = assembled
        .neighbors
        .iter()
        .map(|neighbor| {
            serde_json::json!({
                "via_page": neighbor.from_slug,
                "relation": neighbor.relation,
                "node": neighbor.label,
                "kind": neighbor.node_type,
            })
        })
        .collect();
    let claim_values: Vec<serde_json::Value> = claims
        .iter()
        .map(|(id, statement, source)| {
            serde_json::json!({ "id": id, "statement": statement, "source": source })
        })
        .collect();
    serde_json::json!({
        "sections": sections,
        "related_graph_nodes": related,
        "claims": claim_values,
    })
    .to_string()
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
