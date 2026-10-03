//! Context Builder (§19.3, audit follow-up): assembles retrieval candidates
//! into a synthesis-ready context under explicit budgets — token budget,
//! page diversity, source diversity — plus the graph neighborhood of the
//! selected pages. It is the composition layer the §19.3 four-layer stack
//! (Lexical + Vector + Graph + **Context Builder**) feeds into: lexical hits
//! drive it today, a Vector layer can rank or propose candidates for it
//! later without changing its contract.
//!
//! Selection is deterministic: candidates are ordered by (rank, slug,
//! heading path) and admitted greedily while the page and source diversity
//! caps hold. Everything dropped is counted, never silent.

use std::collections::{BTreeMap, BTreeSet};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::WikiPageId;
use llm_wiki_core::plan::estimate_tokens;
use llm_wiki_storage::{
    generation_page_sources, get_active_build_id, graph_expand_from_page, search_index, Connection,
};

/// The semantic text of one context section — the object the Vector layer
/// embeds and hashes. ONE definition so the embed backfill, the hybrid query
/// path and the candidate resolution can never drift.
pub fn context_section_text(title: &str, heading_path: &[String], body: &str) -> String {
    format!("{title}\n{}\n{body}", heading_path.join(" > "))
}

/// The content hash identifying a context section's text
/// ([`context_section_text`], sha256).
pub fn context_section_hash(title: &str, heading_path: &[String], body: &str) -> String {
    sha256_hex(context_section_text(title, heading_path, body).as_bytes())
}

/// One section of the ACTIVE generation with its identity hash — the row
/// set the Vector layer embeds and searches over.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextSection {
    pub text_hash: String,
    pub page_id: WikiPageId,
    pub slug: String,
    pub title: String,
    pub heading_path: Vec<String>,
    pub body: String,
}

/// Loads the ACTIVE generation's context sections (hash computed here —
/// one definition, see [`context_section_hash`]).
pub fn active_context_sections(conn: &Connection) -> Result<Vec<ContextSection>> {
    let Some(active) = get_active_build_id(conn)? else {
        return Err(WikiError::Index(
            "nothing published; run build first".into(),
        ));
    };
    let mut stmt = conn
        .prepare(
            "SELECT page_id, slug, title, heading_path_json, body
             FROM wiki_page_text WHERE build_id = ?1 ORDER BY page_id, text_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare context sections: {e}")))?;
    let rows = stmt
        .query_map([active.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("context sections: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        let (page_id, slug, title, heading_json, body) =
            row.map_err(|e| WikiError::Storage(e.to_string()))?;
        let heading_path: Vec<String> = serde_json::from_str(&heading_json).unwrap_or_default();
        out.push(ContextSection {
            text_hash: context_section_hash(&title, &heading_path, &body),
            page_id: WikiPageId::from_validated(page_id),
            slug,
            title,
            heading_path,
            body,
        });
    }
    Ok(out)
}

/// A Vector-layer candidate (§19.3): one context-section text hash plus a
/// similarity score (larger is better — cosine). Fusion merges these with
/// the lexical hits; resolution (snippet, page info) happens here.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorCandidate {
    pub text_hash: String,
    pub score: f32,
}

/// Budget and diversity controls for one context assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    /// Hard cap on selected chunks.
    pub max_chunks: usize,
    /// Estimated-token ceiling for the serialized context.
    pub max_tokens: u64,
    /// Distinct pages the context may draw from (page diversity).
    pub max_pages: usize,
    /// Chunks per source path (source diversity; chunks of pages without
    /// citations carry no source cost).
    pub max_per_source: usize,
    /// One-hop graph expansion width per selected page.
    pub graph_limit: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_chunks: 12,
            max_tokens: 24_000,
            max_pages: 4,
            max_per_source: 4,
            graph_limit: 4,
        }
    }
}

/// One selected retrieval chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextChunk {
    pub page_id: WikiPageId,
    pub slug: String,
    pub title: String,
    pub heading_path: Vec<String>,
    pub snippet: String,
    /// RRF-fused retrieval score (larger is better) over the lexical and
    /// vector candidate lists.
    pub score: f32,
    /// Distinct source paths backing this page (empty for uncited pages).
    pub sources: Vec<String>,
}

/// One graph neighbor of a selected page, with the page that reached it.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextNeighbor {
    pub from_slug: String,
    pub node_id: String,
    pub node_type: String,
    pub label: String,
    pub relation: String,
}

/// The assembled context: the final chunk selection, the graph neighborhood
/// around it, and transparency counters.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledContext {
    /// Final selection, in deterministic (rank, slug, heading) order.
    pub chunks: Vec<ContextChunk>,
    /// Graph neighborhood of the selected pages, deduped by node id,
    /// deterministic order.
    pub neighbors: Vec<ContextNeighbor>,
    /// estimate_tokens over the serialized context — what a synthesis prompt
    /// would actually carry.
    pub estimated_tokens: u64,
    /// Candidates that lost to the budgets (chunks + neighbor slots).
    pub dropped: usize,
}

/// Assembles the context for one query over the ACTIVE generation.
/// Deterministic for a given database state; fails honestly when nothing is
/// published or nothing matches.
pub fn build_context(
    conn: &Connection,
    query: &str,
    budget: &ContextBudget,
    vector: &[VectorCandidate],
) -> Result<AssembledContext> {
    build_context_with_reranker(conn, query, budget, vector, None)
}

/// [`build_context`] with an optional reranker applied to the fused
/// candidates before the budget consumes them (PRD §50 V0.5: the rerank
/// abstraction slots between fusion and selection).
pub fn build_context_with_reranker(
    conn: &Connection,
    query: &str,
    budget: &ContextBudget,
    vector: &[VectorCandidate],
    reranker: Option<&dyn crate::rerank::Reranker>,
) -> Result<AssembledContext> {
    let Some(active) = get_active_build_id(conn)? else {
        return Err(WikiError::Index(
            "nothing published; run build first".into(),
        ));
    };

    // Over-fetch so the diversity caps can still fill the budget when the
    // rank-best candidates share pages or sources. The shared analyzer keeps
    // query and index normalization from drifting (PRD §20).
    let fts_query = crate::TextAnalyzer.fts_query(query);
    let lexical = search_index(conn, &fts_query, budget.max_chunks * 3)?;

    // ---- Fusion (§19.3 rank fusion): Reciprocal Rank Fusion over the
    // lexical and vector candidate lists. RRF needs only the ORDER of each
    // list, so bm25 and cosine never have to be comparable. Identity =
    // (page_id, heading path) — the context-section identity. ----
    const RRF_K: f64 = 60.0;
    struct Candidate {
        section: ContextSection,
        snippet: String,
        lexical_rank: Option<usize>,
        vector_rank: Option<usize>,
    }
    let mut by_key: BTreeMap<(String, String), Candidate> = BTreeMap::new();
    for (rank, hit) in lexical.iter().enumerate() {
        let snippet = hit.snippet.clone();
        let key = (
            hit.page_id.as_str().to_owned(),
            hit.heading_path.join("\u{1f}"),
        );
        by_key
            .entry(key)
            .or_insert_with(|| Candidate {
                section: ContextSection {
                    text_hash: String::new(),
                    page_id: hit.page_id.clone(),
                    slug: hit.slug.clone(),
                    title: hit.title.clone(),
                    heading_path: hit.heading_path.clone(),
                    body: String::new(),
                },
                snippet,
                lexical_rank: Some(rank),
                vector_rank: None,
            })
            .lexical_rank = Some(rank);
    }
    if !vector.is_empty() {
        let sections = active_context_sections(conn)?;
        let mut by_hash: BTreeMap<&str, &ContextSection> = BTreeMap::new();
        for section in &sections {
            by_hash.insert(section.text_hash.as_str(), section);
        }
        for (rank, candidate) in vector.iter().enumerate() {
            let Some(section) = by_hash.get(candidate.text_hash.as_str()) else {
                // A stale or foreign hash (other model, pre-embed generation):
                // skipped, never guessed about.
                tracing::debug!(hash = %candidate.text_hash, "vector candidate without a matching context section; skipped");
                continue;
            };
            let snippet = excerpt(&section.body);
            let key = (
                section.page_id.as_str().to_owned(),
                section.heading_path.join("\u{1f}"),
            );
            by_key
                .entry(key)
                .or_insert_with(|| Candidate {
                    section: (*section).clone(),
                    snippet,
                    lexical_rank: None,
                    vector_rank: Some(rank),
                })
                .vector_rank = Some(rank);
        }
    }
    let mut fused: Vec<(f64, &Candidate)> = by_key
        .values()
        .map(|candidate| {
            let mut score = 0.0;
            if let Some(rank) = candidate.lexical_rank {
                score += 1.0 / (RRF_K + rank as f64);
            }
            if let Some(rank) = candidate.vector_rank {
                score += 1.0 / (RRF_K + rank as f64);
            }
            (score, candidate)
        })
        .collect();
    fused.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.section.slug.cmp(&b.1.section.slug))
            .then_with(|| a.1.section.heading_path.cmp(&b.1.section.heading_path))
    });
    // ---- Rerank stage (§50 V0.5): an optional reranker reorders the fused
    // candidates before selection. Noop/absent keeps the fused order, so
    // determinism is unchanged by default. ----
    if let Some(reranker) = reranker {
        let candidates: Vec<crate::rerank::RerankCandidate> = fused
            .iter()
            .map(|(score, candidate)| crate::rerank::RerankCandidate {
                key: (
                    candidate.section.page_id.as_str().to_owned(),
                    candidate.section.heading_path.join("\u{1f}"),
                ),
                slug: candidate.section.slug.clone(),
                title: candidate.section.title.clone(),
                heading_path: candidate.section.heading_path.clone(),
                snippet: candidate.snippet.clone(),
                fused_score: *score as f32,
                score: *score as f32,
            })
            .collect();
        let reranked = reranker.rerank(query, candidates)?;
        let mut order: std::collections::BTreeMap<(String, String), usize> =
            std::collections::BTreeMap::new();
        for (index, candidate) in reranked.into_iter().enumerate() {
            order.insert(candidate.key, index);
        }
        fused.sort_by(|(score_a, candidate_a), (score_b, candidate_b)| {
            let key = |candidate: &Candidate| {
                (
                    candidate.section.page_id.as_str().to_owned(),
                    candidate.section.heading_path.join("\u{1f}"),
                )
            };
            let rank_a = order.get(&key(candidate_a)).copied().unwrap_or(usize::MAX);
            let rank_b = order.get(&key(candidate_b)).copied().unwrap_or(usize::MAX);
            rank_a.cmp(&rank_b).then_with(|| {
                score_b
                    .partial_cmp(score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| candidate_a.section.slug.cmp(&candidate_b.section.slug))
                    .then_with(|| {
                        candidate_a
                            .section
                            .heading_path
                            .cmp(&candidate_b.section.heading_path)
                    })
            })
        });
    }
    if fused.is_empty() {
        return Err(WikiError::Index(format!(
            "no wiki section matches {query:?}; the published generation may not cover this topic"
        )));
    }

    let page_sources = generation_page_sources(conn, &active)?;

    // Greedy diversity-aware selection over the deterministic candidate order.
    let mut chunks: Vec<ContextChunk> = Vec::new();
    let mut pages_used: BTreeSet<String> = BTreeSet::new();
    let mut per_source: BTreeMap<String, usize> = BTreeMap::new();
    let mut tokens = 0u64;
    let mut dropped = 0usize;
    for (fused_score, candidate) in &fused {
        if chunks.len() >= budget.max_chunks {
            dropped += fused.len() - chunks.len();
            break;
        }
        let fresh_page = pages_used.insert(candidate.section.page_id.as_str().to_owned());
        if fresh_page && pages_used.len() > budget.max_pages {
            // A NEW page beyond the diversity cap is rejected; pages already
            // under the cap keep contributing until max_chunks.
            pages_used.remove(candidate.section.page_id.as_str());
            dropped += 1;
            continue;
        }
        let sources = page_sources
            .get(candidate.section.page_id.as_str())
            .cloned()
            .unwrap_or_default();
        if let Some(first_source) = sources.first() {
            let used = per_source.get(first_source).copied().unwrap_or(0);
            if used >= budget.max_per_source {
                dropped += 1;
                continue;
            }
            *per_source.entry(first_source.clone()).or_insert(0) += 1;
        }
        let chunk = ContextChunk {
            page_id: candidate.section.page_id.clone(),
            slug: candidate.section.slug.clone(),
            title: candidate.section.title.clone(),
            heading_path: candidate.section.heading_path.clone(),
            snippet: candidate.snippet.clone(),
            score: *fused_score as f32,
            sources,
        };
        let chunk_tokens = estimate_tokens(&context_chunk_text(&chunk));
        if tokens + chunk_tokens > budget.max_tokens {
            dropped += 1;
            continue;
        }
        tokens += chunk_tokens;
        chunks.push(chunk);
    }

    // Graph neighborhood of the selected pages (FIX-009's bridge makes the
    // semantic component reachable from page hits). Deterministic: page
    // order = chunk order, expansion sorted by the storage layer.
    let mut neighbors: Vec<ContextNeighbor> = Vec::new();
    let mut seen_nodes: BTreeSet<String> = BTreeSet::new();
    let neighbor_slots = budget.graph_limit * 2;
    for chunk in chunks.iter().take(3) {
        if neighbors.len() >= neighbor_slots {
            break;
        }
        for neighbor in graph_expand_from_page(conn, &chunk.page_id, budget.graph_limit)? {
            if neighbors.len() >= neighbor_slots {
                dropped += 1;
                break;
            }
            if seen_nodes.insert(neighbor.node_id.clone()) {
                neighbors.push(ContextNeighbor {
                    from_slug: chunk.slug.clone(),
                    node_id: neighbor.node_id,
                    node_type: neighbor.node_type,
                    label: neighbor.label,
                    relation: neighbor.relation,
                });
            }
        }
    }

    Ok(AssembledContext {
        estimated_tokens: estimate_tokens(&assembled_text(&chunks, &neighbors)),
        chunks,
        neighbors,
        dropped,
    })
}

/// A body excerpt for vector-only candidates (no FTS snippet exists): the
/// first paragraph-ish slice, capped for context economy.
fn excerpt(body: &str) -> String {
    let mut end = body.find("\n\n").unwrap_or(body.len()).min(240);
    while !body.is_char_boundary(end) {
        end -= 1;
    }
    body[..end].trim().to_owned()
}

/// The serialized form of one chunk — what estimate_tokens judges and what a
/// synthesis prompt embeds (keep both in sync by construction).
pub fn context_chunk_text(chunk: &ContextChunk) -> String {
    format!(
        "{} | {} | {} | {}",
        chunk.title,
        chunk.heading_path.join(" > "),
        chunk.sources.join(","),
        chunk.snippet
    )
}

/// The serialized form of the assembled context (chunks + neighbors).
fn assembled_text(chunks: &[ContextChunk], neighbors: &[ContextNeighbor]) -> String {
    let mut text = String::new();
    for chunk in chunks {
        text.push_str(&context_chunk_text(chunk));
        text.push('\n');
    }
    for neighbor in neighbors {
        text.push_str(&format!(
            "{} -{}-> {}\n",
            neighbor.from_slug, neighbor.relation, neighbor.label
        ));
    }
    text
}

/// Estimated tokens of the CHUNK selection alone — what the token budget
/// actually governs (the graph neighborhood is assembled after selection
/// from the already-budgeted pages).
pub fn chunks_tokens(chunks: &[ContextChunk]) -> u64 {
    chunks
        .iter()
        .map(|chunk| estimate_tokens(&context_chunk_text(chunk)))
        .sum()
}
