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
use llm_wiki_core::ids::WikiPageId;
use llm_wiki_core::plan::estimate_tokens;
use llm_wiki_storage::{
    generation_page_sources, get_active_build_id, graph_expand_from_page, search_index, Connection,
};

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
    /// bm25 rank (smaller is better), kept for downstream ordering/display.
    pub rank: f64,
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
    let mut candidates = search_index(conn, &fts_query, budget.max_chunks * 3)?;
    if candidates.is_empty() {
        return Err(WikiError::Index(format!(
            "no wiki section matches {query:?}; the published generation may not cover this topic"
        )));
    }
    // bm25 ties come back in SQLite scan order — pin the order deterministically.
    candidates.sort_by(|a, b| {
        a.rank
            .partial_cmp(&b.rank)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.slug.cmp(&b.slug))
            .then_with(|| a.heading_path.cmp(&b.heading_path))
    });

    let page_sources = generation_page_sources(conn, &active)?;

    // Greedy diversity-aware selection over the deterministic candidate order.
    let mut chunks: Vec<ContextChunk> = Vec::new();
    let mut pages_used: BTreeSet<String> = BTreeSet::new();
    let mut per_source: BTreeMap<String, usize> = BTreeMap::new();
    let mut tokens = 0u64;
    let mut dropped = 0usize;
    for hit in &candidates {
        if chunks.len() >= budget.max_chunks {
            dropped += candidates.len() - chunks.len();
            break;
        }
        let fresh_page = pages_used.insert(hit.page_id.as_str().to_owned());
        if fresh_page && pages_used.len() > budget.max_pages {
            // A NEW page beyond the diversity cap is rejected; pages already
            // under the cap keep contributing until max_chunks.
            pages_used.remove(hit.page_id.as_str());
            dropped += 1;
            continue;
        }
        let sources = page_sources
            .get(hit.page_id.as_str())
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
            page_id: hit.page_id.clone(),
            slug: hit.slug.clone(),
            title: hit.title.clone(),
            heading_path: hit.heading_path.clone(),
            snippet: hit.snippet.clone(),
            rank: hit.rank,
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
