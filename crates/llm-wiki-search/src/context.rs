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

use crate::fusion::{
    tie_break, wiki_identity, Evidence, EvidenceKind, FusedRetrieval, ServedSides, SourceMode,
    SourceRef, RRF_K,
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
    /// Chunks per source (source diversity): wiki chunks are counted by
    /// their first cited source path; raw-source chunks (EPIC A PR3) by
    /// their `source_id` — same cap, per-side counting.
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
    /// The wiki page behind a wiki chunk. For source-kind chunks (EPIC A
    /// PR3) there is no wiki page — this carries the SOURCE id instead, the
    /// chunk's corpus identity (`source_ref` holds the full locator).
    pub page_id: WikiPageId,
    /// The page slug, or the source-relative file path for source chunks.
    pub slug: String,
    pub title: String,
    pub heading_path: Vec<String>,
    pub snippet: String,
    /// Fused retrieval score (larger is better): the wiki-internal
    /// lexical+vector RRF for the legacy path, the top-level source↔wiki RRF
    /// when the context was built from a [`FusedRetrieval`].
    pub score: f32,
    /// Distinct source paths backing this page (empty for uncited pages;
    /// the chunk's own path for source chunks).
    pub sources: Vec<String>,
    /// Which corpus side produced this chunk (EPIC A PR3). The default for
    /// every pre-fusion path is [`EvidenceKind::Wiki`].
    pub evidence_kind: EvidenceKind,
    /// Source-side locator, present iff `evidence_kind` is
    /// [`EvidenceKind::Source`].
    pub source_ref: Option<SourceRef>,
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
    /// Which retrieval mode produced this context (EPIC A PR3). The legacy
    /// paths report [`SourceMode::Wiki`].
    pub served_mode: SourceMode,
    /// Per-side serving metadata when the context was built from a
    /// [`FusedRetrieval`] — any side with a status other than
    /// [`crate::fusion::SideStatus::Served`] is a visible degradation.
    /// `None` for the legacy wiki-only paths (they fail loudly instead of
    /// degrading).
    pub degraded: Option<ServedSides>,
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
    // lexical and vector candidate lists, with the SAME constant the
    // top-level source↔wiki fusion uses (crate::fusion::RRF_K — one knob,
    // two levels). RRF needs only the ORDER of each list, so bm25 and cosine
    // never have to be comparable. Identity = (page_id, heading path) — the
    // context-section identity. ----
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
            // Legacy path: every chunk is a wiki chunk (PR3 default).
            evidence_kind: EvidenceKind::Wiki,
            source_ref: None,
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
        // Legacy path: wiki mode, no side metadata (it errors instead of
        // degrading).
        served_mode: SourceMode::Wiki,
        degraded: None,
    })
}

/// [`build_context_with_reranker`] extended with a fused retrieval (EPIC A
/// PR3, the top-level source↔wiki fusion): the fused result's wiki entries
/// replace this builder's own lexical search, and its source entries join as
/// first-class selection candidates. The wiki-internal lexical+vector RRF
/// keeps its shape — the fused result's wiki side IS the lexical list — and
/// the reranker (if any) reorders only the wiki list; source candidates keep
/// their fused order (reranking them is EPIC G, PRD §1.4).
///
/// `None` keeps the exact legacy behavior. Both sides empty is an HONEST
/// error that names each side's status.
pub fn build_context_with_sources(
    conn: &Connection,
    query: &str,
    budget: &ContextBudget,
    vector: &[VectorCandidate],
    reranker: Option<&dyn crate::rerank::Reranker>,
    fused: Option<&FusedRetrieval>,
) -> Result<AssembledContext> {
    match fused {
        None => build_context_with_reranker(conn, query, budget, vector, reranker),
        Some(fused) => build_from_fused(conn, query, budget, vector, reranker, fused),
    }
}

/// One selection candidate in the fused path: its evidence payload, the
/// display snippet, and the per-level ranks the two RRF stages consume.
struct FusedCandidate {
    evidence: Evidence,
    snippet: String,
    /// Rank within the fused result's wiki side (the lexical list).
    wiki_lexical_rank: Option<usize>,
    vector_rank: Option<usize>,
    /// Rank within the fused result's source side.
    source_rank: Option<usize>,
    exact_match: bool,
}

fn build_from_fused(
    conn: &Connection,
    query: &str,
    budget: &ContextBudget,
    vector: &[VectorCandidate],
    reranker: Option<&dyn crate::rerank::Reranker>,
    fused: &FusedRetrieval,
) -> Result<AssembledContext> {
    // One candidate per fused entry, keyed by the evidence-kind-scoped
    // identity (both sides coexist by design).
    let mut by_key: BTreeMap<String, FusedCandidate> = BTreeMap::new();
    for entry in &fused.entries {
        let snippet = match &entry.evidence {
            Evidence::Wiki(wiki) => wiki.snippet.clone(),
            Evidence::Source(source) => source.snippet.clone(),
        };
        let candidate = by_key
            .entry(entry.evidence.identity())
            .or_insert_with(|| FusedCandidate {
                evidence: entry.evidence.clone(),
                snippet,
                wiki_lexical_rank: None,
                vector_rank: None,
                source_rank: None,
                exact_match: entry.exact_match,
            });
        match entry.evidence.kind() {
            EvidenceKind::Wiki => candidate.wiki_lexical_rank = Some(entry.side_rank),
            EvidenceKind::Source => candidate.source_rank = Some(entry.side_rank),
        }
    }

    // ---- Level 1, vector half (unchanged legacy math): vector candidates
    // fuse with the wiki lexical ranks by section identity. Requires an
    // ACTIVE build — hash resolution reads wiki_page_text, and an
    // unpublished wiki must not fail a source-only context. ----
    let active = get_active_build_id(conn)?;
    if !vector.is_empty() && active.is_some() {
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
            let entry = by_key
                .entry(wiki_identity(
                    section.page_id.as_str(),
                    &section.heading_path,
                ))
                .or_insert_with(|| FusedCandidate {
                    evidence: Evidence::Wiki(crate::fusion::WikiEvidence {
                        page_id: section.page_id.clone(),
                        slug: section.slug.clone(),
                        title: section.title.clone(),
                        heading_path: section.heading_path.clone(),
                        snippet: String::new(),
                        rank: 0.0,
                    }),
                    snippet: excerpt(&section.body),
                    wiki_lexical_rank: None,
                    vector_rank: None,
                    source_rank: None,
                    exact_match: false,
                });
            entry.vector_rank = Some(rank);
        }
    }

    // ---- Level 1, wiki-internal RRF + optional rerank (legacy order and
    // tie-breaks, applied to the wiki list only). ----
    let mut wiki: Vec<(f64, &FusedCandidate)> = by_key
        .values()
        .filter(|candidate| candidate.evidence.kind() == EvidenceKind::Wiki)
        .map(|candidate| {
            let mut score = 0.0;
            if let Some(rank) = candidate.wiki_lexical_rank {
                score += 1.0 / (RRF_K + rank as f64);
            }
            if let Some(rank) = candidate.vector_rank {
                score += 1.0 / (RRF_K + rank as f64);
            }
            (score, candidate)
        })
        .collect();
    wiki.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.evidence.wiki_slug().cmp(b.1.evidence.wiki_slug()))
            .then_with(|| a.1.evidence.wiki_heading().cmp(b.1.evidence.wiki_heading()))
    });
    if let Some(reranker) = reranker {
        let candidates: Vec<crate::rerank::RerankCandidate> = wiki
            .iter()
            .map(|(score, candidate)| match &candidate.evidence {
                Evidence::Wiki(w) => crate::rerank::RerankCandidate {
                    key: (w.page_id.as_str().to_owned(), w.heading_path.join("\u{1f}")),
                    slug: w.slug.clone(),
                    title: w.title.clone(),
                    heading_path: w.heading_path.clone(),
                    snippet: candidate.snippet.clone(),
                    fused_score: *score as f32,
                    score: *score as f32,
                },
                // The rerank stage never sees source candidates (EPIC G).
                Evidence::Source(_) => unreachable!("wiki list holds wiki candidates only"),
            })
            .collect();
        let reranked = reranker.rerank(query, candidates)?;
        let mut order: std::collections::BTreeMap<(String, String), usize> =
            std::collections::BTreeMap::new();
        for (index, candidate) in reranked.into_iter().enumerate() {
            order.insert(candidate.key, index);
        }
        let key = |candidate: &FusedCandidate| match &candidate.evidence {
            Evidence::Wiki(w) => (w.page_id.as_str().to_owned(), w.heading_path.join("\u{1f}")),
            Evidence::Source(_) => unreachable!("wiki list holds wiki candidates only"),
        };
        wiki.sort_by(|(score_a, candidate_a), (score_b, candidate_b)| {
            let rank_a = order.get(&key(candidate_a)).copied().unwrap_or(usize::MAX);
            let rank_b = order.get(&key(candidate_b)).copied().unwrap_or(usize::MAX);
            rank_a.cmp(&rank_b).then_with(|| {
                score_b
                    .partial_cmp(score_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        candidate_a
                            .evidence
                            .wiki_slug()
                            .cmp(candidate_b.evidence.wiki_slug())
                    })
                    .then_with(|| {
                        candidate_a
                            .evidence
                            .wiki_heading()
                            .cmp(candidate_b.evidence.wiki_heading())
                    })
            })
        });
    }

    // ---- Level 2: the top-level RRF. Each side contributes its ordered
    // list; every entry scores 1/(k + position) within its own list. The
    // source list keeps the fused ordering (exact protection first), and the
    // FINAL order re-applies the exact-match front slot. ----
    let mut source: Vec<&FusedCandidate> = by_key
        .values()
        .filter(|candidate| candidate.evidence.kind() == EvidenceKind::Source)
        .collect();
    source.sort_by(|a, b| {
        b.exact_match
            .cmp(&a.exact_match)
            .then_with(|| a.source_rank.cmp(&b.source_rank))
            .then_with(|| a.evidence.identity().cmp(&b.evidence.identity()))
    });

    let mut ordered: Vec<(f64, &FusedCandidate)> = Vec::with_capacity(wiki.len() + source.len());
    for (position, (_, candidate)) in wiki.iter().enumerate() {
        ordered.push((1.0 / (RRF_K + position as f64), candidate));
    }
    for (position, candidate) in source.iter().enumerate() {
        ordered.push((1.0 / (RRF_K + position as f64), candidate));
    }
    ordered.sort_by(|a, b| {
        b.1.exact_match
            .cmp(&a.1.exact_match)
            .then_with(|| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| tie_break(&a.1.evidence, &b.1.evidence))
    });

    // Both sides empty is the honest "no results" error — with the reason
    // each side came up empty (PR3: degradation visible, never silent).
    if ordered.is_empty() {
        return Err(WikiError::Index(format!(
            "no retrieval results for {query:?} (wiki side: {:?}, source side: {:?}); \
             narrow the query or publish a build",
            fused.served.wiki, fused.served.source
        )));
    }

    // Citations exist only for a published wiki generation; a source-only
    // context (or a deactivated workspace) has no page-source map.
    let page_sources = match &active {
        Some(build) => generation_page_sources(conn, build)?,
        None => BTreeMap::new(),
    };

    // Greedy diversity-aware selection over the deterministic candidate
    // order — the legacy loop, with source chunks counted by source_id.
    let mut chunks: Vec<ContextChunk> = Vec::new();
    let mut pages_used: BTreeSet<String> = BTreeSet::new();
    let mut per_source: BTreeMap<String, usize> = BTreeMap::new();
    let mut tokens = 0u64;
    let mut dropped = 0usize;
    for (fused_score, candidate) in &ordered {
        if chunks.len() >= budget.max_chunks {
            dropped += ordered.len() - chunks.len();
            break;
        }
        let (page_id, slug, title, heading_path, evidence_kind, source_ref, sources) =
            match &candidate.evidence {
                Evidence::Wiki(wiki) => (
                    wiki.page_id.clone(),
                    wiki.slug.clone(),
                    wiki.title.clone(),
                    wiki.heading_path.clone(),
                    EvidenceKind::Wiki,
                    None,
                    page_sources
                        .get(wiki.page_id.as_str())
                        .cloned()
                        .unwrap_or_default(),
                ),
                Evidence::Source(source) => (
                    // A raw-source hit has no wiki page: the source id rides
                    // in page_id (the chunk's corpus identity), the file
                    // path in slug (also what `sources` carries).
                    WikiPageId::from_validated(source.source_ref.source_id.as_str().to_owned()),
                    source.source_ref.file_path.clone(),
                    source.title.clone(),
                    source.source_ref.heading_path.clone(),
                    EvidenceKind::Source,
                    Some(source.source_ref.clone()),
                    vec![source.source_ref.file_path.clone()],
                ),
            };
        let fresh_page = pages_used.insert(page_id.as_str().to_owned());
        if fresh_page && pages_used.len() > budget.max_pages {
            // A NEW page beyond the diversity cap is rejected; pages already
            // under the cap keep contributing until max_chunks.
            pages_used.remove(page_id.as_str());
            dropped += 1;
            continue;
        }
        // Source diversity: wiki chunks by their first cited source path
        // (legacy), source chunks by their source_id (PR3) — same cap.
        let diversity_key: Option<String> = match &source_ref {
            Some(source) => Some(source.source_id.as_str().to_owned()),
            None => sources.first().cloned(),
        };
        if let Some(key) = diversity_key {
            let used = per_source.get(&key).copied().unwrap_or(0);
            if used >= budget.max_per_source {
                dropped += 1;
                continue;
            }
            *per_source.entry(key).or_insert(0) += 1;
        }
        let chunk = ContextChunk {
            page_id,
            slug,
            title,
            heading_path,
            snippet: candidate.snippet.clone(),
            score: *fused_score as f32,
            sources,
            evidence_kind,
            source_ref,
        };
        let chunk_tokens = estimate_tokens(&context_chunk_text(&chunk));
        if tokens + chunk_tokens > budget.max_tokens {
            dropped += 1;
            continue;
        }
        tokens += chunk_tokens;
        chunks.push(chunk);
    }

    // Graph neighborhood of the selected WIKI pages (source chunks have no
    // graph node). Deterministic: page order = chunk order, expansion sorted
    // by the storage layer.
    let mut neighbors: Vec<ContextNeighbor> = Vec::new();
    let mut seen_nodes: BTreeSet<String> = BTreeSet::new();
    let neighbor_slots = budget.graph_limit * 2;
    for chunk in chunks
        .iter()
        .filter(|chunk| chunk.evidence_kind == EvidenceKind::Wiki)
        .take(3)
    {
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
        served_mode: fused.mode,
        degraded: Some(fused.served),
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
