//! Rerank abstraction (PRD §50, V0.5 tail).
//!
//! A reranker reorders retrieval candidates AFTER rank fusion and BEFORE the
//! context budget consumes them (and can do the same for raw search hits).
//! The default is [`NoopReranker`] — the fused order is already deterministic
//! and correct; concrete rerankers (LLM-judged, cross-encoder, …) plug in
//! behind the trait without touching retrieval code.

use llm_wiki_core::error::Result;

use crate::SearchHit;

/// One candidate handed to a reranker. `key` is the context-section identity
/// `(page_id, heading_path joined by \u{1f})` — the same key fusion uses.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankCandidate {
    pub key: (String, String),
    pub slug: String,
    pub title: String,
    pub heading_path: Vec<String>,
    pub snippet: String,
    /// RRF-fused retrieval score before reranking (transparency + tie-break).
    pub fused_score: f32,
    /// The reranker's output score. Noop keeps `fused_score`.
    pub score: f32,
}

/// Reorders candidates for one query. Implementations must be deterministic
/// for a given input (PRD §4.8: builds and responses stay reproducible).
pub trait Reranker: Send + Sync {
    /// Stable strategy name (surfaced in responses/logs).
    fn name(&self) -> &str;

    /// Returns the candidates in their new order. The fused score always
    /// travels with the candidate; `score` is what downstream ordering uses.
    fn rerank(&self, query: &str, items: Vec<RerankCandidate>) -> Result<Vec<RerankCandidate>>;
}

/// The identity reranker: fused order in, fused order out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn name(&self) -> &str {
        "none"
    }

    fn rerank(
        &self,
        _query: &str,
        mut items: Vec<RerankCandidate>,
    ) -> Result<Vec<RerankCandidate>> {
        for item in &mut items {
            item.score = item.fused_score;
        }
        Ok(items)
    }
}

/// Applies a reranker to raw search hits (the `/v1/search` path). `None`
/// returns the hits untouched.
pub fn rerank_search_hits(
    query: &str,
    hits: Vec<SearchHit>,
    reranker: Option<&dyn Reranker>,
) -> Result<Vec<SearchHit>> {
    let Some(reranker) = reranker else {
        return Ok(hits);
    };
    let candidates: Vec<RerankCandidate> = hits
        .iter()
        .map(|hit| RerankCandidate {
            key: (
                hit.page_id.as_str().to_owned(),
                hit.heading_path.join("\u{1f}"),
            ),
            slug: hit.slug.clone(),
            title: hit.title.clone(),
            heading_path: hit.heading_path.clone(),
            snippet: String::new(),
            fused_score: -hit.rank as f32,
            score: -hit.rank as f32,
        })
        .collect();
    let reranked = reranker.rerank(query, candidates)?;
    let mut by_key: std::collections::BTreeMap<(String, String), SearchHit> = hits
        .into_iter()
        .map(|hit| {
            (
                (
                    hit.page_id.as_str().to_owned(),
                    hit.heading_path.join("\u{1f}"),
                ),
                hit,
            )
        })
        .collect();
    Ok(reranked
        .into_iter()
        .filter_map(|candidate| by_key.remove(&candidate.key))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_core::ids::WikiPageId;

    fn hit(slug: &str, rank: f64) -> SearchHit {
        SearchHit {
            page_id: WikiPageId::from_validated(format!("wp_{slug}")),
            slug: slug.to_owned(),
            title: slug.to_owned(),
            heading_path: vec!["Overview".into()],
            snippet: format!("snippet of {slug}"),
            rank,
        }
    }

    /// Reverses the input order — a fake "real" reranker for tests.
    struct Reverser;

    impl Reranker for Reverser {
        fn name(&self) -> &str {
            "reverse"
        }

        fn rerank(
            &self,
            _query: &str,
            mut items: Vec<RerankCandidate>,
        ) -> Result<Vec<RerankCandidate>> {
            items.reverse();
            Ok(items)
        }
    }

    #[test]
    fn noop_preserves_order_and_echoes_fused_scores() {
        let items: Vec<RerankCandidate> = ["a", "b", "c"]
            .iter()
            .enumerate()
            .map(|(i, slug)| RerankCandidate {
                key: (format!("wp_{slug}"), "Overview".into()),
                slug: slug.to_string(),
                title: slug.to_string(),
                heading_path: vec!["Overview".into()],
                snippet: String::new(),
                fused_score: 3.0 - i as f32,
                score: 0.0,
            })
            .collect();
        let reranked = NoopReranker.rerank("q", items).unwrap();
        let slugs: Vec<&str> = reranked.iter().map(|c| c.slug.as_str()).collect();
        assert_eq!(slugs, ["a", "b", "c"]);
        assert_eq!(reranked[0].score, 3.0);
    }

    #[test]
    fn rerank_search_hits_reorders_via_the_key_identity() {
        let hits = vec![hit("first", 3.0), hit("second", 2.0), hit("third", 1.0)];
        let reranked = rerank_search_hits("q", hits, Some(&Reverser)).unwrap();
        let slugs: Vec<&str> = reranked.iter().map(|h| h.slug.as_str()).collect();
        assert_eq!(slugs, ["third", "second", "first"]);
        // Without a reranker the input order survives untouched.
        let hits = vec![hit("first", 3.0), hit("second", 2.0)];
        assert_eq!(rerank_search_hits("q", hits, None).unwrap().len(), 2);
    }
}
