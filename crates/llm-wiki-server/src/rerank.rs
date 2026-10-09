//! Config-driven reranker construction (PRD §50 V0.5 + §32 provider split).
//!
//! The `[search] rerank` strategy names the wire protocol; the `[rerank]`
//! config section carries the endpoint (base_url / api_key_env / model /
//! timeout — empty fields inherit `[llm]`), so the rerank model can live at
//! its own provider, independent of `[llm]` (chat) and `[embedding]`.

use std::sync::Arc;

use llm_wiki_core::config::Config;
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_llm::{CohereCompatibleReranker, RerankProvider};
use llm_wiki_search::{RerankCandidate, Reranker};

/// The strategy name for the Cohere-compatible `/rerank` endpoint family
/// (vLLM, Jina, SiliconFlow, Voyage, …).
pub const COHERE_COMPATIBLE: &str = "cohere-compatible";

/// A [`Reranker`] backed by a remote rerank endpoint. The search-layer trait
/// is sync (it sits inside the deterministic, blocking context-assembly
/// pipeline), so the async HTTP call bridges through the runtime handle
/// captured at construction — callers must invoke [`Reranker::rerank`] from
/// a blocking context (`spawn_blocking`), which is where the search and
/// context routes already run their retrieval.
pub struct ApiReranker {
    client: Arc<dyn RerankProvider>,
    model: String,
    handle: tokio::runtime::Handle,
}

impl ApiReranker {
    /// Captures the current runtime handle; must be called from async
    /// context (the route handlers are).
    pub fn new(client: Arc<dyn RerankProvider>, model: String) -> Self {
        Self {
            client,
            model,
            handle: tokio::runtime::Handle::current(),
        }
    }
}

/// The semantic text scored by the rerank model: title, heading path and —
/// when present — the snippet. Raw search hits carry no body text, so their
/// document is titles + headings; context candidates add the snippet.
fn rerank_document(title: &str, heading_path: &[String], snippet: &str) -> String {
    let mut document = format!("{title}\n{}", heading_path.join(" > "));
    if !snippet.trim().is_empty() {
        document.push('\n');
        document.push_str(snippet);
    }
    document
}

impl Reranker for ApiReranker {
    fn name(&self) -> &str {
        COHERE_COMPATIBLE
    }

    fn rerank(&self, query: &str, mut items: Vec<RerankCandidate>) -> Result<Vec<RerankCandidate>> {
        if items.is_empty() {
            return Ok(items);
        }
        let documents: Vec<String> = items
            .iter()
            .map(|item| rerank_document(&item.title, &item.heading_path, &item.snippet))
            .collect();
        let top_n = items.len();
        let results = self
            .handle
            .block_on(self.client.rerank(&self.model, query, &documents, top_n))
            .map_err(|e| WikiError::Llm(e.to_string()))?;
        // Score by document index; unrated candidates keep their fused
        // score, so a partial response degrades to fused order, never NaN.
        let mut scores = vec![None; items.len()];
        for result in results {
            if let Some(slot) = scores.get_mut(result.index) {
                *slot = Some(result.relevance_score);
            } else {
                // Provider misbehavior must stay visible: silently dropping
                // the verdict would read as a rerank-quality regression.
                tracing::warn!(
                    index = result.index,
                    documents = items.len(),
                    "rerank endpoint returned an out-of-range document index; verdict ignored"
                );
            }
        }
        for (item, score) in items.iter_mut().zip(scores) {
            if let Some(score) = score {
                item.score = score;
            }
        }
        // Deterministic order (PRD §4.8): score desc, then the fusion
        // tie-breaks (slug, heading path).
        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.slug.cmp(&b.slug))
                .then_with(|| a.heading_path.cmp(&b.heading_path))
        });
        Ok(items)
    }
}

/// Resolves `[search] rerank` into a concrete reranker using the `[rerank]`
/// endpoint config. `None` = keep the deterministic fused order. Strategy
/// validity (including the required rerank model) is already enforced in
/// `Config::validate`; the match here is the construction dispatch.
pub fn api_reranker_from_config(config: &Config) -> Result<Option<Box<dyn Reranker>>> {
    match config.search.rerank.as_str() {
        "none" => Ok(None),
        COHERE_COMPATIBLE => {
            let endpoint = config.rerank.endpoint(&config.llm);
            let client = CohereCompatibleReranker::new(
                &endpoint.base_url,
                &endpoint.api_key_env,
                endpoint.timeout_seconds,
                2,
            )
            .map_err(|e| WikiError::Config(format!("rerank provider init failed: {e}")))?;
            Ok(Some(Box::new(ApiReranker::new(
                Arc::new(client),
                config.rerank.model.clone(),
            ))))
        }
        other => Err(WikiError::Config(format!(
            "unsupported search.rerank strategy '{other}' (supported: none, cohere-compatible)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use llm_wiki_llm::RerankResult;

    /// Partial-response fake: rates ONLY document 0 (at 0.9), leaving the
    /// other candidates unrated — the degradation path must keep their
    /// fused scores.
    struct PartialFake;

    #[async_trait]
    impl RerankProvider for PartialFake {
        async fn rerank(
            &self,
            _model: &str,
            _query: &str,
            documents: &[String],
            _top_n: usize,
        ) -> std::result::Result<Vec<RerankResult>, llm_wiki_llm::LlmError> {
            if documents.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![RerankResult {
                index: 0,
                relevance_score: 0.9,
            }])
        }
    }

    fn candidate(slug: &str) -> RerankCandidate {
        RerankCandidate {
            key: (format!("page-{slug}"), String::new()),
            slug: slug.to_owned(),
            title: format!("Title {slug}"),
            heading_path: vec!["H".into()],
            snippet: format!("snippet of {slug}"),
            fused_score: 1.0,
            score: 1.0,
        }
    }

    #[tokio::test]
    async fn adapter_orders_by_remote_scores_with_deterministic_ties() {
        let reranker = ApiReranker::new(Arc::new(PartialFake), "rerank-model".into());
        let items = vec![candidate("a"), candidate("b"), candidate("c")];
        // The blocking contract: rerank runs on the blocking pool, exactly
        // like the search/context routes call it.
        let reranked = tokio::task::spawn_blocking(move || reranker.rerank("q", items))
            .await
            .unwrap()
            .unwrap();
        // "a" took the remote 0.9; "b"/"c" were unrated and keep their
        // fused 1.0, ordering before "a" with the slug tie-break.
        assert_eq!(reranked[0].slug, "b");
        assert_eq!(reranked[0].score, 1.0);
        assert_eq!(reranked[1].slug, "c");
        assert_eq!(reranked[1].score, 1.0, "unrated keeps fused score");
        assert_eq!(reranked[2].slug, "a");
        assert_eq!(reranked[2].score, 0.9);
    }

    #[tokio::test]
    async fn empty_candidates_short_circuit() {
        let reranker = ApiReranker::new(Arc::new(PartialFake), "rerank-model".into());
        assert!(reranker.rerank("q", Vec::new()).unwrap().is_empty());
    }

    #[tokio::test]
    async fn config_resolves_none_and_unknown_strategies() {
        let mut config = Config::default();
        assert!(api_reranker_from_config(&config).unwrap().is_none());
        config.search.rerank = "cross-encoder".into();
        let err = match api_reranker_from_config(&config) {
            Err(err) => err,
            Ok(_) => panic!("expected a config error for an unknown rerank strategy"),
        };
        assert!(
            err.to_string().contains("unsupported search.rerank"),
            "{err}"
        );
    }
}
