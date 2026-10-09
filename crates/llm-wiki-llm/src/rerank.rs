//! Cohere/Jina-style rerank API adapter (PRD §50 V0.5 tail).
//!
//! The chat, embedding and rerank models may live at THREE different
//! providers, so this client is constructed independently of the chat
//! provider — its own base URL / API key / timeout come from the `[rerank]`
//! config section.
//!
//! Wire format: `POST {base_url}/rerank` with `{model, query, documents,
//! top_n}` → `{results: [{index, relevance_score}]}`. That is Cohere's
//! `/rerank` — the de-facto standard also implemented by vLLM, Jina,
//! SiliconFlow and Voyage. OpenAI defines no rerank endpoint, so the
//! strategy is deliberately NOT named "openai-compatible".

use async_trait::async_trait;
use serde_json::json;

use crate::transport::HttpTransport;
use crate::LlmError;

/// One rerank verdict: the document's input index and the model's relevance
/// score (higher = more relevant).
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
}

/// Async rerank transport; the model is passed per call (mirrors
/// [`crate::EmbeddingProvider`]). Implementations must be deterministic for
/// a given input (PRD §4.8).
#[async_trait]
pub trait RerankProvider: Send + Sync {
    /// Returns verdicts in response order; the caller orders by score.
    async fn rerank(
        &self,
        model: &str,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>, LlmError>;
}

/// Cohere-compatible `/rerank` client over the shared transport (bearer
/// auth, loopback proxy bypass, bounded 429/5xx backoff).
pub struct CohereCompatibleReranker {
    transport: HttpTransport,
}

impl CohereCompatibleReranker {
    /// `api_key_env` is the config-declared env var name; a missing env var
    /// is only an error once a request actually needs it.
    pub fn new(
        base_url: &str,
        api_key_env: &str,
        timeout_seconds: u64,
        max_retries: u32,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            transport: HttpTransport::new(base_url, api_key_env, timeout_seconds, max_retries)?,
        })
    }
}

/// What a live rerank round trip actually proved, for `llm-wiki doctor`.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankProbeOutcome {
    /// Highest relevance score of the two probe documents — reported so a
    /// degenerate all-zero endpoint (a common misconfiguration: the rerank
    /// path wired to a chat model) is visible in `doctor` output.
    pub top_relevance: Option<f32>,
    pub latency_ms: u128,
}

impl CohereCompatibleReranker {
    /// One query against two trivial documents through the REAL `/rerank`
    /// path, for `doctor`.
    pub async fn probe(&self, model: &str) -> Result<RerankProbeOutcome, LlmError> {
        let started = std::time::Instant::now();
        let results = self
            .rerank(
                model,
                "llm-wiki probe",
                &["llm-wiki probe".to_owned(), "unrelated text".to_owned()],
                2,
            )
            .await?;
        if results.is_empty() {
            return Err(LlmError::InvalidResponse(format!(
                "the rerank endpoint returned no results for a 2-document probe; it is not serving '{model}'"
            )));
        }
        Ok(RerankProbeOutcome {
            top_relevance: results
                .iter()
                .map(|result| result.relevance_score)
                .reduce(f32::max),
            latency_ms: started.elapsed().as_millis(),
        })
    }
}

#[async_trait]
impl RerankProvider for CohereCompatibleReranker {
    async fn rerank(
        &self,
        model: &str,
        query: &str,
        documents: &[String],
        top_n: usize,
    ) -> Result<Vec<RerankResult>, LlmError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        });
        let response = self.transport.post("rerank", &body).await?;
        let payload: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(format!("rerank body is not JSON: {e}")))?;
        parse_rerank_response(payload)
    }
}

fn parse_rerank_response(payload: serde_json::Value) -> Result<Vec<RerankResult>, LlmError> {
    let results = payload
        .get("results")
        .and_then(|results| results.as_array())
        .ok_or_else(|| LlmError::InvalidResponse("rerank response missing results[]".to_owned()))?;
    results
        .iter()
        .map(|entry| {
            let index = entry
                .get("index")
                .and_then(|value| value.as_u64())
                .ok_or_else(|| {
                    LlmError::InvalidResponse("rerank entry missing index".to_owned())
                })?;
            let relevance_score = entry
                .get("relevance_score")
                .and_then(|value| value.as_f64())
                .ok_or_else(|| {
                    LlmError::InvalidResponse("rerank entry missing relevance_score".to_owned())
                })?;
            Ok(RerankResult {
                index: index as usize,
                relevance_score: relevance_score as f32,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An out-of-order `results[]` must map each verdict back to its
    /// document index; the request must hit `/rerank` with the full body.
    #[tokio::test]
    async fn rerank_sends_query_and_documents_and_parses_results() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).unwrap();
            let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
            assert!(raw.contains("/rerank"), "hits the rerank path: {raw}");
            assert!(raw.contains("\"model\":\"rerank-model\""), "{raw}");
            assert!(raw.contains("\"query\":\"sso timeout\""), "{raw}");
            assert!(raw.contains("\"documents\":[\"doc a\",\"doc b\"]"), "{raw}");
            assert!(raw.contains("\"top_n\":2"), "{raw}");
            // results[] deliberately out of input order.
            let body = r#"{"results":[
                {"index":1,"relevance_score":0.9},
                {"index":0,"relevance_score":0.2}
            ]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let reranker =
            CohereCompatibleReranker::new(&format!("http://127.0.0.1:{port}"), "UNUSED_VAR", 10, 0)
                .unwrap();
        let results = reranker
            .rerank(
                "rerank-model",
                "sso timeout",
                &["doc a".to_owned(), "doc b".to_owned()],
                2,
            )
            .await
            .unwrap();
        assert_eq!(
            results,
            vec![
                RerankResult {
                    index: 1,
                    relevance_score: 0.9
                },
                RerankResult {
                    index: 0,
                    relevance_score: 0.2
                },
            ]
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn empty_documents_short_circuits_without_a_request() {
        let reranker =
            CohereCompatibleReranker::new("http://127.0.0.1:1", "UNUSED_VAR", 10, 0).unwrap();
        let results = reranker.rerank("m", "q", &[], 0).await.unwrap();
        assert!(results.is_empty());
    }

    /// The doctor probe drives the real `/rerank` path with two documents and
    /// reports the top score.
    #[tokio::test]
    async fn probe_reports_the_top_relevance_score() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).unwrap();
            let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
            assert!(raw.contains("/rerank"), "{raw}");
            let body = r#"{"results":[{"index":0,"relevance_score":0.7},{"index":1,"relevance_score":0.1}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let reranker =
            CohereCompatibleReranker::new(&format!("http://127.0.0.1:{port}"), "UNUSED_VAR", 10, 0)
                .unwrap();
        let outcome = reranker.probe("rerank-model").await.unwrap();
        assert_eq!(outcome.top_relevance, Some(0.7));
        server.join().unwrap();
    }

    /// An endpoint that answers `/rerank` with an empty `results[]` is not
    /// serving the model; the probe must fail rather than report green.
    #[tokio::test]
    async fn probe_fails_on_an_empty_result_set() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let _ = stream.read(&mut buf).unwrap();
            let body = r#"{"results":[]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let reranker =
            CohereCompatibleReranker::new(&format!("http://127.0.0.1:{port}"), "UNUSED_VAR", 10, 0)
                .unwrap();
        let err = reranker.probe("rerank-model").await.unwrap_err();
        assert!(err.to_string().contains("no results"), "{err}");
        server.join().unwrap();
    }
}
