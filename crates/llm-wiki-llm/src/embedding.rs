//! Embedding provider abstraction (§19.3 Vector layer): text → dense vector.
//!
//! The chat model and the embedding model may live at DIFFERENT providers,
//! so [`OpenAiCompatibleEmbeddings`] is constructed independently of the
//! chat provider — same OpenAI-compatible endpoint family (`POST
//! {base_url}/embeddings`), its own base URL / API key / timeout from the
//! `[embedding]` config section.

use async_trait::async_trait;

use crate::transport::HttpTransport;
use crate::LlmError;

/// One embedding batch: texts in, one vector per text out, same order.
/// Dimensions are model-owned; callers must not mix vectors across models.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Errors are hard (no retry loop here beyond the transport's own
    /// backoff): embedding is an explicit, user-driven step.
    async fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError>;
}

/// Standalone OpenAI-compatible embeddings client for the `[embedding]`
/// config section (which may point at a different provider than `[llm]`).
pub struct OpenAiCompatibleEmbeddings {
    transport: HttpTransport,
}

impl OpenAiCompatibleEmbeddings {
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

/// What a live embedding round trip actually proved, for `llm-wiki doctor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProbeOutcome {
    /// Vector width the model returned — surfaced because the Vector layer
    /// stores width-agnostic blobs, so a mixed-model corpus only shows up as
    /// a dimension mismatch much later.
    pub dimensions: usize,
    pub latency_ms: u128,
}

impl OpenAiCompatibleEmbeddings {
    /// One short text through the REAL embeddings path, for `doctor`.
    pub async fn probe(&self, model: &str) -> Result<EmbeddingProbeOutcome, LlmError> {
        let started = std::time::Instant::now();
        let vectors =
            post_embeddings(&self.transport, model, &["llm-wiki probe".to_owned()]).await?;
        let dimensions = vectors.first().map(Vec::len).unwrap_or(0);
        Ok(EmbeddingProbeOutcome {
            dimensions,
            latency_ms: started.elapsed().as_millis(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiCompatibleEmbeddings {
    async fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        post_embeddings(&self.transport, model, texts).await
    }
}

/// Shared request/parse path for every OpenAI-compatible embedder (the chat
/// provider's [`EmbeddingProvider`] impl delegates here too, so the two
/// cannot drift).
pub(crate) async fn post_embeddings(
    transport: &HttpTransport,
    model: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, LlmError> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let body = serde_json::json!({ "model": model, "input": texts });
    let response = transport.post("embeddings", &body).await?;
    let payload: serde_json::Value = response
        .json()
        .await
        .map_err(|e| LlmError::InvalidResponse(format!("embeddings body is not JSON: {e}")))?;
    parse_embeddings_response(payload, texts.len())
}

/// Re-sorts `data[]` by `index` so output order == input order (providers
/// return entries in arbitrary order).
fn parse_embeddings_response(
    payload: serde_json::Value,
    expected: usize,
) -> Result<Vec<Vec<f32>>, LlmError> {
    let mut entries: Vec<(u64, Vec<f32>)> = payload
        .get("data")
        .and_then(|data| data.as_array())
        .ok_or_else(|| LlmError::InvalidResponse("embeddings response missing data[]".to_owned()))?
        .iter()
        .map(|entry| {
            let index = entry
                .get("index")
                .and_then(|value| value.as_u64())
                .ok_or_else(|| {
                    LlmError::InvalidResponse("embeddings entry missing index".to_owned())
                })?;
            let vector = entry
                .get("embedding")
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    LlmError::InvalidResponse("embeddings entry missing embedding[]".to_owned())
                })?
                .iter()
                .map(|value| {
                    value.as_f64().map(|number| number as f32).ok_or_else(|| {
                        LlmError::InvalidResponse(
                            "embeddings entry holds a non-numeric value".to_owned(),
                        )
                    })
                })
                .collect::<Result<Vec<f32>, LlmError>>()?;
            Ok((index, vector))
        })
        .collect::<Result<Vec<(u64, Vec<f32>)>, LlmError>>()?;
    entries.sort_by_key(|(index, _)| *index);
    if entries.len() != expected {
        return Err(LlmError::InvalidResponse(format!(
            "embeddings returned {} vectors for {} inputs",
            entries.len(),
            expected
        )));
    }
    Ok(entries.into_iter().map(|(_, vector)| vector).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embeddings_batch_returns_vectors_in_input_order() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).unwrap();
            let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
            assert!(raw.contains("/embeddings"), "hits the embeddings path");
            assert!(raw.contains("\"input\":[\"b\",\"a\"]"), "batch body: {raw}");
            // data[] deliberately OUT of input order; index field sorts it.
            let body = r#"{"data":[
                {"index":1,"embedding":[0.4,0.5]},
                {"index":0,"embedding":[0.1,0.2]}
            ]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let provider: std::sync::Arc<dyn EmbeddingProvider> = std::sync::Arc::new(
            OpenAiCompatibleEmbeddings::new(
                &format!("http://127.0.0.1:{port}"),
                "UNUSED_VAR",
                10,
                0,
            )
            .unwrap(),
        );
        let vectors = provider
            .embed("emb-model", &["b".to_owned(), "a".to_owned()])
            .await
            .unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![0.1, 0.2], "input order, not data order");
        assert_eq!(vectors[1], vec![0.4, 0.5]);
        server.join().unwrap();
    }

    /// The doctor probe sends one short text and reports the vector width —
    /// the only cheap way to catch a model swap that silently changes
    /// dimensions underneath a corpus already embedded with the old model.
    #[tokio::test]
    async fn probe_reports_the_returned_dimensions() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let n = stream.read(&mut buf).unwrap();
            let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
            assert!(raw.contains("/embeddings"), "{raw}");
            assert!(raw.contains("\"input\":[\"llm-wiki probe\"]"), "{raw}");
            let body = r#"{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let provider = OpenAiCompatibleEmbeddings::new(
            &format!("http://127.0.0.1:{port}"),
            "UNUSED_VAR",
            10,
            0,
        )
        .unwrap();
        let outcome = provider.probe("emb-model").await.unwrap();
        assert_eq!(outcome.dimensions, 3);
        server.join().unwrap();
    }
}
