//! Deterministic fake provider for tests and CI (PRD §54).
//!
//! Fixed input produces fixed output — no CI run ever needs a real model.
//! Request counting powers the Rebuild Determinism gate (PRD §37.3: "second
//! build adds zero LLM requests").

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::{LlmError, LlmProvider, LlmRequest, LlmResponse};

pub type FakeHandler = Arc<dyn Fn(&LlmRequest) -> Result<String, LlmError> + Send + Sync>;

pub struct FakeLlmProvider {
    model: String,
    handler: FakeHandler,
    requests: AtomicU64,
}

impl FakeLlmProvider {
    pub fn new(model: impl Into<String>, handler: FakeHandler) -> Self {
        Self {
            model: model.into(),
            handler,
            requests: AtomicU64::new(0),
        }
    }

    /// Always answers with the same fixed text.
    pub fn fixed(model: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        Self::new(model, Arc::new(move |_| Ok(text.clone())))
    }

    /// Always answers with the same fixed JSON payload (for structured output
    /// tests).
    pub fn fixed_json(model: impl Into<String>, json: serde_json::Value) -> Self {
        Self::fixed(model, json.to_string())
    }

    pub fn request_count(&self) -> u64 {
        self.requests.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmProvider for FakeLlmProvider {
    fn model(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "fake"
    }

    async fn generate(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let text = (self.handler)(&request)?;
        Ok(LlmResponse {
            text,
            model: self.model.clone(),
            input_tokens: request.prompt.len() as u64 / 4,
            output_tokens: 0,
            finish_reason: Some("stop".to_owned()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fixed_provider_is_deterministic_and_counts() {
        let provider =
            FakeLlmProvider::fixed_json("fake-model", serde_json::json!({"summary": "ok"}));
        let req = LlmRequest::new("document-analysis", "analyze this");
        let a = provider.generate(req.clone()).await.unwrap();
        let b = provider.generate(req).await.unwrap();
        assert_eq!(a.text, b.text);
        assert_eq!(a.model, "fake-model");
        assert_eq!(provider.request_count(), 2);
    }

    #[tokio::test]
    async fn handler_errors_propagate() {
        let provider = FakeLlmProvider::new(
            "fake",
            Arc::new(|_| {
                Err(LlmError::Api {
                    code: 500,
                    message: "boom".into(),
                })
            }),
        );
        let err = provider
            .generate(LlmRequest::new("t", "p"))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::Api { code: 500, .. }));
    }
}
