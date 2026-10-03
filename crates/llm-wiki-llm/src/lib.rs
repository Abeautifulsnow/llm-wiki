#![forbid(unsafe_code)]
//! LLM provider abstraction (PRD §25).
//!
//! [`LlmProvider`] is the only surface the compiler talks to. Structured
//! output, retry, timeout, concurrency limits and token usage accounting are
//! handled here so task code stays deterministic and testable via
//! [`fake::FakeLlmProvider`].

pub mod fake;
pub mod openai;
pub mod structured;

pub use fake::FakeLlmProvider;
pub use openai::OpenAiCompatibleProvider;

pub mod embedding;

pub use embedding::EmbeddingProvider;

use async_trait::async_trait;
use thiserror::Error;

use llm_wiki_core::error::WikiError;

#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// Task tag for telemetry/cache classification (e.g. "document-analysis").
    pub task_tag: String,
    pub system: Option<String>,
    pub prompt: String,
    /// Low default temperature per PRD §44 determinism guidance.
    pub temperature: f32,
    pub max_output_tokens: u32,
    /// Requests JSON-constrained output from the provider.
    pub json_mode: bool,
}

impl LlmRequest {
    pub fn new(task_tag: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            task_tag: task_tag.into(),
            system: None,
            prompt: prompt.into(),
            temperature: 0.0,
            max_output_tokens: 4096,
            json_mode: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("llm http error: {0}")]
    Http(String),
    #[error("llm timeout after {timeout_seconds}s")]
    Timeout { timeout_seconds: u64 },
    #[error("llm rate limited (retry after {retry_after_seconds:?}s)")]
    RateLimited { retry_after_seconds: Option<u64> },
    #[error("llm api error {code}: {message}")]
    Api { code: u16, message: String },
    #[error("llm response was not the expected shape: {0}")]
    InvalidResponse(String),
    #[error("api key env var '{0}' is not set")]
    MissingApiKey(String),
}

impl From<LlmError> for WikiError {
    fn from(err: LlmError) -> Self {
        WikiError::Llm(err.to_string())
    }
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Model identifier; recorded in build metadata and cache keys (PRD §44).
    fn model(&self) -> &str;

    fn provider_name(&self) -> &str;

    async fn generate(&self, request: LlmRequest) -> Result<LlmResponse, LlmError>;
}
