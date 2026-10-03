//! Embedding provider abstraction (§19.3 Vector layer): text → dense vector,
//! served by the same OpenAI-compatible base URL as chat completions.

use async_trait::async_trait;

use crate::LlmError;

/// One embedding batch: texts in, one vector per text out, same order.
/// Dimensions are model-owned; callers must not mix vectors across models.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Errors are hard (no retry loop here beyond the transport's own
    /// backoff): embedding is an explicit, user-driven step.
    async fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError>;
}
