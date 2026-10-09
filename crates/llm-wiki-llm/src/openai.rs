//! OpenAI-compatible chat-completions adapter (PRD §25, §32).
//!
//! Works against any `/v1/chat/completions` endpoint (vLLM, OpenRouter, …).
//! The API key is read from the env var *named* by the config — it never
//! lives in project config. Retry is bounded exponential backoff on
//! 429/5xx/network errors; other statuses fail fast.

use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde_json::json;

use crate::transport::HttpTransport;
use crate::{LlmError, LlmProvider, LlmRequest, LlmResponse};

pub struct OpenAiCompatibleProvider {
    model: String,
    transport: HttpTransport,
    thinking: ThinkingMode,
    thinking_effort: Option<String>,
    /// Set once the provider rejected the thinking parameters (HTTP 4xx):
    /// every later request goes out WITHOUT them. Never re-probed — providers
    /// do not grow support mid-build.
    thinking_downgraded: AtomicBool,
}

/// Provider-agnostic thinking control (T1: reasoning tokens share the output
/// budget; some units need thinking OFF to fit it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingMode {
    /// Send nothing — provider default. Also the fallback after a rejection.
    #[default]
    Auto,
    On,
    Off,
}

impl ThinkingMode {
    /// Tolerant parse (config strings): unknown values degrade to `Auto` with
    /// a warning instead of failing the build — providers differ in support
    /// and a typo must not kill a 40-minute run.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "on" | "enabled" | "true" => Self::On,
            "off" | "disabled" | "false" => Self::Off,
            other => {
                tracing::warn!(
                    value = other,
                    "unknown llm.thinking value; ignoring (expected auto|on|off)"
                );
                Self::Auto
            }
        }
    }
}

impl OpenAiCompatibleProvider {
    /// Builds a provider. `api_key_env` is the config-declared env var name;
    /// a missing env var is only an error once a request actually needs it
    /// (some local endpoints run without keys).
    pub fn new(
        base_url: &str,
        model: &str,
        api_key_env: &str,
        timeout_seconds: u64,
        max_retries: u32,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            model: model.to_owned(),
            transport: HttpTransport::new(base_url, api_key_env, timeout_seconds, max_retries)?,
            thinking: ThinkingMode::Auto,
            thinking_effort: None,
            thinking_downgraded: AtomicBool::new(false),
        })
    }

    /// Applies the config-declared thinking controls (values parsed
    /// tolerantly — see [`ThinkingMode::parse`]). Empty effort sends nothing.
    pub fn with_thinking(mut self, thinking: &str, effort: &str) -> Self {
        self.thinking = ThinkingMode::parse(thinking);
        let effort = effort.trim();
        self.thinking_effort = match effort {
            "" => None,
            other => Some(other.to_ascii_lowercase()),
        };
        self
    }

    async fn send(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
        self.transport.post(path, &body).await
    }
}

/// What a live chat round trip actually proved, for `llm-wiki doctor`
/// (V1.0 field test: `doctor` reported the API KEY was set and said nothing
/// about whether the endpoint served the configured model).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatProbeOutcome {
    /// Model name the endpoint echoed back. A gateway may legally serve a
    /// different revision than the one requested, so the caller reports a
    /// substitution instead of failing it.
    pub served_model: Option<String>,
    /// Whether `GET /models` listed the requested model. `None` when the
    /// catalog is unavailable or unparseable — NOT a failure: self-hosted
    /// gateways commonly vary here, so the live request is the verdict.
    pub model_listed: Option<bool>,
    pub latency_ms: u128,
}

impl OpenAiCompatibleProvider {
    /// One minimal round trip on the REAL chat path, for `doctor`. Unlike
    /// [`LlmProvider::generate`] it does not require non-empty content: a
    /// thinking model given a tiny output budget still proves the endpoint
    /// serves the model, which is all a connectivity probe claims.
    pub async fn probe(&self) -> Result<ChatProbeOutcome, LlmError> {
        let started = std::time::Instant::now();
        let model_listed = self.catalog_lists_model().await;
        let body = json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": "ping" }],
            "max_tokens": 8,
        });
        let response = self.send("chat/completions", body).await?;
        let payload: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(format!("body is not JSON: {e}")))?;
        if payload
            .get("choices")
            .and_then(|choices| choices.get(0))
            .is_none()
        {
            return Err(LlmError::InvalidResponse(format!(
                "the chat response carries no choices[0]; the endpoint answered but is not serving '{}'",
                self.model
            )));
        }
        Ok(ChatProbeOutcome {
            served_model: payload
                .get("model")
                .and_then(|model| model.as_str())
                .map(str::to_owned),
            model_listed,
            latency_ms: started.elapsed().as_millis(),
        })
    }

    /// Best-effort `GET /models`. `None` (unavailable, non-JSON, no `data[]`)
    /// is not an error — see [`ChatProbeOutcome::model_listed`].
    async fn catalog_lists_model(&self) -> Option<bool> {
        let response = self.transport.get("models").await.ok()?;
        let payload: serde_json::Value = response.json().await.ok()?;
        let ids = payload.get("data").and_then(|data| data.as_array())?;
        Some(
            ids.iter().any(|entry| {
                entry.get("id").and_then(|id| id.as_str()) == Some(self.model.as_str())
            }),
        )
    }
}

/// Writes the thinking controls into the request body (Ark/Doubao spelling
/// for the on/off switch — probe-verified against the T1 gateway;
/// OpenAI-style `reasoning_effort` for the effort knob). Returns whether any
/// parameter was applied.
fn apply_thinking_params(
    body: &mut serde_json::Value,
    mode: ThinkingMode,
    effort: Option<&str>,
) -> bool {
    let mut applied = false;
    match mode {
        ThinkingMode::Auto => {}
        ThinkingMode::On => {
            body["thinking"] = json!({ "type": "enabled" });
            applied = true;
        }
        ThinkingMode::Off => {
            body["thinking"] = json!({ "type": "disabled" });
            applied = true;
        }
    }
    if let Some(effort) = effort.filter(|e| !e.is_empty()) {
        body["reasoning_effort"] = json!(effort);
        applied = true;
    }
    applied
}

/// Removes every thinking-related parameter (the downgrade path).
fn strip_thinking_params(body: &mut serde_json::Value) {
    if let Some(obj) = body.as_object_mut() {
        obj.remove("thinking");
        obj.remove("reasoning_effort");
    }
}

#[async_trait]
impl LlmProvider for OpenAiCompatibleProvider {
    fn model(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "openai-compatible"
    }

    async fn generate(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let mut messages = Vec::new();
        if let Some(system) = &request.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        messages.push(json!({ "role": "user", "content": request.prompt }));

        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        });
        if request.json_mode {
            body["response_format"] = json!({ "type": "json_object" });
        }
        // Check the downgrade BEFORE applying: `apply_thinking_params`
        // mutates the body as a side effect, so evaluating it first would
        // re-send the rejected parameters on every later request — and with
        // `params_applied == false` the downgrade retry below no longer
        // fires, failing the build.
        let params_applied = if self.thinking_downgraded.load(Ordering::SeqCst) {
            false
        } else {
            apply_thinking_params(&mut body, self.thinking, self.thinking_effort.as_deref())
        };

        // Fault-tolerant thinking controls (T1): providers implement them
        // differently or not at all. A rejection of the PARAMETIZED request
        // downgrades the provider for the rest of the build and retries the
        // identical request WITHOUT the params — never an error on the user's
        // behalf, always a warning in the log.
        let response = match self.send("chat/completions", body.clone()).await {
            Err(LlmError::Api { code, message })
                if params_applied
                    && (400..500).contains(&code)
                    && !self.thinking_downgraded.swap(true, Ordering::SeqCst) =>
            {
                tracing::warn!(
                    code = code,
                    "provider rejected the thinking parameters (llm.thinking/llm.thinking_effort);                      continuing the rest of this build WITHOUT them (set llm.thinking = \"auto\" to silence)"
                );
                let _ = message;
                strip_thinking_params(&mut body);
                self.send("chat/completions", body).await?
            }
            other => other?,
        };
        let payload: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(format!("body is not JSON: {e}")))?;

        let choice = payload
            .get("choices")
            .and_then(|c| c.get(0))
            .ok_or_else(|| LlmError::InvalidResponse("missing choices[0]".to_owned()))?;
        let text = choice
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| {
                LlmError::InvalidResponse("missing choices[0].message.content".to_owned())
            })?
            .to_owned();
        let finish_reason = choice
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .map(str::to_owned);
        let (input_tokens, output_tokens) = payload
            .get("usage")
            .map(|u| {
                (
                    u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    u.get("completion_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                )
            })
            .unwrap_or((0, 0));

        Ok(LlmResponse {
            text,
            model: self.model.clone(),
            input_tokens,
            output_tokens,
            finish_reason,
        })
    }
}

/// §19.3 Vector layer: the chat provider can double as an embedder on the
/// SAME endpoint. Standalone embedding endpoints (`[embedding] base_url`)
/// use [`crate::OpenAiCompatibleEmbeddings`] instead — both share the
/// request/parse path in [`crate::embedding`].
#[async_trait]
impl crate::EmbeddingProvider for OpenAiCompatibleProvider {
    async fn embed(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        crate::embedding::post_embeddings(&self.transport, model, texts).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_params_apply_and_strip() {
        let mut body = json!({ "model": "m" });
        assert!(!apply_thinking_params(&mut body, ThinkingMode::Auto, None));
        assert!(body.get("thinking").is_none());

        let mut body = json!({ "model": "m" });
        assert!(apply_thinking_params(
            &mut body,
            ThinkingMode::Off,
            Some("low")
        ));
        assert_eq!(body["thinking"]["type"], "disabled");
        assert_eq!(body["reasoning_effort"], "low");
        strip_thinking_params(&mut body);
        assert!(body.get("thinking").is_none());
        assert!(body.get("reasoning_effort").is_none());

        let mut body = json!({ "model": "m" });
        assert!(apply_thinking_params(&mut body, ThinkingMode::On, None));
        assert_eq!(body["thinking"]["type"], "enabled");
    }

    /// Downgrade is permanent for the provider's lifetime: after the first
    /// rejected request, EVERY later generate must go out WITHOUT the
    /// thinking parameters — re-sending them would 400 again and, with the
    /// retry guard disabled by the downgrade, fail the build.
    #[tokio::test]
    async fn downgraded_provider_never_sends_thinking_params_again() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            // Request 1: with thinking → 400; its retry without → 200.
            // Request 2 (a later generate): MUST arrive without thinking.
            for expect_thinking in [true, false, false] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).unwrap();
                let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
                let has_thinking = raw.contains("\"thinking\":{\"type\":\"disabled\"}");
                assert_eq!(has_thinking, expect_thinking, "request body: {raw}");
                let body = if expect_thinking {
                    r#"{"error":{"message":"unknown parameter thinking","type":"invalid_request_error"}}"#
                } else {
                    r#"{"choices":[{"message":{"content":"{\"ok\":true}"},"finish_reason":"stop"}]}"#
                };
                let response = format!(
                    "HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: {}
Connection: close

{}",
                    body.len(),
                    body
                );
                let response = if expect_thinking {
                    response.replace("200 OK", "400 Bad Request")
                } else {
                    response
                };
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let provider = OpenAiCompatibleProvider::new(
            &format!("http://127.0.0.1:{port}"),
            "m",
            "UNUSED_VAR",
            10,
            0,
        )
        .unwrap()
        .with_thinking("off", "");
        let request = LlmRequest {
            task_tag: "test".into(),
            system: None,
            prompt: "ping".into(),
            temperature: 0.0,
            max_output_tokens: 16,
            json_mode: false,
        };
        provider.generate(request.clone()).await.unwrap();
        // The second generate goes straight out without the thinking params.
        provider.generate(request).await.unwrap();
        assert!(provider.thinking_downgraded.load(Ordering::SeqCst));
        server.join().unwrap();
    }

    #[test]
    fn thinking_mode_parse_is_tolerant() {
        assert_eq!(ThinkingMode::parse(""), ThinkingMode::Auto);
        assert_eq!(ThinkingMode::parse(" OFF "), ThinkingMode::Off);
        assert_eq!(ThinkingMode::parse("enabled"), ThinkingMode::On);
        assert_eq!(ThinkingMode::parse("banana"), ThinkingMode::Auto);
    }

    /// A doctor probe must go through the real chat path and report both the
    /// served model and whether the catalog lists the requested one. An empty
    /// `content` (a thinking model that spent its 8-token budget on
    /// chain-of-thought) is NOT a probe failure — the endpoint answered.
    #[tokio::test]
    async fn probe_reports_served_model_and_catalog_hit_with_empty_content() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            // The probe issues GET /models first, then POST /chat/completions.
            for path in ["/models", "/chat/completions"] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).unwrap();
                let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
                assert!(raw.starts_with(if path == "/models" { "GET" } else { "POST" }));
                assert!(raw.contains(path), "request line: {raw}");
                let body = if path == "/models" {
                    r#"{"data":[{"id":"other-model"},{"id":"m"}]}"#.to_owned()
                } else {
                    r#"{"model":"m-2026-01","choices":[{"message":{"content":""},"finish_reason":"length"}]}"#
                        .to_owned()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let provider = OpenAiCompatibleProvider::new(
            &format!("http://127.0.0.1:{port}"),
            "m",
            "UNUSED_VAR",
            10,
            0,
        )
        .unwrap();
        let outcome = provider.probe().await.unwrap();
        assert_eq!(outcome.served_model.as_deref(), Some("m-2026-01"));
        assert_eq!(outcome.model_listed, Some(true));
        server.join().unwrap();
    }

    /// An endpoint that answers without `choices[]` is not serving the model:
    /// the probe must fail loudly rather than report a green check.
    #[tokio::test]
    async fn probe_fails_when_the_endpoint_serves_something_else() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let _ = stream.read(&mut buf).unwrap();
                let body = r#"{"error":{"message":"model not found"}}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let provider = OpenAiCompatibleProvider::new(
            &format!("http://127.0.0.1:{port}"),
            "m",
            "UNUSED_VAR",
            10,
            0,
        )
        .unwrap();
        let err = provider.probe().await.unwrap_err();
        assert!(err.to_string().contains("choices[0]"), "{err}");
        server.join().unwrap();
    }

    /// The `/models` catalog is advisory: a gateway that does not expose one
    /// must leave `model_listed` at `None` and still pass on the live call.
    #[tokio::test]
    async fn probe_treats_a_missing_catalog_as_unknown_not_failure() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).unwrap();
                let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
                let (status, body) = if raw.starts_with("GET") {
                    ("404 Not Found", "not found".to_owned())
                } else {
                    (
                        "200 OK",
                        r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#
                            .to_owned(),
                    )
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let provider = OpenAiCompatibleProvider::new(
            &format!("http://127.0.0.1:{port}"),
            "m",
            "UNUSED_VAR",
            10,
            0,
        )
        .unwrap();
        let outcome = provider.probe().await.unwrap();
        assert_eq!(outcome.model_listed, None);
        server.join().unwrap();
    }

    /// Fault-tolerance contract: a provider that rejects the thinking
    /// parameters must NOT fail the request — the call retries without them
    /// and the provider downgrades for the rest of the build.
    #[tokio::test]
    async fn rejected_thinking_params_downgrade_and_retry() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            for expect_thinking in [true, false] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let n = stream.read(&mut buf).unwrap();
                let raw = String::from_utf8_lossy(&buf[..n]).into_owned();
                let has_thinking = raw.contains("\"thinking\":{\"type\":\"disabled\"}");
                assert_eq!(has_thinking, expect_thinking, "request body: {raw}");
                // Read nothing more; respond.
                let body = if expect_thinking {
                    r#"{"error":{"message":"unknown parameter thinking","type":"invalid_request_error"}}"#
                } else {
                    r#"{"choices":[{"message":{"content":"{\"ok\":true}"},"finish_reason":"stop"}]}"#
                };
                let response = format!(
                    "HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: {}
Connection: close

{}",
                    body.len(),
                    body
                );
                // The downgrade probe answers 400-style via the JSON body the
                // provider already maps to LlmError::Api (status line 400).
                let response = if expect_thinking {
                    response.replace("200 OK", "400 Bad Request")
                } else {
                    response
                };
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let provider = OpenAiCompatibleProvider::new(
            &format!("http://127.0.0.1:{port}"),
            "m",
            "UNUSED_VAR",
            10,
            0,
        )
        .unwrap()
        .with_thinking("off", "");
        let response = provider
            .generate(LlmRequest {
                task_tag: "test".into(),
                system: None,
                prompt: "ping".into(),
                temperature: 0.0,
                max_output_tokens: 16,
                json_mode: false,
            })
            .await
            .unwrap();
        assert_eq!(response.text, "{\"ok\":true}");
        assert!(provider.thinking_downgraded.load(Ordering::SeqCst));
        server.join().unwrap();
    }
}
