//! OpenAI-compatible chat-completions adapter (PRD §25, §32).
//!
//! Works against any `/v1/chat/completions` endpoint (vLLM, OpenRouter, …).
//! The API key is read from the env var *named* by the config — it never
//! lives in project config. Retry is bounded exponential backoff on
//! 429/5xx/network errors; other statuses fail fast.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use crate::{LlmError, LlmProvider, LlmRequest, LlmResponse};

pub struct OpenAiCompatibleProvider {
    base_url: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
    max_retries: u32,
    client: reqwest::Client,
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
        let api_key = match std::env::var(api_key_env) {
            Ok(key) if !key.trim().is_empty() => Some(key.trim().to_owned()),
            Ok(_) => None,
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(LlmError::MissingApiKey(format!("{api_key_env}: {e}"))),
        };
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .map_err(|e| LlmError::Http(format!("client build: {e}")))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            api_key,
            timeout: Duration::from_secs(timeout_seconds),
            max_retries,
            client,
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

    async fn send(&self, body: serde_json::Value) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut attempt = 0u32;
        loop {
            let mut request = self.client.post(&url).json(&body);
            if let Some(key) = &self.api_key {
                request = request.bearer_auth(key);
            }
            let result = request.send().await;

            match result {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return Ok(response);
                    }
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if !retryable || attempt >= self.max_retries {
                        let message = response.text().await.unwrap_or_default();
                        return Err(LlmError::Api {
                            code: status.as_u16(),
                            message,
                        });
                    }
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok());
                    if let Some(seconds) = retry_after {
                        tokio::time::sleep(Duration::from_secs(seconds.min(30))).await;
                    } else {
                        backoff(attempt).await;
                    }
                }
                Err(err) => {
                    let is_timeout = err.is_timeout();
                    if attempt >= self.max_retries {
                        return Err(if is_timeout {
                            LlmError::Timeout {
                                timeout_seconds: self.timeout.as_secs(),
                            }
                        } else {
                            LlmError::Http(err.to_string())
                        });
                    }
                    backoff(attempt).await;
                }
            }
            attempt += 1;
        }
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

async fn backoff(attempt: u32) {
    let millis = 500u64 * (1u64 << attempt.min(4));
    tokio::time::sleep(Duration::from_millis(millis)).await;
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
        let params_applied =
            apply_thinking_params(&mut body, self.thinking, self.thinking_effort.as_deref())
                && !self.thinking_downgraded.load(Ordering::SeqCst);

        // Fault-tolerant thinking controls (T1): providers implement them
        // differently or not at all. A rejection of the PARAMETIZED request
        // downgrades the provider for the rest of the build and retries the
        // identical request WITHOUT the params — never an error on the user's
        // behalf, always a warning in the log.
        let response = match self.send(body.clone()).await {
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
                self.send(body).await?
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

    #[test]
    fn thinking_mode_parse_is_tolerant() {
        assert_eq!(ThinkingMode::parse(""), ThinkingMode::Auto);
        assert_eq!(ThinkingMode::parse(" OFF "), ThinkingMode::Off);
        assert_eq!(ThinkingMode::parse("enabled"), ThinkingMode::On);
        assert_eq!(ThinkingMode::parse("banana"), ThinkingMode::Auto);
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
