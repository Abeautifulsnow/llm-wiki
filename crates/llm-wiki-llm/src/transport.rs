//! Shared HTTP transport for the endpoint adapters (PRD §25, §32).
//!
//! Chat, embeddings and rerank may point at three DIFFERENT providers, so
//! the wire mechanics — auth, timeout, loopback proxy bypass and the bounded
//! 429/5xx backoff — live here once and every adapter reuses them. The API
//! key is read from the env var *named* by the config; it never lives in
//! project config.

use std::time::Duration;

use crate::LlmError;

pub(crate) struct HttpTransport {
    base_url: String,
    api_key: Option<String>,
    timeout: Duration,
    max_retries: u32,
    client: reqwest::Client,
}

impl HttpTransport {
    /// Builds a transport. `api_key_env` is the config-declared env var
    /// name; a missing env var is only an error once a request actually
    /// needs it (some local endpoints run without keys).
    pub fn new(
        base_url: &str,
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
        let mut builder = reqwest::Client::builder().timeout(Duration::from_secs(timeout_seconds));
        if is_loopback_base_url(base_url) {
            builder = builder.no_proxy();
        }
        let client = builder
            .build()
            .map_err(|e| LlmError::Http(format!("client build: {e}")))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key,
            timeout: Duration::from_secs(timeout_seconds),
            max_retries,
            client,
        })
    }

    /// POSTs `{base_url}/{path}` with bearer auth and bounded exponential
    /// backoff on 429/5xx/network errors; other statuses fail fast.
    pub async fn post(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}/{path}", self.base_url);
        let mut attempt = 0u32;
        loop {
            let mut request = self.client.post(&url).json(body);
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

/// True when the base URL points at a loopback host (localhost, 127.0.0.1,
/// ::1): such endpoints are reached directly — a system proxy (http_proxy
/// env) must never intercept them. Local gateways are a first-class target
/// of these adapters.
fn is_loopback_base_url(base_url: &str) -> bool {
    match reqwest::Url::parse(base_url) {
        Ok(url) => matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1")),
        Err(_) => false,
    }
}

pub(crate) async fn backoff(attempt: u32) {
    let millis = 500u64 * (1u64 << attempt.min(4));
    tokio::time::sleep(Duration::from_millis(millis)).await;
}
