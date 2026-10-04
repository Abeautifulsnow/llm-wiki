//! Security middleware (PRD §30): request ids, authentication and per-caller
//! rate limits.
//!
//! - Local mode (`remote_enabled = false`): `/v1` only accepts loopback peers
//!   (defense in depth on top of the loopback bind).
//! - Remote mode: `Authorization: Bearer <token>`; the expected token lives
//!   in the env var named by `server.auth_token_env`. If the env var is
//!   missing at request time every `/v1` request fails closed with 503.
//! - Rate limiting applies to remote mode only (`rate_limit_per_minute`, 0
//!   disables).

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::error::ApiError;
use crate::state::SharedState;

/// Opaque per-request id (also echoed as the `x-request-id` response header).
#[derive(Debug, Clone)]
pub struct RequestId(pub Arc<String>);

pub async fn request_id(mut req: Request, next: Next) -> Response {
    let id = Arc::new(format!("req_{}", ulid::Ulid::new()));
    req.extensions_mut().insert(RequestId(id.clone()));
    let mut response = next.run(req).await;
    if let Ok(value) = axum::http::HeaderValue::from_str(&id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

fn bearer_token(headers: &HeaderMap) -> Option<String> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    value
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
}

/// Constant-time token comparison: the loop always runs over the LONGER
/// input and pads the shorter with zeros, so timing does not track how many
/// leading bytes matched. The final result still encodes the length
/// difference (the caller must not distinguish mismatch reasons anyway).
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = u8::from(a.len() != b.len());
    let max = a.len().max(b.len());
    for index in 0..max {
        let x = a.get(index).copied().unwrap_or(0);
        let y = b.get(index).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0
}

pub async fn auth(State(state): State<SharedState>, req: Request, next: Next) -> Response {
    let remote = state.config().server.remote_enabled;
    if remote {
        let expected = std::env::var(&state.config().server.auth_token_env).unwrap_or_default();
        if expected.trim().is_empty() {
            // Fail closed: remote mode without its token serves nothing.
            return ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "auth_unavailable",
                "remote mode is enabled but the auth token env is not set; refusing all requests",
            )
            .into_response();
        }
        let Some(provided) = bearer_token(req.headers()) else {
            return ApiError::unauthorized("missing bearer token").into_response();
        };
        if !constant_time_eq(&provided, expected.trim()) {
            return ApiError::unauthorized("invalid bearer token").into_response();
        }
    } else if let Some(ConnectInfo(addr)) = req.extensions().get::<ConnectInfo<SocketAddr>>() {
        if !addr.ip().is_loopback() {
            return ApiError::forbidden_remote().into_response();
        }
    }
    next.run(req).await
}

/// The rate-limit key for a request: peer IP when available, one shared
/// bucket otherwise (oneshot tests / unix sockets).
fn caller_key(req: &Request) -> String {
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.ip().to_string())
        .unwrap_or_else(|| "local".to_owned())
}

/// Rate-limit buckets past this age are dead entries (several expired
/// windows old) and get evicted when the map grows (#S01: the per-caller map
/// must not grow unboundedly on a long-lived remote server).
const BUCKET_RETENTION_SECS: u64 = 300;
const BUCKET_EVICT_THRESHOLD: usize = 1024;

pub async fn rate_limit(State(state): State<SharedState>, req: Request, next: Next) -> Response {
    let per_minute = state.config().server.rate_limit_per_minute;
    let remote = state.config().server.remote_enabled;
    if remote && per_minute > 0 {
        let key = caller_key(&req);
        let mut buckets = state.0.rate_buckets.lock().await;
        let now = std::time::Instant::now();
        if buckets.len() >= BUCKET_EVICT_THRESHOLD {
            buckets.retain(|_, bucket| {
                now.duration_since(bucket.window_started_at).as_secs() < BUCKET_RETENTION_SECS
            });
        }
        let bucket = buckets.entry(key).or_insert(crate::state::RateBucket {
            window_started_at: now,
            count: 0,
        });
        if now.duration_since(bucket.window_started_at).as_secs() >= 60 {
            bucket.window_started_at = now;
            bucket.count = 0;
        }
        bucket.count += 1;
        if bucket.count > per_minute {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"error": {"code": "rate_limited",
                    "message": "per-caller request budget exceeded; retry later"}})),
            )
                .into_response();
        }
    }
    next.run(req).await
}
