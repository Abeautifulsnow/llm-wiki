//! Stable API error codes + JSON envelope (PRD §30/§34).
//!
//! Every error response is `{"error": {"code", "message"}}` with the
//! `x-request-id` header attached by the request-id middleware. Codes are
//! part of the API contract — never rename one without a version bump.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use llm_wiki_core::error::WikiError;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "unauthorized", message)
    }

    /// 403 for requests that would be fine locally but hit a non-loopback
    /// peer while `remote_enabled = false`.
    pub fn forbidden_remote() -> Self {
        Self::new(
            StatusCode::FORBIDDEN,
            "remote_disabled",
            "this server is local-only (server.remote_enabled = false); \
             /v1 endpoints only accept loopback connections",
        )
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", message)
    }

    pub fn conflict(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, code, message)
    }

    pub fn queue_full(limit: u32) -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "job_queue_full",
            format!("the job queue is full ({limit} queued); retry later"),
        )
    }

    pub fn rate_limited() -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limited",
            "per-caller request budget exceeded; retry later",
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
    }

    pub fn nothing_published() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "nothing_published",
            "no generation is published; run a build first",
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "code": self.code,
                    "message": self.message,
                }
            })),
        )
            .into_response()
    }
}

/// Maps the domain error model (PRD §34) onto HTTP: the category decides the
/// status + stable code; the message is preserved verbatim.
///
/// KNOWN EXPOSURE (review #S04, accepted for the local-first v1): `Storage`/
/// `Index` messages can embed SQL or path detail in 500 bodies. Local mode is
/// the primary deployment; before exposing a server on a reachable interface
/// (remote mode), scrub those two variants to a generic message and keep the
/// detail in the `tracing` log.
impl From<WikiError> for ApiError {
    fn from(err: WikiError) -> Self {
        let (status, code) = match &err {
            WikiError::Config(_) | WikiError::InvalidId { .. } => {
                (StatusCode::BAD_REQUEST, "config_error")
            }
            WikiError::Source(_) | WikiError::PathCollision { .. } => {
                (StatusCode::BAD_REQUEST, "source_error")
            }
            WikiError::Parse { .. } => (StatusCode::UNPROCESSABLE_ENTITY, "parse_error"),
            WikiError::Storage(_) => (StatusCode::INTERNAL_SERVER_ERROR, "storage_error"),
            WikiError::Llm(_) => (StatusCode::BAD_GATEWAY, "llm_error"),
            WikiError::SchemaValidation(_) | WikiError::EvidenceValidation(_) => {
                (StatusCode::BAD_GATEWAY, "validation_error")
            }
            WikiError::Planning(_) => (StatusCode::UNPROCESSABLE_ENTITY, "planning_error"),
            WikiError::ReplanRequired { .. } => (StatusCode::CONFLICT, "replan_required"),
            WikiError::Compilation(_) | WikiError::Index(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
            }
            WikiError::BudgetExceeded(_) => (StatusCode::UNPROCESSABLE_ENTITY, "budget_exceeded"),
            WikiError::PublishRecovery(_) => (StatusCode::CONFLICT, "publish_recovery"),
            WikiError::Lint { .. } => (StatusCode::UNPROCESSABLE_ENTITY, "lint_failed"),
            WikiError::Cancelled => (StatusCode::CONFLICT, "cancelled"),
        };
        Self::new(status, code, err.to_string())
    }
}
