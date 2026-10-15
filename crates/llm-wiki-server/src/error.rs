//! Stable API error codes + JSON envelope (PRD §30/§34).
//!
//! Every error response is `{"error": {"code", "message"}}` with the
//! `x-request-id` header attached by the request-id middleware. Codes are
//! part of the API contract — never rename one without a version bump.
//!
//! Messages are split in two (review #S04 remediation): `message` is the
//! public, client-safe text; `internal_detail` (never serialized) carries the
//! full error for the `tracing` log, correlated by request id. `Storage` /
//! `Index` / `Compilation` / `Llm` messages can embed SQL, paths or upstream
//! bodies, so those four scrub to fixed text in every deployment mode.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use llm_wiki_core::error::WikiError;

use crate::middleware::REQUEST_ID;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    /// Log-only companion of `message`: the full internal error. Never
    /// serialized into the response body.
    pub internal_detail: Option<String>,
}

/// The generic public text for scrubbed internal failures; points operators
/// at the log, which carries the detail plus the request id.
const INTERNAL_MESSAGE: &str = "internal error; see server logs";
/// The generic public text for scrubbed upstream model failures.
const LLM_MESSAGE: &str = "upstream model request failed";

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            internal_detail: None,
        }
    }

    /// A scrubbed error: fixed public text, full detail only in the log.
    pub fn scrubbed(
        status: StatusCode,
        code: &'static str,
        public_message: impl Into<String>,
        detail: impl std::fmt::Display,
    ) -> Self {
        let detail = detail.to_string();
        log_internal(status, code, &detail);
        Self {
            status,
            code,
            message: public_message.into(),
            internal_detail: Some(detail),
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

    /// 500 with a scrubbed public message: the detail goes to the log only.
    /// Every hand-built internal error funnels through here so the scrubbing
    /// policy cannot be bypassed by constructing `ApiError::new` directly.
    pub fn internal(detail: impl std::fmt::Display) -> Self {
        Self::scrubbed(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            INTERNAL_MESSAGE,
            detail,
        )
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
/// status + stable code. Messages that can embed internals (`Storage` /
/// `Compilation` / `Index` may carry SQL or paths; `Llm` may carry upstream
/// response bodies) scrub to fixed text in every deployment mode — local and
/// remote alike, so flipping `remote_enabled` never changes what leaks. The
/// 4xx/409/422 categories keep their actionable text (per-site audited for
/// path/credential echo). The full detail is logged with the request id.
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
        let public_message = match &err {
            WikiError::Storage(_) | WikiError::Compilation(_) | WikiError::Index(_) => {
                return Self::scrubbed(status, code, INTERNAL_MESSAGE, &err)
            }
            WikiError::Llm(_) => return Self::scrubbed(status, code, LLM_MESSAGE, &err),
            _ => err.to_string(),
        };
        Self::new(status, code, public_message)
    }
}

/// Logs a scrubbed error's internal detail, correlated by request id. The
/// task-local is only visible within the request scope (it does not cross
/// `spawn_blocking`), so conversions must happen there — `db()` converts
/// outside its blocking closure for exactly this reason.
fn log_internal(status: StatusCode, code: &str, detail: &str) {
    let request_id = REQUEST_ID.try_with(|id| (**id).clone()).unwrap_or_default();
    tracing::error!(
        request_id = %request_id,
        status = %status.as_u16(),
        code,
        detail = %redact(detail),
        "request failed (detail scrubbed from the response body)"
    );
}

/// Masks credential-shaped substrings before internal detail reaches the
/// log: the value run after `api_key`/`token`/`key`/`Bearer` markers becomes
/// `***`. Deliberately conservative — it never removes text, it only
/// shortens runs that follow a credential marker.
pub(crate) fn redact(text: &str) -> String {
    // Credential-shaped keys only: a bare `key=` would also mask innocent
    // values like `monkey=banana`, and `secret=`/`password=` are at least as
    // sensitive as `api_key=`. `Bearer ` runs BEFORE `Authorization: ` so the
    // token is masked (the other order would mask the scheme and leave it).
    const MARKERS: [&str; 7] = [
        "api_key=",
        "api-key=",
        "token=",
        "secret=",
        "password=",
        "Bearer ",
        "Authorization: ",
    ];
    // A credential value ends at the next delimiter: whitespace, a query
    // separator, or a quote — so `api_key=sk-1&next=2` masks only `sk-1`.
    const DELIMS: [char; 6] = [' ', '\t', '\n', '&', '"', '\''];
    let mut out = text.to_owned();
    for marker in MARKERS {
        let mut search_from = 0;
        while let Some(at) = out[search_from..].find(marker) {
            let start = search_from + at + marker.len();
            let end = out[start..]
                .find(|c: char| DELIMS.contains(&c))
                .map(|offset| start + offset)
                .unwrap_or(out.len());
            out.replace_range(start..end, "***");
            search_from = start + "***".len();
            if search_from >= out.len() {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_and_llm_errors_scrub_their_detail() {
        let storage = ApiError::from(WikiError::Storage(
            "prepare context sections: /abs/secret/path SELECT * FROM x".into(),
        ));
        assert_eq!(storage.code, "storage_error");
        assert_eq!(storage.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(storage.message, INTERNAL_MESSAGE);
        assert!(storage
            .internal_detail
            .unwrap()
            .contains("/abs/secret/path"));

        let llm = ApiError::from(WikiError::Llm(
            "llm api error 500: upstream said booms".into(),
        ));
        assert_eq!(llm.status, StatusCode::BAD_GATEWAY);
        assert_eq!(llm.message, LLM_MESSAGE);
    }

    #[test]
    fn client_actionable_errors_keep_their_text() {
        let config = ApiError::from(WikiError::Config("unknown cursor insight abc".into()));
        assert_eq!(config.status, StatusCode::BAD_REQUEST);
        assert_eq!(config.message, "config error: unknown cursor insight abc");
        assert!(config.internal_detail.is_none());
    }

    #[test]
    fn internal_constructor_is_scrubbed() {
        let err = ApiError::internal("task join: panic at /abs/path");
        assert_eq!(err.message, INTERNAL_MESSAGE);
        assert_eq!(
            err.internal_detail.as_deref(),
            Some("task join: panic at /abs/path")
        );
    }

    #[tokio::test]
    async fn response_body_carries_only_the_public_message() {
        let err = ApiError::from(WikiError::Index("index error: E:\\secret\\path".into()));
        let body = axum::body::to_bytes(err.into_response().into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("internal_error"));
        assert!(!text.contains("secret"));
    }

    #[test]
    fn redact_masks_credential_markers() {
        assert_eq!(
            redact("GET https://x.io/v1?api_key=sk-12345&k=v Bearer abc.def"),
            "GET https://x.io/v1?api_key=***&k=v Bearer ***"
        );
        assert_eq!(redact("no markers here"), "no markers here");
        // secret=/password= are masked; the Authorization header form keeps
        // the scheme name but masks both the scheme and the token value.
        assert_eq!(
            redact("token=abc secret=def password=ghi"),
            "token=*** secret=*** password=***"
        );
        assert_eq!(
            redact("Authorization: Bearer xyz.123"),
            "Authorization: *** ***"
        );
        // A bare `key=` must NOT mask innocent values.
        assert_eq!(redact("monkey=banana"), "monkey=banana");
    }
}
