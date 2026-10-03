#![forbid(unsafe_code)]
//! Axum HTTP transport over the application services (PRD §7.8, §30, §31).
//!
//! This crate is a pure transport adapter, exactly as the CLI is: every
//! handler resolves to `llm-wiki-compiler` / `llm-wiki-search` /
//! `llm-wiki-storage` calls; no business logic lives here. Security posture
//! (PRD §30 "API 契约与远程暴露"):
//!
//! - Default bind is loopback only; `server.remote_enabled` must be explicit
//!   and requires an auth token from the configured environment variable
//!   (never from the config file). TLS is the deployment layer's job.
//! - Remote mode enforces per-caller rate limits; all modes cap request body
//!   size and the job queue.
//! - Write operations accept an `Idempotency-Key`; the same key returns the
//!   SAME job instead of starting a second LLM run.
//! - `source_id` values that are not configured source roots are rejected
//!   with a 4xx — the server never scans an unvetted path.

pub mod error;
pub mod jobs;
pub mod middleware;
pub mod routes;
pub mod state;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use llm_wiki_core::config::Config;
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_llm::LlmProvider;

pub use error::ApiError;
pub use state::{JobManager, SharedState};

/// Everything [`serve`] needs: the workspace (holding `.llm-wiki/config.toml`
/// conventions), the effective config, optional CLI host override and the
/// (optional) LLM provider for build/query endpoints.
pub struct ServeOptions {
    pub workspace: PathBuf,
    pub config: Config,
    /// CLI `--host` override; defaults to `server.bind`.
    pub host: Option<String>,
    pub port: u16,
    /// Built from `[llm]` config when `llm.model` is set; absent → build and
    /// query endpoints return a 400 config error (fail-closed, never a
    /// half-configured LLM call).
    pub provider: Option<Arc<dyn LlmProvider>>,
}

/// True when `host` only exposes loopback interfaces.
pub fn is_loopback_host(host: &str) -> bool {
    match host.trim().trim_start_matches('[').trim_end_matches(']') {
        "localhost" => true,
        other => other
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false),
    }
}

/// Startup security validation (PRD §30): the server refuses to start on an
/// unsafe configuration instead of degrading silently.
pub fn validate_security(config: &Config, host: &str) -> Result<()> {
    if !config.server.remote_enabled && !is_loopback_host(host) {
        return Err(WikiError::Config(format!(
            "server.bind = {host} is not a loopback address; binding a reachable \
             interface requires server.remote_enabled = true in .llm-wiki/config.toml"
        )));
    }
    if config.server.remote_enabled {
        let token = std::env::var(&config.server.auth_token_env).unwrap_or_default();
        if token.trim().is_empty() {
            return Err(WikiError::Config(format!(
                "server.remote_enabled = true requires a non-empty token in env {} \
                 (the token is never read from the config file)",
                config.server.auth_token_env
            )));
        }
    }
    Ok(())
}

/// Runs the HTTP server until Ctrl-C. Startup recovery marks QUEUED/RUNNING
/// jobs INTERRUPTED (PRD §31: no fake RUNNING state across restarts).
pub async fn serve(options: ServeOptions) -> Result<()> {
    let body_limit = options.config.server.max_body_bytes as usize;
    let remote_enabled = options.config.server.remote_enabled;
    let host = options
        .host
        .unwrap_or_else(|| options.config.server.bind.clone());
    validate_security(&options.config, &host)?;

    let state = SharedState::new(options.workspace.clone(), options.config, options.provider);

    // §31 startup recovery over the persisted job rows.
    let db_path = state.db_path();
    if db_path.exists() {
        let conn = llm_wiki_storage::open(&db_path)?;
        let recovered = llm_wiki_storage::mark_stale_jobs_interrupted(&conn)?;
        if recovered > 0 {
            tracing::warn!(
                count = recovered,
                "recovered stale server jobs as INTERRUPTED"
            );
        }
    }

    let app = build_router(state).layer(axum::extract::DefaultBodyLimit::max(body_limit));
    let listener = tokio::net::TcpListener::bind((host.as_str(), options.port))
        .await
        .map_err(|e| WikiError::Source(format!("cannot bind {host}:{}: {e}", options.port)))?;
    tracing::info!(
        addr = %listener.local_addr().map_err(|e| WikiError::Source(e.to_string()))?,
        remote_enabled,
        "llm-wiki server listening"
    );
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
    })
    .await
    .map_err(|e| WikiError::Source(format!("server error: {e}")))?;
    Ok(())
}

/// The full v1 API (PRD §30). `/health` is unauthenticated; everything under
/// `/v1` passes the security middleware (loopback restriction in local mode,
/// bearer token in remote mode).
pub fn build_router(state: SharedState) -> Router {
    let protected = Router::new()
        .route("/v1/status", get(routes::status))
        .route("/v1/build", post(routes::build))
        .route("/v1/jobs", get(routes::list_jobs))
        .route("/v1/jobs/{job_id}", get(routes::get_job))
        .route("/v1/jobs/{job_id}/cancel", post(routes::cancel_job))
        .route("/v1/search", post(routes::search))
        .route("/v1/context", post(routes::context))
        .route("/v1/query", post(routes::query))
        .route("/v1/pages", get(routes::list_pages))
        .route("/v1/pages/{id}", get(routes::get_page))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::auth,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::rate_limit,
        ));
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/health", get(routes::health))
        .merge(protected)
        .layer(axum::middleware::from_fn(middleware::request_id))
        .with_state(state)
}

/// Convenience for tests and embedders: a workspace's state db path.
pub fn state_db_path(workspace: &Path) -> PathBuf {
    workspace.join(".llm-wiki").join("state.db")
}
