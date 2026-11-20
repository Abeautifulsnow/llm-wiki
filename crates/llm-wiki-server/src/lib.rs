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
pub mod rerank;
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

/// The frozen agent-protocol version carried by every `/v1` response body
/// (V1.x contract: additive-only changes bump this number). See
/// `docs/agent-protocol.md`.
pub const PROTOCOL_VERSION: u32 = 1;

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
/// Startup version guard for the workspace this server owns.
///
/// Two questions, two severities: STRUCTURE (can this binary read the db?) is
/// fatal, CONTENT (was the published generation compiled by this binary?) is
/// graded by `server.strict_version_check`. Both log the concrete versions —
/// the whole point is that this mismatch is otherwise silent: pages keep
/// serving, just missing whatever the newer pipeline started producing.
fn verify_workspace_versions(
    conn: &llm_wiki_storage::Connection,
    config: &Config,
) -> Result<()> {
    let versions = llm_wiki_storage::check_workspace_versions(conn)?;
    check_db_schema(&versions)?;
    check_generation(&versions, config)
}

/// Structural compatibility.
///
/// A db written by a NEWER release is fatal, and `server.strict_version_check`
/// does not apply: migrations only move forward, so the tables this binary
/// knows may already have been replaced — reading them would fail at the first
/// query instead of here.
fn check_db_schema(versions: &llm_wiki_storage::WorkspaceVersions) -> Result<()> {
    if versions.db_is_from_the_future() {
        tracing::error!(
            db_schema = versions.db_schema_version,
            supported = versions.supported_schema_version,
            "workspace state db was written by a newer release"
        );
        return Err(WikiError::Storage(format!(
            "state db schema v{} is newer than this binary supports (v{}); \
             upgrade the binary or restore a matching backup",
            versions.db_schema_version, versions.supported_schema_version
        )));
    }
    if versions.db_is_from_the_past() {
        tracing::info!(
            from = versions.db_schema_version,
            to = versions.supported_schema_version,
            "workspace state db migrated forward"
        );
    }
    Ok(())
}

/// Content freshness.
///
/// A generation compiled by another binary version is still grounded and
/// correct — it just predates this pipeline — so the default is a loud warning;
/// `server.strict_version_check` makes it fatal.
///
/// The comparison is a crate VERSION STRING, not a binary digest: two builds of
/// the same version number are indistinguishable here. That is deliberate (the
/// guard must not force a rebuild on every recompile), and it means a workspace
/// rebuilt without a version bump will not register as stale.
fn check_generation(
    versions: &llm_wiki_storage::WorkspaceVersions,
    config: &Config,
) -> Result<()> {
    let binary_version = env!("CARGO_PKG_VERSION");
    if !versions.generation_is_stale(binary_version) {
        return Ok(());
    }
    let compiled_by = versions
        .generation_compiler_version
        .as_deref()
        .unwrap_or("<unknown>");
    if config.server.strict_version_check {
        tracing::error!(
            compiled_by,
            binary_version,
            "published generation predates this binary and strict_version_check is on"
        );
        return Err(WikiError::Storage(format!(
            "workspace generation was compiled by {compiled_by}, this binary is \
             {binary_version}; run `llm-wiki build` first or disable \
             server.strict_version_check"
        )));
    }
    tracing::warn!(
        compiled_by,
        binary_version,
        "published generation predates this binary; serving it as-is — rebuild to pick up the new pipeline"
    );
    Ok(())
}

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
        // Version guard BEFORE any recovery work: a db this binary cannot read
        // must fail here, not halfway through a job mutation.
        verify_workspace_versions(&conn, state.config())?;
        let recovered = llm_wiki_storage::mark_stale_jobs_interrupted(&conn)?;
        if recovered > 0 {
            tracing::warn!(
                count = recovered,
                "recovered stale server jobs as INTERRUPTED"
            );
        }
    }

    // Opt-in auto-resume (PRD §31 "根据未来策略恢复"): jobs that died with
    // the previous server process are re-enqueued and re-run one at a time.
    // Off by default — resuming spends LLM budget on restart, which must
    // never come as a surprise.
    if state.config().server.resume_interrupted_jobs {
        if state.0.provider.is_some() {
            let resume_state = state.clone();
            tokio::spawn(async move {
                let resumed = jobs::recover_interrupted_jobs(&resume_state).await;
                if resumed > 0 {
                    tracing::info!(count = resumed, "resumed interrupted build jobs");
                }
            });
        } else {
            tracing::warn!(
                "server.resume_interrupted_jobs is enabled but no LLM provider is configured; \
                 interrupted jobs stay INTERRUPTED"
            );
        }
    }

    // Embed resume is a SEPARATE opt-in (embedding spend is often larger
    // than a build's) and additionally requires `[embedding] model` —
    // otherwise interrupted embed jobs stay INTERRUPTED, visible via
    // `/v1/jobs`.
    if state.config().server.resume_embed_jobs {
        let resume_state = state.clone();
        tokio::spawn(async move {
            let resumed = jobs::recover_interrupted_embed_jobs(&resume_state).await;
            if resumed > 0 {
                tracing::info!(count = resumed, "resumed interrupted embed jobs");
            }
        });
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
///
/// The agent-facing read surface (`/v1/search`, `/v1/context`, `/v1/query`,
/// `/v1/insights`, `/v1/embed`) is the frozen V1.x protocol (PRD §50):
/// additive-only changes, every response carries `protocol_version` and the
/// caller's `request_id` (also on the `x-request-id` header). See
/// `docs/agent-protocol.md`.
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
        .route("/v1/insights", get(routes::list_insights))
        .route("/v1/embed", post(routes::embed))
        .route("/v1/embed/jobs", post(routes::embed_job))
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

#[cfg(test)]
mod version_guard_tests {
    use super::*;

    #[test]
    fn newer_schema_refuses_to_start() {
        let conn = llm_wiki_storage::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA user_version = 999;").unwrap();
        let error = verify_workspace_versions(&conn, &Config::default())
            .expect_err("a state db from the future must not be served");
        assert!(
            error.to_string().contains("newer than this binary"),
            "{error}"
        );
    }

    #[test]
    fn matching_versions_start_cleanly() {
        let conn = llm_wiki_storage::open_in_memory().unwrap();
        verify_workspace_versions(&conn, &Config::default())
            .expect("a fresh workspace starts cleanly");
    }

    /// One COMPLETED build compiled by `version`, activated as the generation.
    fn workspace_with_generation(version: &str) -> llm_wiki_storage::Connection {
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        let build_id = llm_wiki_storage::start_build(
            &mut conn,
            &llm_wiki_storage::BuildDraft {
                source_snapshot_hash: None,
                model: Some("fake".to_owned()),
                prompt_version: None,
                compiler_version: Some(version.to_owned()),
                parser_version: None,
                schema_version: Some("1".to_owned()),
                config_hash: None,
                build_fingerprint: None,
            },
        )
        .unwrap();
        llm_wiki_storage::finish_build(&mut conn, &build_id, "COMPLETED", None, None).unwrap();
        llm_wiki_storage::activate_build(&mut conn, &build_id).unwrap();
        conn
    }

    #[test]
    fn stale_generation_is_fatal_only_in_strict_mode() {
        let conn = workspace_with_generation("0.0.1");
        let versions = llm_wiki_storage::check_workspace_versions(&conn).unwrap();
        assert!(versions.generation_is_stale(env!("CARGO_PKG_VERSION")));

        // Default: the old generation keeps serving, loudly.
        check_generation(&versions, &Config::default())
            .expect("a stale generation still serves by default");

        // Opt-in: refuse until it is rebuilt.
        let mut strict = Config::default();
        strict.server.strict_version_check = true;
        let error = check_generation(&versions, &strict)
            .expect_err("strict mode must refuse a stale generation");
        assert!(error.to_string().contains("0.0.1"), "{error}");
    }

    #[test]
    fn active_pointer_to_a_missing_build_counts_as_stale() {
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        // The pointer names a build whose row does not exist: unknown, so it
        // must be reported rather than assumed current.
        let ghost = llm_wiki_core::ids::BuildId::generate();
        llm_wiki_storage::set_active_build(&mut conn, Some(&ghost)).unwrap();

        let versions = llm_wiki_storage::check_workspace_versions(&conn).unwrap();
        assert_eq!(
            versions.active_build_id.as_ref().map(|id| id.as_str()),
            Some(ghost.as_str())
        );
        assert!(versions.generation_is_stale(env!("CARGO_PKG_VERSION")));
    }
}
