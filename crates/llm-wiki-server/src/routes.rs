//! HTTP handlers (PRD §30). Every handler is a thin transport: resolve →
//! application-service call → serialize. SQLite access happens on the
//! blocking pool with a fresh connection per request (WAL keeps readers
//! concurrent); LLM-bound endpoints take a permit from the shared semaphore.

use std::collections::BTreeMap;

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use axum::Extension;

use llm_wiki_core::cancel::CancelFlag;
use llm_wiki_core::config::SearchSourceMode;
use llm_wiki_core::ids::{InsightId, JobId};
use llm_wiki_llm::LlmProvider;
use llm_wiki_search::fusion::{self, Evidence};
use llm_wiki_search::{rerank_search_hits, FullTextSearch, SqliteFullTextSearch};
use llm_wiki_storage::{
    count_jobs_by_status, count_sources, finish_job, get_active_build_id,
    get_job as storage_get_job, get_job_by_idempotency_key, insert_job, insight_exists,
    latest_build, list_insights_paged, list_jobs as storage_list_jobs, load_generation_page_view,
    load_generation_pages, open, FAILURE_CANCELLED, JOB_STATUSES,
};

use crate::error::ApiError;
use crate::jobs::spawn_build_job;
use crate::middleware::RequestId;
use crate::rerank::api_reranker_from_config;
use crate::state::{ActiveJob, SharedState};
use crate::PROTOCOL_VERSION;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn parse_body<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, ApiError> {
    if bytes.iter().all(|b| b.is_ascii_whitespace()) {
        return Err(ApiError::bad_request("a JSON request body is required"));
    }
    serde_json::from_slice(bytes)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))
}

/// Blocks on a SQLite closure; storage errors map to 500.
async fn db<F, T>(state: &SharedState, f: F) -> Result<T, ApiError>
where
    F: FnOnce(&llm_wiki_storage::Connection) -> llm_wiki_core::error::Result<T> + Send + 'static,
    T: Send + 'static,
{
    let path = state.db_path();
    let output = tokio::task::spawn_blocking(move || -> Result<T, ApiError> {
        let conn = open(&path).map_err(ApiError::from)?;
        f(&conn).map_err(ApiError::from)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))?;
    output
}

fn job_json(record: &llm_wiki_storage::ServerJobRecord) -> serde_json::Value {
    json!({
        "job_id": record.job_id.as_str(),
        "kind": record.kind,
        "status": record.status,
        "phase": record.phase,
        "build_id": record.build_id.as_ref().map(|b| b.as_str()),
        "failure_code": record.failure_code,
        "retryable": record.retryable,
        "error": record.error,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    })
}

// ---------------------------------------------------------------------------
// Health & status
// ---------------------------------------------------------------------------

pub async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

#[derive(Deserialize)]
pub struct PageQuery {
    pub limit: Option<usize>,
    pub cursor: Option<String>,
}

/// `GET /v1/status`: sources, latest build, active generation, job depth.
pub async fn status(
    State(state): State<SharedState>,
    Extension(request_id): Extension<RequestId>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let path = state.db_path();
    let workspace_name = state.config().project.name.clone();
    let request_id = request_id.0.as_str().to_owned();
    let payload = tokio::task::spawn_blocking(move || {
        let conn = open(&path).map_err(ApiError::from)?;
        let sources = count_sources(&conn).map_err(ApiError::from)?;
        let latest = latest_build(&conn).map_err(ApiError::from)?;
        let active = get_active_build_id(&conn).map_err(ApiError::from)?;
        let jobs = count_jobs_by_status(&conn, JOB_STATUSES).map_err(ApiError::from)?;
        Result::<_, ApiError>::Ok(json!({
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "workspace": workspace_name,
            "sources": sources,
            "latest_build": latest.map(|build| json!({
                "build_id": build.build_id.as_str(),
                "status": build.status,
                "started_at": build.started_at,
            })),
            "active_build_id": active.as_ref().map(|b| b.as_str()),
            "jobs": jobs,
        }))
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))??;
    Ok(Json(payload))
}

// ---------------------------------------------------------------------------
// Build + jobs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
pub struct BuildRequest {
    /// Must map to a configured source root; the server accepts only the
    /// workspace's configured source (PRD §30: no arbitrary local paths).
    pub source_id: Option<String>,
}

/// Idempotency replay (PRD §30): the same `Idempotency-Key` returns the SAME
/// job instead of starting a second LLM run. Keys currently have NO TTL —
/// retention equals job-row retention (no cleanup exists yet), which is
/// stricter than the contract's "within the validity period".
async fn replay_idempotent(
    state: &SharedState,
    key: Option<String>,
) -> Result<Option<Response>, ApiError> {
    let Some(key) = key else {
        return Ok(None);
    };
    let existing = db(state, move |conn| get_job_by_idempotency_key(conn, &key)).await?;
    Ok(existing.map(|record| {
        (
            StatusCode::OK,
            Json(json!({"job": job_json(&record), "replayed": true})),
        )
            .into_response()
    }))
}

/// Admission control before a job id is consumed: provider configured, queue
/// cap, single-slot build lock.
async fn ensure_build_admission(state: &SharedState) -> Result<Arc<dyn LlmProvider>, ApiError> {
    let provider = state.0.provider.clone().ok_or_else(|| {
        ApiError::bad_request(
            "llm.model is not configured in .llm-wiki/config.toml; the server cannot build",
        )
    })?;
    let queued = db(state, |conn| count_jobs_by_status(conn, &["QUEUED"]))
        .await?
        .remove("QUEUED")
        .unwrap_or(0);
    let cap = state.config().server.max_queued_jobs;
    if queued >= cap {
        return Err(ApiError::queue_full(cap));
    }
    if let Some(running) = state.0.jobs.running().await {
        return Err(ApiError::conflict(
            "build_already_running",
            format!("build job {running} is already running; one build at a time"),
        ));
    }
    Ok(provider)
}

/// `POST /v1/build`: queue a build job. Honors `Idempotency-Key`, the job
/// queue cap and the single-slot build lock.
///
/// Length note: ~55 lines — thin orchestration over parse/admission helpers;
/// the tail is the accept path (insert row → claim slot → spawn).
pub async fn build(
    State(state): State<SharedState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let request: BuildRequest = if body.iter().all(|b| b.is_ascii_whitespace()) {
        BuildRequest { source_id: None }
    } else {
        parse_body(&body)?
    };
    match request.source_id.as_deref() {
        None | Some("default") => {}
        Some(other) => {
            return Err(ApiError::bad_request(format!(
                "unknown source_id {other:?}; the server only builds the configured source root ('default')"
            )));
        }
    }

    let idempotency_key = headers
        .get("idempotency-key")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    if let Some(replayed) = replay_idempotent(&state, idempotency_key.clone()).await? {
        return Ok(replayed);
    }

    let provider = ensure_build_admission(&state).await?;

    let job_id = JobId::generate();
    let record = llm_wiki_storage::ServerJobRecord {
        job_id: job_id.clone(),
        kind: "build".into(),
        status: "QUEUED".into(),
        phase: None,
        build_id: None,
        failure_code: None,
        retryable: false,
        error: None,
        request_id: None,
        idempotency_key,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        finished_at: None,
    };
    db(&state, move |conn| insert_job(conn, &record)).await?;

    let cancel = CancelFlag::new();
    if !state
        .0
        .jobs
        .start(ActiveJob {
            job_id: job_id.clone(),
            cancel: cancel.clone(),
        })
        .await
    {
        // A build started between the check and the claim: drop the queued
        // row back out and report the conflict.
        let cancel_job_id = job_id.clone();
        db(&state, move |conn| {
            finish_job(
                conn,
                &cancel_job_id,
                "CANCELLED",
                Some(FAILURE_CANCELLED),
                true,
                Some("superseded by a concurrently started build"),
            )
        })
        .await?;
        return Err(ApiError::conflict(
            "build_already_running",
            "another build started concurrently; retry with the same idempotency key",
        ));
    }

    spawn_build_job(state.clone(), job_id.clone(), cancel, provider);
    tracing::info!(job = %job_id, "build job queued");
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({"job_id": job_id.as_str(), "status": "queued"})),
    )
        .into_response())
}

/// `GET /v1/jobs/{job_id}`.
pub async fn get_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let job_id = JobId::parse(job_id).map_err(ApiError::from)?;
    let lookup_id = job_id.clone();
    let record = db(&state, move |conn| storage_get_job(conn, &lookup_id))
        .await?
        .ok_or_else(|| ApiError::not_found(format!("no job {job_id}")))?;
    Ok(Json(job_json(&record)))
}

/// `GET /v1/jobs?status=&limit=&cursor=` — cursor-paginated, newest first.
pub async fn list_jobs(
    State(state): State<SharedState>,
    Query(query): Query<PageQuery>,
    Query(filter): Query<JobFilter>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let limit = query.limit.unwrap_or(20).clamp(1, 100);
    let cursor = match &query.cursor {
        Some(raw) => Some(JobId::parse(raw).map_err(ApiError::from)?),
        None => None,
    };
    let records = db(&state, move |conn| {
        storage_list_jobs(conn, filter.status.as_deref(), limit + 1, cursor.as_ref())
    })
    .await?;
    let truncated = records.len() > limit;
    let page: Vec<&llm_wiki_storage::ServerJobRecord> = records.iter().take(limit).collect();
    let next_cursor = page
        .last()
        .filter(|_| truncated)
        .map(|record| record.job_id.as_str().to_owned());
    Ok(Json(json!({
        "jobs": page.iter().map(|r| job_json(r)).collect::<Vec<_>>(),
        "next_cursor": next_cursor,
        "truncated": truncated,
    })))
}

#[derive(Debug, Deserialize)]
pub struct JobFilter {
    pub status: Option<String>,
}

/// `POST /v1/jobs/{job_id}/cancel` — cooperative (PRD §31): RUNNING jobs
/// stop at the next pipeline checkpoint; QUEUED jobs cancel immediately;
/// terminal jobs are a 409.
///
/// Cancellation NEVER releases the build slot (#I01): a cancelled pipeline
/// is still unwinding toward its checkpoint, and only the job task itself
/// frees the slot when its pipeline has ended — otherwise a new build could
/// be admitted against a workspace that still has a live pipeline.
pub async fn cancel_job(
    State(state): State<SharedState>,
    Path(job_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let job_id = JobId::parse(job_id).map_err(ApiError::from)?;
    let lookup_id = job_id.clone();
    let record = db(&state, move |conn| storage_get_job(conn, &lookup_id))
        .await?
        .ok_or_else(|| ApiError::not_found(format!("no job {job_id}")))?;
    match record.status.as_str() {
        "QUEUED" => {
            // Signal the in-memory flag too: the task may be between the
            // accept and its RUNNING transition; the flag makes its first
            // pipeline checkpoint bail out.
            if let Some(cancel) = state.0.jobs.cancel_flag_of(&job_id).await {
                cancel.cancel();
            }
            let cancel_id = job_id.clone();
            let cancelled = db(&state, move |conn| {
                finish_job(
                    conn,
                    &cancel_id,
                    "CANCELLED",
                    Some(FAILURE_CANCELLED),
                    true,
                    Some("cancelled before start"),
                )
            })
            .await?;
            if cancelled {
                Ok(Json(
                    json!({"job_id": job_id.as_str(), "status": "cancelled"}),
                ))
            } else {
                // Lost the race: the task flipped the row to RUNNING between
                // our read and write — fall through to the cooperative path.
                cancel_running(&state, &job_id).await
            }
        }
        "RUNNING" => cancel_running(&state, &job_id).await,
        other => Err(ApiError::conflict(
            "job_not_cancellable",
            format!("job is already terminal ({other})"),
        )),
    }
}

/// Signals a RUNNING job's cancel flag without releasing its slot.
fn cancel_running<'a>(
    state: &'a SharedState,
    job_id: &'a JobId,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Json<serde_json::Value>, ApiError>> + Send + 'a>,
> {
    Box::pin(async move {
        let Some(cancel) = state.0.jobs.cancel_flag_of(job_id).await else {
            return Err(ApiError::conflict(
                "job_not_cancellable",
                "the job is RUNNING but not owned by this server instance",
            ));
        };
        cancel.cancel();
        Ok(Json(
            json!({"job_id": job_id.as_str(), "status": "cancelling"}),
        ))
    })
}

// ---------------------------------------------------------------------------
// Search / Context / Query / Pages
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<usize>,
    /// Optional per-request retrieval corpus override (EPIC A PR4). Absent →
    /// the configured `search.source_mode` (default `wiki`); an invalid value
    /// fails deserialization → 400 invalid_request, never a panic.
    pub source_mode: Option<SearchSourceMode>,
}

/// Resolves the effective retrieval mode: the per-request field wins, else
/// the configured default (`search.source_mode`, default `wiki` — an
/// unconfigured server keeps the exact pre-fusion protocol behavior).
fn effective_source_mode(
    request: Option<SearchSourceMode>,
    configured: SearchSourceMode,
) -> SearchSourceMode {
    request.unwrap_or(configured)
}

/// Maps the config/protocol vocabulary onto the fusion engine's enum (the
/// wire strings are identical; core cannot depend on the search crate, so
/// the two enums mirror each other deliberately).
fn fusion_mode(mode: SearchSourceMode) -> fusion::SourceMode {
    match mode {
        SearchSourceMode::Source => fusion::SourceMode::Source,
        SearchSourceMode::Wiki => fusion::SourceMode::Wiki,
        SearchSourceMode::Fusion => fusion::SourceMode::Fusion,
    }
}

/// Distinct citation counts per page across the ACTIVE generation (the
/// `citation_count` field of wiki hits). Extracted so both search paths and
/// tests share one definition.
fn citation_counts(
    conn: &llm_wiki_storage::Connection,
) -> llm_wiki_core::error::Result<BTreeMap<String, u32>> {
    fn db_err(e: rusqlite::Error) -> llm_wiki_core::error::WikiError {
        llm_wiki_core::error::WikiError::Storage(e.to_string())
    }
    let Some(active) = get_active_build_id(conn)? else {
        return Ok(BTreeMap::new());
    };
    let mut stmt = conn
        .prepare(
            "SELECT page_id, COUNT(*) FROM page_citations \
             WHERE page_id IN (SELECT page_id FROM wiki_pages WHERE build_id = ?1) \
             GROUP BY page_id",
        )
        .map_err(db_err)?;
    let mut rows = stmt.query([active.as_str()]).map_err(db_err)?;
    let mut counts = BTreeMap::new();
    while let Some(row) = rows.next().map_err(db_err)? {
        let page_id: String = row.get(0).map_err(db_err)?;
        let count: u32 = row.get(1).map_err(db_err)?;
        counts.insert(page_id, count);
    }
    Ok(counts)
}

/// `POST /v1/search` (PRD §5.5): page/section hits — never answers. Returns
/// the generation actually used, a truncation flag and per-hit citation
/// counts. EPIC A PR4: the optional `source_mode` field picks the corpus —
/// `wiki` (default) keeps the legacy path field-for-field, `source`/`fusion`
/// route through `fusion::retrieve` with visible per-side degradation.
pub async fn search(
    State(state): State<SharedState>,
    Extension(request_id): Extension<RequestId>,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let request: SearchRequest = parse_body(&body)?;
    if request.query.trim().is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    let limit = request.limit.unwrap_or(10).clamp(1, 50);
    let mode = effective_source_mode(request.source_mode, state.config().search.source_mode);
    let query = request.query;
    if mode == SearchSourceMode::Wiki {
        search_wiki(state, request_id, query, limit).await
    } else {
        search_fused(state, request_id, query, limit, mode).await
    }
}

/// The wiki-only search path — the pre-PR4 handler verbatim (FTS over the
/// ACTIVE generation → optional API rerank → truncate), plus the additive
/// `evidence_kind:"wiki"` hit marker and the top-level `source_mode`/`served`
/// metadata. Nothing here degrades: no published generation stays a 404.
///
/// Length note: ~55 lines — FTS + rerank hop + serialization.
async fn search_wiki(
    state: SharedState,
    request_id: RequestId,
    query: String,
    limit: usize,
) -> Result<Json<serde_json::Value>, ApiError> {
    let reranker = api_reranker_from_config(state.config())?;
    let full_text = state.config().search.full_text;
    // FTS + graph stay behind the blocking pool (the search crate's own
    // blocking hop needs a live runtime, which the async context provides).
    let path = state.db_path();
    let conn = tokio::task::spawn_blocking(move || open(&path).map_err(ApiError::from))
        .await
        .map_err(|e| ApiError::internal(format!("task join: {e}")))?;
    let fts = SqliteFullTextSearch::new(conn?, full_text);
    if !fts.has_published().map_err(ApiError::from)? {
        return Err(ApiError::nothing_published());
    }
    let mut hits = fts
        .search(&query, limit + 1)
        .await
        .map_err(ApiError::from)?;
    // The API-backed reranker blocks on HTTP: keep it off the async workers.
    hits = match reranker {
        None => hits,
        Some(reranker) => {
            let query = query.clone();
            tokio::task::spawn_blocking(move || {
                rerank_search_hits(&query, hits, Some(reranker.as_ref()))
            })
            .await
            .map_err(|e| ApiError::internal(format!("task join: {e}")))?
            .map_err(ApiError::from)?
        }
    };
    let truncated = hits.len() > limit;
    if truncated {
        hits.truncate(limit);
    }

    let citation_counts = db(&state, citation_counts).await?;
    let generation = db(&state, get_active_build_id).await?;

    let served = fusion::ServedSides {
        wiki: if hits.is_empty() {
            fusion::SideStatus::NoMatches
        } else {
            fusion::SideStatus::Served
        },
        source: fusion::SideStatus::Disabled,
    };
    let hits_json: Vec<serde_json::Value> = hits
        .iter()
        .map(|hit| {
            json!({
                "page_id": hit.page_id.as_str(),
                "slug": hit.slug,
                "title": hit.title,
                "heading_path": hit.heading_path,
                "snippet": hit.snippet,
                "rank": hit.rank,
                "citation_count": citation_counts.get(hit.page_id.as_str()).copied().unwrap_or(0),
                // EPIC A PR4 additive marker; the legacy fields are untouched.
                "evidence_kind": "wiki",
            })
        })
        .collect();
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "generation": generation.as_ref().map(llm_wiki_core::ids::BuildId::as_str),
        "source_mode": SearchSourceMode::Wiki,
        "served": served,
        "hits": hits_json,
        "truncated": truncated,
    })))
}

/// The `source`/`fusion` search path (EPIC A PR4): retrieval order is owned
/// by `fusion::retrieve` (top-level RRF + exact-match protection) and is
/// never re-sorted here; the reranker does not apply (it is a wiki-side
/// feature — source-side reranking is EPIC G). Degradation is visible: both
/// sides' statuses travel in `served`, including a never-built workspace
/// (`200` + `not_published` instead of the wiki path's 404).
///
/// Length note: ~50 lines — one blocking fusion call + serialization.
async fn search_fused(
    state: SharedState,
    request_id: RequestId,
    query: String,
    limit: usize,
    mode: SearchSourceMode,
) -> Result<Json<serde_json::Value>, ApiError> {
    let path = state.db_path();
    let outcome = tokio::task::spawn_blocking(move || {
        let conn = open(&path).map_err(ApiError::from)?;
        let generation = get_active_build_id(&conn).map_err(ApiError::from)?;
        // Over-fetch per side (the wiki path's limit+1 shape) so `truncated`
        // stays honest after the top-level merge.
        let fused = fusion::retrieve(&conn, &query, fusion_mode(mode), limit + 1)
            .map_err(ApiError::from)?;
        let counts = citation_counts(&conn).map_err(ApiError::from)?;
        Ok::<_, ApiError>((generation, fused, counts))
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))??;
    let (generation, fused, citation_counts) = outcome;

    let truncated = fused.entries.len() > limit;
    let hits_json: Vec<serde_json::Value> = fused
        .entries
        .iter()
        .take(limit)
        .map(|entry| match &entry.evidence {
            Evidence::Wiki(wiki) => json!({
                "page_id": wiki.page_id.as_str(),
                "slug": wiki.slug,
                "title": wiki.title,
                "heading_path": wiki.heading_path,
                "snippet": wiki.snippet,
                "rank": wiki.rank,
                "citation_count": citation_counts.get(wiki.page_id.as_str()).copied().unwrap_or(0),
                "evidence_kind": "wiki",
            }),
            Evidence::Source(source) => json!({
                // Containment rule (PR3/PR4): a source hit NEVER carries
                // page_id/slug — its identity is the source_ref locator.
                "evidence_kind": "source",
                "source_ref": source.source_ref,
                "file_path": source.source_ref.file_path,
                "title": source.title,
                "heading_path": source.source_ref.heading_path,
                "snippet": source.snippet,
                "rank": source.rank,
            }),
        })
        .collect();
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "generation": generation.as_ref().map(llm_wiki_core::ids::BuildId::as_str),
        "source_mode": mode,
        "served": fused.served,
        "hits": hits_json,
        "truncated": truncated,
    })))
}

#[derive(Debug, Deserialize)]
pub struct ContextRequest {
    pub query: String,
    #[serde(default)]
    pub hybrid: bool,
    pub budget: Option<ContextBudgetRequest>,
    /// Optional per-request retrieval corpus override (EPIC A PR4). Absent →
    /// the configured `search.source_mode` (default `wiki` — the legacy
    /// builder, untouched); `source`/`fusion` route through the PR3 fused
    /// retrieval with `per_side_limit = budget.max_chunks`.
    pub source_mode: Option<SearchSourceMode>,
}

#[derive(Debug, Deserialize)]
pub struct ContextBudgetRequest {
    pub max_chunks: Option<usize>,
    pub max_tokens: Option<u64>,
    pub max_pages: Option<usize>,
    pub max_per_source: Option<usize>,
    pub graph_limit: Option<usize>,
}

/// Resolves the request budget over the safe defaults (absent fields keep
/// the default; the request can only tighten or relax the documented knobs).
fn requested_budget(over: Option<&ContextBudgetRequest>) -> llm_wiki_search::ContextBudget {
    let mut budget = llm_wiki_search::ContextBudget::default();
    let Some(over) = over else {
        return budget;
    };
    if let Some(v) = over.max_chunks {
        budget.max_chunks = v;
    }
    if let Some(v) = over.max_tokens {
        budget.max_tokens = v;
    }
    if let Some(v) = over.max_pages {
        budget.max_pages = v;
    }
    if let Some(v) = over.max_per_source {
        budget.max_per_source = v;
    }
    if let Some(v) = over.graph_limit {
        budget.graph_limit = v;
    }
    budget
}

/// The §19.3 vector candidates for one query, split by Send-ness: the
/// coverage check and cosine scoring run on the blocking pool; the embed
/// call is the only async piece, and an uncovered workspace never spends it.
async fn vector_candidates_for(
    state: &SharedState,
    query: &str,
) -> Result<Vec<llm_wiki_search::VectorCandidate>, ApiError> {
    let (embedding_provider, model) = hybrid_context(state, None)?;
    let coverage_path = state.db_path();
    let model_for_coverage = model.clone();
    let covered = tokio::task::spawn_blocking(move || {
        let conn = open(&coverage_path).map_err(ApiError::from)?;
        llm_wiki_compiler::embedding_coverage(&conn, &model_for_coverage).map_err(ApiError::from)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))?;
    if !covered? {
        return Ok(Vec::new());
    }
    let hybrid = llm_wiki_compiler::HybridContext {
        provider: &embedding_provider,
        model: model.clone(),
    };
    let query_vector = llm_wiki_compiler::embed_query_vector(&hybrid, query)
        .await
        .map_err(ApiError::from)?;
    let path = state.db_path();
    tokio::task::spawn_blocking(move || {
        let conn = open(&path).map_err(ApiError::from)?;
        llm_wiki_compiler::top_cosine_candidates(&conn, &model, query_vector.as_deref())
            .map_err(ApiError::from)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))?
}

/// `POST /v1/context` (PRD §24/§30, the Agent-integration surface): the
/// budgeted, diversity-aware retrieval bundle with citations and truncation
/// transparency. EPIC A PR4: the optional `source_mode` field selects the
/// corpus; the response fields themselves shipped additively in PR3
/// (`evidence_kind`/`source_ref` per chunk, `served_mode`/`degraded` at the
/// top level).
///
/// Length note: ~85 lines — parse → budget/vector helpers → one blocking
/// assembly call → serialization (incl. the PR3 additive pass-through
/// fields); no logic deeper than the outcome `??`s.
pub async fn context(
    State(state): State<SharedState>,
    Extension(request_id): Extension<RequestId>,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let request: ContextRequest = parse_body(&body)?;
    if request.query.trim().is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    let budget = requested_budget(request.budget.as_ref());
    let mode = effective_source_mode(request.source_mode, state.config().search.source_mode);
    let reranker = api_reranker_from_config(state.config())?;

    let vector = if request.hybrid {
        vector_candidates_for(&state, &request.query).await?
    } else {
        Vec::new()
    };

    let path = state.db_path();
    let query = request.query.clone();
    let assembled = tokio::task::spawn_blocking(move || {
        let conn = open(&path).map_err(ApiError::from)?;
        let generation = get_active_build_id(&conn).map_err(ApiError::from)?;
        // Wiki mode keeps the legacy builder untouched (fused = None);
        // source/fusion run the PR3 fused retrieval first, per-side limit =
        // the chunk budget (PRD §1.2).
        let fused = if mode == SearchSourceMode::Wiki {
            None
        } else {
            Some(
                fusion::retrieve(&conn, &query, fusion_mode(mode), budget.max_chunks)
                    .map_err(ApiError::from)?,
            )
        };
        llm_wiki_search::build_context_with_sources(
            &conn,
            &query,
            &budget,
            &vector,
            reranker.as_deref(),
            fused.as_ref(),
        )
        .map(|assembled| (generation, assembled))
        .map_err(ApiError::from)
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))??;
    let (generation, assembled) = assembled;

    let chunks: Vec<serde_json::Value> = assembled
        .chunks
        .iter()
        .map(|chunk| {
            json!({
                "slug": chunk.slug,
                "title": chunk.title,
                "heading_path": chunk.heading_path,
                "snippet": chunk.snippet,
                "score": chunk.score,
                "sources": chunk.sources,
                // EPIC A PR3 additive evidence fields; source chunks carry
                // their locator in source_ref and NEVER page_id/slug (PR4
                // wired the source_mode request field that selects them).
                "evidence_kind": chunk.evidence_kind,
                "source_ref": chunk.source_ref,
            })
        })
        .collect();
    let neighbors: Vec<serde_json::Value> = assembled
        .neighbors
        .iter()
        .map(|neighbor| {
            json!({
                "from_slug": neighbor.from_slug,
                "node_id": neighbor.node_id,
                "node_type": neighbor.node_type,
                "label": neighbor.label,
                "relation": neighbor.relation,
            })
        })
        .collect();
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "generation": generation.as_ref().map(llm_wiki_core::ids::BuildId::as_str),
        "chunks": chunks,
        "neighbors": neighbors,
        "estimated_tokens": assembled.estimated_tokens,
        "dropped": assembled.dropped,
        "truncated": assembled.dropped > 0,
        // EPIC A PR3 pass-through: the retrieval mode and the per-side
        // serving metadata (null for the legacy wiki-only path).
        "served_mode": assembled.served_mode,
        "degraded": assembled.degraded,
    })))
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    /// Persist the verified insight (audit FIX-020 write-back).
    #[serde(default)]
    pub write_back: bool,
    /// Vector-layer candidates in the retrieval fusion (§19.3).
    #[serde(default)]
    pub hybrid: bool,
    pub embedding_model: Option<String>,
}

/// `POST /v1/query` (PRD §24): a grounded, citation-verified answer over the
/// published generation — the same service as `llm-wiki ask`.
pub async fn query(
    State(state): State<SharedState>,
    Extension(request_id): Extension<RequestId>,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let request: QueryRequest = parse_body(&body)?;
    if request.query.trim().is_empty() {
        return Err(ApiError::bad_request("query must not be empty"));
    }
    let provider = state
        .0
        .provider
        .clone()
        .ok_or_else(|| {
            ApiError::bad_request("llm.model is not configured; the server cannot answer queries")
        })?
        .clone();
    // The embedding provider must outlive the ask call: hold the owning Arc
    // and let the task construct HybridContext locally.
    let embedding = if request.hybrid {
        Some(hybrid_context(&state, request.embedding_model.as_deref())?)
    } else {
        None
    };

    // Bound concurrent synthesis by the shared LLM semaphore.
    let _permit = state
        .0
        .llm_permits
        .acquire()
        .await
        .map_err(|e| ApiError::internal(format!("llm semaphore: {e}")))?;

    let workspace = state.0.workspace.clone();
    let config = state.0.config.clone();
    let query = request.query.clone();
    let write_back = request.write_back;
    // run_ask holds its SQLite connection across awaits (a !Send future by
    // design — the CLI block_on's it): give it the blocking pool with a
    // block_on handle instead of poisoning this handler's Send future.
    let embedding_for_task = embedding.clone();
    let report = tokio::task::spawn_blocking(move || {
        let hybrid = embedding_for_task
            .as_ref()
            .map(
                |(embedding_provider, model)| llm_wiki_compiler::HybridContext {
                    provider: embedding_provider,
                    model: model.clone(),
                },
            );
        tokio::runtime::Handle::current().block_on(llm_wiki_compiler::run_ask(
            &workspace, &config, provider, &query, write_back, hybrid,
        ))
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))?
    .map_err(ApiError::from)?;

    let generation = db(&state, get_active_build_id).await?;
    let Some(report) = report else {
        return Err(ApiError::nothing_published());
    };
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "generation": generation.as_ref().map(llm_wiki_core::ids::BuildId::as_str),
        "answer": report.answer,
        "citations": report.citations.iter().map(|c| json!({
            "claim_node_id": c.claim_node_id.as_str(),
            "source_id": c.source_id.as_str(),
            "section_id": c.section_id.as_ref().map(|s| s.as_str()),
            "source_hash": c.source_hash,
            "evidence_digest": c.evidence_digest,
            "heading_path": c.heading_path,
        })).collect::<Vec<_>>(),
        "sources": report.sources,
        "llm_request_count": report.llm_request_count,
        "insight_id": report.insight_id,
    })))
}

/// Resolves the embedding provider + model for `hybrid` requests (same rules
/// as the CLI: `[embedding]` section — empty fields inherit `[llm]`, so the
/// embedding model may live at its own provider; then the request's
/// `embedding_model` override, `LLM_WIKI_EMBEDDING_MODEL`, and
/// `[embedding] model`). Returns the owning Arc so the caller can hold
/// it across the await.
fn hybrid_context(
    state: &SharedState,
    model_override: Option<&str>,
) -> Result<(std::sync::Arc<dyn llm_wiki_llm::EmbeddingProvider>, String), ApiError> {
    let config = state.config();
    let model = model_override
        .map(str::to_owned)
        .or_else(|| std::env::var("LLM_WIKI_EMBEDDING_MODEL").ok())
        .filter(|m| !m.trim().is_empty())
        .or_else(|| {
            let configured = config.embedding.model.trim();
            (!configured.is_empty()).then(|| configured.to_owned())
        })
        .ok_or_else(|| {
            ApiError::bad_request(
                "hybrid retrieval needs an embedding model: pass embedding_model, set LLM_WIKI_EMBEDDING_MODEL, or set [embedding] model",
            )
        })?;
    // An injected provider (tests / embedders) wins over the config-derived
    // one; the model resolution above still guards the model name.
    if let Some(injected) = state.0.embedding.clone() {
        return Ok((injected, model));
    }
    let endpoint = config.embedding.endpoint(&config.llm);
    let provider = match config.embedding.resolved_provider(&config.llm) {
        "openai-compatible" => Arc::new(
            llm_wiki_llm::OpenAiCompatibleEmbeddings::new(
                &endpoint.base_url,
                &endpoint.api_key_env,
                endpoint.timeout_seconds,
                2,
            )
            .map_err(|e| ApiError::bad_request(format!("embedding provider init failed: {e}")))?,
        ),
        other => {
            return Err(ApiError::bad_request(format!(
                "unsupported embedding.provider '{other}'"
            )));
        }
    };
    Ok((
        provider as std::sync::Arc<dyn llm_wiki_llm::EmbeddingProvider>,
        model,
    ))
}

// ---------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------

/// `GET /v1/pages?limit=&cursor=` — the ACTIVE generation's pages (metadata
/// only), slug-ordered, cursor-paginated.
pub async fn list_pages(
    State(state): State<SharedState>,
    Extension(request_id): Extension<RequestId>,
    Query(query): Query<PageQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let limit = query.limit.unwrap_or(20).clamp(1, 100);
    let cursor = query.cursor.clone();
    let outcome = db(&state, move |conn| {
        let Some(active) = get_active_build_id(conn)? else {
            return Ok(None);
        };
        let mut records = load_generation_pages(conn, &active)?;
        records.sort_by(|a, b| a.slug.cmp(&b.slug));
        let after = match cursor.as_deref() {
            Some(cursor) => match records.iter().position(|page| page.slug == cursor) {
                Some(index) => Some(index),
                None => {
                    return Err(llm_wiki_core::error::WikiError::Config(format!(
                        "unknown cursor page {cursor:?}"
                    )));
                }
            },
            None => None,
        };
        let remaining = match after {
            Some(index) => &records[index + 1..],
            None => records.as_slice(),
        };
        let truncated = remaining.len() > limit;
        let page: Vec<serde_json::Value> = remaining
            .iter()
            .take(limit)
            .map(|page| {
                json!({
                    "page_id": page.page_id.as_str(),
                    "slug": page.slug,
                    "title": page.title,
                    "category": page.category,
                    "language": page.language,
                    "body_hash": page.body_hash,
                })
            })
            .collect();
        let next_cursor = remaining
            .iter()
            .take(limit)
            .next_back()
            .filter(|_| truncated)
            .map(|page| page.slug.clone());
        Ok(Some((active, page, next_cursor, truncated)))
    })
    .await?;
    let Some((generation, pages, next_cursor, truncated)) = outcome else {
        return Err(ApiError::nothing_published());
    };
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "generation": generation.as_str(),
        "pages": pages,
        "next_cursor": next_cursor,
        "truncated": truncated,
    })))
}

/// `GET /v1/pages/{id}` — one full page (id or slug) of the ACTIVE
/// generation, with citations and links.
pub async fn get_page(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let found = db(&state, move |conn| {
        let Some(active) = get_active_build_id(conn)? else {
            return Ok(None);
        };
        load_generation_page_view(conn, &active, &id)
    })
    .await?;
    let page = found.ok_or_else(|| {
        ApiError::not_found("no page with that id or slug in the active generation")
    })?;
    Ok(Json(json!({
        "page_id": page.page_id.as_str(),
        "slug": page.slug,
        "title": page.title,
        "category": page.category,
        "language": page.language,
        "body_hash": page.body_hash,
        "content": page.content,
        "knowledge_refs": page.knowledge_refs.iter().map(|n| n.as_str()).collect::<Vec<_>>(),
        "citations": page.citations.iter().map(|c| json!({
            "claim_node_id": c.claim_node_id.as_str(),
            "source_id": c.source_id.as_str(),
            "section_id": c.section_id.as_ref().map(|s| s.as_str()),
            "range": [c.range.start, c.range.end],
            "source_hash": c.source_hash,
            "evidence_digest": c.evidence_digest,
            "heading_path": c.heading_path,
        })).collect::<Vec<_>>(),
        "links": page.links.iter().map(|l| json!({
            "to_page_id": l.to_page_id.as_str(),
            "target_title": l.target_title,
        })).collect::<Vec<_>>(),
    })))
}

// ---------------------------------------------------------------------------
// Insights (write-back loop read surface) + Embed (vector layer maintenance)
// ---------------------------------------------------------------------------

/// Serializes one insight record for the API.
fn insight_json(record: &llm_wiki_storage::InsightRecord) -> serde_json::Value {
    json!({
        "insight_id": record.insight_id.as_str(),
        "build_id": record.build_id.as_str(),
        "query": record.query,
        "answer": record.answer,
        "citations": record.citations.iter().map(|c| json!({
            "claim_node_id": c.claim_node_id,
            "source": c.source,
            "heading_path": c.heading_path,
            "range": [c.range.0, c.range.1],
            "evidence_digest": c.evidence_digest,
        })).collect::<Vec<_>>(),
        "created_at": record.created_at,
    })
}

/// `GET /v1/insights?limit=&cursor=` — the curated insight layer, newest
/// first, cursor-paginated (frozen V1.x protocol).
pub async fn list_insights(
    State(state): State<SharedState>,
    Extension(request_id): Extension<RequestId>,
    Query(query): Query<PageQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let limit = query.limit.unwrap_or(20).clamp(1, 100);
    let cursor = match &query.cursor {
        Some(raw) => Some(InsightId::parse(raw).map_err(ApiError::from)?),
        None => None,
    };
    let outcome = db(&state, move |conn| {
        // Insights span generations (a curated layer), so the active build is
        // only a "something was published" signal here.
        if get_active_build_id(conn)?.is_none() {
            return Ok(None);
        }
        // An unknown cursor is a CLIENT error, not a storage failure (§30:
        // 4xx for malformed requests).
        if let Some(cursor) = &cursor {
            if !insight_exists(conn, cursor)? {
                return Err(llm_wiki_core::error::WikiError::Config(format!(
                    "unknown cursor insight {cursor}"
                )));
            }
        }
        let mut records = list_insights_paged(conn, limit + 1, cursor.as_ref())?;
        let truncated = records.len() > limit;
        if truncated {
            records.truncate(limit);
        }
        let next_cursor = records
            .last()
            .filter(|_| truncated)
            .map(|record| record.insight_id.as_str().to_owned());
        Ok(Some((records, next_cursor, truncated)))
    })
    .await?;
    let Some((records, next_cursor, truncated)) = outcome else {
        return Err(ApiError::nothing_published());
    };
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "insights": records.iter().map(insight_json).collect::<Vec<_>>(),
        "next_cursor": next_cursor,
        "truncated": truncated,
    })))
}

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    /// Embedding model override (falls back to $LLM_WIKI_EMBEDDING_MODEL).
    pub model: Option<String>,
    /// Sections per embedding request (default 16).
    pub batch: Option<usize>,
}

/// `POST /v1/embed` — incremental embedding backfill for the ACTIVE
/// generation (the service counterpart of `llm-wiki embed`). Synchronous by
/// design: embedding is incremental (fully covered generations issue zero
/// requests), so the first backfill is the only long call, and the §31 job
/// state machine is build-shaped — embedding is maintenance, not a build.
/// Bounded by the shared LLM semaphore.
pub async fn embed(
    State(state): State<SharedState>,
    Extension(request_id): Extension<RequestId>,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    let request: EmbedRequest = if body.iter().all(|b| b.is_ascii_whitespace()) {
        EmbedRequest {
            model: None,
            batch: None,
        }
    } else {
        parse_body(&body)?
    };
    let batch = request.batch.unwrap_or(16).clamp(1, 64);
    let (embedding_provider, model) = hybrid_context(&state, request.model.as_deref())?;

    let _permit = state
        .0
        .llm_permits
        .acquire()
        .await
        .map_err(|e| ApiError::internal(format!("llm semaphore: {e}")))?;

    let workspace = state.0.workspace.clone();
    let config = state.0.config.clone();
    // run_embed holds its SQLite connection across awaits (a !Send future by
    // design): give it the blocking pool with a block_on handle.
    let report = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(llm_wiki_compiler::run_embed(
            &workspace,
            &config,
            embedding_provider,
            &model,
            batch,
        ))
    })
    .await
    .map_err(|e| ApiError::internal(format!("task join: {e}")))?
    .map_err(ApiError::from)?;
    let Some(report) = report else {
        return Err(ApiError::nothing_published());
    };
    let generation = db(&state, get_active_build_id).await?;
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "generation": generation.as_ref().map(llm_wiki_core::ids::BuildId::as_str),
        "model": report.model,
        "total_sections": report.total_sections,
        "covered_before": report.covered_before,
        "embedded": report.embedded,
    })))
}
