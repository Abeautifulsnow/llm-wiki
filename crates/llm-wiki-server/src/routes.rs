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
use llm_wiki_core::ids::{InsightId, JobId};
use llm_wiki_llm::LlmProvider;
use llm_wiki_search::{
    rerank_search_hits, reranker_from_config, FullTextSearch, SqliteFullTextSearch,
};
use llm_wiki_storage::{
    count_jobs_by_status, count_sources, finish_job, get_active_build_id,
    get_job as storage_get_job, get_job_by_idempotency_key, insert_job, insight_exists,
    latest_build, list_insights_paged, list_jobs as storage_list_jobs, load_generation_pages, open,
    FAILURE_CANCELLED, JOB_STATUSES,
};

use crate::error::ApiError;
use crate::jobs::spawn_build_job;
use crate::middleware::RequestId;
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
}

/// `POST /v1/search` (PRD §5.5): page/section hits — never answers. Returns
/// the generation actually used, a truncation flag and per-hit citation
/// counts.
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
    let reranker = reranker_from_config(&state.config().search.rerank)?;

    let path = state.db_path();
    let query = request.query.clone();
    let full_text = state.config().search.full_text;
    // FTS + graph stay behind the blocking pool (the search crate's own
    // blocking hop needs a live runtime, which the async context provides).
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
    hits = rerank_search_hits(&query, hits, reranker.as_deref())?;
    let truncated = hits.len() > limit;
    if truncated {
        hits.truncate(limit);
    }

    let citation_counts = db(&state, |conn| {
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
    })
    .await?;
    let generation = db(&state, get_active_build_id).await?;

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
            })
        })
        .collect();
    Ok(Json(json!({
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id.0.as_str(),
        "generation": generation.as_ref().map(llm_wiki_core::ids::BuildId::as_str),
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
/// transparency.
///
/// Length note: ~75 lines — parse → budget/vector helpers → one blocking
/// assembly call → serialization; no logic deeper than the outcome `??`s.
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
    let reranker = reranker_from_config(&state.config().search.rerank)?;

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
        llm_wiki_search::build_context_with_reranker(
            &conn,
            &query,
            &budget,
            &vector,
            reranker.as_deref(),
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
/// as the CLI: `[llm]` endpoint family, `--embedding-model` override, then
/// `LLM_WIKI_EMBEDDING_MODEL`). Returns the owning Arc so the caller can hold
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
        .ok_or_else(|| {
            ApiError::bad_request(
                "hybrid retrieval needs an embedding model: pass embedding_model or set LLM_WIKI_EMBEDDING_MODEL",
            )
        })?;
    // An injected provider (tests / embedders) wins over the config-derived
    // one; the config check below still guards the model name.
    if let Some(injected) = state.0.embedding.clone() {
        return Ok((injected, model));
    }
    if config.llm.model.trim().is_empty() {
        return Err(ApiError::bad_request(
            "llm.model must be configured before hybrid retrieval",
        ));
    }
    let provider = match config.llm.provider.as_str() {
        "openai-compatible" => Arc::new(
            llm_wiki_llm::OpenAiCompatibleProvider::new(
                &config.llm.base_url,
                &config.llm.model,
                &config.llm.api_key_env,
                config.llm.timeout_seconds,
                2,
            )
            .map_err(|e| ApiError::bad_request(format!("embedding provider init failed: {e}")))?,
        ),
        other => {
            return Err(ApiError::bad_request(format!(
                "unsupported llm.provider '{other}'"
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
        let records = load_generation_pages(conn, &active)?;
        Ok(records
            .into_iter()
            .find(|page| page.slug == id || page.page_id.as_str() == id))
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
