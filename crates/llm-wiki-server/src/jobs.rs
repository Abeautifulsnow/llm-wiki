//! Build job runner (PRD §31): owns the background task behind
//! `POST /v1/build`, mirrors §31 stage transitions onto the persisted job row
//! and terminalizes the job from the pipeline's outcome.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use llm_wiki_compiler::{
    BuildOptions, BuildProgress, BuildProgressSink, EmbedOptions, EmbedProgress, EmbedProgressSink,
};
use llm_wiki_core::cancel::CancelFlag;
use llm_wiki_core::error::WikiError;
use llm_wiki_core::ids::{BuildId, JobId};
use llm_wiki_llm::{EmbeddingProvider, LlmProvider};
use llm_wiki_storage::{
    attach_job_build, finish_job, list_resumable_jobs, open, requeue_job, set_job_phase,
    set_job_progress, set_job_running, FAILURE_CANCELLED, FAILURE_INTERNAL, FAILURE_LLM,
    FAILURE_PLANNING, FAILURE_PUBLISH, FAILURE_REPLAN_REQUIRED,
};

use crate::state::SharedState;

/// Failure-code mapping is exported for tests.
pub fn failure_of(err: &WikiError) -> (&'static str, &'static str, bool) {
    match err {
        WikiError::Cancelled => ("CANCELLED", FAILURE_CANCELLED, true),
        WikiError::ReplanRequired { .. } => ("REPLAN_REQUIRED", FAILURE_REPLAN_REQUIRED, false),
        WikiError::Llm(_) | WikiError::SchemaValidation(_) | WikiError::EvidenceValidation(_) => {
            ("FAILED", FAILURE_LLM, true)
        }
        WikiError::Planning(_) => ("FAILED", FAILURE_PLANNING, false),
        WikiError::PublishRecovery(_) => ("FAILED", FAILURE_PUBLISH, false),
        _ => ("FAILED", FAILURE_INTERNAL, false),
    }
}

/// The §31 progress sink: mirrors every pipeline stage transition onto the
/// job row and attaches the build id on the first transition.
fn progress_sink(db_path: PathBuf, job_id: JobId) -> BuildProgressSink {
    let attached = Arc::new(AtomicBool::new(false));
    let attach_flag = attached.clone();
    Arc::new(move |progress: &BuildProgress| {
        // A dedicated short-lived connection per transition: the build
        // holds its own; WAL + busy timeout make the two safe.
        let Ok(conn) = open(&db_path) else {
            return;
        };
        if let Err(err) = set_job_phase(&conn, &job_id, progress.phase) {
            tracing::warn!(job = %job_id, error = %err, "phase mirror failed");
        }
        if !attach_flag.swap(true, Ordering::SeqCst) {
            if let Ok(build_id) = BuildId::parse(&progress.build_id) {
                if let Err(err) = attach_job_build(&conn, &job_id, &build_id) {
                    tracing::warn!(job = %job_id, error = %err, "build attach failed");
                }
            }
        }
    })
}

/// Spawns the background task that runs `run_build_with_options` for a queued
/// job: RUNNING → phase mirror (progress sink) → terminal status. Releases
/// the job slot when the task unwinds, whatever the outcome.
///
/// Length note: ~70 lines — flat orchestration over three helpers (failure_of,
/// progress_sink, terminalize) with no nesting beyond the outcome match.
pub fn spawn_build_job(
    state: SharedState,
    job_id: JobId,
    cancel: CancelFlag,
    provider: Arc<dyn LlmProvider>,
) {
    tokio::spawn(async move {
        let db_path = state.db_path();

        // Claim the QUEUED row. A `false` transition means the row is no
        // longer QUEUED (cancelled between accept and task start): the job
        // must NOT run — exit without touching the row or the pipeline.
        let started = open(&db_path)
            .ok()
            .and_then(|conn| set_job_running(&conn, &job_id).ok())
            .unwrap_or(false);
        if !started {
            tracing::info!(job = %job_id, "job no longer queued; not starting the build");
            state.0.jobs.take(&job_id).await;
            return;
        }

        let sink = progress_sink(db_path.clone(), job_id.clone());
        let workspace = state.0.workspace.clone();
        let config = state.0.config.clone();
        // The pipeline runs on a NESTED task and the outer task joins its
        // JoinHandle: a PANIC inside the pipeline surfaces as a JoinError
        // instead of aborting this task, so the job is terminalized FAILED
        // and the build slot is ALWAYS released (review #I01 — otherwise a
        // panic would hold the slot until restart, 409-ing every build and
        // deadlocking the auto-resume wait loop).
        let pipeline = tokio::spawn(async move {
            llm_wiki_compiler::run_build_with_options(
                &workspace,
                &config,
                provider,
                BuildOptions {
                    cancel: Some(cancel),
                    on_progress: Some(sink),
                },
            )
            .await
        });
        let result = match pipeline.await {
            Ok(result) => result,
            Err(join_err) => Err(llm_wiki_core::error::WikiError::Llm(format!(
                "build task panicked: {join_err}"
            ))),
        };

        match &result {
            Ok(report) => {
                tracing::info!(
                    job = %job_id,
                    build = %report.build_id,
                    pages = report.pages,
                    "build job completed"
                );
                if let Ok(conn) = open(&db_path) {
                    let _ = finish_job(&conn, &job_id, "COMPLETED", None, false, None);
                }
            }
            Err(err) => {
                let (status, failure_code, retryable) = failure_of(err);
                tracing::warn!(
                    job = %job_id,
                    status,
                    failure_code,
                    error = %crate::error::redact(&err.to_string()),
                    "build job terminated"
                );
                if let Ok(conn) = open(&db_path) {
                    let _ = finish_job(
                        &conn,
                        &job_id,
                        status,
                        Some(failure_code),
                        retryable,
                        Some(&err.to_string()),
                    );
                }
            }
        }
        state.0.jobs.take(&job_id).await;
    });
}

/// Auto-resume (PRD §31 "根据未来策略恢复", config-gated): re-enqueues build
/// jobs that died with the previous server process and re-runs them ONE at a
/// time under the single-slot lock. Runs as a background task from `serve`;
/// tests call it directly. Returns how many jobs were resumed.
///
/// The same job ROW is reused (not duplicated), so its `idempotency_key`
/// mapping survives: a client replaying the original request still resolves
/// to the same job, never to a second LLM run.
///
/// Length note: ~45 lines — a flat claim→requeue→spawn→wait loop.
pub async fn recover_interrupted_jobs(state: &SharedState) -> usize {
    let Some(provider) = state.0.provider.clone() else {
        return 0;
    };
    let mut resumed = 0usize;
    loop {
        let next: Option<llm_wiki_storage::ServerJobRecord> = match open(&state.db_path()) {
            Ok(conn) => match list_resumable_jobs(&conn, "build") {
                Ok(list) => list.into_iter().next(),
                Err(err) => {
                    tracing::warn!(error = %err, "could not list resumable jobs; recovery stops");
                    break;
                }
            },
            Err(err) => {
                tracing::warn!(error = %err, "could not open state db; recovery stops");
                break;
            }
        };
        let Some(record) = next else {
            break;
        };
        let job_id = record.job_id.clone();
        let cancel = CancelFlag::new();
        if !state
            .0
            .jobs
            .start(crate::state::ActiveJob {
                job_id: job_id.clone(),
                cancel: cancel.clone(),
            })
            .await
        {
            // A build claimed the slot while we were recovering (manual
            // request or a concurrent recovery): leave the rest queued and
            // stop — racing the slot would break the single-build contract.
            tracing::info!(job = %job_id, "build slot busy; remaining interrupted jobs stay INTERRUPTED");
            break;
        }
        let requeued = match open(&state.db_path()) {
            Ok(conn) => requeue_job(&conn, &job_id).unwrap_or(false),
            Err(_) => false,
        };
        if !requeued {
            // The row changed underneath us (terminalized some other way);
            // nothing to run — release and move on.
            state.0.jobs.take(&job_id).await;
            continue;
        }
        tracing::info!(job = %job_id, "resuming build job interrupted by restart");
        spawn_build_job(state.clone(), job_id.clone(), cancel, provider.clone());
        // Wait for the task to release the slot before considering the next
        // job — resumed builds are strictly sequential by contract.
        while state
            .0
            .jobs
            .running()
            .await
            .map(|running| running.as_str().to_owned())
            == Some(job_id.as_str().to_owned())
        {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        resumed += 1;
    }
    resumed
}

// ---------------------------------------------------------------------------
// EMBED jobs (0014): async, persistent, cancellable backfills over the SAME
// job rows and state machine as builds, with their OWN single slot.
// ---------------------------------------------------------------------------

/// Minimum interval between mid-run progress writes. The sink fires once per
/// completed batch; on a large backfill that would be thousands of UPDATEs in
/// the collector loop, so the mid-run mirror is throttled. The exact final
/// counters are written by the runner once, before the terminal transition.
const PROGRESS_WRITE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Live state of the progress mirror: ONE connection for the whole run
/// (opened lazily on the first callback — that runs on the collector's
/// blocking thread) plus the throttle clock.
struct ProgressMirror {
    conn: Option<llm_wiki_storage::Connection>,
    phase_written: bool,
    last_write: Option<std::time::Instant>,
}

/// The EMBED progress sink: mirrors the run's counters onto the job row —
/// phase once, `progress_json` at most once per [`PROGRESS_WRITE_INTERVAL`].
///
/// The connection is opened ONCE and reused: the callback fires per batch, so
/// a per-call `open()` would mean one connection (PRAGMA + migrate) per batch
/// on large corpora — a DB round-trip in the batch loop.
fn embed_progress_sink(db_path: PathBuf, job_id: JobId) -> EmbedProgressSink {
    let mirror = Arc::new(std::sync::Mutex::new(ProgressMirror {
        conn: None,
        phase_written: false,
        last_write: None,
    }));
    Arc::new(move |progress: &EmbedProgress| {
        let Ok(mut mirror) = mirror.lock() else {
            return;
        };
        if mirror.conn.is_none() {
            match open(&db_path) {
                Ok(conn) => mirror.conn = Some(conn),
                Err(_) => return,
            }
        }
        // Decide what to write under the &mut borrow, then borrow the
        // connection for the writes.
        let write_phase = !mirror.phase_written;
        let write_progress = mirror
            .last_write
            .is_none_or(|last| last.elapsed() >= PROGRESS_WRITE_INTERVAL);
        if !write_phase && !write_progress {
            return;
        }
        if write_phase {
            mirror.phase_written = true;
        }
        if write_progress {
            mirror.last_write = Some(std::time::Instant::now());
        }
        let Some(conn) = mirror.conn.as_ref() else {
            return;
        };
        if write_phase {
            if let Err(err) = set_job_phase(conn, &job_id, "EMBEDDING") {
                tracing::warn!(job = %job_id, error = %err, "phase mirror failed");
            }
        }
        if write_progress {
            let json = serde_json::json!({
                "total": progress.total,
                "covered_before": progress.covered_before,
                "embedded": progress.embedded,
            });
            if let Err(err) = set_job_progress(conn, &job_id, &json.to_string()) {
                tracing::warn!(job = %job_id, error = %err, "progress mirror failed");
            }
        }
    })
}

/// Spawns the background task behind `POST /v1/embed/jobs`: RUNNING →
/// progress mirror → terminal status. In-flight batches draw from the shared
/// LLM semaphore; the embed slot is released whatever the outcome.
///
/// Length note: ~95 lines — flat orchestration (claim RUNNING → sink →
/// blocking run → terminalize → release slot) mirroring `spawn_build_job`;
/// no nesting beyond the outcome match.
pub fn spawn_embed_job(
    state: SharedState,
    job_id: JobId,
    cancel: CancelFlag,
    provider: Arc<dyn EmbeddingProvider>,
    model: String,
    batch_size: usize,
) {
    tokio::spawn(async move {
        let db_path = state.db_path();
        let started = open(&db_path)
            .ok()
            .and_then(|conn| set_job_running(&conn, &job_id).ok())
            .unwrap_or(false);
        if !started {
            tracing::info!(job = %job_id, "embed job no longer queued; not starting");
            state.0.embed_jobs.take(&job_id).await;
            return;
        }

        let sink = embed_progress_sink(db_path.clone(), job_id.clone());
        let workspace = state.0.workspace.clone();
        let config = state.0.config.clone();
        let batch = batch_size.max(1);
        let permits = state.0.llm_permits.clone();
        let max_retries = config.embedding.max_retries;
        // Same !Send discipline as the sync endpoint: run_embed holds its
        // connections across awaits, so it gets the blocking pool.
        let run = tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(llm_wiki_compiler::run_embed(
                &workspace,
                &config,
                provider,
                &model,
                EmbedOptions {
                    batch_size: batch,
                    max_retries,
                    permits: Some(permits),
                    cancel: Some(cancel),
                    on_progress: Some(sink),
                    // `max_concurrency` stays at the serial default: with
                    // `permits` set, the shared semaphore is the only
                    // concurrency limit (see `EmbedOptions`).
                    ..EmbedOptions::serial(batch)
                },
            ))
        });
        let result = match run.await {
            Ok(result) => result,
            Err(join_err) => Err(WikiError::Llm(format!("embed task panicked: {join_err}"))),
        };

        match &result {
            Ok(report) => {
                if let Ok(conn) = open(&db_path) {
                    // Final counters while the row is still RUNNING, then the
                    // terminal transition. `None` (nothing published between
                    // admission and the run) still completes: the backfill is
                    // a no-op by definition.
                    if let Some(report) = report {
                        tracing::info!(
                            job = %job_id,
                            embedded = report.embedded,
                            "embed job completed"
                        );
                        let _ = set_job_progress(
                            &conn,
                            &job_id,
                            &serde_json::json!({
                                "total": report.total_sections,
                                "covered_before": report.covered_before,
                                "embedded": report.embedded,
                            })
                            .to_string(),
                        );
                    }
                    let _ = finish_job(&conn, &job_id, "COMPLETED", None, false, None);
                }
            }
            Err(err) => {
                let (status, failure_code, retryable) = failure_of(err);
                tracing::warn!(
                    job = %job_id,
                    status,
                    failure_code,
                    error = %crate::error::redact(&err.to_string()),
                    "embed job terminated"
                );
                if let Ok(conn) = open(&db_path) {
                    let _ = finish_job(
                        &conn,
                        &job_id,
                        status,
                        Some(failure_code),
                        retryable,
                        Some(&err.to_string()),
                    );
                }
            }
        }
        state.0.embed_jobs.take(&job_id).await;
    });
}

/// Config-gated EMBED auto-resume (`server.resume_embed_jobs`): re-enqueues
/// embed jobs that died with the previous process, ONE at a time under the
/// embed slot. Requires `[embedding] model` — a resumed backfill must target
/// the same model — otherwise the jobs stay INTERRUPTED (visible via
/// `/v1/jobs`).
///
/// Length note: ~95 lines — flat claim→requeue→spawn→wait loop mirroring
/// `recover_interrupted_jobs`; the provider is built once, outside the loop.
pub async fn recover_interrupted_embed_jobs(state: &SharedState) -> usize {
    let config = state.config();
    let model = config.embedding.model.trim().to_owned();
    if model.is_empty() {
        tracing::warn!(
            "server.resume_embed_jobs is on but [embedding] model is unset; \
             interrupted embed jobs stay INTERRUPTED"
        );
        return 0;
    }
    // An injected provider (tests / embedders) wins over the config-derived
    // one, mirroring the hybrid endpoints.
    let provider: Option<Arc<dyn EmbeddingProvider>> = state.0.embedding.clone().or_else(|| {
        let endpoint = config.embedding.endpoint(&config.llm);
        match config.embedding.resolved_provider(&config.llm) {
            "openai-compatible" => llm_wiki_llm::OpenAiCompatibleEmbeddings::new(
                &endpoint.base_url,
                &endpoint.api_key_env,
                endpoint.timeout_seconds,
                2,
            )
            .ok()
            .map(|provider| Arc::new(provider) as Arc<dyn EmbeddingProvider>),
            _ => None,
        }
    });
    let Some(provider) = provider else {
        tracing::warn!(
            "server.resume_embed_jobs is on but the embedding provider is unavailable; \
             interrupted embed jobs stay INTERRUPTED"
        );
        return 0;
    };
    // A resumed backfill uses the CONFIGURED batch: the original request's
    // `batch` is not persisted on the row. It only groups requests, so the
    // difference is cosmetic.
    let batch = config.embedding.batch_size.max(1) as usize;
    let mut resumed = 0usize;
    loop {
        let next: Option<llm_wiki_storage::ServerJobRecord> = match open(&state.db_path()) {
            Ok(conn) => match list_resumable_jobs(&conn, "embed") {
                Ok(list) => list.into_iter().next(),
                Err(err) => {
                    tracing::warn!(error = %err, "could not list resumable embed jobs; recovery stops");
                    break;
                }
            },
            Err(err) => {
                tracing::warn!(error = %err, "could not open state db; recovery stops");
                break;
            }
        };
        let Some(record) = next else {
            break;
        };
        let job_id = record.job_id.clone();
        let cancel = CancelFlag::new();
        if !state
            .0
            .embed_jobs
            .start(crate::state::ActiveJob {
                job_id: job_id.clone(),
                cancel: cancel.clone(),
            })
            .await
        {
            tracing::info!("embed slot busy; remaining interrupted embed jobs stay INTERRUPTED");
            break;
        }
        let requeued = match open(&state.db_path()) {
            Ok(conn) => requeue_job(&conn, &job_id).unwrap_or(false),
            Err(_) => false,
        };
        if !requeued {
            state.0.embed_jobs.take(&job_id).await;
            continue;
        }
        tracing::info!(job = %job_id, "resuming embed job interrupted by restart");
        spawn_embed_job(
            state.clone(),
            job_id.clone(),
            cancel,
            provider.clone(),
            model.clone(),
            batch,
        );
        // Wait for the task to release the slot before considering the next
        // job — resumed backfills are strictly sequential, like builds.
        while state
            .0
            .embed_jobs
            .running()
            .await
            .map(|running| running.as_str().to_owned())
            == Some(job_id.as_str().to_owned())
        {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        resumed += 1;
    }
    resumed
}
