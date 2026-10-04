//! Build job runner (PRD §31): owns the background task behind
//! `POST /v1/build`, mirrors §31 stage transitions onto the persisted job row
//! and terminalizes the job from the pipeline's outcome.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use llm_wiki_compiler::{BuildOptions, BuildProgress, BuildProgressSink};
use llm_wiki_core::cancel::CancelFlag;
use llm_wiki_core::error::WikiError;
use llm_wiki_core::ids::{BuildId, JobId};
use llm_wiki_llm::LlmProvider;
use llm_wiki_storage::{
    attach_job_build, finish_job, list_resumable_jobs, open, requeue_job, set_job_phase,
    set_job_running, FAILURE_CANCELLED, FAILURE_INTERNAL, FAILURE_LLM, FAILURE_PLANNING,
    FAILURE_PUBLISH, FAILURE_REPLAN_REQUIRED,
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
                    error = %err,
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
            Ok(conn) => match list_resumable_jobs(&conn) {
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
