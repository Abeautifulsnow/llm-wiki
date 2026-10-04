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
    attach_job_build, finish_job, open, set_job_phase, set_job_running, FAILURE_CANCELLED,
    FAILURE_INTERNAL, FAILURE_LLM, FAILURE_PLANNING, FAILURE_PUBLISH, FAILURE_REPLAN_REQUIRED,
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
        let result = llm_wiki_compiler::run_build_with_options(
            &workspace,
            &config,
            provider,
            BuildOptions {
                cancel: Some(cancel),
                on_progress: Some(sink),
            },
        )
        .await;

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
