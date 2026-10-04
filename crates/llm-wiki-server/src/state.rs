//! Shared server state: the workspace/config pair plus the single-slot job
//! manager (one build at a time — the §31 pipeline is serialized by design:
//! concurrent builds over one workspace would race the registry).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::Mutex;

use llm_wiki_core::cancel::CancelFlag;
use llm_wiki_core::config::Config;
use llm_wiki_core::ids::JobId;
use llm_wiki_llm::{EmbeddingProvider, LlmProvider};

/// Cloneable handle to the server state.
#[derive(Clone)]
pub struct SharedState(pub Arc<Inner>);

pub struct Inner {
    pub workspace: PathBuf,
    pub config: Config,
    /// Present when `[llm]` was configured at startup; `None` → build/query
    /// endpoints fail with a 400 config error.
    pub provider: Option<Arc<dyn LlmProvider>>,
    /// Embedding provider override (tests / embedders). When absent, the
    /// `/v1/embed` and hybrid endpoints construct one from `[llm]` config.
    pub embedding: Option<Arc<dyn EmbeddingProvider>>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub jobs: JobManager,
    /// LLM semaphore for request-time endpoints (`/v1/query`): bounds
    /// concurrent synthesis independent of the build pipeline.
    pub llm_permits: Arc<tokio::sync::Semaphore>,
    /// Remote-mode per-caller rate-limit buckets (PRD §30).
    pub rate_buckets: Mutex<std::collections::HashMap<String, RateBucket>>,
}

#[derive(Debug, Clone, Copy)]
pub struct RateBucket {
    pub window_started_at: std::time::Instant,
    pub count: u32,
}

impl SharedState {
    pub fn new(workspace: PathBuf, config: Config, provider: Option<Arc<dyn LlmProvider>>) -> Self {
        let permits = config.llm.max_concurrency.max(1) as usize;
        Self(Arc::new(Inner {
            workspace,
            config,
            provider,
            embedding: None,
            started_at: chrono::Utc::now(),
            jobs: JobManager::new(),
            llm_permits: Arc::new(tokio::sync::Semaphore::new(permits)),
            rate_buckets: Mutex::new(std::collections::HashMap::new()),
        }))
    }

    /// Overrides the config-derived embedding provider (tests, embedders).
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        let inner = Arc::get_mut(&mut self.0).expect("state not yet shared");
        inner.embedding = Some(provider);
        self
    }

    pub fn db_path(&self) -> PathBuf {
        state_db(&self.0.workspace)
    }

    pub fn config(&self) -> &Config {
        &self.0.config
    }
}

pub fn state_db(workspace: &Path) -> PathBuf {
    workspace.join(".llm-wiki").join("state.db")
}

/// The single build slot. `start` fails while another job holds it; `take`
/// releases it when the job task unwinds.
#[derive(Default)]
pub struct JobManager {
    slot: Mutex<Option<ActiveJob>>,
}

#[derive(Debug, Clone)]
pub struct ActiveJob {
    pub job_id: JobId,
    pub cancel: CancelFlag,
}

impl JobManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// The currently running job id, if any.
    pub async fn running(&self) -> Option<JobId> {
        self.slot
            .lock()
            .await
            .as_ref()
            .map(|job| job.job_id.clone())
    }

    /// Claims the slot for `job`; `false` when another job holds it.
    pub async fn start(&self, job: ActiveJob) -> bool {
        let mut slot = self.slot.lock().await;
        if slot.is_some() {
            return false;
        }
        *slot = Some(job);
        true
    }

    /// Returns the cancel flag of the RUNNING job with `job_id` and releases
    /// the slot for it (the caller decides whether to signal).
    pub async fn detach_running(&self, job_id: &JobId) -> Option<CancelFlag> {
        let mut slot = self.slot.lock().await;
        if slot.as_ref().is_some_and(|job| &job.job_id == job_id) {
            slot.take().map(|job| job.cancel)
        } else {
            None
        }
    }

    /// The cancel flag of the job holding the slot, WITHOUT releasing it.
    ///
    /// Cancellation must not free the build slot: a cooperatively-cancelled
    /// pipeline is still unwinding toward its next checkpoint, and a new
    /// build admitted in that window would race it over the registry (the
    /// slot is the single-build contract, §31). Only the job task itself
    /// releases the slot, via [`Self::take`], when its pipeline has ended.
    pub async fn cancel_flag_of(&self, job_id: &JobId) -> Option<CancelFlag> {
        self.slot
            .lock()
            .await
            .as_ref()
            .filter(|job| &job.job_id == job_id)
            .map(|job| job.cancel.clone())
    }

    /// Releases the slot if it still belongs to `job_id` (end-of-job path).
    pub async fn take(&self, job_id: &JobId) -> bool {
        self.detach_running(job_id).await.is_some()
    }
}
