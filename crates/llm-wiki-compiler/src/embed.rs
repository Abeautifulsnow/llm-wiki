//! `llm-wiki embed` (§19.3 Vector layer): the explicit backfill that gives
//! the active generation's context sections their vectors. Embeddings cost
//! model calls, so this is user-driven — the build pipeline never embeds.
//!
//! Content addressing makes the backfill incremental: a section's hash
//! (title + heading path + body, ONE definition in the search crate) keys
//! the stored vector, so unchanged sections re-embed to nothing and a
//! re-run over a fully covered generation issues zero requests. The model
//! is part of the storage key: switching models never mixes vector spaces.
//!
//! Batches run with BOUNDED concurrency — the same JoinSet windowing idiom
//! as the build pipeline (PR §27) — while the run itself stays a `!Send`
//! future by design: the SQLite connections (reader + the ONE writer) never
//! leave the collector. Batch tasks carry no connection, only the provider
//! Arc, the texts and a semaphore permit, which is exactly what makes them
//! spawnable. Concurrency is accounted against the caller's budget: the
//! HTTP server passes its shared `llm_permits` semaphore so in-flight embed
//! batches, builds and queries all draw from the same LLM allowance.

use std::path::Path;
use std::sync::Arc;

use llm_wiki_core::cancel::CancelFlag;
use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_llm::{EmbeddingProvider, LlmError};
use llm_wiki_storage::insert_section_embeddings;

use crate::publish::read_current_pointer;

/// What one embed run did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedReport {
    pub model: String,
    /// Context sections of the ACTIVE generation.
    pub total_sections: usize,
    /// Sections that already had a stored vector for this model.
    pub covered_before: usize,
    /// Sections embedded by this run.
    pub embedded: usize,
    /// Retry attempts spent on transient provider failures (429/5xx/timeout).
    pub retries: usize,
}

/// Live counters for one embed run, mirrored to the sink after every batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedProgress {
    /// Sections of the ACTIVE generation.
    pub total: usize,
    /// Sections already covered when the run started.
    pub covered_before: usize,
    /// Sections embedded (and persisted) so far.
    pub embedded: usize,
}

pub type EmbedProgressSink = Arc<dyn Fn(&EmbedProgress) + Send + Sync>;

/// How one embed run executes (mirrors `BuildOptions`).
#[derive(Clone, Default)]
pub struct EmbedOptions {
    /// Sections per embedding request.
    pub batch_size: usize,
    /// Cap on in-flight batches when `permits` is absent (CLI/local path);
    /// 1 reproduces the historical serial behavior. Ignored when `permits`
    /// is set — the semaphore is the limit there.
    pub max_concurrency: usize,
    /// Retries per batch for transient provider failures (429/5xx/timeout),
    /// exponential backoff; 0 preserves the historical fail-fast.
    pub max_retries: u32,
    /// The LLM budget: one permit per in-flight batch, held for the duration
    /// of the provider call. Servers pass their shared semaphore; `None` →
    /// a local semaphore sized `max_concurrency`.
    pub permits: Option<Arc<tokio::sync::Semaphore>>,
    /// Cooperative cancellation: checked before scheduling and before every
    /// retry; already-stored vectors stay valid, a re-run only fills gaps.
    pub cancel: Option<CancelFlag>,
    pub on_progress: Option<EmbedProgressSink>,
}

impl EmbedOptions {
    /// The historical behavior: one batch in flight, no retries, no cancel.
    pub fn serial(batch_size: usize) -> Self {
        Self {
            batch_size,
            max_concurrency: 1,
            max_retries: 0,
            permits: None,
            cancel: None,
            on_progress: None,
        }
    }
}

/// Backfills embeddings for the ACTIVE generation's context sections under
/// `model`. Returns `Ok(None)` when nothing is published.
///
/// Length note: ~120 lines — flat orchestration (validate → load coverage →
/// seed the batch window → collect completions → report), the same shape as
/// the build pipeline's entry points; nesting never exceeds the collector's
/// outcome match.
pub async fn run_embed(
    workspace_root: &Path,
    config: &Config,
    provider: Arc<dyn EmbeddingProvider>,
    model: &str,
    options: EmbedOptions,
) -> Result<Option<EmbedReport>> {
    let state_db = workspace_root.join(".llm-wiki").join("state.db");
    if !state_db.exists() {
        return Ok(None);
    }
    let wiki_dir = lexical_absolute(workspace_root, &config.project.wiki_dir);
    if read_current_pointer(&crate::publish::PublishPaths::new(&wiki_dir))?.is_none() {
        return Ok(None);
    }
    let conn = llm_wiki_storage::open(&state_db)?;
    let sections = llm_wiki_search::active_context_sections(&conn)?;

    let hashes: Vec<String> = sections.iter().map(|s| s.text_hash.clone()).collect();
    let stored = llm_wiki_storage::section_embeddings_by_hash(&conn, model, &hashes)?;
    let covered_before = stored.len();

    // Hash → section text, for the missing set.
    let missing: Vec<&llm_wiki_search::ContextSection> = sections
        .iter()
        .filter(|section| !stored.contains_key(&section.text_hash))
        .collect();

    let batch_size = options.batch_size.max(1);
    // Already cancelled at entry: no batch is scheduled, no request spent —
    // a silent empty report would look like success.
    if options
        .cancel
        .as_ref()
        .is_some_and(CancelFlag::is_cancelled)
    {
        return Err(WikiError::Cancelled);
    }
    // Fully covered (the common incremental case): nothing to embed, and no
    // need to open the write connection at all.
    if missing.is_empty() {
        return Ok(Some(EmbedReport {
            model: model.to_owned(),
            total_sections: sections.len(),
            covered_before,
            embedded: 0,
            retries: 0,
        }));
    }
    let permits = options
        .permits
        .clone()
        .unwrap_or_else(|| Arc::new(tokio::sync::Semaphore::new(options.max_concurrency.max(1))));
    // The single writer: batch tasks never touch SQLite; completed batches
    // are persisted here, in completion order, over ONE connection — no
    // per-batch reopen, no concurrent-writer lock contention.
    let mut writable = llm_wiki_storage::open(&state_db)?;

    let mut embedded = 0usize;
    let mut retries = 0usize;
    let mut join_set = tokio::task::JoinSet::new();
    for batch in missing.chunks(batch_size) {
        // Stop SCHEDULING new batches once cancelled; in-flight ones notice
        // the flag before their provider call and return without spending.
        if options
            .cancel
            .as_ref()
            .is_some_and(CancelFlag::is_cancelled)
        {
            break;
        }
        let texts: Vec<String> = batch
            .iter()
            .map(|section| {
                llm_wiki_search::context_section_text(
                    &section.title,
                    &section.heading_path,
                    &section.body,
                )
            })
            .collect();
        let batch_hashes: Vec<String> = batch
            .iter()
            .map(|section| section.text_hash.clone())
            .collect();
        join_set.spawn(embed_batch(
            provider.clone(),
            model.to_owned(),
            texts,
            batch_hashes,
            permits.clone(),
            options.cancel.clone(),
            options.max_retries,
        ));
    }

    let mut cancelled = false;
    while let Some(done) = join_set.join_next().await {
        match done {
            Ok(Ok((pairs, batch_retries))) => {
                retries += batch_retries;
                insert_section_embeddings(&mut writable, model, &pairs)?;
                embedded += pairs.len();
                if let Some(sink) = &options.on_progress {
                    sink(&EmbedProgress {
                        total: sections.len(),
                        covered_before,
                        embedded,
                    });
                }
                tracing::info!(
                    embedded,
                    remaining = missing.len() - embedded,
                    "embeddings stored"
                );
            }
            Ok(Err(WikiError::Cancelled)) => {
                cancelled = true;
                join_set.abort_all();
                break;
            }
            Ok(Err(err)) => {
                join_set.abort_all();
                return Err(err);
            }
            Err(join_err) => {
                join_set.abort_all();
                return Err(WikiError::Llm(format!(
                    "embed batch task panicked: {join_err}"
                )));
            }
        }
    }
    if cancelled {
        return Err(WikiError::Cancelled);
    }

    Ok(Some(EmbedReport {
        model: model.to_owned(),
        total_sections: sections.len(),
        covered_before,
        embedded,
        retries,
    }))
}

/// One batch's provider round: acquire a budget permit, retry transient
/// failures with backoff, return hash→vector pairs in batch order. Never
/// touches SQLite (the collector is the single writer), which keeps the
/// future `Send` and spawnable.
async fn embed_batch(
    provider: Arc<dyn EmbeddingProvider>,
    model: String,
    texts: Vec<String>,
    batch_hashes: Vec<String>,
    permits: Arc<tokio::sync::Semaphore>,
    cancel: Option<CancelFlag>,
    max_retries: u32,
) -> std::result::Result<(Vec<(String, Vec<f32>)>, usize), WikiError> {
    let _permit = permits
        .acquire_owned()
        .await
        .map_err(|e| WikiError::Llm(format!("embedding semaphore closed: {e}")))?;
    if cancel.as_ref().is_some_and(CancelFlag::is_cancelled) {
        return Err(WikiError::Cancelled);
    }
    let mut retries = 0usize;
    loop {
        match provider.embed(&model, &texts).await {
            Ok(vectors) => {
                if vectors.len() != texts.len() {
                    return Err(WikiError::Llm(format!(
                        "embedding provider returned {} vectors for {} sections",
                        vectors.len(),
                        texts.len()
                    )));
                }
                return Ok((batch_hashes.into_iter().zip(vectors).collect(), retries));
            }
            Err(err) => {
                let backoff = retry_backoff(&err, retries);
                let Some(backoff) = backoff else {
                    return Err(WikiError::from(err));
                };
                if retries >= max_retries as usize {
                    return Err(WikiError::from(err));
                }
                retries += 1;
                if cancel.as_ref().is_some_and(CancelFlag::is_cancelled) {
                    return Err(WikiError::Cancelled);
                }
                tracing::warn!(model = %model, attempt = retries, "transient embedding failure; retrying");
                tokio::time::sleep(backoff).await;
            }
        }
    }
}

/// Backoff for transient provider failures; `None` means permanent (do not
/// retry). 429 honors the server's Retry-After when present; other transient
/// classes back off exponentially from 500ms, capped at 8s.
fn retry_backoff(err: &LlmError, attempt: usize) -> Option<std::time::Duration> {
    const BASE: std::time::Duration = std::time::Duration::from_millis(500);
    const CAP: std::time::Duration = std::time::Duration::from_secs(8);
    match err {
        LlmError::RateLimited {
            retry_after_seconds: Some(secs),
        } if *secs > 0 => Some(std::time::Duration::from_secs(*secs)),
        LlmError::RateLimited { .. }
        | LlmError::Timeout { .. }
        | LlmError::Http(_)
        | LlmError::Api {
            code: 429 | 500.., ..
        } => Some(std::cmp::min(BASE * (1_u32 << attempt.min(4)), CAP)),
        LlmError::Api { .. } | LlmError::InvalidResponse(_) | LlmError::MissingApiKey(_) => None,
    }
}
