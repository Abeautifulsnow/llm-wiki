//! `llm-wiki embed` (§19.3 Vector layer): the explicit backfill that gives
//! the active generation's context sections their vectors. Embeddings cost
//! model calls, so this is user-driven — the build pipeline never embeds.
//!
//! Content addressing makes the backfill incremental: a section's hash
//! (title + heading path + body, ONE definition in the search crate) keys
//! the stored vector, so unchanged sections re-embed to nothing and a
//! re-run over a fully covered generation issues zero requests. The model
//! is part of the storage key: switching models never mixes vector spaces.

use std::path::Path;
use std::sync::Arc;

use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_llm::EmbeddingProvider;
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
}

/// Backfills embeddings for the ACTIVE generation's context sections under
/// `model`. Returns `Ok(None)` when nothing is published.
pub async fn run_embed(
    workspace_root: &Path,
    config: &Config,
    provider: Arc<dyn EmbeddingProvider>,
    model: &str,
    batch_size: usize,
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

    let batch_size = batch_size.max(1);
    let mut embedded = 0usize;
    for batch in missing.chunks(batch_size) {
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
        let vectors = provider
            .embed(model, &texts)
            .await
            .map_err(WikiError::from)?;
        if vectors.len() != batch.len() {
            return Err(WikiError::Llm(format!(
                "embedding provider returned {} vectors for {} sections",
                vectors.len(),
                batch.len()
            )));
        }
        let pairs: Vec<(String, Vec<f32>)> = batch
            .iter()
            .zip(vectors)
            .map(|(section, vector)| (section.text_hash.clone(), vector))
            .collect();
        let mut writable = llm_wiki_storage::open(&state_db)?;
        insert_section_embeddings(&mut writable, model, &pairs)?;
        embedded += pairs.len();
        tracing::info!(
            embedded,
            remaining = missing.len() - embedded,
            "embeddings stored"
        );
    }

    Ok(Some(EmbedReport {
        model: model.to_owned(),
        total_sections: sections.len(),
        covered_before,
        embedded,
    }))
}
