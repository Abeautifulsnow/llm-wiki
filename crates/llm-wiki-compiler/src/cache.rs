//! LLM response cache (PRD §28) and stage-cache plumbing.
//!
//! The [`LlmCache`] decorator area provides exact-key lookup over the
//! `llm_cache` table: `cache_key = hash(task_type + canonical request payload
//! hash + model + prompt_version + schema_version + parser_version +
//! effective_config_hash)`. Every input/model/prompt/schema/parser/config
//! change changes the key (§28: any drift invalidates).
//!
//! Only **validated** responses enter the cache: stages call
//! [`remember_validated`] after their shape → referential → semantic
//! validation accepted the response (analysis: schema+evidence verified;
//! planner: parsed+validated; compiler: grounded+citations valid). Rejected or
//! repair-failed responses are never cached. Cache hits do not count as new
//! LLM requests (§37.3 rebuild-determinism gate).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_llm::{LlmProvider, LlmRequest, LlmResponse};
use llm_wiki_storage::{get_cached_response, put_cached_response, CacheRow};

/// Hit/miss counters surfaced on `BuildReport` (PRD §28/§37.3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

/// Stage-facing cache surface: request-keyed LLM responses plus raw-keyed
/// entries (the plan-identity store, §45). Implemented by [`LlmCache`].
pub trait StageCache: Send + Sync {
    /// Cached response for `request`, or `None` on a miss.
    fn lookup(&self, request: &LlmRequest) -> Option<LlmResponse>;
    /// Stores the response for `request`. Called ONLY after the stage
    /// validated the response (PRD §28).
    fn remember(&self, request: &LlmRequest, response: &LlmResponse);
    /// Raw byte-string entry by exact cache key (plan persistence).
    fn lookup_raw(&self, cache_key: &str) -> Option<String>;
    fn remember_raw(&self, cache_key: &str, value: &str);
}

/// Per-stage key material, supplied by `run_build` (PRD §28: the decorator
/// needs model, prompt version, schema/parser versions and config hash).
#[derive(Debug, Clone)]
pub struct CacheContext {
    pub model: String,
    /// `task_tag → "name@version"`; the prompt version baked into the key.
    pub prompt_versions: BTreeMap<String, String>,
    pub schema_version: String,
    pub parser_version: String,
    pub config_hash: String,
}

/// Computes the §28 cache key for one request.
pub fn cache_key(request: &LlmRequest, context: &CacheContext, prompt_version: &str) -> String {
    // Canonical request payload: system + prompt + params (embeds source
    // content and, for planning, the sorted §14 input sets).
    let payload = format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{:.6}\u{1f}{}\u{1f}{}",
        request.task_tag,
        request.system.as_deref().unwrap_or(""),
        request.prompt,
        request.temperature,
        request.max_output_tokens,
        u8::from(request.json_mode),
    );
    sha256_hex(
        format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
            request.task_tag,
            sha256_hex(payload.as_bytes()),
            context.model,
            prompt_version,
            context.schema_version,
            context.parser_version,
            context.config_hash,
        )
        .as_bytes(),
    )
}

/// Identity suffix for plan-identity cache keys (§45): model + schema +
/// effective config over the planner's reconciliation key.
pub fn plan_cache_identity(config_hash: &str, model: &str, schema_version: &str) -> String {
    sha256_hex(format!("{config_hash}\u{1f}{model}\u{1f}{schema_version}").as_bytes())
}

/// Cache key under which the full validated `WikiPlan` JSON (including page
/// IDs) is stored for one reconciliation key.
pub fn plan_cache_key(reconciliation_key: &str, identity: &str) -> String {
    sha256_hex(format!("wiki-plan\u{1f}{reconciliation_key}\u{1f}{identity}").as_bytes())
}

/// SQLite-backed stage cache. Owns its own connection so the build connection
/// stays owned by the pipeline; WAL + busy timeout make the two safe.
pub struct LlmCache {
    conn: Mutex<rusqlite::Connection>,
    context: CacheContext,
    source_snapshot_hash: Mutex<String>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl LlmCache {
    pub fn new(conn: rusqlite::Connection, context: CacheContext) -> Self {
        Self {
            conn: Mutex::new(conn),
            context,
            source_snapshot_hash: Mutex::new(String::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Opens (and migrates) the state db for caching — used by `run_build`.
    pub fn open(db_path: &Path, context: CacheContext) -> Result<Self> {
        Ok(Self::new(llm_wiki_storage::open(db_path)?, context))
    }

    /// The source snapshot hash recorded on cache rows (§28); set once the
    /// scan produced the manifest.
    pub fn set_source_snapshot_hash(&self, hash: &str) {
        *self
            .source_snapshot_hash
            .lock()
            .expect("source snapshot lock") = hash.to_owned();
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::SeqCst),
            misses: self.misses.load(Ordering::SeqCst),
        }
    }

    fn prompt_version_for(&self, task_tag: &str) -> Option<String> {
        self.context.prompt_versions.get(task_tag).cloned()
    }
}

impl StageCache for LlmCache {
    fn lookup(&self, request: &LlmRequest) -> Option<LlmResponse> {
        let prompt_version = self.prompt_version_for(&request.task_tag)?;
        let key = cache_key(request, &self.context, &prompt_version);
        let conn = self.conn.lock().expect("cache connection lock");
        let text = match get_cached_response(&conn, &key) {
            Ok(Some(text)) => text,
            Ok(None) => {
                self.misses.fetch_add(1, Ordering::SeqCst);
                return None;
            }
            Err(err) => {
                // A cache read failure must never fail a build (§28: cache is
                // an optimization; the inner provider remains the fallback).
                tracing::warn!(error = %err, "llm cache lookup failed; treating as miss");
                self.misses.fetch_add(1, Ordering::SeqCst);
                return None;
            }
        };
        drop(conn);
        self.hits.fetch_add(1, Ordering::SeqCst);
        Some(LlmResponse {
            text,
            model: self.context.model.clone(),
            input_tokens: 0,
            output_tokens: 0,
            finish_reason: Some("cache".to_owned()),
        })
    }

    fn remember(&self, request: &LlmRequest, response: &LlmResponse) {
        let Some(prompt_version) = self.prompt_version_for(&request.task_tag) else {
            return;
        };
        let key = cache_key(request, &self.context, &prompt_version);
        let row = CacheRow {
            cache_key: key,
            task_type: request.task_tag.clone(),
            model: self.context.model.clone(),
            prompt_version,
            schema_version: self.context.schema_version.clone(),
            parser_version: self.context.parser_version.clone(),
            config_hash: self.context.config_hash.clone(),
            source_snapshot_hash: self
                .source_snapshot_hash
                .lock()
                .expect("source snapshot lock")
                .clone(),
            response: response.text.clone(),
        };
        let mut conn = self.conn.lock().expect("cache connection lock");
        if let Err(err) = put_cached_response(&mut conn, &row) {
            tracing::warn!(error = %err, "llm cache write failed; continuing without it");
        }
    }

    fn lookup_raw(&self, cache_key: &str) -> Option<String> {
        let conn = self.conn.lock().expect("cache connection lock");
        match get_cached_response(&conn, cache_key) {
            Ok(Some(text)) => Some(text),
            Ok(None) => None,
            Err(err) => {
                tracing::warn!(error = %err, "raw cache lookup failed; treating as miss");
                None
            }
        }
    }

    fn remember_raw(&self, cache_key: &str, value: &str) {
        let row = CacheRow {
            cache_key: cache_key.to_owned(),
            task_type: "wiki-plan".to_owned(),
            model: self.context.model.clone(),
            prompt_version: "plan-identity".to_owned(),
            schema_version: self.context.schema_version.clone(),
            parser_version: self.context.parser_version.clone(),
            config_hash: self.context.config_hash.clone(),
            source_snapshot_hash: self
                .source_snapshot_hash
                .lock()
                .expect("source snapshot lock")
                .clone(),
            response: value.to_owned(),
        };
        let mut conn = self.conn.lock().expect("cache connection lock");
        if let Err(err) = put_cached_response(&mut conn, &row) {
            tracing::warn!(error = %err, "raw cache write failed; continuing without it");
        }
    }
}

/// One provider request with cache short-circuit. Returns the response and how
/// many NEW LLM requests it consumed (0 on a hit) — hits must not count toward
/// `llm_request_count` (PRD §37.3).
pub(crate) async fn generate_cached(
    provider: &Arc<dyn LlmProvider>,
    cache: Option<&Arc<dyn StageCache>>,
    request: LlmRequest,
) -> Result<(LlmResponse, u32)> {
    if let Some(cache) = cache {
        if let Some(response) = cache.lookup(&request) {
            return Ok((response, 0));
        }
    }
    let response = provider
        .generate(request.clone())
        .await
        .map_err(WikiError::from)?;
    Ok((response, 1))
}

/// Caches a response AFTER the stage validated it (PRD §28). No-op without a
/// cache wired.
pub(crate) fn remember_validated(
    cache: Option<&Arc<dyn StageCache>>,
    request: &LlmRequest,
    response: &LlmResponse,
) {
    if let Some(cache) = cache {
        cache.remember(request, response);
    }
}

/// Builds the single repair request for a rejected attempt: machine-readable
/// reasons go into the untouched `{{REPAIR_NOTES}}` slot (PRD §11).
pub(crate) fn repair_request(base: &LlmRequest, template: &str, reasons: &[String]) -> LlmRequest {
    let mut repair = base.clone();
    let notes = format!(
        "## Previous attempt rejected\nYour previous reply failed validation:\n{}\n\nFix every issue and resend the COMPLETE JSON object.",
        reasons
            .iter()
            .map(|reason| format!("- {reason}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    repair.prompt = template.replace("{{REPAIR_NOTES}}", &notes);
    repair
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> CacheContext {
        CacheContext {
            model: "fake-model".into(),
            prompt_versions: BTreeMap::from([(
                "document-analysis".to_owned(),
                "document-analysis@1".to_owned(),
            )]),
            schema_version: "1".into(),
            parser_version: "0.1.0".into(),
            config_hash: "cfg-hash".into(),
        }
    }

    fn request(prompt: &str) -> LlmRequest {
        LlmRequest {
            task_tag: "document-analysis".into(),
            system: None,
            prompt: prompt.into(),
            temperature: 0.0,
            max_output_tokens: 4096,
            json_mode: true,
        }
    }

    #[test]
    fn cache_key_is_stable_and_input_sensitive() {
        let ctx = context();
        let base = cache_key(&request("prompt-a"), &ctx, "document-analysis@1");
        assert_eq!(
            base,
            cache_key(&request("prompt-a"), &ctx, "document-analysis@1")
        );

        // Any input/model/prompt/schema/parser/config change changes the key.
        assert_ne!(
            base,
            cache_key(&request("prompt-b"), &ctx, "document-analysis@1")
        );
        assert_ne!(
            base,
            cache_key(
                &LlmRequest {
                    temperature: 0.5,
                    ..request("prompt-a")
                },
                &ctx,
                "document-analysis@1"
            )
        );
        let mut ctx_model = context();
        ctx_model.model = "other".into();
        assert_ne!(
            base,
            cache_key(&request("prompt-a"), &ctx_model, "document-analysis@1")
        );
        let mut ctx_prompt = context();
        ctx_prompt
            .prompt_versions
            .insert("document-analysis".into(), "document-analysis@2".into());
        assert_ne!(
            base,
            cache_key(&request("prompt-a"), &ctx_prompt, "document-analysis@2")
        );
        let mut ctx_schema = context();
        ctx_schema.schema_version = "2".into();
        assert_ne!(
            base,
            cache_key(&request("prompt-a"), &ctx_schema, "document-analysis@1")
        );
        let mut ctx_parser = context();
        ctx_parser.parser_version = "9.9.9".into();
        assert_ne!(
            base,
            cache_key(&request("prompt-a"), &ctx_parser, "document-analysis@1")
        );
        let mut ctx_config = context();
        ctx_config.config_hash = "changed".into();
        assert_ne!(
            base,
            cache_key(&request("prompt-a"), &ctx_config, "document-analysis@1")
        );
    }

    #[test]
    fn plan_cache_identity_and_key_are_input_sensitive() {
        let identity = plan_cache_identity("cfg", "m", "1");
        assert_eq!(identity, plan_cache_identity("cfg", "m", "1"));
        assert_ne!(identity, plan_cache_identity("cfg2", "m", "1"));
        assert_ne!(identity, plan_cache_identity("cfg", "m2", "1"));
        let key = plan_cache_key("reconcile-key", &identity);
        assert_eq!(key, plan_cache_key("reconcile-key", &identity));
        assert_ne!(key, plan_cache_key("reconcile-key-2", &identity));
    }

    #[test]
    fn llm_cache_roundtrip_counts_hits_and_misses() {
        let cache = LlmCache::new(llm_wiki_storage::open_in_memory().unwrap(), context());
        cache.set_source_snapshot_hash("snap-1");
        let req = request("analyze this");
        assert!(cache.lookup(&req).is_none(), "miss on fresh cache");
        assert_eq!(cache.stats(), CacheStats { hits: 0, misses: 1 });

        cache.remember(
            &req,
            &LlmResponse {
                text: "{\"summary\":\"ok\"}".into(),
                model: "fake-model".into(),
                input_tokens: 1,
                output_tokens: 1,
                finish_reason: Some("stop".into()),
            },
        );
        let hit = cache.lookup(&req).expect("second lookup is a hit");
        assert_eq!(hit.text, "{\"summary\":\"ok\"}");
        assert_eq!(hit.finish_reason.as_deref(), Some("cache"));
        assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 1 });

        // Unknown task tags are never cached (no prompt version → no key).
        let unknown = LlmRequest {
            task_tag: "unknown-task".into(),
            ..request("x")
        };
        assert!(cache.lookup(&unknown).is_none());
        cache.remember(
            &unknown,
            &LlmResponse {
                text: "y".into(),
                model: "m".into(),
                input_tokens: 0,
                output_tokens: 0,
                finish_reason: None,
            },
        );
        assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 1 });
    }

    #[test]
    fn raw_entries_roundtrip_for_plan_identity() {
        let cache = LlmCache::new(llm_wiki_storage::open_in_memory().unwrap(), context());
        let key = plan_cache_key("rk", "id");
        assert!(cache.lookup_raw(&key).is_none());
        cache.remember_raw(&key, "{\"pages\":[]}");
        assert_eq!(cache.lookup_raw(&key).as_deref(), Some("{\"pages\":[]}"));
    }
}
