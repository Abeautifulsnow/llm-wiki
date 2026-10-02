//! Build pipeline orchestration (PRD §29/§31/§51 steps 15–16): chains scan →
//! parse → analyze → plan → compile → publish, persisting every §31 stage
//! transition. The LLM provider is injected so tests run on
//! `FakeLlmProvider` (PRD §54); the CLI stays a thin transport (PRD §7.7).
//!
//! Failure semantics (PRD §34/§35): any error marks the build FAILED (or
//! REPLAN_REQUIRED for `WikiError::ReplanRequired`) and leaves the previously
//! published generation and pointer untouched.
//!
//! Incremental builds (PRD §19, gated on `build.incremental`, default true):
//! after the scan, a pure ChangeSet (§19.1) classifies sources; with changes,
//! the §19 pipeline runs — BuildFingerprint guard (§18.1/§19.2) → deleted
//! sources retired without ghost knowledge (§19.3) → selective re-analysis of
//! added+modified sources → deterministic mapping onto the CURRENT generation
//! → partial compile + verbatim carry-over. Changes that cannot be localized
//! deterministically stop at REPLAN_REQUIRED (trigger recorded in
//! `plan_decisions`, migration 0006). With no changes and an intact active
//! generation, the build is a true no-op: the active generation is returned
//! unchanged (no new generation, no index rebuild, 0 requests — §37.3).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::{BuildId, KnowledgeNodeId, SourceId, WikiPageId};
use llm_wiki_core::matcher::{match_sections, PrevSection, SectionIdentity, SectionOutcome};
use llm_wiki_core::model::{WikiPagePlan, WikiPlan};
use llm_wiki_llm::LlmProvider;
use llm_wiki_markdown::parse_document;
use llm_wiki_source::{ScanDiagnostic, ScanOutput, Scanner, SourceManifest};
use llm_wiki_storage::{
    apply_section_matches, finish_build, generation_stats, get_active_build_id,
    insert_plan_decision, latest_completed_build, list_source_active_node_sections, list_sources,
    load_active_sections, load_generation_pages, load_generation_view, load_knowledge_base,
    load_plan_input, mark_removed, mark_stale_builds_interrupted, open, persist_generation,
    retire_source_knowledge, start_build, update_build_status, upsert_sources_batch, BuildDraft,
    GenerationPageView, GenerationStats, PlanDecision, SourceRecord, SourceUpsert, WikiPageRecord,
    OUTCOME_FAST_PATH, OUTCOME_LOCAL_UPDATE, OUTCOME_REPLAN_REQUIRED, TRIGGER_FINGERPRINT_CHANGED,
    TRIGGER_STRUCTURAL_CHANGE,
};

use crate::analysis::{AnalysisOutcome, AnalyzedDocument, DocumentAnalyzer};
use crate::cache::{plan_cache_identity, CacheContext, CacheStats, LlmCache, StageCache};
use crate::changeset::{
    diff_manifest, finalize_change_set, BuildFingerprint, ChangeSet, FileOutcome, RegisteredSource,
    ScannedSource,
};
use crate::compile::{CompilerConfig, WikiCompiler};
use crate::incremental::{map_incremental_change, MappingDecision, MappingInput};
use crate::persist::{persist_outcome, PersistOptions};
use crate::plan::{PlannerConfig, WikiPlanner};
use crate::prompt::{load_prompt, PromptDocument};
use crate::publish::{publish, recover_if_needed, PublishPaths};

/// Response schema version recorded on builds and cache rows (§28); shared
/// with the replan pipeline so cache keys and fingerprints agree.
pub(crate) const SCHEMA_VERSION: &str = "1";

/// §19 incremental outcome summary surfaced on the build report and CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncrementalSummary {
    /// added + modified sources (the re-analyzed set).
    pub changed: usize,
    /// Deleted (removed + knowledge-retired) sources.
    pub deleted: usize,
    /// Pages recompiled by the partial compile.
    pub recompiled: usize,
    /// Pages carried over verbatim from the previous generation.
    pub carried: usize,
    /// Pages excluded from the new generation (§19.3.5: zero refs left).
    pub obsolete: usize,
}

/// Successful end of a build; the CLI prints this.
#[derive(Debug, Clone)]
pub struct BuildReport {
    pub build_id: BuildId,
    pub sources: usize,
    pub pages: usize,
    pub citations: usize,
    pub links: usize,
    /// All NEW LLM requests spent by analysis, planning and compilation
    /// (§28 cache hits are not counted, PRD §37.3).
    pub llm_request_count: u32,
    /// §28 cache behavior of this build.
    pub cache: CacheStats,
    pub published_path: PathBuf,
    /// Human-readable summary of publish recovery performed before this build
    /// started, if any (PRD §35: recovery must be reported).
    pub recovery: Option<String>,
    /// Present when the §19 incremental pipeline ran: what changed, what was
    /// recompiled, carried and dropped.
    pub incremental: Option<IncrementalSummary>,
}

/// Prompts, hashes, fingerprint and §28 cache shared by `run_build` and the
/// explicit re-plan (`replan.rs`): both commands must derive IDENTICAL cache
/// keys and BuildFingerprints so a dry-run warms the cache for the execute
/// and fingerprints stay comparable (PRD §28/§19.2).
pub(crate) struct PipelineEnv {
    pub analysis_prompt: PromptDocument,
    pub planning_prompt: PromptDocument,
    pub compilation_prompt: PromptDocument,
    pub config_hash: String,
    /// `name@version` of the analysis prompt.
    pub prompt_version: String,
    /// Canonical BuildFingerprint JSON (PRD §18.1/§19.2).
    pub fingerprint_json: String,
    pub cache: Arc<LlmCache>,
}

/// Loads prompts, derives the config hash + BuildFingerprint and opens the
/// §28 LLM cache over `state_dir/state.db`.
pub(crate) fn prepare_pipeline_env(
    state_dir: &Path,
    config: &Config,
    provider: &Arc<dyn LlmProvider>,
) -> Result<PipelineEnv> {
    let analysis_prompt = load_prompt("document-analysis", None)?;
    let planning_prompt = load_prompt("wiki-planning", None)?;
    let compilation_prompt = load_prompt("wiki-compilation", None)?;

    let config_hash = sha256_hex(
        serde_json::to_vec(config)
            .map_err(|e| WikiError::Config(format!("config serialize: {e}")))?
            .as_slice(),
    );
    let prompt_version = format!(
        "{name}@{version}",
        name = analysis_prompt.name,
        version = analysis_prompt.version
    );

    // §18.1/§19.2 BuildFingerprint: the planning-relevant inputs of THIS
    // build. Stored on the build row; the incremental pipeline compares it
    // against the last COMPLETED build and stops at REPLAN_REQUIRED on any
    // drift (a stale plan must never be silently extended).
    let fingerprint = BuildFingerprint {
        analysis_prompt: analysis_prompt.fingerprint_tag(),
        planning_prompt: planning_prompt.fingerprint_tag(),
        compilation_prompt: compilation_prompt.fingerprint_tag(),
        schema_version: SCHEMA_VERSION.to_owned(),
        parser_version: llm_wiki_markdown::PARSER_VERSION.to_owned(),
        config_hash: config_hash.clone(),
    };

    // §28 LLM cache: its own connection to the same state db (WAL + busy
    // timeout keep the two safe). The key material covers model, per-task
    // prompt versions, schema/parser versions and the effective config hash —
    // any change invalidates (§28: cross-version reuse is impossible).
    let cache = Arc::new(LlmCache::open(
        &state_dir.join("state.db"),
        CacheContext {
            model: provider.model().to_owned(),
            prompt_versions: BTreeMap::from([
                (
                    analysis_prompt.name.clone(),
                    analysis_prompt.fingerprint_tag(),
                ),
                (
                    planning_prompt.name.clone(),
                    planning_prompt.fingerprint_tag(),
                ),
                (
                    compilation_prompt.name.clone(),
                    compilation_prompt.fingerprint_tag(),
                ),
            ]),
            schema_version: SCHEMA_VERSION.to_owned(),
            parser_version: llm_wiki_markdown::PARSER_VERSION.to_owned(),
            config_hash: config_hash.clone(),
        },
    )?);

    Ok(PipelineEnv {
        analysis_prompt,
        planning_prompt,
        compilation_prompt,
        config_hash,
        prompt_version,
        fingerprint_json: fingerprint.to_json(),
        cache,
    })
}

/// Length note: ~95 lines — thin orchestration (recoveries + env prep + error terminalization); the heavy lifting lives in build_full_pipeline/build_incremental.
/// Runs the whole V0.1 build pipeline. `workspace_root` is the project
/// directory holding `.llm-wiki/config.toml`; paths inside `config` are
/// resolved against it.
pub async fn run_build(
    workspace_root: &Path,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
) -> Result<BuildReport> {
    config.validate()?;
    let root = lexical_absolute(workspace_root, &config.source.root);
    let wiki_dir = lexical_absolute(workspace_root, &config.project.wiki_dir);

    let state_dir = workspace_root.join(".llm-wiki");
    std::fs::create_dir_all(&state_dir)
        .map_err(|e| WikiError::Source(format!("cannot create state dir: {e}")))?;
    let mut conn = open(&state_dir.join("state.db"))?;

    // Startup recovery: stale builds from a dead process are INTERRUPTED and
    // a possibly-interrupted publish is resolved before anything new starts.
    let stale = mark_stale_builds_interrupted(&mut conn)?;
    if stale > 0 {
        tracing::warn!(count = stale, "marked stale builds INTERRUPTED");
    }
    let recovery = recover_if_needed(&mut conn, &wiki_dir)?;
    let recovery_note = recovery.map(|report| report.detail);

    // Shared pipeline environment (prompts, config hash, fingerprint, §28
    // cache) — identical for `build` and `replan` so cache keys and
    // fingerprints agree across the two commands.
    let env = prepare_pipeline_env(&state_dir, config, &provider)?;
    let analysis_prompt = env.analysis_prompt.clone();
    let planning_prompt = env.planning_prompt.clone();
    let compilation_prompt = env.compilation_prompt.clone();
    let config_hash = env.config_hash.clone();
    let prompt_version = env.prompt_version.clone();
    let fingerprint_json = env.fingerprint_json.clone();
    let cache = env.cache.clone();

    let build_id = start_build(
        &mut conn,
        &BuildDraft {
            model: Some(provider.model().to_owned()),
            prompt_version: Some(prompt_version.clone()),
            compiler_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
            schema_version: Some(SCHEMA_VERSION.to_owned()),
            config_hash: Some(config_hash.clone()),
            build_fingerprint: Some(fingerprint_json.clone()),
            ..BuildDraft::default()
        },
    )?;
    tracing::info!(build = %build_id, "build started");

    update_build_status(&mut conn, &build_id, "SCANNING")?;
    let result = build_inner(
        &mut conn,
        &root,
        &wiki_dir,
        &build_id,
        config,
        provider,
        &cache,
        &config_hash,
        &analysis_prompt,
        &planning_prompt,
        &compilation_prompt,
        &prompt_version,
        &fingerprint_json,
    )
    .await;

    match result {
        Ok((stats, sources, llm_request_count, incremental, fast_path)) => {
            // The no-change fast path keeps the ACTIVE generation published:
            // the report points at its directory, not the (empty) audit build.
            let published_build = fast_path.as_ref().unwrap_or(&build_id);
            let published_path = PublishPaths::new(&wiki_dir).generation_dir(published_build);
            Ok(BuildReport {
                build_id,
                sources,
                pages: stats.pages,
                citations: stats.citations,
                links: stats.links,
                llm_request_count,
                cache: cache.stats(),
                published_path,
                recovery: recovery_note,
                incremental,
            })
        }
        Err(err) => {
            // The previous generation and pointer stay untouched (PRD §35).
            let status = terminal_status_for(&err);
            let marked = finish_build(&mut conn, &build_id, status, None, None);
            if let Err(mark_err) = marked {
                tracing::error!(build = %build_id, error = %mark_err, "could not mark build terminal");
            }
            tracing::warn!(build = %build_id, status, error = %err, "build failed");
            Err(err)
        }
    }
}

/// Maps an error to the §31 terminal status the build must be marked with.
pub(crate) fn terminal_status_for(err: &WikiError) -> &'static str {
    match err {
        WikiError::ReplanRequired { .. } => "REPLAN_REQUIRED",
        _ => "FAILED",
    }
}

/// Scan (§8): snapshot hash recorded on the build row and the §28 cache; the
/// §19.1 ChangeSet is diffed BEFORE any upsert overwrites content hashes.
fn scan_and_diff(
    conn: &mut rusqlite::Connection,
    root: &Path,
    wiki_dir: &Path,
    build_id: &BuildId,
    config: &Config,
    cache: &Arc<LlmCache>,
) -> Result<(ScanOutput, Vec<FileOutcome>, Vec<SourceId>)> {
    let wiki_dir_rel = normalized_rel_of(root, wiki_dir);
    let scanner = Scanner::new(
        root,
        &config.source.include,
        &config.source.exclude,
        wiki_dir_rel,
    )?;
    let output = scanner.scan()?;
    for diagnostic in &output.diagnostics {
        warn_diagnostic(diagnostic);
    }
    let manifest = SourceManifest::from_scanned(&output.files);
    llm_wiki_storage::set_build_snapshot_hash(conn, build_id, &manifest.snapshot_hash())?;
    // Cache rows record the source snapshot hash so cross-snapshot reuse is
    // impossible (PRD §28).
    cache.set_source_snapshot_hash(&manifest.snapshot_hash());

    // ---- §19.1 ChangeSet: pure diff BEFORE the upsert overwrites hashes. ----
    let registry_before = list_sources(conn)?;
    let scanned_sources: Vec<ScannedSource<'_>> = output
        .files
        .iter()
        .map(|file| ScannedSource {
            locator_key: file.locator_key.as_str(),
            content_hash: &file.content_hash,
        })
        .collect();
    let registered_sources: Vec<RegisteredSource<'_>> = registry_before
        .iter()
        .map(|record| RegisteredSource {
            source_id: &record.source_id,
            locator_key: record.locator_key.as_str(),
            content_hash: &record.content_hash,
        })
        .collect();
    let (file_outcomes, deleted_ids) = diff_manifest(&scanned_sources, &registered_sources);
    Ok((output, file_outcomes, deleted_ids))
}

/// §19.2 BuildFingerprint guard: ANY drift in a planning-relevant field vs
/// the last COMPLETED build stops the build at REPLAN_REQUIRED — the current
/// plan is stale and must never be silently extended (the trigger is recorded
/// for `replan --dry-run` auditing). Applies before the no-change fast path
/// too: a prompt/config/schema change invalidates the PLAN, not just the
/// incremental mapping. `build.incremental = false` is the explicit
/// full-rebuild escape hatch (V0.1 behavior, no guard).
fn fingerprint_guard(
    conn: &mut rusqlite::Connection,
    build_id: &BuildId,
    fingerprint_json: &str,
) -> Result<()> {
    let Some(previous) = latest_completed_build(conn)? else {
        return Ok(());
    };
    let drifted = match &previous.build_fingerprint {
        Some(previous_fp) => previous_fp != fingerprint_json,
        // A completed build without a fingerprint predates §19
        // auditing: plan compatibility cannot be proven.
        None => true,
    };
    if !drifted {
        return Ok(());
    }
    let notes = format!(
        "build fingerprint differs from the last completed build {} (prompts, schema, parser, model or effective config changed)",
        previous.build_id
    );
    record_decision(
        conn,
        build_id,
        None,
        OUTCOME_REPLAN_REQUIRED,
        Some(TRIGGER_FINGERPRINT_CHANGED),
        0,
        notes.clone(),
    )?;
    tracing::warn!(build = %build_id, "{notes}");
    Err(WikiError::ReplanRequired {
        reason: format!(
            "{notes} (trigger: {TRIGGER_FINGERPRINT_CHANGED}); the previous generation stays visible"
        ),
    })
}

/// The mapped outcome: the post-analysis knowledge base (compile base),
/// the pages to recompile, their updated refs, the obsolete pages and the
/// reloaded previous-generation pages.
type MappedIncremental = (
    llm_wiki_core::plan::KnowledgeBase,
    BTreeSet<WikiPageId>,
    BTreeMap<WikiPageId, Vec<KnowledgeNodeId>>,
    BTreeSet<WikiPageId>,
    Vec<GenerationPageView>,
);

/// Runs the deterministic §19.2 mapping over the post-analysis knowledge
/// state. A REPLAN_REQUIRED outcome is recorded and propagated; a local
/// update returns the new knowledge base (the compile base) plus the
/// recompile/updated-refs/obsolete sets and the reloaded previous pages.
#[allow(clippy::too_many_arguments)]
fn map_incremental_or_fail(
    conn: &mut rusqlite::Connection,
    build_id: &BuildId,
    prev_build_id: &BuildId,
    prev_pages: Vec<GenerationPageView>,
    prev_kb: llm_wiki_core::plan::KnowledgeBase,
    deletion_gone: BTreeSet<KnowledgeNodeId>,
    change_set: &ChangeSet,
    modified_ids: BTreeSet<SourceId>,
) -> Result<MappedIncremental> {
    let new_kb = load_knowledge_base(conn)?;
    let mapping = map_incremental_change(&MappingInput {
        prev_pages: prev_pages.clone(),
        // The PRE-analysis knowledge state: what counts as "brand-new" is
        // decided against it, never against the post-analysis base.
        prev_kb,
        new_kb: new_kb.clone(),
        deletion_gone,
        modified_pre: modified_pre_of(conn, &change_set.modified)?,
        changed_post: changed_post_of(conn, change_set)?,
        modified_ids,
    });
    match mapping {
        MappingDecision::LocalUpdate {
            recompile,
            updated_refs,
            obsolete,
        } => {
            // Load the previous generation's pages again for the assembly
            // step (the mapping consumed the view; page data is unchanged).
            let pages = load_generation_view(conn, prev_build_id)?;
            Ok((new_kb, recompile, updated_refs, obsolete, pages))
        }
        MappingDecision::ReplanRequired { trigger, reason } => {
            record_decision(
                conn,
                build_id,
                None,
                OUTCOME_REPLAN_REQUIRED,
                Some(trigger),
                0,
                reason.clone(),
            )?;
            tracing::warn!(build = %build_id, trigger, "{reason}");
            Err(WikiError::ReplanRequired {
                reason: format!("{trigger}: {reason}"),
            })
        }
    }
}

/// Compile → topology guard → decision row → merge carried pages → index →
/// publish: the second half of the incremental pipeline (§19.2/§35), split
/// out so `build_incremental` stays an orchestrator.
#[allow(clippy::too_many_arguments)]
async fn compile_and_publish_incremental(
    conn: &mut rusqlite::Connection,
    wiki_dir: &Path,
    build_id: &BuildId,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    cache: &Arc<LlmCache>,
    compilation_prompt: &PromptDocument,
    change_set: &ChangeSet,
    prev_pages: &[GenerationPageView],
    prev_build_id: &BuildId,
    new_kb: &llm_wiki_core::plan::KnowledgeBase,
    plan: &WikiPlan,
    surviving_ids: BTreeSet<WikiPageId>,
    recompile: BTreeSet<WikiPageId>,
    obsolete: BTreeSet<WikiPageId>,
    llm_request_count: u32,
) -> Result<(GenerationStats, u32, IncrementalSummary)> {
    let prev_meta = load_generation_pages(conn, prev_build_id)?;
    let language_of: BTreeMap<String, String> = prev_meta
        .iter()
        .map(|page| (page.page_id.as_str().to_owned(), page.language.clone()))
        .collect();

    // ---- Partial compile (§19.2): ONLY affected pages consume requests. ----
    update_build_status(conn, build_id, "COMPILING")?;
    let stage_cache: Arc<dyn StageCache> = cache.clone();
    let compiler = WikiCompiler::new(
        provider.clone(),
        compilation_prompt.clone(),
        CompilerConfig {
            max_input_tokens: config.analysis.max_input_tokens as u64,
            min_output_tokens: config.llm.max_output_tokens,
            ..CompilerConfig::default()
        },
        config.llm.max_concurrency as usize,
    )
    .with_cache(stage_cache);
    let compiled = compiler
        .compile_plan_subset(plan, new_kb, build_id, &recompile)
        .await?;
    let llm_request_count = llm_request_count + compiled.llm_request_count;
    if compiled.pages.len() != recompile.len() {
        return Err(WikiError::Compilation(format!(
            "partial compile produced {} page(s) for {} affected page(s)",
            compiled.pages.len(),
            recompile.len()
        )));
    }

    // ---- Related-page topology check among surviving pages (§19.2): the
    // recompiled pages' outbound links must equal their previous ones.
    // Carried pages match by construction (verbatim copy). ----
    verify_related_page_topology(
        conn,
        build_id,
        prev_pages,
        &compiled.pages,
        &surviving_ids,
        &recompile,
    )?;

    let carried = surviving_ids.len() - recompile.len();
    let candidates = recompile.len() + obsolete.len();

    // Record the judgment (PRD §19.2: every decision is recorded, with
    // candidate count, recompiled count and outcome). This happens only once
    // the mapping is FINAL — the post-compile topology check above may still
    // revoke it into a replan-required row, and a build must never carry a
    // local-update row for a wiki it did not publish.
    record_decision(
        conn,
        build_id,
        None,
        OUTCOME_LOCAL_UPDATE,
        None,
        recompile.len(),
        format!(
            "{} source(s) changed ({} added, {} modified, {} deleted); {} candidate page(s), {} recompiled, {} carried, {} obsolete",
            change_set.changed_count() + change_set.deleted.len(),
            change_set.added.len(),
            change_set.modified.len(),
            change_set.deleted.len(),
            candidates,
            recompile.len(),
            carried,
            obsolete.len()
        ),
    )?;

    // ---- Final page set: recompiled records + carried rows copied VERBATIM
    // from the previous generation (content/body_hash/refs/citations/links;
    // the frontmatter keeps the ORIGINAL build id — never rewritten, so the
    // on-disk file is byte-identical and §36 lint stays consistent). ----
    let new_pages = merge_compiled_and_carried(
        prev_pages,
        &obsolete,
        &compiled.pages,
        &language_of,
        &surviving_ids,
    )?;
    if new_pages.is_empty() {
        return Err(WikiError::Compilation(
            "every page became obsolete; refusing to publish an empty generation".into(),
        ));
    }

    // ---- Index + Publish (§35, unchanged contract): new immutable
    // generation, pointer swap, previous generation intact on failure. ----
    update_build_status(conn, build_id, "INDEXING")?;
    let stats = persist_generation(conn, build_id, &new_pages)?;
    publish(
        conn,
        wiki_dir,
        build_id,
        &new_pages,
        config.build.keep_generations,
    )?;

    let summary = IncrementalSummary {
        changed: change_set.changed_count(),
        deleted: change_set.deleted.len(),
        recompiled: recompile.len(),
        carried,
        obsolete: obsolete.len(),
    };
    Ok((stats, llm_request_count, summary))
}

/// Persists one §19.2 decision row (every judgment is recorded).
pub(crate) fn record_decision(
    conn: &mut rusqlite::Connection,
    build_id: &BuildId,
    source_id: Option<&SourceId>,
    outcome: &str,
    trigger: Option<&str>,
    affected_pages: usize,
    notes: String,
) -> Result<()> {
    insert_plan_decision(
        conn,
        &PlanDecision {
            build_id: build_id.clone(),
            source_id: source_id.cloned(),
            outcome: outcome.to_owned(),
            trigger: trigger.map(str::to_owned),
            affected_pages: affected_pages as u32,
            notes,
        },
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn build_inner(
    conn: &mut rusqlite::Connection,
    root: &Path,
    wiki_dir: &Path,
    build_id: &BuildId,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    cache: &Arc<LlmCache>,
    config_hash: &str,
    analysis_prompt: &PromptDocument,
    planning_prompt: &PromptDocument,
    compilation_prompt: &PromptDocument,
    analysis_prompt_version: &str,
    fingerprint_json: &str,
) -> Result<(
    GenerationStats,
    usize,
    u32,
    Option<IncrementalSummary>,
    Option<BuildId>,
)> {
    // ---- Scan (§8) + §19.1 ChangeSet (before the upsert overwrites hashes).
    let (output, file_outcomes, deleted_ids) =
        scan_and_diff(conn, root, wiki_dir, build_id, config, cache)?;
    let scan_has_changes = !deleted_ids.is_empty()
        || file_outcomes
            .iter()
            .any(|outcome| matches!(outcome, FileOutcome::Added | FileOutcome::Modified(_)));

    let incremental_enabled = config.build.incremental;
    let active_build = if incremental_enabled {
        get_active_build_id(conn)?
    } else {
        None
    };

    // ---- §19.2 BuildFingerprint guard (any drift → REPLAN_REQUIRED, before
    // the fast path too: a prompt/config/schema change invalidates the PLAN,
    // not just the incremental mapping). ----
    if incremental_enabled {
        fingerprint_guard(conn, build_id, fingerprint_json)?;
    }

    if incremental_enabled && !scan_has_changes {
        if let Some(active) = active_build.as_ref() {
            // §37.3 fast path (audit FIX-006): nothing changed and the active
            // generation is intact on disk — return it unchanged instead of
            // re-running the whole cached pipeline. The started build row
            // stays a pure audit record (FAST_PATH decision, COMPLETED
            // status); the pointer, generation files, FTS and graph are
            // untouched. O(scan) + one stats query + one directory check.
            let active_dir = PublishPaths::new(wiki_dir).generation_dir(active);
            if active_dir.is_dir() {
                record_decision(
                    conn,
                    build_id,
                    None,
                    OUTCOME_FAST_PATH,
                    None,
                    0,
                    "no source changes since the last build; returning the active generation unchanged (no new generation, no index rebuild)".to_owned(),
                )?;
                let stats = generation_stats(conn, active)?;
                // finish_build overwrites the fingerprint column, so the
                // value recorded at start_build must be restated here.
                finish_build(conn, build_id, "COMPLETED", Some(fingerprint_json), None)?;
                tracing::info!(
                    build = %build_id,
                    active = %active,
                    "fast path: no source changes; the active generation stays published"
                );
                return Ok((stats, output.files.len(), 0, None, Some(active.clone())));
            }
            // Unhealthy state: the pointer names a generation whose files are
            // gone. Fall through to the full pipeline, which rebuilds and
            // republishes from scratch (self-healing).
            tracing::warn!(
                build = %build_id,
                active = %active,
                "active generation directory is missing; running the full pipeline to republish"
            );
        }
    }

    // ---- §19 incremental pipeline when the workspace actually changed. ----
    let prev_pages = if incremental_enabled && scan_has_changes {
        match &active_build {
            Some(previous) => Some(load_generation_view(conn, previous)?),
            None => None,
        }
    } else {
        None
    };
    let incremental_path = incremental_enabled
        && scan_has_changes
        && prev_pages.as_ref().is_some_and(|pages| !pages.is_empty());
    if incremental_path {
        let previous = active_build
            .as_ref()
            .expect("incremental path implies an active build");
        return build_incremental(
            conn,
            root,
            wiki_dir,
            build_id,
            config,
            provider,
            cache,
            analysis_prompt,
            compilation_prompt,
            analysis_prompt_version,
            &output,
            &file_outcomes,
            deleted_ids,
            previous,
            prev_pages.expect("checked above"),
        )
        .await;
    }

    // ---- Full pipeline (V0.1 behavior): first build, `build.incremental =
    // false`, or no previous generation to carry over. NOTE: with incremental
    // enabled this path also serves as the explicit global rebuild — a build
    // run after REPLAN_REQUIRED (or with `build.incremental = false`) re-plans
    // through the planner when the registry revision or the plan cache keys
    // moved. That is the user-driven escape hatch until `llm-wiki replan`
    // lands (next V0.2 slice). ----
    build_full_pipeline(
        conn,
        root,
        wiki_dir,
        build_id,
        config,
        provider,
        cache,
        config_hash,
        analysis_prompt,
        planning_prompt,
        compilation_prompt,
        analysis_prompt_version,
        &output,
    )
    .await
    .map(|(stats, sources, count)| (stats, sources, count, None, None))
}

/// The V0.1 full pipeline: upsert ALL sources, parse + analyze everything,
/// plan from scratch, compile every planned page, persist and publish
/// (§31/§35). Served the whole workspace until V0.2's incremental path split
/// out; now reached on first build, `build.incremental = false`, or when no
/// previous generation exists to carry over.
#[allow(clippy::too_many_arguments)]
async fn build_full_pipeline(
    conn: &mut rusqlite::Connection,
    root: &Path,
    wiki_dir: &Path,
    build_id: &BuildId,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    cache: &Arc<LlmCache>,
    config_hash: &str,
    analysis_prompt: &PromptDocument,
    planning_prompt: &PromptDocument,
    compilation_prompt: &PromptDocument,
    analysis_prompt_version: &str,
    output: &ScanOutput,
) -> Result<(GenerationStats, usize, u32)> {
    let batch: Vec<SourceUpsert> = output
        .files
        .iter()
        .map(|file| SourceUpsert {
            locator_key: &file.locator_key,
            rel_path: &file.rel_path,
            content_hash: &file.content_hash,
            size: file.size as i64,
        })
        .collect();
    let upserted = upsert_sources_batch(conn, &batch, Some(build_id.as_str()))?;
    let sources = upserted.len();
    if sources == 0 {
        return Err(WikiError::Source(format!(
            "no markdown sources found under {}; nothing to build",
            root.display()
        )));
    }

    // ---- Parse (§9): analyzed text, sections, diagnostics. ----
    update_build_status(conn, build_id, "PARSING")?;
    let mut parsed = Vec::with_capacity(sources);
    for (file, (source_id, _created)) in output.files.iter().zip(upserted) {
        let absolute = root.join(&file.rel_path);
        let raw = std::fs::read_to_string(&absolute)
            .map_err(|e| WikiError::Source(format!("cannot read {}: {e}", absolute.display())))?;
        let parsed_doc = parse_document(&raw, &file.rel_path);
        for diagnostic in &parsed_doc.diagnostics {
            tracing::warn!(
                source = %file.rel_path,
                kind = ?diagnostic.kind,
                "{}",
                diagnostic.message
            );
        }
        parsed.push((file, source_id, parsed_doc));
    }

    // ---- Analyze (§10/§11) + persist the knowledge outcome. ----
    update_build_status(conn, build_id, "ANALYZING")?;
    // §28 stage cache: shared by analysis, planning and compilation; each
    // stage writes a response only after its own validation accepted it.
    let stage_cache: Arc<dyn StageCache> = cache.clone();
    let analyzer = Arc::new(
        DocumentAnalyzer::new(
            provider.clone(),
            analysis_prompt.clone(),
            config.analysis.section_target_tokens,
            config.analysis.max_rejected_claim_ratio,
            config.llm.max_output_tokens,
            config.llm.max_concurrency,
        )
        .with_cache(stage_cache.clone()),
    );
    let doc_concurrency = config.llm.max_concurrency.max(1) as usize;
    // ---- Document-parallel analysis (T1 finding #12) ----
    // Bounded document-parallel prefetch, dependency-free: slide a window of
    // `doc_concurrency` join handles (`llm.max_concurrency`, PRD §27 —
    // previously parsed but never enforced). SQLite work (register/persist)
    // never crosses a task boundary — Connection is !Sync — only the LLM
    // calls run inside spawned tasks; results are persisted in document
    // order so the wiki stays deterministic.
    let mut llm_request_count = 0u32;
    let mut doc_inputs = Vec::with_capacity(parsed.len());
    for (file, source_id, parsed_doc) in &parsed {
        let sections = register_sections(conn, source_id, &parsed_doc.sections, Some(build_id))?;
        doc_inputs.push(AnalyzedDocument {
            source_id: source_id.clone(),
            rel_path: file.rel_path.clone(),
            content_hash: file.content_hash.clone(),
            language: parsed_doc.language.clone(),
            sections,
        });
    }
    let mut outcomes: Vec<(usize, AnalysisOutcome)> = Vec::with_capacity(doc_inputs.len());
    let mut next_to_spawn = 0usize;
    let mut join_set = tokio::task::JoinSet::new();
    while next_to_spawn < doc_inputs.len() || !join_set.is_empty() {
        let build_id_owned = build_id.clone();
        while next_to_spawn < doc_inputs.len() && join_set.len() < doc_concurrency {
            let doc = doc_inputs[next_to_spawn].clone();
            let build_id_owned = build_id_owned.clone();
            let index = next_to_spawn;
            let doc_analyzer = Arc::clone(&analyzer);
            join_set.spawn(async move {
                let outcome = doc_analyzer
                    .clone()
                    .analyze_document(&doc, Some(&build_id_owned))
                    .await?;
                Ok::<_, WikiError>((index, outcome))
            });
            next_to_spawn += 1;
        }
        if let Some(joined) = join_set.join_next().await {
            let (index, outcome) =
                joined.map_err(|e| WikiError::Llm(format!("analysis task panicked: {e}")))??;
            outcomes.push((index, outcome));
        }
    }
    outcomes.sort_by_key(|(index, _)| *index);

    // Deterministic persistence in document order.
    for ((index, outcome), doc) in outcomes.iter().zip(doc_inputs.iter()) {
        debug_assert_eq!(doc_inputs[*index].rel_path, doc.rel_path);
        llm_request_count += outcome.llm_request_count;
        persist_outcome(
            conn,
            doc,
            outcome,
            &PersistOptions {
                build_id: Some(build_id.clone()),
                model: Some(provider.model().to_owned()),
                prompt_version: Some(analysis_prompt_version.to_owned()),
                replace_source: true,
            },
        )?;
    }

    // ---- Plan (§14). ----
    update_build_status(conn, build_id, "PLANNING")?;
    let (base, registry_revision) = load_plan_input(conn)?;
    let planner_config = PlannerConfig {
        hierarchical: config.planning.hierarchical,
        max_cluster_nodes: config.planning.max_cluster_nodes as usize,
        max_plan_input_tokens: config.analysis.max_plan_input_tokens as u64,
        max_output_tokens: config.llm.max_output_tokens,
        max_concurrency: config.llm.max_concurrency.max(1) as usize,
        ..PlannerConfig::default()
    };
    let planner = WikiPlanner::new(provider.clone(), planning_prompt.clone(), planner_config)
        .with_plan_cache(
            stage_cache.clone(),
            plan_cache_identity(config_hash, provider.model(), SCHEMA_VERSION),
        );
    let plan_outcome = planner.plan(&base, registry_revision).await?;
    llm_request_count += plan_outcome.llm_request_count;

    // ---- Compile (§15/§16). ----
    update_build_status(conn, build_id, "COMPILING")?;
    let compiler_config = CompilerConfig {
        max_input_tokens: config.analysis.max_input_tokens as u64,
        ..CompilerConfig::default()
    };
    let compiler = WikiCompiler::new(
        provider.clone(),
        compilation_prompt.clone(),
        compiler_config,
        config.llm.max_concurrency as usize,
    )
    .with_cache(stage_cache);
    let generation = compiler
        .compile_plan(&plan_outcome.plan, &base, build_id)
        .await?;
    llm_request_count += generation.llm_request_count;
    if generation.pages.is_empty() {
        return Err(WikiError::Compilation(
            "no pages were compiled from the plan; refusing to publish an empty generation".into(),
        ));
    }

    // ---- Index (§31): machine-side generation rows in one transaction. ----
    update_build_status(conn, build_id, "INDEXING")?;
    let stats = persist_generation(conn, build_id, &generation.pages)?;

    // ---- Publish (§35): READY → pointer swap → COMPLETED. ----
    publish(
        conn,
        wiki_dir,
        build_id,
        &generation.pages,
        config.build.keep_generations,
    )?;

    Ok((stats, sources, llm_request_count))
}

/// The §19 incremental pipeline: deletions retired → selective re-analysis →
/// deterministic mapping onto the current generation → partial compile with
/// verbatim carry-over → normal §35 publish. Any non-localizable outcome
/// stops at REPLAN_REQUIRED with the previous generation intact (PRD §19.2).
#[allow(clippy::too_many_arguments)]
async fn build_incremental(
    conn: &mut rusqlite::Connection,
    root: &Path,
    wiki_dir: &Path,
    build_id: &BuildId,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    cache: &Arc<LlmCache>,
    analysis_prompt: &PromptDocument,
    compilation_prompt: &PromptDocument,
    analysis_prompt_version: &str,
    output: &ScanOutput,
    file_outcomes: &[FileOutcome],
    deleted_ids: Vec<SourceId>,
    prev_build_id: &BuildId,
    prev_pages: Vec<GenerationPageView>,
) -> Result<(
    GenerationStats,
    usize,
    u32,
    Option<IncrementalSummary>,
    Option<BuildId>,
)> {
    // Previous knowledge state BEFORE this build touches anything: the
    // mapping compares it against the post-build state (§19.2).
    let prev_kb = load_knowledge_base(conn)?;

    // ---- §19.3 deleted sources: retire knowledge, mark removed. No ghost
    // claims may survive (§53 DoD #5). The registry is read BEFORE the upsert
    // below overwrites the content hashes (§19.1). ----
    let registry_before = list_sources(conn)?;
    let deletion_gone = retire_deleted_sources(conn, &deleted_ids, &registry_before, build_id)?;

    // ---- Registry upsert: refresh hashes/paths, mint ids for added files. ----
    let (upserted, change_set) =
        upsert_scan_outputs(conn, output, build_id, file_outcomes, deleted_ids)?;
    let sources = upserted.len();

    // Changed sources = added + modified; everything else is skipped entirely
    // (§19.2: only re-analyze the affected set; unchanged sections and
    // SectionIds stay as-is).
    let modified_ids: BTreeSet<SourceId> = change_set.modified.iter().cloned().collect();
    let changed_files: Vec<usize> = output
        .files
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            matches!(
                file_outcomes[*index],
                FileOutcome::Added | FileOutcome::Modified(_)
            )
        })
        .map(|(index, _)| index)
        .collect();

    // ---- Selective re-analysis (§19.2) + the mapping over its outcome. ----
    update_build_status(conn, build_id, "PARSING")?;
    let llm_request_count = analyze_changed_sources(
        conn,
        root,
        build_id,
        config,
        provider.clone(),
        cache,
        analysis_prompt,
        analysis_prompt_version,
        output,
        &upserted,
        &changed_files,
    )
    .await?;

    // ---- Deterministic mapping attempt (§19.2 fixed order). Zero planner
    // LLM calls: the current plan's page identities are reused. ----
    update_build_status(conn, build_id, "PLANNING")?;
    let (new_kb, recompile, updated_refs, obsolete, prev_pages) = map_incremental_or_fail(
        conn,
        build_id,
        prev_build_id,
        prev_pages,
        prev_kb,
        deletion_gone,
        &change_set,
        modified_ids,
    )?;
    // ---- Assemble the new generation: survivors keep their page identity,
    // slugs, titles and categories; recompiled pages take the mapped refs. ----
    let (plan, surviving_ids) =
        assemble_surviving_plan(&prev_pages, &obsolete, &updated_refs, &new_kb);

    let (stats, llm_request_count, summary) = compile_and_publish_incremental(
        conn,
        wiki_dir,
        build_id,
        config,
        provider,
        cache,
        compilation_prompt,
        &change_set,
        &prev_pages,
        prev_build_id,
        &new_kb,
        &plan,
        surviving_ids,
        recompile,
        obsolete,
        llm_request_count,
    )
    .await?;
    tracing::info!(
        build = %build_id,
        changed = summary.changed,
        recompiled = summary.recompiled,
        carried = summary.carried,
        obsolete = summary.obsolete,
        "incremental build published"
    );
    Ok((stats, sources, llm_request_count, Some(summary), None))
}

/// §19.3 deleted sources: retire their knowledge and mark them removed in the
/// registry. Returns the node ids whose support vanished (mapping input).
fn retire_deleted_sources(
    conn: &mut rusqlite::Connection,
    deleted_ids: &[SourceId],
    registry_before: &[SourceRecord],
    build_id: &BuildId,
) -> Result<BTreeSet<KnowledgeNodeId>> {
    let mut deletion_gone: BTreeSet<KnowledgeNodeId> = BTreeSet::new();
    for source_id in deleted_ids {
        let record = registry_before
            .iter()
            .find(|record| &record.source_id == source_id)
            .ok_or_else(|| {
                WikiError::Storage(format!(
                    "deleted source {source_id} missing from the registry"
                ))
            })?;
        let retired = retire_source_knowledge(conn, source_id, Some(build_id.as_str()))?;
        deletion_gone.extend(retired.affected_nodes);
        mark_removed(conn, &record.locator_key)?;
        tracing::info!(
            source = %record.rel_path,
            claims = retired.retired_claims,
            relations = retired.retired_relations,
            nodes_retired = retired.retired_registry_nodes.len(),
            "deleted source retired"
        );
    }
    Ok(deletion_gone)
}

/// Upserts the scanned sources (hashes refresh, added files mint ids) and
/// finalizes the ChangeSet. MUST run AFTER the ChangeSet diff — the upsert
/// overwrites content hashes (§19.1).
fn upsert_scan_outputs(
    conn: &mut rusqlite::Connection,
    output: &ScanOutput,
    build_id: &BuildId,
    file_outcomes: &[FileOutcome],
    deleted_ids: Vec<SourceId>,
) -> Result<(Vec<(SourceId, bool)>, ChangeSet)> {
    let batch: Vec<SourceUpsert> = output
        .files
        .iter()
        .map(|file| SourceUpsert {
            locator_key: &file.locator_key,
            rel_path: &file.rel_path,
            content_hash: &file.content_hash,
            size: file.size as i64,
        })
        .collect();
    let upserted = upsert_sources_batch(conn, &batch, Some(build_id.as_str()))?;
    if upserted.is_empty() {
        return Err(WikiError::Source(format!(
            "no markdown sources found under {}; nothing to build",
            output
                .files
                .first()
                .map(|f| f.rel_path.as_str())
                .unwrap_or("<unknown>")
        )));
    }
    let change_set = finalize_change_set(file_outcomes, deleted_ids, &upserted);
    Ok((upserted, change_set))
}

/// Previous node ownership of modified sources, BEFORE re-analysis retires
/// their old claims (§19.2 mapping input).
fn modified_pre_of(
    conn: &rusqlite::Connection,
    modified_ids: &[SourceId],
) -> Result<BTreeMap<SourceId, Vec<llm_wiki_storage::SourceNodeSection>>> {
    let mut modified_pre: BTreeMap<SourceId, Vec<llm_wiki_storage::SourceNodeSection>> =
        BTreeMap::new();
    for source_id in modified_ids {
        modified_pre.insert(
            source_id.clone(),
            list_source_active_node_sections(conn, source_id)?,
        );
    }
    Ok(modified_pre)
}

/// Post-build node state of every changed source (mapping input).
fn changed_post_of(
    conn: &rusqlite::Connection,
    change_set: &ChangeSet,
) -> Result<BTreeMap<SourceId, Vec<llm_wiki_storage::SourceNodeSection>>> {
    let mut changed_post: BTreeMap<SourceId, Vec<llm_wiki_storage::SourceNodeSection>> =
        BTreeMap::new();
    for source_id in change_set.added.iter().chain(change_set.modified.iter()) {
        changed_post.insert(
            source_id.clone(),
            list_source_active_node_sections(conn, source_id)?,
        );
    }
    Ok(changed_post)
}

/// Parses + analyzes ONLY the changed sources (§19.2 selective re-analysis;
/// modified sources replace their previous active knowledge — no stale
/// claims, §53 DoD #4). Returns the LLM requests spent.
#[allow(clippy::too_many_arguments)]
async fn analyze_changed_sources(
    conn: &mut rusqlite::Connection,
    root: &Path,
    build_id: &BuildId,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    cache: &Arc<LlmCache>,
    analysis_prompt: &PromptDocument,
    analysis_prompt_version: &str,
    output: &ScanOutput,
    upserted: &[(SourceId, bool)],
    changed_files: &[usize],
) -> Result<u32> {
    update_build_status(conn, build_id, "ANALYZING")?;
    let stage_cache: Arc<dyn StageCache> = cache.clone();
    let analyzer = DocumentAnalyzer::new(
        provider.clone(),
        analysis_prompt.clone(),
        config.analysis.section_target_tokens,
        config.analysis.max_rejected_claim_ratio,
        config.llm.max_output_tokens,
        config.llm.max_concurrency,
    )
    .with_cache(stage_cache);
    // Same document-parallel pattern as the full pipeline: serial register,
    // concurrent LLM, ordered persist.
    let mut doc_inputs = Vec::with_capacity(changed_files.len());
    for index in changed_files {
        let file = &output.files[*index];
        let (source_id, _created) = upserted[*index].clone();
        let absolute = root.join(&file.rel_path);
        let raw = std::fs::read_to_string(&absolute)
            .map_err(|e| WikiError::Source(format!("cannot read {}: {e}", absolute.display())))?;
        let parsed_doc = parse_document(&raw, &file.rel_path);
        for diagnostic in &parsed_doc.diagnostics {
            tracing::warn!(
                source = %file.rel_path,
                kind = ?diagnostic.kind,
                "{}",
                diagnostic.message
            );
        }
        let sections = register_sections(conn, &source_id, &parsed_doc.sections, Some(build_id))?;
        doc_inputs.push(AnalyzedDocument {
            source_id: source_id.clone(),
            rel_path: file.rel_path.clone(),
            content_hash: file.content_hash.clone(),
            language: parsed_doc.language.clone(),
            sections,
        });
    }
    let analyzer = Arc::new(analyzer);
    let doc_concurrency = config.llm.max_concurrency.max(1) as usize;
    let mut outcomes: Vec<(usize, AnalysisOutcome)> = Vec::with_capacity(doc_inputs.len());
    let mut next_to_spawn = 0usize;
    let mut join_set = tokio::task::JoinSet::new();
    while next_to_spawn < doc_inputs.len() || !join_set.is_empty() {
        while next_to_spawn < doc_inputs.len() && join_set.len() < doc_concurrency {
            let analyzer = Arc::clone(&analyzer);
            let doc = doc_inputs[next_to_spawn].clone();
            let build_id_owned = build_id.clone();
            let index = next_to_spawn;
            join_set.spawn(async move {
                let outcome = analyzer
                    .analyze_document(&doc, Some(&build_id_owned))
                    .await?;
                Ok::<_, WikiError>((index, outcome))
            });
            next_to_spawn += 1;
        }
        if let Some(joined) = join_set.join_next().await {
            let (index, outcome) =
                joined.map_err(|e| WikiError::Llm(format!("analysis task panicked: {e}")))??;
            outcomes.push((index, outcome));
        }
    }
    outcomes.sort_by_key(|(index, _)| *index);

    let mut llm_request_count = 0u32;
    for ((index, outcome), doc) in outcomes.iter().zip(doc_inputs.iter()) {
        debug_assert_eq!(doc_inputs[*index].rel_path, doc.rel_path);
        llm_request_count += outcome.llm_request_count;
        persist_outcome(
            conn,
            doc,
            outcome,
            &PersistOptions {
                build_id: Some(build_id.clone()),
                model: Some(provider.model().to_owned()),
                prompt_version: Some(analysis_prompt_version.to_owned()),
                replace_source: true,
            },
        )?;
    }
    Ok(llm_request_count)
}

/// Builds the surviving-page plan from the previous generation: survivors
/// keep their page identity, slugs, titles and categories; recompiled pages
/// take the mapped refs. Returns (plan, surviving ids); the language map is
/// resolved by the caller from `load_generation_pages`.
fn assemble_surviving_plan(
    prev_pages: &[GenerationPageView],
    obsolete: &BTreeSet<WikiPageId>,
    updated_refs: &BTreeMap<WikiPageId, Vec<KnowledgeNodeId>>,
    new_kb: &llm_wiki_core::plan::KnowledgeBase,
) -> (WikiPlan, BTreeSet<WikiPageId>) {
    let mut plan_pages: Vec<WikiPagePlan> = Vec::new();
    let mut surviving_ids: BTreeSet<WikiPageId> = BTreeSet::new();
    for page in prev_pages {
        if obsolete.contains(&page.page_id) {
            continue; // §19.3.5: zero refs left — excluded from the generation
        }
        surviving_ids.insert(page.page_id.clone());
        let knowledge_refs = updated_refs
            .get(&page.page_id)
            .cloned()
            .unwrap_or_else(|| page.knowledge_refs.clone());
        // Source refs come from the NEW knowledge state (frontmatter of
        // recompiled pages; carried pages keep their original frontmatter).
        let mut source_refs: Vec<SourceId> = knowledge_refs
            .iter()
            .filter_map(|node_id| new_kb.nodes.get(node_id))
            .flat_map(|node| node.anchors.iter())
            .map(|anchor| anchor.source_id.clone())
            .collect();
        source_refs.sort();
        source_refs.dedup();
        // Related pages: the page's previous outbound links, restricted to
        // surviving pages (links into obsolete pages disappear with them).
        let related_pages: Vec<WikiPageId> = page
            .links
            .iter()
            .map(|link| link.to_page_id.clone())
            .filter(|target| surviving_ids.contains(target))
            .collect();
        plan_pages.push(WikiPagePlan {
            id: page.page_id.clone(),
            slug: page.slug.clone(),
            title: page.title.clone(),
            category: page.category.clone(),
            // Purpose is not persisted on page rows; recompiled pages render
            // an empty purpose field (the body is grounded in the refs).
            purpose: String::new(),
            knowledge_refs,
            source_refs,
            related_pages,
        });
    }
    (WikiPlan { pages: plan_pages }, surviving_ids)
}

/// §19.2 related-page topology guard: every recompiled page's outbound links
/// must equal its previous ones (carried pages match by construction). A
/// divergence is a structural change → REPLAN_REQUIRED with the trigger
/// recorded; the previous generation stays visible.
fn verify_related_page_topology(
    conn: &mut rusqlite::Connection,
    build_id: &BuildId,
    prev_pages: &[GenerationPageView],
    compiled: &[WikiPageRecord],
    surviving_ids: &BTreeSet<WikiPageId>,
    recompile: &BTreeSet<WikiPageId>,
) -> Result<()> {
    let previous_topology: BTreeMap<&WikiPageId, BTreeSet<String>> = prev_pages
        .iter()
        .map(|page| {
            (
                &page.page_id,
                page.links
                    .iter()
                    .filter(|link| surviving_ids.contains(&link.to_page_id))
                    .map(|link| link.to_page_id.as_str().to_owned())
                    .collect::<BTreeSet<String>>(),
            )
        })
        .collect();
    for record in compiled {
        if !recompile.contains(&record.page_id) {
            continue;
        }
        let targets: BTreeSet<String> = record
            .links
            .iter()
            .map(|link| link.to_page_id.as_str().to_owned())
            .collect();
        if previous_topology.get(&record.page_id) != Some(&targets) {
            let notes = format!(
                "page '{}' outbound links changed from {:?} to {targets:?}; related-page topology is not stable",
                record.slug,
                previous_topology.get(&record.page_id)
            );
            record_decision(
                conn,
                build_id,
                None,
                OUTCOME_REPLAN_REQUIRED,
                Some(TRIGGER_STRUCTURAL_CHANGE),
                recompile.len(),
                notes.clone(),
            )?;
            return Err(WikiError::ReplanRequired {
                reason: format!("{TRIGGER_STRUCTURAL_CHANGE}: {notes}"),
            });
        }
    }
    Ok(())
}

/// Final page set: recompiled records + carried rows copied VERBATIM from
/// the previous generation (content/body_hash/refs/citations/links; the
/// frontmatter keeps the ORIGINAL build id — never rewritten, so the on-disk
/// file is byte-identical and §36 lint stays consistent).
fn merge_compiled_and_carried(
    prev_pages: &[GenerationPageView],
    obsolete: &BTreeSet<WikiPageId>,
    compiled: &[WikiPageRecord],
    language_of: &BTreeMap<String, String>,
    surviving_ids: &BTreeSet<WikiPageId>,
) -> Result<Vec<WikiPageRecord>> {
    let compiled_by_id: BTreeMap<String, &WikiPageRecord> = compiled
        .iter()
        .map(|page| (page.page_id.as_str().to_owned(), page))
        .collect();
    let mut new_pages: Vec<WikiPageRecord> = Vec::with_capacity(prev_pages.len() - obsolete.len());
    for page in prev_pages {
        if obsolete.contains(&page.page_id) {
            continue;
        }
        if let Some(record) = compiled_by_id.get(page.page_id.as_str()) {
            new_pages.push((*record).clone());
            continue;
        }
        new_pages.push(WikiPageRecord {
            page_id: page.page_id.clone(),
            slug: page.slug.clone(),
            title: page.title.clone(),
            category: page.category.clone(),
            language: language_of
                .get(page.page_id.as_str())
                .cloned()
                .unwrap_or_else(|| "und".to_owned()),
            body_hash: page.body_hash.clone(),
            content: page.content.clone(),
            knowledge_refs: page.knowledge_refs.clone(),
            citations: page.citations.clone(),
            links: page
                .links
                .iter()
                .filter(|link| surviving_ids.contains(&link.to_page_id))
                .cloned()
                .collect(),
        });
    }
    Ok(new_pages)
}

/// Persists the parsed sections through the Section Registry (PRD §45):
/// identities are carried across builds deterministically; ambiguous matches
/// are re-analyzed under a FRESH identity (never reusing the old id), and
/// unmatched previous sections retire.
pub(crate) fn register_sections(
    conn: &mut rusqlite::Connection,
    source_id: &llm_wiki_core::ids::SourceId,
    sections: &[llm_wiki_markdown::SectionOutput],
    build_id: Option<&BuildId>,
) -> Result<Vec<crate::analysis::AnalysisSection>> {
    let prev: Vec<PrevSection> = load_active_sections(conn, source_id)?
        .into_iter()
        .map(|stored| PrevSection {
            id: stored.section_id,
            identity: SectionIdentity {
                heading_path: stored.heading_path,
                fingerprint: stored.fingerprint,
            },
        })
        .collect();
    let current: Vec<SectionIdentity> = sections
        .iter()
        .map(|section| SectionIdentity::from_parts(&section.heading_path, &section.content))
        .collect();
    let ranges: Vec<llm_wiki_core::model::SourceRange> = sections
        .iter()
        .map(|section| section.source_range)
        .collect();

    let mut report = match_sections(&prev, &current);
    if report.ambiguous {
        // PRD §45: ambiguity retires the old identity; this build re-analyzes
        // the source, so the ambiguous sections get brand-new identities.
        tracing::warn!(source = %source_id, "ambiguous section matches; assigning fresh identities");
        for assignment in &mut report.assignments {
            if assignment.outcome == SectionOutcome::Ambiguous {
                assignment.outcome = SectionOutcome::Created;
            }
        }
    }
    let mut created_ids: BTreeMap<usize, llm_wiki_core::ids::SectionId> = BTreeMap::new();
    for assignment in &report.assignments {
        if assignment.outcome == SectionOutcome::Created {
            created_ids.insert(
                assignment.cur_index,
                llm_wiki_core::ids::SectionId::generate(),
            );
        }
    }
    apply_section_matches(
        conn,
        source_id,
        &report,
        &current,
        &ranges,
        &created_ids,
        build_id.map(|id| id.as_str().to_owned()).as_deref(),
    )?;

    Ok(sections
        .iter()
        .zip(&report.assignments)
        .map(|(section, assignment)| {
            let section_id = match &assignment.outcome {
                SectionOutcome::Carried { prev } => prev.clone(),
                SectionOutcome::Created => created_ids[&assignment.cur_index].clone(),
                SectionOutcome::Ambiguous => {
                    // Unreachable: ambiguous outcomes were rewritten above.
                    llm_wiki_core::ids::SectionId::generate()
                }
            };
            crate::analysis::AnalysisSection {
                section_id,
                heading_path: section.heading_path.clone(),
                range: section.source_range,
                content: section.content.clone(),
            }
        })
        .collect())
}

/// Relative path of `wiki_dir` inside the source root (hard-exclude defense
/// in depth, PRD §8.4); `None` when wiki_dir lives outside the source tree.
pub(crate) fn normalized_rel_of(root: &Path, wiki_dir: &Path) -> Option<String> {
    wiki_dir.strip_prefix(root).ok().map(|rel| {
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
    })
}

pub(crate) fn warn_diagnostic(diagnostic: &ScanDiagnostic) {
    tracing::warn!(
        source = %diagnostic.rel_path,
        kind = ?diagnostic.kind,
        "{}",
        diagnostic.message
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_storage::list_plan_decisions;

    #[test]
    fn replan_required_maps_to_replan_status_and_everything_else_to_failed() {
        // §31/§34: ReplanRequired is an actionable terminal state of its own
        // (CLI exit code 7, trigger reason attached); every other failure
        // marks the build FAILED. Neither touches the published generation.
        let replan = WikiError::ReplanRequired {
            reason: "sources restructured".into(),
        };
        assert_eq!(terminal_status_for(&replan), "REPLAN_REQUIRED");
        assert_eq!(replan.exit_code(), 7);

        assert_eq!(
            terminal_status_for(&WikiError::Planning("bad plan".into())),
            "FAILED"
        );
        assert_eq!(
            terminal_status_for(&WikiError::Llm("down".into())),
            "FAILED"
        );
        assert_eq!(
            terminal_status_for(&WikiError::Storage("db gone".into())),
            "FAILED"
        );
        assert_eq!(
            terminal_status_for(&WikiError::PublishRecovery("mismatch".into())),
            "FAILED"
        );
    }

    #[test]
    fn decision_recording_writes_rows_for_the_audit_view() {
        // The plan_decisions rows (migration 0006) are the audit substrate:
        // record_decision must round-trip through list_plan_decisions.
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        record_decision(
            &mut conn,
            &build,
            None,
            OUTCOME_LOCAL_UPDATE,
            None,
            2,
            "two pages recompiled".into(),
        )
        .unwrap();
        let rows = list_plan_decisions(&conn, &build).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].outcome, "local-update");
        assert_eq!(rows[0].affected_pages, 2);
        assert_eq!(rows[0].trigger, None);
    }
}
