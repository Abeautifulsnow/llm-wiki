//! Explicit global re-plan (PRD §19.2/§29): `llm-wiki replan` separates
//! high-cost, high-impact knowledge-architecture changes from daily
//! maintenance builds. `--dry-run` audits the global plan change and the cost
//! estimate WITHOUT calling the Compiler or publishing; the bare command
//! executes hierarchical planning, a stable-ID plan diff (§45) and a partial
//! recompilation, then publishes through the unchanged §35 contract.
//!
//! The plan diff is Registry-ID-anchored (§19.2 fixed order: Registry ID,
//! existing `knowledge_refs`, page category, local relation rules): every
//! old page's knowledge "flows" to the new page with the largest ref overlap
//! (deterministic tie-breaks: larger overlap, then slug order), and every
//! new page keeps the `WikiPageId` of the old page it continues — unless the
//! old page was split (fresh ids for every successor) or the new page merges
//! several old pages (the DOMINANT predecessor's id survives, all
//! predecessors recorded). Semantic-unchanged pages are carried
//! byte-identical, so their `WikiPageId`s demonstrably survive (§53 #8).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, KnowledgeNodeId, SourceId, WikiPageId};
use llm_wiki_core::model::{WikiPagePlan, WikiPlan};
use llm_wiki_llm::LlmProvider;
use llm_wiki_markdown::parse_document;
use llm_wiki_source::{ScanOutput, Scanner, SourceManifest};
use llm_wiki_storage::{
    carry_source_chunks, finish_build, get_active_build_id, insert_page_id_maps,
    latest_completed_build, list_recent_plan_decisions_by_outcome, list_sources,
    load_generation_pages, load_generation_view, load_knowledge_base, load_plan_input,
    mark_removed, mark_stale_builds_interrupted, open, persist_generation, retire_source_knowledge,
    start_build, update_build_status, upsert_sources_batch, BuildDraft, GenerationPageView,
    PageIdMapRow, SourceRecord, SourceUpsert, WikiPageRecord, MAP_KIND_KEEP, MAP_KIND_MERGE,
    MAP_KIND_RETIRE, MAP_KIND_SPLIT, OUTCOME_REPLAN_DRY_RUN, OUTCOME_REPLAN_EXECUTED,
    OUTCOME_REPLAN_REQUIRED,
};

use crate::analysis::{AnalysisOutcome, AnalyzedDocument, DocumentAnalyzer};
use crate::build::{
    normalized_rel_of, prepare_pipeline_env, record_decision, register_sections,
    terminal_status_for, warn_diagnostic, write_source_chunks, PipelineEnv,
};
use crate::cache::plan_cache_identity;
use crate::cache::CacheStats;
use crate::changeset::{diff_manifest, FileOutcome, RegisteredSource, ScannedSource};
use crate::compile::{CompilerConfig, WikiCompiler, MAX_PAGE_OUTPUT_TOKENS};
use crate::plan::{PlannerConfig, WikiPlanner};
use crate::publish::recover_if_needed;

// ---------------------------------------------------------------------------
// Plan diff (§19.2/§45)
// ---------------------------------------------------------------------------

/// A new plan page whose identity continues from an old page (keep the old
/// `WikiPageId`): `unchanged` (carried) or `modified` (recompiled).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuedPage {
    /// Index into `WikiPlan.pages`.
    pub index: usize,
    /// The FINAL id: the predecessor's stable `WikiPageId`.
    pub page_id: WikiPageId,
    /// The old page the identity continues from (same as `page_id`).
    pub predecessor_id: WikiPageId,
    pub title: String,
    pub slug: String,
}

/// One merge: a new page absorbing several old pages (PRD §19.2). The DOMINANT
/// predecessor's `WikiPageId` survives; ALL predecessors are recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeOutcome {
    /// The merged new page; `page_id` is the dominant predecessor's id.
    pub continued: ContinuedPage,
    /// ALL predecessor ids, DOMINANT FIRST (largest ref overlap, then old
    /// slug order). `predecessors[0] == continued.predecessor_id`.
    pub predecessors: Vec<WikiPageId>,
}

/// One split: an old page whose knowledge is now covered by several new pages
/// (PRD §19.2). Every successor gets a FRESH `WikiPageId`; the old page is
/// retired with all successors recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitOutcome {
    /// The old page that was split.
    pub retired: WikiPageId,
    pub slug: String,
    /// Fresh ids of the new pages sharing its knowledge, in plan order.
    pub successors: Vec<WikiPageId>,
}

/// A brand-new plan page with no old-page match: fresh `WikiPageId`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreatedPage {
    /// Index into `WikiPlan.pages`.
    pub index: usize,
    /// The fresh id the planner assigned (kept as-is).
    pub page_id: WikiPageId,
    pub title: String,
    pub slug: String,
}

/// An old page no new page continues: retired (PRD §19.3.5/§45).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetiredPage {
    pub page_id: WikiPageId,
    pub slug: String,
    /// Refs of the retired page that NO new page covers — empty when the
    /// knowledge was absorbed elsewhere; non-empty flags potential knowledge
    /// loss (the ref's source was deleted, §19.3, or the knowledge is gone).
    pub lost_refs: Vec<KnowledgeNodeId>,
}

/// The deterministic, Registry-ID-anchored diff between the current
/// generation and a freshly planned `WikiPlan` (PRD §19.2/§45/§53 #8).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanDiff {
    /// New pages identical to their predecessor (refs set + title +
    /// category): keep the id, carry byte-identical, no recompile.
    pub unchanged: Vec<ContinuedPage>,
    /// New pages continuing a predecessor with changes: keep the id,
    /// recompile.
    pub modified: Vec<ContinuedPage>,
    /// New pages absorbing several old pages: keep the DOMINANT id,
    /// recompile, record all predecessors.
    pub merged: Vec<MergeOutcome>,
    /// Old pages shared by several new pages: retire with successors, every
    /// successor recompiles under a fresh id.
    pub split: Vec<SplitOutcome>,
    /// New pages with no old match: fresh ids, recompile.
    pub created: Vec<CreatedPage>,
    /// Old pages no new page continues (absorbed pages are part of `merged`
    /// instead): retire.
    pub retired: Vec<RetiredPage>,
}

impl PlanDiff {
    /// True when the fresh plan matches the current generation page-for-page
    /// (everything unchanged): no replan is needed (PRD §19.2).
    pub fn is_empty(&self) -> bool {
        self.modified.is_empty()
            && self.merged.is_empty()
            && self.split.is_empty()
            && self.created.is_empty()
            && self.retired.is_empty()
    }

    /// Pages the execute mode must compile: modified + merged + split
    /// successors + created (PRD §19.2 dry-run cost estimate).
    pub fn recompile_count(&self) -> usize {
        self.modified.len()
            + self.merged.len()
            + self
                .split
                .iter()
                .map(|split| split.successors.len())
                .sum::<usize>()
            + self.created.len()
    }
}

/// The old/new ref sets shared by the matching passes and classifiers.
struct RefSets<'a> {
    old_refs: &'a [BTreeSet<KnowledgeNodeId>],
    new_refs: &'a [BTreeSet<KnowledgeNodeId>],
}

/// Outcome of the two matching passes over the ref sets.
struct MatchPass {
    /// Each new page's best old page (larger overlap first, then old slug
    /// order).
    best_old: Vec<Option<usize>>,
    /// Old pages named best by which new pages (split detection).
    claimants: Vec<Vec<usize>>,
    /// Each old page's knowledge flows to ONE new page (larger overlap
    /// first, then new slug order — slugs are unique within a plan).
    flows_to: Vec<Option<usize>>,
}

fn match_pass(
    old_refs: &[BTreeSet<KnowledgeNodeId>],
    new_refs: &[BTreeSet<KnowledgeNodeId>],
    old_pages: &[GenerationPageView],
    new_plan: &WikiPlan,
) -> MatchPass {
    let mut best_old: Vec<Option<usize>> = Vec::with_capacity(new_plan.pages.len());
    let mut claimants: Vec<Vec<usize>> = vec![Vec::new(); old_pages.len()];
    for (i, new_set) in new_refs.iter().enumerate() {
        let mut candidates: Vec<(usize, usize)> = old_refs
            .iter()
            .enumerate()
            .map(|(j, old_set)| (j, new_set.intersection(old_set).count()))
            .filter(|(_, overlap)| *overlap > 0)
            .collect();
        candidates.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| old_pages[a.0].slug.cmp(&old_pages[b.0].slug))
        });
        let best = candidates.first().map(|(j, _)| *j);
        if let Some(j) = best {
            claimants[j].push(i);
        }
        best_old.push(best);
    }

    let mut flows_to: Vec<Option<usize>> = vec![None; old_pages.len()];
    for (j, old_set) in old_refs.iter().enumerate() {
        let mut best: Option<usize> = None;
        for (i, new_set) in new_refs.iter().enumerate() {
            let overlap = new_set.intersection(old_set).count();
            if overlap == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some(current) => {
                    let current_overlap = old_set.intersection(&new_refs[current]).count();
                    overlap > current_overlap
                        || (overlap == current_overlap
                            && new_plan.pages[i].slug < new_plan.pages[current].slug)
                }
            };
            if better {
                best = Some(i);
            }
        }
        flows_to[j] = best;
    }

    MatchPass {
        best_old,
        claimants,
        flows_to,
    }
}

/// Identity continues only when the continuation covers BOTH sides: at least
/// half the old page's nodes survive into the new page AND at least half the
/// new page's nodes come from that old page (audit FIX-014: a 1-node overlap
/// must never carry a 100-node page's `WikiPageId`). Inclusive halves, integer
/// math; the overlap itself must be non-empty.
fn continuation_covers(
    old_set: &BTreeSet<KnowledgeNodeId>,
    new_set: &BTreeSet<KnowledgeNodeId>,
) -> bool {
    let intersection = new_set.intersection(old_set).count();
    intersection > 0 && 2 * intersection >= old_set.len() && 2 * intersection >= new_set.len()
}

/// Classifies one new plan page (§19.2 fixed order): split successors are
/// handled by the caller; the remaining classes are merge (≥2 flowing old
/// pages, dominant first), unchanged/modified (mutual best match), created.
fn classify_new_page(
    index: usize,
    page: &WikiPagePlan,
    refs: &RefSets<'_>,
    old_pages: &[GenerationPageView],
    pass: &MatchPass,
    split_primaries: &BTreeSet<usize>,
    diff: &mut PlanDiff,
) {
    let RefSets { old_refs, new_refs } = *refs;
    if pass.best_old[index].is_some_and(|primary| split_primaries.contains(&primary)) {
        return; // split successor: fresh plan id kept, recorded by the caller
    }
    // Old pages whose knowledge flows into THIS page (split primaries belong
    // to their split successors, never to a merge).
    let mut predecessors: Vec<usize> = (0..old_pages.len())
        .filter(|&j| pass.flows_to[j] == Some(index) && !split_primaries.contains(&j))
        .collect();
    // Dominant first: larger ref overlap, then old slug order.
    predecessors.sort_by(|&a, &b| {
        let overlap_a = new_refs[index].intersection(&old_refs[a]).count();
        let overlap_b = new_refs[index].intersection(&old_refs[b]).count();
        overlap_b
            .cmp(&overlap_a)
            .then_with(|| old_pages[a].slug.cmp(&old_pages[b].slug))
    });
    if predecessors.len() >= 2 && continuation_covers(&old_refs[predecessors[0]], &new_refs[index])
    {
        let dominant = predecessors[0];
        diff.merged.push(MergeOutcome {
            continued: ContinuedPage {
                index,
                page_id: old_pages[dominant].page_id.clone(),
                predecessor_id: old_pages[dominant].page_id.clone(),
                title: page.title.clone(),
                slug: page.slug.clone(),
            },
            predecessors: predecessors
                .iter()
                .map(|&j| old_pages[j].page_id.clone())
                .collect(),
        });
        return;
    }
    // Mutual best match with the primary, and the continuation covers both
    // sides: the identity continues. A weak-overlap match (the audit's
    // degenerate case) falls through to a fresh id instead.
    if let Some(primary) = pass.best_old[index] {
        if pass.flows_to[primary] == Some(index)
            && continuation_covers(&old_refs[primary], &new_refs[index])
        {
            let identical = new_refs[index] == old_refs[primary]
                && page.title.trim() == old_pages[primary].title.trim()
                && page.category == old_pages[primary].category;
            let entry = ContinuedPage {
                index,
                page_id: old_pages[primary].page_id.clone(),
                predecessor_id: old_pages[primary].page_id.clone(),
                title: page.title.clone(),
                slug: page.slug.clone(),
            };
            if identical {
                diff.unchanged.push(entry);
            } else {
                diff.modified.push(entry);
            }
            return;
        }
    }
    // No old page continues this new page: brand-new knowledge placement.
    diff.created.push(CreatedPage {
        index,
        page_id: page.id.clone(),
        title: page.title.clone(),
        slug: page.slug.clone(),
    });
}

/// Retirements: old pages neither kept (unchanged/modified), nor merge
/// predecessors, nor split primaries — with the refs no new page covers
/// (§19.2 ghost check).
fn collect_retirements(
    old_pages: &[GenerationPageView],
    old_refs: &[BTreeSet<KnowledgeNodeId>],
    new_refs: &[BTreeSet<KnowledgeNodeId>],
    split_primaries: &BTreeSet<usize>,
    diff: &PlanDiff,
) -> Vec<RetiredPage> {
    let kept: BTreeSet<WikiPageId> = diff
        .unchanged
        .iter()
        .chain(&diff.modified)
        .map(|entry| entry.predecessor_id.clone())
        .collect();
    let merged_predecessors: BTreeSet<WikiPageId> = diff
        .merged
        .iter()
        .flat_map(|merge| merge.predecessors.iter().cloned())
        .collect();
    let covered_refs: BTreeSet<KnowledgeNodeId> = new_refs.iter().flatten().cloned().collect();
    let mut retired = Vec::new();
    for (j, page) in old_pages.iter().enumerate() {
        if split_primaries.contains(&j)
            || kept.contains(&page.page_id)
            || merged_predecessors.contains(&page.page_id)
        {
            continue;
        }
        let lost_refs = old_refs[j]
            .iter()
            .filter(|node_id| !covered_refs.contains(*node_id))
            .cloned()
            .collect();
        retired.push(RetiredPage {
            page_id: page.page_id.clone(),
            slug: page.slug.clone(),
            lost_refs,
        });
    }
    retired
}

/// Classifies every new plan page against the current generation's pages
/// (deterministic: fixed iteration order, BTree tie-breaks — same input,
/// same classification).
///
/// Matching is ref-overlap based with the §19.2 tie-breaks (larger overlap
/// first, then slug order):
///
/// 1. each NEW page names its best old page (`best_old`, overlap > 0);
/// 2. each OLD page's knowledge flows to its best new page (`flows_to`);
/// 3. old pages named best by ≥ 2 new pages are SPLIT (fresh ids for every
///    claimant, retired with successors) — split wins over every other class
///    for those pages;
/// 4. new pages receiving the flow of ≥ 2 old pages MERGE (the DOMINANT
///    predecessor's id is kept, all predecessors recorded);
/// 5. remaining new pages whose best old page flows back to them are
///    UNCHANGED (identical refs set + title + category) or MODIFIED (keep
///    the id, recompile);
/// 6. everything else: created (new) / retired (old).
pub fn plan_diff(old_pages: &[GenerationPageView], new_plan: &WikiPlan) -> PlanDiff {
    let old_refs: Vec<BTreeSet<KnowledgeNodeId>> = old_pages
        .iter()
        .map(|page| page.knowledge_refs.iter().cloned().collect())
        .collect();
    let new_refs: Vec<BTreeSet<KnowledgeNodeId>> = new_plan
        .pages
        .iter()
        .map(|page| page.knowledge_refs.iter().cloned().collect())
        .collect();

    let pass = match_pass(&old_refs, &new_refs, old_pages, new_plan);
    let split_primaries: BTreeSet<usize> = pass
        .claimants
        .iter()
        .enumerate()
        .filter(|(_, claim)| claim.len() >= 2)
        .map(|(j, _)| j)
        .collect();

    let mut diff = PlanDiff::default();

    // Split outcomes: plan-order successors per split primary.
    for (j, claim) in pass.claimants.iter().enumerate() {
        if claim.len() >= 2 {
            diff.split.push(SplitOutcome {
                retired: old_pages[j].page_id.clone(),
                slug: old_pages[j].slug.clone(),
                successors: claim
                    .iter()
                    .map(|&i| new_plan.pages[i].id.clone())
                    .collect(),
            });
        }
    }

    // Merge / unchanged / modified / created per new page (plan order).
    for (i, page) in new_plan.pages.iter().enumerate() {
        classify_new_page(
            i,
            page,
            &RefSets {
                old_refs: &old_refs,
                new_refs: &new_refs,
            },
            old_pages,
            &pass,
            &split_primaries,
            &mut diff,
        );
    }

    diff.retired = collect_retirements(old_pages, &old_refs, &new_refs, &split_primaries, &diff);

    diff
}

// ---------------------------------------------------------------------------
// Replan report
// ---------------------------------------------------------------------------

/// Successful end of a replan (dry-run audit or executed replan); the CLI
/// prints this (PRD §19.2: `BuildReport`-style output).
#[derive(Debug, Clone)]
pub struct ReplanReport {
    pub dry_run: bool,
    /// The fresh plan matches the current generation page-for-page: nothing
    /// to compile or publish ("no replan needed", exit 0).
    pub no_replan_needed: bool,
    /// WHY a replan may be needed: recorded `replan-required` decision rows,
    /// fingerprint drift and the diff class counts (PRD §19.2).
    pub triggers: Vec<String>,
    pub unchanged: usize,
    pub modified: usize,
    pub merged: usize,
    pub split: usize,
    pub created: usize,
    pub retired: usize,
    /// Per-class page lists for the dry-run audit (slugs).
    pub unchanged_pages: Vec<String>,
    pub modified_pages: Vec<String>,
    /// (new page slug, absorbed predecessor slugs — dominant first)
    pub merged_pages: Vec<(String, Vec<String>)>,
    /// (old slug, successor slugs)
    pub split_pages: Vec<(String, Vec<String>)>,
    pub created_pages: Vec<String>,
    /// (old slug, refs no new page covers)
    pub retired_pages: Vec<(String, Vec<String>)>,
    /// §19.2 ghost-check warnings: retired pages whose refs no new page
    /// covers. Dry-run surfaces them as warnings; execute proceeds and the
    /// decision row records them (PRD §19.2: correctness first).
    pub knowledge_loss_warnings: Vec<String>,
    /// Dry-run cost estimate: compile LLM calls the execute would spend.
    pub estimated_compile_llm_calls: u64,
    /// Dry-run cost upper bound: calls × per-page output ceiling
    /// (`MAX_PAGE_OUTPUT_TOKENS`, PRD §19.2).
    pub estimated_max_output_tokens: u64,
    // ---- Present after an executed replan (dry_run = false) ----
    pub build_id: Option<BuildId>,
    pub sources: usize,
    pub pages: usize,
    pub citations: usize,
    pub links: usize,
    pub recompiled: usize,
    pub carried: usize,
    pub llm_request_count: u32,
    pub cache: CacheStats,
    pub published_path: Option<PathBuf>,
    /// Human-readable summary of publish recovery performed first, if any.
    pub recovery: Option<String>,
}

fn empty_report(dry_run: bool, cache: CacheStats, recovery: Option<String>) -> ReplanReport {
    ReplanReport {
        dry_run,
        no_replan_needed: false,
        triggers: Vec::new(),
        unchanged: 0,
        modified: 0,
        merged: 0,
        split: 0,
        created: 0,
        retired: 0,
        unchanged_pages: Vec::new(),
        modified_pages: Vec::new(),
        merged_pages: Vec::new(),
        split_pages: Vec::new(),
        created_pages: Vec::new(),
        retired_pages: Vec::new(),
        knowledge_loss_warnings: Vec::new(),
        estimated_compile_llm_calls: 0,
        estimated_max_output_tokens: 0,
        build_id: None,
        sources: 0,
        pages: 0,
        citations: 0,
        links: 0,
        recompiled: 0,
        carried: 0,
        llm_request_count: 0,
        cache,
        published_path: None,
        recovery,
    }
}

/// Why the workspace may need a re-plan: recorded `replan-required` decision
/// rows, fingerprint drift vs the last COMPLETED build, and the diff counts.
fn collect_triggers(
    conn: &rusqlite::Connection,
    env: &PipelineEnv,
    diff: &PlanDiff,
) -> Result<Vec<String>> {
    let mut triggers = Vec::new();
    let pending = list_recent_plan_decisions_by_outcome(conn, OUTCOME_REPLAN_REQUIRED, 10)?;
    if !pending.is_empty() {
        triggers.push(format!(
            "{} recorded 'replan-required' decision row(s); latest: {}",
            pending.len(),
            pending[0].notes
        ));
    }
    if let Some(previous) = latest_completed_build(conn)? {
        let drifted = match &previous.build_fingerprint {
            Some(previous_fp) => *previous_fp != env.fingerprint_json,
            // A completed build without a fingerprint predates §19 auditing:
            // plan compatibility cannot be proven.
            None => true,
        };
        if drifted {
            triggers.push(format!(
                "build fingerprint differs from the last completed build {} (prompts, schema, parser, model or effective config changed)",
                previous.build_id
            ));
        }
    }
    triggers.push(format!(
        "plan diff: {} unchanged, {} modified, {} merged, {} split, {} created, {} retired",
        diff.unchanged.len(),
        diff.modified.len(),
        diff.merged.len(),
        diff.split.len(),
        diff.created.len(),
        diff.retired.len()
    ));
    Ok(triggers)
}

/// Dry-run audit row (PRD §19.2 step 5): anchored to the CURRENT active build
/// because `plan_decisions.build_id` is a NOT NULL foreign key and a dry-run
/// must not create a §31 build row — the audit records "this judgment was
/// made against the generation of build X".
fn record_dry_run_decision(
    conn: &mut rusqlite::Connection,
    active_build: &BuildId,
    report: &ReplanReport,
) -> Result<()> {
    let notes = if report.no_replan_needed {
        "dry-run: no replan needed — the fresh plan matches the current generation".to_owned()
    } else {
        format!(
            "dry-run: {} unchanged, {} modified, {} merged, {} split, {} created, {} retired; estimated compile llm calls {}, cost upper bound {} tokens",
            report.unchanged,
            report.modified,
            report.merged,
            report.split,
            report.created,
            report.retired,
            report.estimated_compile_llm_calls,
            report.estimated_max_output_tokens
        )
    };
    record_decision(
        conn,
        active_build,
        None,
        OUTCOME_REPLAN_DRY_RUN,
        None,
        report.estimated_compile_llm_calls as usize,
        notes,
    )
}

// ---------------------------------------------------------------------------
// Orchestration (PRD §19.2 steps 1–7)
// ---------------------------------------------------------------------------

/// Runs the explicit global re-plan. `dry_run` audits the plan diff and cost
/// WITHOUT compiling, generating or publishing; the bare command executes the
/// replan and publishes through the unchanged §35 contract. The explicit
/// command IS the user confirmation (PRD §19.2).
pub async fn replan(
    workspace_root: &Path,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    dry_run: bool,
) -> Result<ReplanReport> {
    config.validate()?;
    let root = lexical_absolute(workspace_root, &config.source.root);
    let wiki_dir = lexical_absolute(workspace_root, &config.project.wiki_dir);

    let state_dir = workspace_root.join(".llm-wiki");
    std::fs::create_dir_all(&state_dir)
        .map_err(|e| WikiError::Source(format!("cannot create state dir: {e}")))?;
    let mut conn = open(&state_dir.join("state.db"))?;

    // Preconditions identical to build (§35): stale builds from a dead
    // process are INTERRUPTED and a possibly-interrupted publish is resolved
    // before anything new starts.
    let stale = mark_stale_builds_interrupted(&mut conn)?;
    if stale > 0 {
        tracing::warn!(count = stale, "marked stale builds INTERRUPTED");
    }
    let recovery = recover_if_needed(&mut conn, &wiki_dir)?;
    let recovery_note = recovery.map(|report| report.detail);

    // Nothing published → nothing to diff the fresh plan against.
    let Some(active_build) = get_active_build_id(&conn)? else {
        return Err(WikiError::Config(
            "nothing to replan; run build first (a replan diffs the fresh plan against the CURRENT generation)".into(),
        ));
    };

    // Shared pipeline environment: identical prompts/config-hash/fingerprint
    // and the same §28 cache as `build`, so a dry-run warms the cache the
    // execute hits (PRD §28: planning is paid for once across the pair).
    let env = prepare_pipeline_env(&state_dir, config, &provider)?;

    // The executed replan owns a §31 build row from the start (stages are
    // persisted; failures mark it terminal). A dry-run creates NO build row.
    let build_id = if dry_run {
        None
    } else {
        let build_id = start_build(
            &mut conn,
            &BuildDraft {
                model: Some(provider.model().to_owned()),
                prompt_version: Some(env.prompt_version.clone()),
                compiler_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
                schema_version: Some(crate::build::SCHEMA_VERSION.to_owned()),
                config_hash: Some(env.config_hash.clone()),
                build_fingerprint: Some(env.fingerprint_json.clone()),
                ..BuildDraft::default()
            },
        )?;
        tracing::info!(build = %build_id, "replan started");
        update_build_status(&mut conn, &build_id, "SCANNING")?;
        Some(build_id)
    };

    let result = replan_inner(
        &mut conn,
        &root,
        &wiki_dir,
        &active_build,
        build_id.as_ref(),
        &env,
        config,
        &provider,
        dry_run,
        recovery_note.clone(),
    )
    .await;

    match result {
        Ok(report) => Ok(report),
        Err(err) => {
            if let Some(build_id) = &build_id {
                let status = terminal_status_for(&err);
                let marked = finish_build(&mut conn, build_id, status, None, None);
                if let Err(mark_err) = marked {
                    tracing::error!(
                        build = %build_id,
                        error = %mark_err,
                        "could not mark replan build terminal"
                    );
                }
            }
            tracing::warn!(dry_run, error = %err, "replan failed");
            Err(err)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn replan_inner(
    conn: &mut rusqlite::Connection,
    root: &Path,
    wiki_dir: &Path,
    active_build: &BuildId,
    build_id: Option<&BuildId>,
    env: &PipelineEnv,
    config: &Config,
    provider: &Arc<dyn LlmProvider>,
    dry_run: bool,
    recovery_note: Option<String>,
) -> Result<ReplanReport> {
    // ---- Scan (§8) + §19.1 ChangeSet + §19.3 deletions + registry upsert:
    // identical machinery to `build`, so cache keys and the knowledge state
    // agree across the two commands (PRD §28). ----
    let (output, file_outcomes, deleted_ids, upserted, sources) =
        replan_sync_sources(conn, root, wiki_dir, build_id, env, config)?;

    // EPIC A PR2: sources whose chunk rows this replan rewrites (added +
    // modified) or retires (deleted) — `carry_source_chunks` excludes them so
    // the publish-time source_fts rebuild serves exactly this build's rows.
    let mut carry_exclude: std::collections::BTreeSet<SourceId> =
        deleted_ids.iter().cloned().collect();
    for (outcome, (source_id, _created)) in file_outcomes.iter().zip(&upserted) {
        if matches!(outcome, FileOutcome::Added | FileOutcome::Modified(_)) {
            carry_exclude.insert(source_id.clone());
        }
    }
    let carry_exclude: Vec<SourceId> = carry_exclude.into_iter().collect();

    // ---- Selective re-analysis (§19.2): added + modified sources only, so
    // the fresh plan reflects current knowledge. ----
    let mut llm_request_count = replan_analyze_pending(
        conn,
        root,
        build_id,
        env,
        config,
        provider,
        &output,
        &upserted,
        &file_outcomes,
    )
    .await?;

    // ---- FRESH planning (§19.2 step 3): the plan-identity short-circuit is
    // DISABLED so the plan is re-derived even when the reconciliation key
    // still matches. The request-level §28 cache stays intact: a dry-run
    // warms it, and the follow-up execute replans with 0 NEW planning
    // requests (planning is paid for once across the pair). ----
    if let Some(build_id) = build_id {
        update_build_status(conn, build_id, "PLANNING")?;
    }
    let stage_cache: Arc<dyn crate::cache::StageCache> = env.cache.clone();
    let (base, registry_revision) = load_plan_input(conn)?;
    let planner_config = PlannerConfig {
        hierarchical: config.planning.hierarchical,
        max_cluster_nodes: config.planning.max_cluster_nodes as usize,
        max_plan_input_tokens: config.analysis.max_plan_input_tokens as u64,
        max_output_tokens: config.llm.max_output_tokens,
        max_concurrency: config.llm.max_concurrency.max(1) as usize,
        ..PlannerConfig::default()
    };
    let planner = WikiPlanner::new(
        provider.clone(),
        env.planning_prompt.clone(),
        planner_config,
    )
    .with_plan_cache(
        stage_cache.clone(),
        plan_cache_identity(
            &env.config_hash,
            provider.model(),
            crate::build::SCHEMA_VERSION,
        ),
    )
    .with_force_fresh(true);
    let plan_outcome = planner.plan(&base, registry_revision).await?;
    llm_request_count += plan_outcome.llm_request_count;

    // ---- Stable-ID plan diff vs the CURRENT generation (§19.2 step 4). ----
    let old_pages = load_generation_view(conn, active_build)?;
    let diff = plan_diff(&old_pages, &plan_outcome.plan);

    let tables = ReplanTables::of(&diff, old_pages.as_slice(), &plan_outcome.plan);
    let knowledge_loss_warnings = knowledge_loss_warnings(&diff);

    if diff.is_empty() {
        // Empty diff → "no replan needed", exit 0, in BOTH modes (PRD §19.2).
        // The dry-run still records its audit row; the executed replan marks
        // its build row CANCELLED (started, nothing to do, nothing published).
        let triggers = collect_triggers(conn, env, &diff)?;
        let mut report = empty_report(dry_run, env.cache.stats(), recovery_note);
        report.no_replan_needed = true;
        report.triggers = triggers;
        report.unchanged = diff.unchanged.len();
        report.unchanged_pages = tables.unchanged;
        if dry_run {
            record_dry_run_decision(conn, active_build, &report)?;
        } else if let Some(build_id) = build_id {
            finish_build(conn, build_id, "CANCELLED", None, None)?;
        }
        tracing::info!(dry_run, "replan found nothing to change");
        return Ok(report);
    }

    // ---- Dry-run report (§19.2 step 5): NO compiler calls, NO generation,
    // NO publish, NO build row. ----
    if dry_run {
        let triggers = collect_triggers(conn, env, &diff)?;
        let estimated_calls = diff.recompile_count() as u64;
        let mut report = empty_report(true, env.cache.stats(), recovery_note);
        report.triggers = triggers;
        report.unchanged = diff.unchanged.len();
        report.modified = diff.modified.len();
        report.merged = diff.merged.len();
        report.split = diff.split.len();
        report.created = diff.created.len();
        report.retired = diff.retired.len();
        report.unchanged_pages = tables.unchanged;
        report.modified_pages = tables.modified;
        report.merged_pages = tables.merged;
        report.split_pages = tables.split;
        report.created_pages = tables.created;
        report.retired_pages = tables.retired;
        report.knowledge_loss_warnings = knowledge_loss_warnings;
        report.estimated_compile_llm_calls = estimated_calls;
        report.estimated_max_output_tokens = estimated_calls * u64::from(MAX_PAGE_OUTPUT_TOKENS);
        record_dry_run_decision(conn, active_build, &report)?;
        tracing::info!(
            unchanged = report.unchanged,
            modified = report.modified,
            merged = report.merged,
            split = report.split,
            created = report.created,
            retired = report.retired,
            "replan dry-run audited"
        );
        return Ok(report);
    }

    replan_execute(
        conn,
        wiki_dir,
        active_build,
        build_id.expect("execute mode owns a build row"),
        env,
        config,
        provider,
        diff,
        plan_outcome.plan,
        llm_request_count,
        sources,
        tables,
        knowledge_loss_warnings,
        recovery_note,
        carry_exclude,
    )
    .await
}

/// Execute mode (§19.2 step 6): compile ALL non-unchanged pages, carry
/// unchanged pages byte-identical, write the §45 page-identity map and
/// publish through the unchanged §35 contract.
#[allow(clippy::too_many_arguments)]
async fn replan_execute(
    conn: &mut rusqlite::Connection,
    wiki_dir: &Path,
    active_build: &BuildId,
    build_id: &BuildId,
    env: &PipelineEnv,
    config: &Config,
    provider: &Arc<dyn LlmProvider>,
    diff: PlanDiff,
    plan: WikiPlan,
    mut llm_request_count: u32,
    sources: usize,
    tables: ReplanTables,
    knowledge_loss_warnings: Vec<String>,
    recovery_note: Option<String>,
    carry_exclude: Vec<SourceId>,
) -> Result<ReplanReport> {
    let new_kb = load_knowledge_base(conn)?;
    let (remapped_plan, recompile) = replan_remap_plan(&diff, plan);

    update_build_status(conn, build_id, "COMPILING")?;
    let stage_cache: Arc<dyn crate::cache::StageCache> = env.cache.clone();
    let compiler = WikiCompiler::new(
        provider.clone(),
        env.compilation_prompt.clone(),
        CompilerConfig {
            max_input_tokens: config.analysis.max_input_tokens as u64,
            min_output_tokens: config.llm.max_output_tokens,
            ..CompilerConfig::default()
        },
        config.llm.max_concurrency as usize,
    )
    .with_cache(stage_cache);
    let compiled = compiler
        .compile_plan_subset(&remapped_plan, &new_kb, build_id, &recompile)
        .await?;
    llm_request_count += compiled.llm_request_count;
    if compiled.pages.len() != recompile.len() {
        return Err(WikiError::Compilation(format!(
            "replan partial compile produced {} page(s) for {} affected page(s)",
            compiled.pages.len(),
            recompile.len()
        )));
    }

    // Final page set: compiled records + carried rows copied VERBATIM from
    // the current generation (frontmatter keeps the ORIGINAL build id; link
    // rows are filtered to surviving pages for FK, like §19).
    let new_pages = replan_assemble_pages(conn, active_build, &remapped_plan, &compiled)?;

    // Decision row AFTER the final check (compile succeeded), mirroring the
    // incremental contract (§19.2: one final judgment row per build).
    let notes = format!(
        "executed: {} unchanged (carried), {} modified, {} merged, {} split, {} created, {} retired; {} recompiled, {} carried",
        diff.unchanged.len(),
        diff.modified.len(),
        diff.merged.len(),
        diff.split.len(),
        diff.created.len(),
        diff.retired.len(),
        recompile.len(),
        diff.unchanged.len()
    );
    record_decision(
        conn,
        build_id,
        None,
        OUTCOME_REPLAN_EXECUTED,
        None,
        recompile.len(),
        if knowledge_loss_warnings.is_empty() {
            notes
        } else {
            format!(
                "{notes}; knowledge-loss: {}",
                knowledge_loss_warnings.join(" | ")
            )
        },
    )?;

    // Index (§31) + stable page-identity rows (§45) + publish (§35).
    update_build_status(conn, build_id, "INDEXING")?;
    // EPIC A PR2: same carry contract as the incremental build — only the
    // changed sources re-staged during analysis; the rest copy forward from
    // the active build so the publish-time source_fts rebuild (inside the
    // activate transaction) keeps them searchable.
    {
        let tx = conn
            .transaction()
            .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
        let carried = carry_source_chunks(&tx, active_build, build_id, &carry_exclude)?;
        tx.commit()
            .map_err(|e| WikiError::Storage(format!("commit carry_source_chunks: {e}")))?;
        tracing::info!(build = %build_id, carried, "source chunks carried forward");
    }
    let stats = persist_generation(conn, build_id, &new_pages)?;

    // page_id_map (§45): written AFTER persist_generation so successor
    // references resolve, BEFORE publish so a publish failure never leaves
    // an error path after the pointer moved.
    replan_write_map_rows(conn, build_id, &diff)?;

    let published = crate::publish::publish(
        conn,
        wiki_dir,
        build_id,
        &new_pages,
        config.build.keep_generations,
    )?;

    tracing::info!(
        build = %build_id,
        recompiled = recompile.len(),
        carried = diff.unchanged.len(),
        "replan published"
    );
    Ok(ReplanReport {
        dry_run: false,
        no_replan_needed: false,
        triggers: collect_triggers(conn, env, &diff)?,
        unchanged: diff.unchanged.len(),
        modified: diff.modified.len(),
        merged: diff.merged.len(),
        split: diff.split.len(),
        created: diff.created.len(),
        retired: diff.retired.len(),
        unchanged_pages: tables.unchanged,
        modified_pages: tables.modified,
        merged_pages: tables.merged,
        split_pages: tables.split,
        created_pages: tables.created,
        retired_pages: tables.retired,
        knowledge_loss_warnings,
        estimated_compile_llm_calls: 0,
        estimated_max_output_tokens: 0,
        build_id: Some(build_id.clone()),
        sources,
        pages: stats.pages,
        citations: stats.citations,
        links: stats.links,
        recompiled: recompile.len(),
        carried: diff.unchanged.len(),
        llm_request_count,
        cache: env.cache.stats(),
        published_path: Some(published.published_path),
        recovery: recovery_note,
    })
}

/// Final ids per plan page: predecessors' stable ids for
/// unchanged/modified/merge-dominant pages, fresh planner ids otherwise;
/// related-page links are remapped to the final ids.
fn replan_remap_plan(diff: &PlanDiff, plan: WikiPlan) -> (WikiPlan, BTreeSet<WikiPageId>) {
    let mut final_ids: Vec<WikiPageId> = plan.pages.iter().map(|page| page.id.clone()).collect();
    for entry in diff.unchanged.iter().chain(&diff.modified) {
        final_ids[entry.index] = entry.page_id.clone();
    }
    for merge in &diff.merged {
        final_ids[merge.continued.index] = merge.continued.page_id.clone();
    }
    let plan_id_to_final: BTreeMap<String, WikiPageId> = plan
        .pages
        .iter()
        .enumerate()
        .map(|(i, page)| (page.id.as_str().to_owned(), final_ids[i].clone()))
        .collect();
    let pages: Vec<WikiPagePlan> = plan
        .pages
        .iter()
        .enumerate()
        .map(|(i, page)| {
            let mut page = page.clone();
            page.id = final_ids[i].clone();
            page.related_pages = page
                .related_pages
                .iter()
                .map(|related| {
                    plan_id_to_final
                        .get(related.as_str())
                        .cloned()
                        .unwrap_or_else(|| related.clone())
                })
                .collect();
            page
        })
        .collect();

    let mut recompile: BTreeSet<WikiPageId> = BTreeSet::new();
    for entry in &diff.modified {
        recompile.insert(entry.page_id.clone());
    }
    for merge in &diff.merged {
        recompile.insert(merge.continued.page_id.clone());
    }
    for split in &diff.split {
        for successor in &split.successors {
            recompile.insert(successor.clone());
        }
    }
    for created in &diff.created {
        recompile.insert(created.page_id.clone());
    }
    (WikiPlan { pages }, recompile)
}

/// Final page set: compiled records + unchanged rows carried VERBATIM from
/// the current generation (frontmatter keeps the ORIGINAL build id; link
/// rows are filtered to surviving pages for FK, like §19).
fn replan_assemble_pages(
    conn: &rusqlite::Connection,
    active_build: &BuildId,
    remapped_plan: &WikiPlan,
    compiled: &crate::compile::CompiledGeneration,
) -> Result<Vec<WikiPageRecord>> {
    let compiled_by_id: BTreeMap<String, &WikiPageRecord> = compiled
        .pages
        .iter()
        .map(|page| (page.page_id.as_str().to_owned(), page))
        .collect();
    let old_pages = load_generation_view(conn, active_build)?;
    let carried_by_id: BTreeMap<String, &GenerationPageView> = old_pages
        .iter()
        .map(|page| (page.page_id.as_str().to_owned(), page))
        .collect();
    let language_of: BTreeMap<String, String> = load_generation_pages(conn, active_build)?
        .into_iter()
        .map(|page| (page.page_id.as_str().to_owned(), page.language))
        .collect();
    let surviving: BTreeSet<WikiPageId> = remapped_plan
        .pages
        .iter()
        .map(|page| page.id.clone())
        .collect();

    let mut new_pages: Vec<WikiPageRecord> = Vec::with_capacity(remapped_plan.pages.len());
    let mut slugs: BTreeSet<String> = BTreeSet::new();
    for page in &remapped_plan.pages {
        if !slugs.insert(page.slug.clone()) {
            return Err(WikiError::Compilation(format!(
                "replan produced duplicate slug '{}'; slugs must be unique within a generation",
                page.slug
            )));
        }
        if let Some(record) = compiled_by_id.get(page.id.as_str()) {
            new_pages.push((*record).clone());
            continue;
        }
        // Unchanged: carried byte-identical from the current generation.
        let previous = carried_by_id.get(page.id.as_str()).ok_or_else(|| {
            WikiError::Compilation(format!(
                "page '{}' ({}) is unchanged but has no current-generation row to carry",
                page.slug, page.id
            ))
        })?;
        new_pages.push(WikiPageRecord {
            page_id: previous.page_id.clone(),
            slug: previous.slug.clone(),
            title: previous.title.clone(),
            category: previous.category.clone(),
            language: language_of
                .get(previous.page_id.as_str())
                .cloned()
                .unwrap_or_else(|| "und".to_owned()),
            body_hash: previous.body_hash.clone(),
            content: previous.content.clone(),
            knowledge_refs: previous.knowledge_refs.clone(),
            citations: previous.citations.clone(),
            links: previous
                .links
                .iter()
                .filter(|link| surviving.contains(&link.to_page_id))
                .cloned()
                .collect(),
        });
    }
    if new_pages.is_empty() {
        return Err(WikiError::Compilation(
            "every page was retired by the replan; refusing to publish an empty generation".into(),
        ));
    }
    Ok(new_pages)
}

/// §45 page-identity rows: keep / merge (dominant-first, pred == succ) /
/// split / retire — written AFTER persist_generation so successor references
/// resolve, BEFORE publish so a publish failure never leaves an error path
/// after the pointer moved.
fn replan_write_map_rows(
    conn: &mut rusqlite::Connection,
    build_id: &BuildId,
    diff: &PlanDiff,
) -> Result<()> {
    let mut map_rows: Vec<PageIdMapRow> = Vec::new();
    for entry in diff.unchanged.iter().chain(&diff.modified) {
        map_rows.push(PageIdMapRow {
            predecessor_page_id: Some(entry.page_id.clone()),
            successor_page_id: Some(entry.page_id.clone()),
            kind: MAP_KIND_KEEP.to_owned(),
            build_id: build_id.clone(),
        });
    }
    for merge in &diff.merged {
        // One row per predecessor, DOMINANT FIRST (predecessors[0] is the
        // dominant one; its row has predecessor == successor, §45).
        for (position, predecessor) in merge.predecessors.iter().enumerate() {
            map_rows.push(PageIdMapRow {
                predecessor_page_id: Some(predecessor.clone()),
                successor_page_id: Some(if position == 0 {
                    predecessor.clone()
                } else {
                    merge.continued.page_id.clone()
                }),
                kind: MAP_KIND_MERGE.to_owned(),
                build_id: build_id.clone(),
            });
        }
    }
    for split in &diff.split {
        for successor in &split.successors {
            map_rows.push(PageIdMapRow {
                predecessor_page_id: Some(split.retired.clone()),
                successor_page_id: Some(successor.clone()),
                kind: MAP_KIND_SPLIT.to_owned(),
                build_id: build_id.clone(),
            });
        }
    }
    for retired in &diff.retired {
        map_rows.push(PageIdMapRow {
            predecessor_page_id: Some(retired.page_id.clone()),
            successor_page_id: None,
            kind: MAP_KIND_RETIRE.to_owned(),
            build_id: build_id.clone(),
        });
    }
    insert_page_id_maps(conn, &map_rows)?;
    Ok(())
}

/// Scan (§8) + §19.1 ChangeSet (BEFORE the upsert overwrites hashes) +
/// §19.3 deletions + registry upsert. The retirement mutates the knowledge
/// registry in BOTH modes — the visible wiki is only ever touched by the
/// §35 publish of the execute (PRD §19.2).
#[allow(clippy::type_complexity)]
fn replan_sync_sources(
    conn: &mut rusqlite::Connection,
    root: &Path,
    wiki_dir: &Path,
    build_id: Option<&BuildId>,
    env: &PipelineEnv,
    config: &Config,
) -> Result<(
    ScanOutput,
    Vec<FileOutcome>,
    Vec<SourceId>,
    Vec<(SourceId, bool)>,
    usize,
)> {
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
    if let Some(build_id) = build_id {
        llm_wiki_storage::set_build_snapshot_hash(conn, build_id, &manifest.snapshot_hash())?;
    }
    // Cache rows record the source snapshot hash so cross-snapshot reuse is
    // impossible (§28) — and so the execute hits the dry-run's warm entries.
    env.cache
        .set_source_snapshot_hash(&manifest.snapshot_hash());

    let registry_before: Vec<SourceRecord> = list_sources(conn)?;
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

    // §19.3 deleted sources: replan after deletion is a legitimate
    // structural change (PRD §19.2).
    for source_id in &deleted_ids {
        let record = registry_before
            .iter()
            .find(|record| &record.source_id == source_id)
            .ok_or_else(|| {
                WikiError::Storage(format!(
                    "deleted source {source_id} missing from the registry"
                ))
            })?;
        let retired = retire_source_knowledge(conn, source_id, build_id.map(|b| b.as_str()))?;
        mark_removed(conn, &record.locator_key)?;
        tracing::info!(
            source = %record.rel_path,
            claims = retired.retired_claims,
            nodes_retired = retired.retired_registry_nodes.len(),
            "replan retired deleted source"
        );
    }

    // Registry upsert: refresh hashes/paths, mint ids for added files.
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
    let upserted = upsert_sources_batch(conn, &batch, build_id.map(|b| b.as_str()))?;
    let sources = upserted.len();
    if sources == 0 {
        return Err(WikiError::Source(format!(
            "no markdown sources found under {}; nothing to replan",
            root.display()
        )));
    }
    Ok((output, file_outcomes, deleted_ids, upserted, sources))
}

/// Selective re-analysis (§19.2): added + modified sources only, so the
/// fresh plan reflects current knowledge. Returns the LLM requests spent.
#[allow(clippy::too_many_arguments)]
async fn replan_analyze_pending(
    conn: &mut rusqlite::Connection,
    root: &Path,
    build_id: Option<&BuildId>,
    env: &PipelineEnv,
    config: &Config,
    provider: &Arc<dyn LlmProvider>,
    output: &ScanOutput,
    upserted: &[(SourceId, bool)],
    file_outcomes: &[FileOutcome],
) -> Result<u32> {
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

    if let Some(build_id) = build_id {
        update_build_status(conn, build_id, "PARSING")?;
    }
    let stage_cache: Arc<dyn crate::cache::StageCache> = env.cache.clone();
    let mut parsed = Vec::with_capacity(changed_files.len());
    for index in changed_files {
        let file = &output.files[index];
        let (source_id, _created) = upserted[index].clone();
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

    if let Some(build_id) = build_id {
        update_build_status(conn, build_id, "ANALYZING")?;
    }
    let analyzer = Arc::new(
        DocumentAnalyzer::new(
            provider.clone(),
            env.analysis_prompt.clone(),
            config.analysis.section_target_tokens,
            config.analysis.max_rejected_claim_ratio,
            config.llm.max_output_tokens,
            config.llm.max_concurrency,
        )
        .with_cache(stage_cache),
    );
    // Document-parallel analysis, same contract as build_full_pipeline
    // (audit Phase 2: replan's changed-source analysis must match the build
    // pipeline's concurrency — it previously ran the documents serially).
    // SQLite work stays on this task; only the LLM calls run inside the
    // bounded JoinSet window, and outcomes persist in document order so the
    // knowledge state stays deterministic.
    let mut llm_request_count = 0u32;
    let mut doc_inputs = Vec::with_capacity(parsed.len());
    for (file, source_id, parsed_doc) in &parsed {
        let sections = register_sections(conn, source_id, &parsed_doc.sections, build_id)?;
        // EPIC A PR2: the execute build stages the changed sources' retrieval
        // chunks here (the unchanged ones carry forward before publish); a
        // dry-run owns NO build artifacts and stages nothing.
        if let Some(build_id) = build_id {
            write_source_chunks(
                conn,
                source_id,
                build_id,
                &file.rel_path,
                Some(&parsed_doc.language),
                &parsed_doc.sections,
                config.analysis.section_target_tokens,
            )?;
        }
        doc_inputs.push(AnalyzedDocument {
            source_id: source_id.clone(),
            rel_path: file.rel_path.clone(),
            content_hash: file.content_hash.clone(),
            language: parsed_doc.language.clone(),
            sections,
        });
    }
    let doc_concurrency = config.llm.max_concurrency.max(1) as usize;
    let mut outcomes: Vec<(usize, AnalysisOutcome)> = Vec::with_capacity(doc_inputs.len());
    let mut next_to_spawn = 0usize;
    let mut join_set = tokio::task::JoinSet::new();
    while next_to_spawn < doc_inputs.len() || !join_set.is_empty() {
        while next_to_spawn < doc_inputs.len() && join_set.len() < doc_concurrency {
            let doc = doc_inputs[next_to_spawn].clone();
            let build_id_owned = build_id.cloned();
            let index = next_to_spawn;
            let doc_analyzer = Arc::clone(&analyzer);
            join_set.spawn(async move {
                let outcome = doc_analyzer
                    .clone()
                    .analyze_document(&doc, build_id_owned.as_ref())
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
    for ((index, outcome), doc) in outcomes.iter().zip(doc_inputs.iter()) {
        debug_assert_eq!(doc_inputs[*index].rel_path, doc.rel_path);
        llm_request_count += outcome.llm_request_count;
        crate::persist::persist_outcome(
            conn,
            doc,
            outcome,
            &crate::persist::PersistOptions {
                build_id: build_id.cloned(),
                model: Some(provider.model().to_owned()),
                prompt_version: Some(env.prompt_version.clone()),
                replace_source: true,
            },
        )?;
    }
    Ok(llm_request_count)
}

/// Per-class page tables for the dry-run/execute reports (identical in both
/// modes): slugs only, deterministic order.
struct ReplanTables {
    unchanged: Vec<String>,
    modified: Vec<String>,
    merged: Vec<(String, Vec<String>)>,
    split: Vec<(String, Vec<String>)>,
    created: Vec<String>,
    retired: Vec<(String, Vec<String>)>,
}

impl ReplanTables {
    fn of(diff: &PlanDiff, old_pages: &[GenerationPageView], plan: &WikiPlan) -> Self {
        Self {
            unchanged: diff.unchanged.iter().map(|e| e.slug.clone()).collect(),
            modified: diff.modified.iter().map(|e| e.slug.clone()).collect(),
            merged: diff
                .merged
                .iter()
                .map(|merge| {
                    (
                        merge.continued.slug.clone(),
                        old_slugs_of(&merge.predecessors, old_pages),
                    )
                })
                .collect(),
            split: diff
                .split
                .iter()
                .map(|split| (split.slug.clone(), plan_slugs_of(&split.successors, plan)))
                .collect(),
            created: diff.created.iter().map(|page| page.slug.clone()).collect(),
            retired: diff
                .retired
                .iter()
                .map(|page| {
                    (
                        page.slug.clone(),
                        page.lost_refs
                            .iter()
                            .map(|id| id.as_str().to_owned())
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

fn old_slugs_of(ids: &[WikiPageId], old_pages: &[GenerationPageView]) -> Vec<String> {
    ids.iter()
        .filter_map(|id| {
            old_pages
                .iter()
                .find(|page| &page.page_id == id)
                .map(|page| page.slug.clone())
        })
        .collect()
}

fn plan_slugs_of(ids: &[WikiPageId], plan: &WikiPlan) -> Vec<String> {
    ids.iter()
        .filter_map(|id| {
            plan.pages
                .iter()
                .find(|page| &page.id == id)
                .map(|page| page.slug.clone())
        })
        .collect()
}

/// Warnings for retired pages whose refs no new page covers (§19.2 ghost
/// check): either the underlying claims retired with their deleted sources
/// (legitimate) or the knowledge is gone — recorded on the decision row.
fn knowledge_loss_warnings(diff: &PlanDiff) -> Vec<String> {
    diff.retired
        .iter()
        .filter(|page| !page.lost_refs.is_empty())
        .map(|page| {
            format!(
                "retired page '{}' ({}) has knowledge no new page covers: {}",
                page.slug,
                page.page_id,
                page.lost_refs
                    .iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_core::ids::SourceId;
    use llm_wiki_storage::{PageCitationRecord, PageLinkRecord};

    const C1: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
    const C2: &str = "01BX5ZZKBKACTAV9WEVGEMMVRZ";
    const C3: &str = "01CZZZZZZZZZZZZZZZZZZZZZZZ";
    const C4: &str = "01DZZZZZZZZZZZZZZZZZZZZZZZ";
    const C9: &str = "01EZZZZZZZZZZZZZZZZZZZZZZZ";

    fn node(tag: &str) -> KnowledgeNodeId {
        KnowledgeNodeId::from_validated(format!("kn_{tag}"))
    }

    fn refs(tags: &[&str]) -> Vec<KnowledgeNodeId> {
        tags.iter().map(|tag| node(tag)).collect()
    }

    /// A current-generation page row (identity + refs is what the diff sees).
    fn old_page(id: &str, slug: &str, title: &str, tags: &[&str]) -> GenerationPageView {
        GenerationPageView {
            page_id: WikiPageId::from_validated(format!("wp_{id}")),
            slug: slug.to_owned(),
            title: title.to_owned(),
            category: "concepts".into(),
            language: "en".into(),
            body_hash: "hash".into(),
            content: format!("# {title}"),
            knowledge_refs: refs(tags),
            citations: Vec::<PageCitationRecord>::new(),
            links: Vec::<PageLinkRecord>::new(),
            inbound_links: 0,
        }
    }

    /// A planned page (fresh planner-assigned identity).
    fn new_page(id: &str, slug: &str, tags: &[&str]) -> WikiPagePlan {
        WikiPagePlan {
            id: WikiPageId::from_validated(format!("wp_{id}")),
            slug: slug.to_owned(),
            title: slug.to_owned(),
            category: "concepts".into(),
            purpose: "p".into(),
            knowledge_refs: refs(tags),
            source_refs: Vec::<SourceId>::new(),
            related_pages: Vec::new(),
        }
    }

    fn plan(pages: Vec<WikiPagePlan>) -> WikiPlan {
        WikiPlan { pages }
    }

    #[test]
    fn identical_page_is_unchanged_and_keeps_its_id() {
        let old = vec![old_page("a", "alpha", "alpha", &[C1, C2])];
        let new = plan(vec![new_page("fresh", "alpha", &[C1, C2])]);
        let diff = plan_diff(&old, &new);
        assert!(diff.is_empty(), "identical plan needs no replan");
        assert_eq!(diff.unchanged.len(), 1);
        assert_eq!(
            diff.unchanged[0].page_id,
            WikiPageId::from_validated("wp_a")
        );
        assert_eq!(diff.unchanged[0].predecessor_id.as_str(), "wp_a");
        assert!(diff.modified.is_empty() && diff.merged.is_empty() && diff.split.is_empty());
        assert!(diff.created.is_empty() && diff.retired.is_empty());
        assert_eq!(diff.recompile_count(), 0);
    }

    #[test]
    fn changed_refs_or_title_or_category_is_a_modification() {
        // Refs grew.
        let old = vec![old_page("a", "alpha", "alpha", &[C1, C2])];
        let new = plan(vec![new_page("fresh", "alpha", &[C1, C2, C3])]);
        let diff = plan_diff(&old, &new);
        assert!(!diff.is_empty());
        assert_eq!(diff.modified.len(), 1);
        assert_eq!(diff.modified[0].page_id.as_str(), "wp_a", "id is kept");

        // Ref set identical but title differs.
        let old = vec![old_page("a", "alpha", "alpha", &[C1])];
        let new = plan(vec![new_page("fresh", "alpha-renamed", &[C1])]);
        let diff = plan_diff(&old, &new);
        // The title changed → modified (keep id, recompile), never unchanged.
        // NOTE: the new page's slug differs, so the mutual-best match holds
        // via refs; the title check flips the class.
        assert_eq!(diff.modified.len(), 1, "{diff:?}");
        assert_eq!(diff.modified[0].page_id.as_str(), "wp_a");

        // Category differs.
        let mut categorized = new_page("fresh", "alpha", &[C1]);
        categorized.category = "guides".into();
        let diff = plan_diff(&old, &plan(vec![categorized]));
        assert_eq!(diff.modified.len(), 1, "{diff:?}");
        assert_eq!(diff.modified[0].page_id.as_str(), "wp_a");
    }

    #[test]
    fn merge_keeps_dominant_predecessor_by_ref_overlap() {
        // Old alpha owns 1 ref, beta owns 2; the merged page covers all three
        // → beta is dominant (larger overlap) even though alpha's slug sorts
        // first.
        let old = vec![
            old_page("a", "alpha", "alpha", &[C1]),
            old_page("b", "beta", "beta", &[C2, C3]),
        ];
        let new = plan(vec![new_page("fresh", "merged", &[C1, C2, C3])]);
        let diff = plan_diff(&old, &new);
        assert_eq!(diff.merged.len(), 1, "{diff:?}");
        let merge = &diff.merged[0];
        assert_eq!(merge.continued.page_id.as_str(), "wp_b", "dominant id kept");
        assert_eq!(
            merge.predecessors,
            vec![
                WikiPageId::from_validated("wp_b"),
                WikiPageId::from_validated("wp_a")
            ],
            "dominant first, all predecessors recorded"
        );
        assert_eq!(diff.recompile_count(), 1);
    }

    #[test]
    fn merge_tie_breaks_by_old_slug_order() {
        // Equal overlaps → the slug-sorted first old page is dominant.
        let old = vec![
            old_page("b", "beta", "beta", &[C3, C4]),
            old_page("a", "alpha", "alpha", &[C1, C2]),
        ];
        let new = plan(vec![new_page("fresh", "merged", &[C1, C2, C3, C4])]);
        let diff = plan_diff(&old, &new);
        assert_eq!(diff.merged.len(), 1, "{diff:?}");
        assert_eq!(diff.merged[0].continued.page_id.as_str(), "wp_a");
        assert_eq!(
            diff.merged[0].predecessors,
            vec![
                WikiPageId::from_validated("wp_a"),
                WikiPageId::from_validated("wp_b")
            ]
        );
    }

    #[test]
    fn split_gives_fresh_ids_and_records_successors() {
        let old = vec![old_page("a", "alpha", "alpha", &[C1, C2, C3])];
        let new = plan(vec![
            new_page("p1", "part-one", &[C1]),
            new_page("p2", "part-two", &[C2, C3]),
        ]);
        let diff = plan_diff(&old, &new);
        assert_eq!(diff.split.len(), 1, "{diff:?}");
        assert_eq!(diff.split[0].retired.as_str(), "wp_a");
        assert_eq!(
            diff.split[0].successors.len(),
            2,
            "both successors recorded"
        );
        assert_eq!(diff.split[0].slug, "alpha");
        // The successors keep their FRESH planner ids (never wp_a).
        assert_ne!(diff.split[0].successors[0].as_str(), "wp_a");
        assert_ne!(diff.split[0].successors[1].as_str(), "wp_a");
        assert!(diff.unchanged.is_empty() && diff.modified.is_empty() && diff.merged.is_empty());
        assert_eq!(diff.recompile_count(), 2);
        // The old page is split, not plain-retired.
        assert!(diff.retired.is_empty(), "{diff:?}");
    }

    #[test]
    fn created_page_has_no_predecessor() {
        let old = vec![old_page("a", "alpha", "alpha", &[C1])];
        let new = plan(vec![
            new_page("fresh", "alpha", &[C1]),
            new_page("brand", "brand-new", &[C9]),
        ]);
        let diff = plan_diff(&old, &new);
        assert_eq!(diff.unchanged.len(), 1);
        assert_eq!(diff.created.len(), 1, "{diff:?}");
        assert_eq!(diff.created[0].page_id.as_str(), "wp_brand");
        assert_eq!(diff.created[0].index, 1);
        assert_eq!(diff.recompile_count(), 1);
    }

    #[test]
    fn retired_page_flags_lost_refs_but_absorbed_knowledge_does_not() {
        // Beta vanishes entirely: its ref is lost (knowledge gone or retired
        // with a deleted source — the caller decides).
        let old = vec![
            old_page("a", "alpha", "alpha", &[C1]),
            old_page("b", "beta", "beta", &[C3]),
        ];
        let new = plan(vec![new_page("fresh", "alpha", &[C1])]);
        let diff = plan_diff(&old, &new);
        assert_eq!(diff.retired.len(), 1, "{diff:?}");
        assert_eq!(diff.retired[0].page_id.as_str(), "wp_b");
        assert_eq!(diff.retired[0].lost_refs, refs(&[C3]));
        assert_eq!(diff.unchanged.len(), 1);

        // Beta is absorbed into the merged page: retired is NOT flagged —
        // the knowledge moved, it did not vanish.
        let old = vec![
            old_page("a", "alpha", "alpha", &[C1]),
            old_page("b", "beta", "beta", &[C2]),
        ];
        let new = plan(vec![new_page("fresh", "merged", &[C1, C2])]);
        let diff = plan_diff(&old, &new);
        assert_eq!(diff.merged.len(), 1, "{diff:?}");
        assert!(
            diff.retired.iter().all(|page| page.lost_refs.is_empty()),
            "absorbed refs are covered: {diff:?}"
        );
    }

    #[test]
    fn same_input_yields_the_same_classification() {
        let old = vec![
            old_page("b", "beta", "beta", &[C2, C3]),
            old_page("a", "alpha", "alpha", &[C1]),
        ];
        let new = plan(vec![
            new_page("fresh", "alpha", &[C1]),
            new_page("m", "merged", &[C2, C3, C4]),
            new_page("x", "extra", &[C4]),
        ]);
        let first = plan_diff(&old, &new);
        let second = plan_diff(&old, &new);
        assert_eq!(first, second, "plan_diff must be deterministic");
    }

    #[test]
    fn empty_plan_retries_everything_as_retired() {
        let old = vec![old_page("a", "alpha", "alpha", &[C1])];
        let diff = plan_diff(&old, &WikiPlan::default());
        assert_eq!(diff.retired.len(), 1);
        assert_eq!(diff.retired[0].lost_refs, refs(&[C1]));
        assert_eq!(diff.recompile_count(), 0);
    }

    #[test]
    fn continuation_requires_half_coverage_on_both_sides() {
        let four: BTreeSet<KnowledgeNodeId> = refs(&[C1, C2, C3, C4]).into_iter().collect();
        let two: BTreeSet<KnowledgeNodeId> = refs(&[C1, C2]).into_iter().collect();
        let one: BTreeSet<KnowledgeNodeId> = refs(&[C1]).into_iter().collect();
        let unrelated: BTreeSet<KnowledgeNodeId> = refs(&[C9]).into_iter().collect();

        // Audit FIX-014 degenerate case: a 1-node overlap must never carry a
        // 4-node page's WikiPageId (1/4 old coverage).
        assert!(!continuation_covers(&four, &one));
        // Inclusive halves continue: 2/2 and 2/4.
        assert!(continuation_covers(&two, &two));
        assert!(continuation_covers(&four, &two));
        // No overlap continues nothing, and neither do empty ref sets.
        assert!(!continuation_covers(&four, &unrelated));
        assert!(!continuation_covers(&BTreeSet::new(), &BTreeSet::new()));
    }
}
