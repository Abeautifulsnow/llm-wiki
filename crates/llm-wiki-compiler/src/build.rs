//! Build pipeline orchestration (PRD §29/§31/§51 steps 15–16): chains scan →
//! parse → analyze → plan → compile → publish, persisting every §31 stage
//! transition. The LLM provider is injected so tests run on
//! `FakeLlmProvider` (PRD §54); the CLI stays a thin transport (PRD §7.7).
//!
//! Failure semantics (PRD §34/§35): any error marks the build FAILED (or
//! REPLAN_REQUIRED for `WikiError::ReplanRequired`) and leaves the previously
//! published generation and pointer untouched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::BuildId;
use llm_wiki_core::matcher::{match_sections, PrevSection, SectionIdentity, SectionOutcome};
use llm_wiki_llm::LlmProvider;
use llm_wiki_markdown::parse_document;
use llm_wiki_source::{ScanDiagnostic, Scanner, SourceManifest};
use llm_wiki_storage::{
    apply_section_matches, finish_build, load_active_sections, load_plan_input,
    mark_stale_builds_interrupted, open, persist_generation, start_build, update_build_status,
    upsert_sources_batch, BuildDraft, GenerationStats, SourceUpsert,
};

use crate::analysis::{AnalyzedDocument, DocumentAnalyzer};
use crate::compile::{CompilerConfig, WikiCompiler};
use crate::persist::persist_outcome;
use crate::persist::PersistOptions;
use crate::plan::{PlannerConfig, WikiPlanner};
use crate::prompt::load_prompt;
use crate::publish::{publish, recover_if_needed, PublishPaths};

/// Successful end of a build; the CLI prints this.
#[derive(Debug, Clone)]
pub struct BuildReport {
    pub build_id: BuildId,
    pub sources: usize,
    pub pages: usize,
    pub citations: usize,
    pub links: usize,
    /// All LLM requests spent by analysis, planning and compilation.
    pub llm_request_count: u32,
    pub published_path: PathBuf,
    /// Human-readable summary of publish recovery performed before this build
    /// started, if any (PRD §35: recovery must be reported).
    pub recovery: Option<String>,
}

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
    let build_id = start_build(
        &mut conn,
        &BuildDraft {
            model: Some(provider.model().to_owned()),
            prompt_version: Some(prompt_version.clone()),
            compiler_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
            schema_version: Some("1".to_owned()),
            config_hash: Some(config_hash),
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
        &analysis_prompt,
        &planning_prompt,
        &compilation_prompt,
        &prompt_version,
    )
    .await;

    match result {
        Ok((stats, sources, llm_request_count)) => {
            let published_path = PublishPaths::new(&wiki_dir).generation_dir(&build_id);
            Ok(BuildReport {
                build_id,
                sources,
                pages: stats.pages,
                citations: stats.citations,
                links: stats.links,
                llm_request_count,
                published_path,
                recovery: recovery_note,
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
fn terminal_status_for(err: &WikiError) -> &'static str {
    match err {
        WikiError::ReplanRequired { .. } => "REPLAN_REQUIRED",
        _ => "FAILED",
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_inner(
    conn: &mut rusqlite::Connection,
    root: &Path,
    wiki_dir: &Path,
    build_id: &BuildId,
    config: &Config,
    provider: Arc<dyn LlmProvider>,
    analysis_prompt: &crate::prompt::PromptDocument,
    planning_prompt: &crate::prompt::PromptDocument,
    compilation_prompt: &crate::prompt::PromptDocument,
    analysis_prompt_version: &str,
) -> Result<(GenerationStats, usize, u32)> {
    // ---- Scan (§8): sources are upserted in ONE transaction. ----
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
    let analyzer = DocumentAnalyzer::new(
        provider.clone(),
        analysis_prompt.clone(),
        config.analysis.section_target_tokens,
        config.analysis.max_rejected_claim_ratio,
    );
    let mut llm_request_count = 0u32;
    for (file, source_id, parsed_doc) in &parsed {
        let sections = register_sections(conn, source_id, &parsed_doc.sections, build_id)?;
        let doc = AnalyzedDocument {
            source_id: source_id.clone(),
            rel_path: file.rel_path.clone(),
            content_hash: file.content_hash.clone(),
            language: parsed_doc.language.clone(),
            sections,
        };
        let outcome = analyzer.analyze_document(&doc, Some(build_id)).await?;
        llm_request_count += outcome.llm_request_count;
        persist_outcome(
            conn,
            &doc,
            &outcome,
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
        ..PlannerConfig::default()
    };
    let planner = WikiPlanner::new(provider.clone(), planning_prompt.clone(), planner_config);
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
    );
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

/// Persists the parsed sections through the Section Registry (PRD §45):
/// identities are carried across builds deterministically; ambiguous matches
/// are re-analyzed under a FRESH identity (never reusing the old id), and
/// unmatched previous sections retire.
fn register_sections(
    conn: &mut rusqlite::Connection,
    source_id: &llm_wiki_core::ids::SourceId,
    sections: &[llm_wiki_markdown::SectionOutput],
    build_id: &BuildId,
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
        Some(build_id.as_str()),
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
fn normalized_rel_of(root: &Path, wiki_dir: &Path) -> Option<String> {
    wiki_dir.strip_prefix(root).ok().map(|rel| {
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
    })
}

fn warn_diagnostic(diagnostic: &ScanDiagnostic) {
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
}
