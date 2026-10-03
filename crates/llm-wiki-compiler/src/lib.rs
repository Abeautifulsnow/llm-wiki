#![forbid(unsafe_code)]
//! Application service layer for the Knowledge Compiler (PRD §7 application
//! logic; §51 steps 10+). Hosts the stage-one document analyzer, the
//! hierarchical wiki planner, the page compiler, the atomic publisher (§35)
//! and the end-to-end `run_build` orchestration (§29/§31). Depends on
//! core/llm/markdown/storage — never the reverse.

pub mod analysis;
pub mod build;
pub mod cache;
pub mod changeset;
pub mod compile;
pub mod incremental;
pub mod lint;
pub mod persist;
pub mod plan;
pub mod prompt;
pub mod publish;
pub mod replan;

pub use analysis::{
    AnalysisOutcome, AnalysisSection, AnalyzedDocument, DocumentAnalyzer, VerifiedRelation,
};
pub use build::{run_build, BuildReport, IncrementalSummary};
pub use cache::{CacheContext, CacheStats, LlmCache, StageCache};
pub use changeset::{
    diff_manifest, finalize_change_set, BuildFingerprint, ChangeSet, FileOutcome, RegisteredSource,
    ScannedSource,
};
pub use compile::{CompiledGeneration, CompilerConfig, WikiCompiler};
pub use incremental::{map_incremental_change, MappingDecision, MappingInput};
pub use lint::{run_lint, LintCheck, LintFinding, LintReport, LintSeverity};
pub use persist::{persist_outcome, PersistOptions};
pub use plan::{PlanCacheKeys, PlanOutcome, PlannerConfig, WikiPlanner};
pub use prompt::{load_prompt, PromptDocument};
pub use publish::{
    atomic_write, cleanup_generations, journal_exists, page_file_name, publish,
    read_current_pointer, read_journal, recover_if_needed, write_current_pointer, write_generation,
    write_journal, CurrentPointer, PublishJournal, PublishPaths, PublishReport, RecoveryAction,
    RecoveryReport, RENAME_ATTEMPTS, RENAME_RETRY_DELAY,
};
pub use replan::{plan_diff, replan, PlanDiff, ReplanReport};
