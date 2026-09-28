#![forbid(unsafe_code)]
//! Application service layer for the Knowledge Compiler (PRD §7 application
//! logic; §51 steps 10+). Hosts the stage-one document analyzer, the
//! hierarchical wiki planner and the page compiler; the atomic publisher
//! lands here in a later slice. Depends on core/llm/markdown/storage — never
//! the reverse.

pub mod analysis;
pub mod compile;
pub mod persist;
pub mod plan;
pub mod prompt;

pub use analysis::{
    AnalysisOutcome, AnalysisSection, AnalyzedDocument, DocumentAnalyzer, VerifiedRelation,
};
pub use compile::{CompiledGeneration, CompilerConfig, WikiCompiler};
pub use persist::{persist_outcome, PersistOptions};
pub use plan::{PlanCacheKeys, PlanOutcome, PlannerConfig, WikiPlanner};
pub use prompt::{load_prompt, PromptDocument};
