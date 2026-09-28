#![forbid(unsafe_code)]
//! Application service layer for the Knowledge Compiler (PRD §7 application
//! logic; §51 steps 10+). Hosts the stage-one document analyzer today; the
//! hierarchical planner, wiki compiler and atomic publisher land here in
//! later slices. Depends on core/llm/markdown/storage — never the reverse.

pub mod analysis;
pub mod persist;
pub mod prompt;

pub use analysis::{
    AnalysisOutcome, AnalysisSection, AnalyzedDocument, DocumentAnalyzer, VerifiedRelation,
};
pub use persist::{persist_outcome, PersistOptions};
pub use prompt::{load_prompt, PromptDocument};
