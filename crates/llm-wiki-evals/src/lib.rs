#![forbid(unsafe_code)]
//! Eval scoring for the Knowledge Compiler (PRD §37.3).
//!
//! Pure scoring functions over the pipeline's persisted state plus the
//! checked-in fixture annotations (`evals/dataset.yaml`,
//! `evals/expected/pages.yaml`). Gate tests in this crate run the full
//! `run_build` pipeline over `evals/corpus/` with a `FakeLlmProvider` and
//! assert the V0.1 release gates — no real model in CI (PRD §54).

pub mod fixtures;
pub mod gates;
pub mod scoring;
pub mod stages;

pub use fixtures::{
    load_fixtures, Dataset, ExpectedPage, ExpectedPages, Fact, FixtureError, MIN_CJK, MIN_DOCS,
    MIN_MDX, MIN_QUESTIONS,
};
pub use gates::{evaluate_gates, thresholds, GateFailure, GateMetrics, GateReport, SourceIndex};
pub use scoring::{
    audit_citations, coverage, cross_document_synthesis, generation_manifest, hallucination,
    CitationAudit, Coverage, Hallucination, SynthesisResult,
};
pub use stages::{eval_llm, eval_workspace, evals_dir, span_grouping};
