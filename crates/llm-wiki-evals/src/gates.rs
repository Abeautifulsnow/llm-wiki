//! V0.1 release gates (PRD §37.3). Each gate consumes a §37.3 metric and
//! fails with numerator/denominator/offending items — threshold changes are a
//! product contract change and live in `evals/README.md`.

use std::collections::BTreeMap;
use std::fmt;

use llm_wiki_storage::SourceRecord;

use crate::fixtures::{Dataset, ExpectedPages};
use crate::scoring::{
    audit_citations, coverage, cross_document_synthesis, generation_manifest, hallucination,
    CitationAudit, Coverage, Hallucination, SynthesisResult,
};

/// Versioned thresholds — MUST match `evals/README.md` (v1, 2026-09-28).
pub mod thresholds {
    pub const SOURCE_COVERAGE: f64 = 0.90;
    pub const CITATION_CORRECTNESS: f64 = 0.95;
    pub const HALLUCINATION_RATE: f64 = 0.05;
}

/// Source sizes come straight from the registry; the audit needs
/// `content_hash` + `size` per source id.
pub type SourceIndex = BTreeMap<String, SourceRecord>;

#[derive(Debug, Clone)]
pub enum GateFailure {
    Coverage {
        ratio: f64,
        uncovered: Vec<String>,
        total: usize,
    },
    CitationCorrectness {
        ratio: f64,
        invalid: Vec<(String, String, String)>,
        checked: usize,
    },
    Hallucination {
        ratio: f64,
        unbacked: Vec<String>,
        total: usize,
    },
    Synthesis {
        results: Vec<SynthesisResult>,
    },
    Determinism {
        new_requests: u32,
        manifest_first_len: usize,
        manifest_second_len: usize,
        diverged_at: Option<usize>,
    },
}

impl fmt::Display for GateFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateFailure::Coverage {
                ratio,
                uncovered,
                total,
            } => write!(
                f,
                "source coverage {}: covered {}/{} high facts (need >= {}); uncovered: {:?}",
                ratio,
                total - uncovered.len(),
                total,
                thresholds::SOURCE_COVERAGE,
                uncovered
            ),
            GateFailure::CitationCorrectness {
                ratio,
                invalid,
                checked,
            } => write!(
                f,
                "citation correctness {}: {} invalid of {} checked (need >= {} and ZERO invalid); invalid: {:?}",
                ratio,
                invalid.len(),
                checked,
                thresholds::CITATION_CORRECTNESS,
                invalid
            ),
            GateFailure::Hallucination {
                ratio,
                unbacked,
                total,
            } => write!(
                f,
                "hallucination rate {}: {} unbacked of {} claims (need <= {})",
                ratio,
                unbacked.len(),
                total,
                thresholds::HALLUCINATION_RATE
            ),
            GateFailure::Synthesis { results } => {
                let bad: Vec<&SynthesisResult> = results
                    .iter()
                    .filter(|r| r.missing || r.shortfall)
                    .collect();
                write!(f, "cross-document synthesis failed for {bad:?}")
            }
            GateFailure::Determinism {
                new_requests,
                manifest_first_len,
                manifest_second_len,
                diverged_at,
            } => write!(
                f,
                "rebuild determinism: {new_requests} new LLM requests on rebuild (need 0); \
                 manifest sizes {manifest_first_len} vs {manifest_second_len}, diverged_at {diverged_at:?}"
            ),
        }
    }
}

/// The aggregate §37.3 result: empty `failures` means the release gates pass.
#[derive(Debug, Clone, Default)]
pub struct GateReport {
    pub failures: Vec<GateFailure>,
}

impl GateReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Summary line per metric for the failure report.
#[derive(Debug, Clone)]
pub struct GateMetrics {
    pub coverage: Coverage,
    pub citations: CitationAudit,
    pub hallucination: Hallucination,
    pub synthesis: Vec<SynthesisResult>,
}

/// Evaluates every §37.3 gate except determinism (needs two builds, so it is
/// supplied by the caller as precomputed inputs).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_gates(
    dataset: &Dataset,
    expected_pages: &ExpectedPages,
    pages: &mut [llm_wiki_storage::GenerationPageView],
    sources: &SourceIndex,
    claim_statements: &[String],
    knowledge: &llm_wiki_core::plan::KnowledgeBase,
    new_requests_on_rebuild: u32,
    manifest_first: &[String],
    manifest_second: &[String],
) -> (GateReport, GateMetrics) {
    let mut failures = Vec::new();

    let cov = coverage(dataset, claim_statements);
    if cov.ratio() < thresholds::SOURCE_COVERAGE {
        failures.push(GateFailure::Coverage {
            ratio: cov.ratio(),
            uncovered: cov.uncovered.clone(),
            total: cov.total,
        });
    }

    let audit = audit_citations(pages, sources);
    if audit.ratio() < thresholds::CITATION_CORRECTNESS || !audit.invalid.is_empty() {
        failures.push(GateFailure::CitationCorrectness {
            ratio: audit.ratio(),
            invalid: audit.invalid.clone(),
            checked: audit.checked,
        });
    }

    let hall = hallucination(knowledge);
    if hall.ratio() > thresholds::HALLUCINATION_RATE {
        failures.push(GateFailure::Hallucination {
            ratio: hall.ratio(),
            unbacked: hall.unbacked.clone(),
            total: hall.total_claims,
        });
    }

    let synthesis_results: Vec<SynthesisResult> = expected_pages
        .pages
        .iter()
        .map(|page| cross_document_synthesis(page, pages, sources))
        .collect();
    if synthesis_results.iter().any(|r| r.missing || r.shortfall) {
        failures.push(GateFailure::Synthesis {
            results: synthesis_results.clone(),
        });
    }

    if new_requests_on_rebuild != 0 || manifest_first != manifest_second {
        let diverged_at = manifest_first
            .iter()
            .zip(manifest_second.iter())
            .position(|(a, b)| a != b);
        failures.push(GateFailure::Determinism {
            new_requests: new_requests_on_rebuild,
            manifest_first_len: manifest_first.len(),
            manifest_second_len: manifest_second.len(),
            diverged_at,
        });
    }

    let _ = generation_manifest(pages); // canonical sort for future consumers
    (
        GateReport { failures },
        GateMetrics {
            coverage: cov,
            citations: audit,
            hallucination: hall,
            synthesis: synthesis_results,
        },
    )
}
