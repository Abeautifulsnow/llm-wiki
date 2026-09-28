//! Pure §37.3 metric functions: each takes persisted pipeline state plus the
//! fixture annotations and reports numerator/denominator/offending items.
//! No I/O — unit-testable in isolation (PRD §37: every metric defines its
//! denominator).

use std::collections::{BTreeMap, BTreeSet};

use llm_wiki_core::plan::KnowledgeBase;
use llm_wiki_storage::{GenerationPageView, SourceRecord};

use crate::fixtures::{Dataset, ExpectedPage};

// ---------------------------------------------------------------------------
// Source Coverage (§37.3): annotated high facts represented by a claim
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    pub covered: Vec<String>,
    pub uncovered: Vec<String>,
    pub total: usize,
}

impl Coverage {
    pub fn ratio(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.covered.len() as f64 / self.total as f64
    }
}

/// A high fact is covered when some claim node's statement contains the
/// verbatim span (the fixture fakes quote spans, so equality is the norm;
/// `contains` tolerates benign wrapping).
pub fn coverage(dataset: &Dataset, claim_statements: &[String]) -> Coverage {
    let mut covered = Vec::new();
    let mut uncovered = Vec::new();
    let high: Vec<_> = dataset.high_facts().collect();
    for fact in high {
        if claim_statements
            .iter()
            .any(|stmt| stmt.contains(&fact.span))
        {
            covered.push(fact.id.clone());
        } else {
            uncovered.push(fact.id.clone());
        }
    }
    let total = covered.len() + uncovered.len();
    Coverage {
        covered,
        uncovered,
        total,
    }
}

// ---------------------------------------------------------------------------
// Citation Correctness (§37.3): valid ratio ≥95% AND zero invalid range/hash
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CitationAudit {
    pub checked: usize,
    /// `(page_slug, claim_node_id, problem)` for every invalid citation.
    pub invalid: Vec<(String, String, String)>,
}

impl CitationAudit {
    pub fn ratio(&self) -> f64 {
        if self.checked == 0 {
            return 1.0;
        }
        (self.checked - self.invalid.len()) as f64 / self.checked as f64
    }
}

/// Audits every citation of every page: range inside the source document,
/// source hash matching the registry, digest well-formed (64 hex).
pub fn audit_citations(
    pages: &[GenerationPageView],
    sources: &BTreeMap<String, SourceRecord>,
) -> CitationAudit {
    let mut checked = 0usize;
    let mut invalid = Vec::new();
    for page in pages {
        for citation in &page.citations {
            checked += 1;
            let Some(source) = sources.get(citation.source_id.as_str()) else {
                invalid.push((
                    page.slug.clone(),
                    citation.claim_node_id.as_str().to_owned(),
                    "source missing from registry".into(),
                ));
                continue;
            };
            if citation.source_hash != source.content_hash {
                invalid.push((
                    page.slug.clone(),
                    citation.claim_node_id.as_str().to_owned(),
                    format!(
                        "source hash drifted: citation {} vs registry {}",
                        &citation.source_hash[..12.min(citation.source_hash.len())],
                        &source.content_hash[..12.min(source.content_hash.len())]
                    ),
                ));
            }
            if citation.range.end < citation.range.start
                || citation.range.end as u64 > source.size.max(0) as u64
            {
                invalid.push((
                    page.slug.clone(),
                    citation.claim_node_id.as_str().to_owned(),
                    format!(
                        "range {}..{} outside source ({} bytes)",
                        citation.range.start, citation.range.end, source.size
                    ),
                ));
            }
            if citation.evidence_digest.len() != 64
                || !citation
                    .evidence_digest
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit())
            {
                invalid.push((
                    page.slug.clone(),
                    citation.claim_node_id.as_str().to_owned(),
                    format!("evidence digest malformed: {}", citation.evidence_digest),
                ));
            }
        }
    }
    CitationAudit { checked, invalid }
}

// ---------------------------------------------------------------------------
// Hallucination Rate (§37.3): falsifiable claims without source backing ≤5%
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hallucination {
    pub total_claims: usize,
    /// Claim node ids with no grounding anchor (nothing ties the statement to
    /// a source section).
    pub unbacked: Vec<String>,
}

impl Hallucination {
    pub fn ratio(&self) -> f64 {
        if self.total_claims == 0 {
            return 0.0;
        }
        self.unbacked.len() as f64 / self.total_claims as f64
    }
}

/// A claim node is falsifiable-by-construction in the knowledge model; it is
/// "backed" when it carries at least one grounding anchor.
pub fn hallucination(base: &KnowledgeBase) -> Hallucination {
    let mut total_claims = 0usize;
    let mut unbacked = Vec::new();
    for node in base.nodes.values() {
        if node.kind != "claim" {
            continue;
        }
        total_claims += 1;
        if node.anchors.is_empty() {
            unbacked.push(node.id.as_str().to_owned());
        }
    }
    Hallucination {
        total_claims,
        unbacked,
    }
}

// ---------------------------------------------------------------------------
// Cross-document Synthesis (§37.3): expected pages cite ≥ min distinct sources
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesisResult {
    pub page_title: String,
    /// Distinct source rel_paths cited by the compiled page.
    pub cited_sources: BTreeSet<String>,
    pub missing: bool,
    pub shortfall: bool,
}

/// Resolves a page's citations to source rel_paths and compares against the
/// expected page's requirement.
pub fn cross_document_synthesis(
    expected: &ExpectedPage,
    pages: &[GenerationPageView],
    sources: &BTreeMap<String, SourceRecord>,
) -> SynthesisResult {
    let target = pages
        .iter()
        .find(|page| page.title.trim().to_lowercase() == expected.title.trim().to_lowercase());
    let Some(page) = target else {
        return SynthesisResult {
            page_title: expected.title.clone(),
            cited_sources: BTreeSet::new(),
            missing: true,
            shortfall: true,
        };
    };
    let cited: BTreeSet<String> = page
        .citations
        .iter()
        .filter_map(|citation| sources.get(citation.source_id.as_str()))
        .map(|source| source.rel_path.clone())
        .collect();
    let covered = expected
        .sources
        .iter()
        .filter(|src| cited.contains(*src))
        .count();
    SynthesisResult {
        page_title: expected.title.clone(),
        cited_sources: cited,
        missing: false,
        shortfall: covered < expected.min_sources_merged,
    }
}

// ---------------------------------------------------------------------------
// Rebuild Determinism (§37.3 / DoD #16)
// ---------------------------------------------------------------------------

/// One manifest line per page: ids, refs, citation tuples and links — the
/// structured checklist §37.3 compares between rebuilds. Page content is
/// deliberately excluded: §15.2 frontmatter carries the owning build id.
pub fn manifest_entry(page: &GenerationPageView) -> String {
    let citations: Vec<String> = page
        .citations
        .iter()
        .map(|c| {
            format!(
                "{}/{}/{}/{}/{}",
                c.claim_node_id, c.source_id, c.range.start, c.range.end, c.evidence_digest
            )
        })
        .collect();
    let links: Vec<String> = page
        .links
        .iter()
        .map(|l| format!("{}/{}", l.to_page_id, l.target_title))
        .collect();
    format!(
        "{}|{}|{}|{:?}|{:?}|{:?}",
        page.page_id, page.slug, page.title, page.knowledge_refs, citations, links
    )
}

/// The full sorted manifest of a generation.
pub fn generation_manifest(pages: &mut [GenerationPageView]) -> Vec<String> {
    pages.sort_by(|a, b| a.slug.cmp(&b.slug));
    pages.iter().map(manifest_entry).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset_with(spans: &[(&str, &str)]) -> Dataset {
        Dataset {
            facts: spans
                .iter()
                .map(|(id, span)| crate::fixtures::Fact {
                    id: id.to_string(),
                    span: span.to_string(),
                    importance: "high".into(),
                    doc_path: "doc.md".into(),
                })
                .collect(),
            doc_paths: vec!["doc.md".into()],
        }
    }

    #[test]
    fn coverage_counts_only_covered_high_facts() {
        let dataset = dataset_with(&[("F-1", "alpha fact"), ("F-2", "beta fact")]);
        let cov = coverage(
            &dataset,
            &["the wiki says alpha fact here".into(), "other".into()],
        );
        assert_eq!(cov.total, 2);
        assert_eq!(cov.covered, vec!["F-1"]);
        assert_eq!(cov.uncovered, vec!["F-2"]);
        assert!((cov.ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn citation_audit_flags_missing_source_and_bad_range() {
        // Constructed directly to avoid a DB fixture for two flags.
        let audit = CitationAudit {
            checked: 3,
            invalid: vec![
                (
                    "p".into(),
                    "kn_a".into(),
                    "source missing from registry".into(),
                ),
                (
                    "p".into(),
                    "kn_b".into(),
                    "range 5..2 outside source (10 bytes)".into(),
                ),
            ],
        };
        assert!((audit.ratio() - (1.0 / 3.0)).abs() < 1e-9);
        assert_eq!(audit.invalid.len(), 2);
    }

    #[test]
    fn hallucination_counts_unanchored_claims() {
        let mut base = KnowledgeBase::default();
        for (id, anchored) in [("kn_a", true), ("kn_b", false), ("kn_c", false)] {
            let mut node = llm_wiki_core::plan::PlanNode {
                id: llm_wiki_core::ids::KnowledgeNodeId::parse(id).unwrap(),
                kind: "claim".into(),
                name: id.into(),
                entity_type: None,
                description: None,
                statement: Some("s".into()),
                anchors: Vec::new(),
            };
            if anchored {
                node.anchors.push(llm_wiki_core::plan::PlanAnchor {
                    source_id: llm_wiki_core::ids::SourceId::parse("src_a").unwrap(),
                    rel_path: "doc.md".into(),
                    section_id: None,
                    heading_path: vec![],
                    range: llm_wiki_core::model::SourceRange::new(0, 1),
                    evidence_digest: "d".into(),
                    source_hash: "h".into(),
                });
            }
            base.nodes.insert(node.id.clone(), node);
        }
        let hall = hallucination(&base);
        assert_eq!(hall.total_claims, 3);
        assert_eq!(hall.unbacked.len(), 2);
        assert!((hall.ratio() - (2.0 / 3.0)).abs() < 1e-9);
    }
}
