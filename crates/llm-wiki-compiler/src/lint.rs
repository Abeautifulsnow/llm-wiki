//! Wiki lint (PRD §36/§29): checks the *currently published* generation —
//! the build `current.json` points at — using the DB rows (`wiki_pages`,
//! `page_citations`, `page_links`) plus the generation's Markdown files on
//! disk. Lint is a read-only product capability: it never repairs anything.
//!
//! Checks and default severities (PRD §36): `missing-source` (Error),
//! `stale-citation` (Warning), `broken-link` (Error), `orphan-page`
//! (Warning), `unsupported-section` (Warning), `duplicate-concept`
//! (Warning), `hand-edited-file` (Error — PRD §52 DoD #19: manual edits are
//! surfaced, never silently lost).
//!
//! Findings are deterministically ordered (check, severity, page slug,
//! message); a workspace that was never built lints to "nothing published"
//! and is not an error.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use llm_wiki_core::config::{lexical_absolute, Config};
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::BuildId;
use llm_wiki_markdown::parse_document;
use llm_wiki_storage::list_sources;
use llm_wiki_storage::{list_active_relation_pairs, list_insights, load_generation_view};

use crate::compile::{scan_citations, scan_wikilinks, strip_citations};
use crate::publish::{page_file_name, read_current_pointer, PublishPaths};

/// A section body is "substantive" once its prose (citation comments and
/// WikiLinks stripped) reaches this many characters; shorter bodies carry no
/// obligation to cite (PRD §36 unsupported-section).
pub const SUBSTANTIVE_SECTION_CHARS: usize = 160;

/// Severity of a lint finding (PRD §36: errors fail the command, warnings do
/// not).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintSeverity {
    Error,
    Warning,
}

impl LintSeverity {
    pub fn label(&self) -> &'static str {
        match self {
            LintSeverity::Error => "error",
            LintSeverity::Warning => "warning",
        }
    }
}

/// One lint check, in the PRD §36 table order (also the report order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintCheck {
    MissingSource,
    StaleCitation,
    BrokenLink,
    OrphanPage,
    UnsupportedSection,
    DuplicateConcept,
    StaleInsight,
    HandEditedFile,
}

impl LintCheck {
    pub fn label(&self) -> &'static str {
        match self {
            LintCheck::MissingSource => "missing-source",
            LintCheck::StaleCitation => "stale-citation",
            LintCheck::BrokenLink => "broken-link",
            LintCheck::OrphanPage => "orphan-page",
            LintCheck::UnsupportedSection => "unsupported-section",
            LintCheck::DuplicateConcept => "duplicate-concept",
            LintCheck::StaleInsight => "stale-insight",
            LintCheck::HandEditedFile => "hand-edited-file",
        }
    }
}

/// One deterministic lint finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub check: LintCheck,
    pub severity: LintSeverity,
    /// Slug of the page the finding belongs to (`"(workspace)"` for
    /// generation-wide findings).
    pub page_slug: String,
    pub message: String,
}

/// The full result of one lint run. Findings are stably sorted by
/// (check, severity, page_slug, message) so reports never shuffle.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LintReport {
    pub findings: Vec<LintFinding>,
}

impl LintReport {
    pub fn errors(&self) -> u32 {
        self.findings
            .iter()
            .filter(|f| f.severity == LintSeverity::Error)
            .count() as u32
    }

    pub fn warnings(&self) -> u32 {
        self.findings
            .iter()
            .filter(|f| f.severity == LintSeverity::Warning)
            .count() as u32
    }

    fn push(&mut self, finding: LintFinding) {
        self.findings.push(finding);
    }

    fn finish(mut self) -> Self {
        // Stable sort so equal keys keep their deterministic push order.
        self.findings.sort_by(|a, b| {
            a.check
                .cmp(&b.check)
                .then_with(|| a.severity.cmp(&b.severity))
                .then_with(|| a.page_slug.cmp(&b.page_slug))
                .then_with(|| a.message.cmp(&b.message))
        });
        self.findings.dedup();
        self
    }
}

/// Lints the current visible generation. Returns `Ok(None)` when the
/// workspace has nothing published (no pointer file) — that is not an error.
pub fn run_lint(workspace_root: &Path, config: &Config) -> Result<Option<LintReport>> {
    let state_db = workspace_root.join(".llm-wiki").join("state.db");
    if !state_db.exists() {
        return Ok(None);
    }
    let wiki_dir = lexical_absolute(workspace_root, &config.project.wiki_dir);
    let paths = PublishPaths::new(&wiki_dir);
    let Some(pointer) = read_current_pointer(&paths)? else {
        return Ok(None);
    };
    let build_id = BuildId::parse(&pointer.build_id)?;
    let conn = llm_wiki_storage::open(&state_db)?;
    let pages = load_generation_view(&conn, &build_id)?;
    if pages.is_empty() {
        // The visible pointer names a generation the database knows nothing
        // about: report the publish-state problem instead of guessing (§35).
        return Err(WikiError::PublishRecovery(format!(
            "current.json points at build {} but no page rows exist for it; run `llm-wiki doctor`",
            pointer.build_id
        )));
    }

    let sources = list_sources(&conn)?;
    let source_by_id: BTreeMap<String, &llm_wiki_storage::SourceRecord> = sources
        .iter()
        .map(|source| (source.source_id.as_str().to_owned(), source))
        .collect();
    let relation_pairs = list_active_relation_pairs(&conn)?;

    let mut report = LintReport::default();
    check_citation_sources(&pages, &source_by_id, &mut report);
    check_links(&pages, &mut report);
    check_orphans(&pages, &relation_pairs, &mut report);
    check_sections(&pages, &mut report);
    check_duplicates(&pages, &mut report);
    check_insights(&conn, &pages, &mut report);
    check_files_on_disk(&paths, &build_id, &pages, &mut report);
    Ok(Some(report.finish()))
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

fn check_citation_sources(
    pages: &[llm_wiki_storage::GenerationPageView],
    source_by_id: &BTreeMap<String, &llm_wiki_storage::SourceRecord>,
    report: &mut LintReport,
) {
    for page in pages {
        for citation in &page.citations {
            let source_id = citation.source_id.as_str();
            match source_by_id.get(source_id) {
                Some(source) if source.status == "active" => {
                    if source.content_hash != citation.source_hash {
                        report.push(LintFinding {
                            check: LintCheck::StaleCitation,
                            severity: LintSeverity::Warning,
                            page_slug: page.slug.clone(),
                            message: format!(
                                "citation of claim {} quotes source {} whose content changed since this page was compiled",
                                citation.claim_node_id.as_str(),
                                source.rel_path
                            ),
                        });
                    }
                }
                Some(source) => report.push(LintFinding {
                    check: LintCheck::MissingSource,
                    severity: LintSeverity::Error,
                    page_slug: page.slug.clone(),
                    message: format!(
                        "citation of claim {} references source {} which has been removed from the source registry",
                        citation.claim_node_id.as_str(),
                        source.rel_path
                    ),
                }),
                None => report.push(LintFinding {
                    check: LintCheck::MissingSource,
                    severity: LintSeverity::Error,
                    page_slug: page.slug.clone(),
                    message: format!(
                        "citation of claim {} references source id {source_id} which is absent from the source registry",
                        citation.claim_node_id.as_str()
                    ),
                }),
            }
        }
    }
}

fn check_links(pages: &[llm_wiki_storage::GenerationPageView], report: &mut LintReport) {
    // Same case-insensitive title resolution the compiler used (§15.2).
    let titles: BTreeSet<String> = pages
        .iter()
        .map(|page| page.title.trim().to_lowercase())
        .collect();
    let page_ids: BTreeSet<&str> = pages.iter().map(|page| page.page_id.as_str()).collect();
    for page in pages {
        for link in scan_wikilinks(&page.content) {
            let folded = link.target.trim().to_lowercase();
            if !titles.contains(&folded) {
                report.push(LintFinding {
                    check: LintCheck::BrokenLink,
                    severity: LintSeverity::Error,
                    page_slug: page.slug.clone(),
                    message: format!(
                        "WikiLink [[{link_target}]] has no target in the published generation",
                        link_target = link.target
                    ),
                });
            }
        }
        for link in &page.links {
            if !page_ids.contains(link.to_page_id.as_str()) {
                report.push(LintFinding {
                    check: LintCheck::BrokenLink,
                    severity: LintSeverity::Error,
                    page_slug: page.slug.clone(),
                    message: format!(
                        "resolved link to '{}' points at a page that is not part of this generation",
                        link.target_title
                    ),
                });
            }
        }
    }
}

fn check_orphans(
    pages: &[llm_wiki_storage::GenerationPageView],
    relation_pairs: &[(
        llm_wiki_core::ids::KnowledgeNodeId,
        llm_wiki_core::ids::KnowledgeNodeId,
    )],
    report: &mut LintReport,
) {
    for page in pages {
        if page.inbound_links > 0 {
            continue;
        }
        let touched = relation_pairs.iter().any(|(source, target)| {
            page.knowledge_refs.contains(source) || page.knowledge_refs.contains(target)
        });
        if !touched {
            report.push(LintFinding {
                check: LintCheck::OrphanPage,
                severity: LintSeverity::Warning,
                page_slug: page.slug.clone(),
                message:
                    "page has no inbound WikiLinks and no knowledge relation touches its nodes"
                        .to_owned(),
            });
        }
    }
}

fn check_sections(pages: &[llm_wiki_storage::GenerationPageView], report: &mut LintReport) {
    for page in pages {
        let parsed = parse_document(&page.content, &format!("{}.md", page.slug));
        for section in &parsed.sections {
            // Only H2/H3 sections are linted (PRD §36); the H1 title and any
            // preamble belong to the page shell, not a knowledge section.
            if section.heading_level < 2 || section.heading_level > 3 {
                continue;
            }
            if !scan_citations(&section.content).is_empty() {
                continue;
            }
            let prose = strip_wikilinks(&strip_citations(&section.content));
            if prose.chars().count() >= SUBSTANTIVE_SECTION_CHARS {
                report.push(LintFinding {
                    check: LintCheck::UnsupportedSection,
                    severity: LintSeverity::Warning,
                    page_slug: page.slug.clone(),
                    message: format!(
                        "section '{}' has a substantive body but no citations",
                        section.heading_path.join(" > ")
                    ),
                });
            }
        }
    }
}

/// Normalized title for duplicate detection: NFKC, whitespace-collapsed,
/// lowercased (same fold family as the registry's §13 canonical key).
fn normalized_title(title: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let nfkc: String = title.nfkc().collect();
    nfkc.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn check_duplicates(pages: &[llm_wiki_storage::GenerationPageView], report: &mut LintReport) {
    // Pages are loaded slug-ordered, so "first" is deterministic.
    let mut by_title: BTreeMap<String, String> = BTreeMap::new();
    for page in pages {
        let normalized = normalized_title(&page.title);
        match by_title.get(&normalized) {
            Some(first_slug) => report.push(LintFinding {
                check: LintCheck::DuplicateConcept,
                severity: LintSeverity::Warning,
                page_slug: page.slug.clone(),
                message: format!(
                    "title '{}' normalizes to the same concept as page '{first_slug}'",
                    page.title
                ),
            }),
            None => {
                by_title.insert(normalized.clone(), page.slug.clone());
                // V0.1 pages carry no alias field, so the alias arm compares
                // against slugs — the page's durable alias (PRD §45).
                if normalized != page.slug {
                    if let Some(other_slug) = pages
                        .iter()
                        .find(|other| other.slug == normalized)
                        .map(|other| other.slug.clone())
                    {
                        report.push(LintFinding {
                            check: LintCheck::DuplicateConcept,
                            severity: LintSeverity::Warning,
                            page_slug: page.slug.clone(),
                            message: format!(
                                "title '{}' normalizes onto the slug of page '{other_slug}'",
                                page.title
                            ),
                        });
                    }
                }
            }
        }
    }
}

/// Insight-consumption check (the write-back loop's read side): every stored
/// insight is re-verified against the ACTIVE generation. An insight whose
/// cited claim no longer exists in the generation, or whose recorded
/// evidence digest drifted, can no longer be traced to its provenance —
/// flagged as a warning so the curator can re-ask or retire it. Deterministic,
/// no model involvement.
fn check_insights(
    conn: &rusqlite::Connection,
    pages: &[llm_wiki_storage::GenerationPageView],
    report: &mut LintReport,
) {
    let insights = match list_insights(conn) {
        Ok(insights) => insights,
        Err(err) => {
            tracing::warn!(error = %err, "could not list insights; stale-insight check skipped");
            return;
        }
    };
    if insights.is_empty() {
        return;
    }
    // Current claim digest map over the active generation: claim node id →
    // evidence digest (a claim is cited by at most one page, but multiple
    // citations of it agree by construction; last wins deterministically).
    let mut claim_digests: BTreeMap<String, String> = BTreeMap::new();
    for page in pages {
        for citation in &page.citations {
            claim_digests.insert(
                citation.claim_node_id.as_str().to_owned(),
                citation.evidence_digest.clone(),
            );
        }
    }
    for insight in insights {
        for citation in &insight.citations {
            let Some(current) = claim_digests.get(&citation.claim_node_id) else {
                report.push(LintFinding {
                    check: LintCheck::StaleInsight,
                    severity: LintSeverity::Warning,
                    page_slug: "(insight)".to_owned(),
                    message: format!(
                        "insight {} ({:?}) cites claim {} which is no longer part of the published generation",
                        insight.insight_id.as_str(),
                        insight.query,
                        citation.claim_node_id
                    ),
                });
                continue;
            };
            if current != &citation.evidence_digest {
                report.push(LintFinding {
                    check: LintCheck::StaleInsight,
                    severity: LintSeverity::Warning,
                    page_slug: "(insight)".to_owned(),
                    message: format!(
                        "insight {} ({:?}) quotes evidence of claim {} that changed since the insight was written",
                        insight.insight_id.as_str(),
                        insight.query,
                        citation.claim_node_id
                    ),
                });
            }
        }
    }
}

fn check_files_on_disk(
    paths: &PublishPaths,
    build_id: &BuildId,
    pages: &[llm_wiki_storage::GenerationPageView],
    report: &mut LintReport,
) {
    let generation_dir = paths.generation_dir(build_id);
    for page in pages {
        let file = generation_dir.join(page_file_name(&page.slug, page.page_id.as_str()));
        match std::fs::read_to_string(&file) {
            Err(err) => report.push(LintFinding {
                check: LintCheck::HandEditedFile,
                severity: LintSeverity::Error,
                page_slug: page.slug.clone(),
                message: format!(
                    "generation file {} cannot be read ({err}); the published wiki is incomplete",
                    file.display()
                ),
            }),
            Ok(text) => {
                if sha256_hex(text.as_bytes()) != page.body_hash {
                    report.push(LintFinding {
                        check: LintCheck::HandEditedFile,
                        severity: LintSeverity::Error,
                        page_slug: page.slug.clone(),
                        message: format!(
                            "file {} was modified after publish (content hash no longer matches wiki_pages); manual edits are surfaced, never silently lost",
                            file.display()
                        ),
                    });
                }
            }
        }
    }
}

/// Removes `[[...]]` spans so link lists do not count as prose.
fn strip_wikilinks(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut cursor = 0;
    for span in scan_delimited(body, "[[", "]]") {
        out.push_str(&body[cursor..span.start]);
        cursor = span.end;
    }
    out.push_str(&body[cursor..]);
    out
}

/// Byte spans of every `open...close` pair, markers included, ordered by
/// position (mirror of the compiler's scanner, kept local so lint never
/// depends on compilation internals beyond the shared helpers).
fn scan_delimited(body: &str, open: &str, close: &str) -> Vec<std::ops::Range<usize>> {
    let mut spans = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = body[search_from..].find(open) {
        let open_start = search_from + offset;
        let content_start = open_start + open.len();
        let Some(close_offset) = body[content_start..].find(close) else {
            break;
        };
        let end = content_start + close_offset + close.len();
        spans.push(open_start..end);
        search_from = end;
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_order_matches_the_prd_table() {
        // Covered via the enum order: matches the PRD §36 table.
        let ordered = [
            LintCheck::MissingSource,
            LintCheck::StaleCitation,
            LintCheck::BrokenLink,
            LintCheck::OrphanPage,
            LintCheck::UnsupportedSection,
            LintCheck::DuplicateConcept,
            LintCheck::StaleInsight,
            LintCheck::HandEditedFile,
        ];
        for pair in ordered.windows(2) {
            assert!(pair[0] < pair[1]);
        }
    }

    #[test]
    fn normalized_title_folds_case_whitespace_and_unicode() {
        assert_eq!(normalized_title("Plugin  System"), "plugin system");
        assert_eq!(normalized_title("  PLUGIN system "), "plugin system");
        assert_eq!(
            normalized_title("Ｐｌｕｇｉｎ System"),
            normalized_title("plugin system"),
        );
    }

    #[test]
    fn unsupported_section_requires_substantive_prose() {
        let body = "## Related\n\n- [[Other]]\n- [[More]]\n".repeat(3);
        let prose = strip_wikilinks(&strip_citations(&body));
        assert!(prose.chars().count() < SUBSTANTIVE_SECTION_CHARS);

        let cited = "## Deep\n\n<!-- llm-wiki:cite claim=\"kn_x\" -->\n".to_owned();
        assert!(!scan_citations(&cited).is_empty());
    }

    #[test]
    fn report_counts_and_sorts_deterministically() {
        let mut report = LintReport::default();
        report.push(LintFinding {
            check: LintCheck::OrphanPage,
            severity: LintSeverity::Warning,
            page_slug: "b-page".into(),
            message: "m2".into(),
        });
        report.push(LintFinding {
            check: LintCheck::BrokenLink,
            severity: LintSeverity::Error,
            page_slug: "a-page".into(),
            message: "m1".into(),
        });
        report.push(LintFinding {
            check: LintCheck::BrokenLink,
            severity: LintSeverity::Error,
            page_slug: "a-page".into(),
            message: "m1".into(),
        });
        let report = report.finish();
        assert_eq!(report.findings.len(), 2, "duplicates removed");
        assert_eq!(report.errors(), 1);
        assert_eq!(report.warnings(), 1);
        assert_eq!(report.findings[0].check, LintCheck::BrokenLink);
        assert_eq!(report.findings[1].check, LintCheck::OrphanPage);
    }
}
