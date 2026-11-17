//! Deterministic aggregation pass (V0.6, PRD §14.2).
//!
//! Turns the flat compiled page set into a three-level structure:
//!
//! ```text
//! overview ──> domain ──> page ──> knowledge nodes
//!      index ──┘
//! ```
//!
//! Aggregate pages are ORDINARY `wiki_pages` rows that carry a reserved
//! `category` (`overview` / `index` / `domain`). That is the whole trick: FTS,
//! the Wiki Graph, `/v1/pages` and lint cover them with **no** schema change
//! and **no** new node vocabulary — they were already reading every page of
//! the active generation.
//!
//! Three invariants this module must not break:
//!
//! - **No LLM.** Every number here is a SQL-shaped fact over the page set
//!   (counts, inbound degree, knowledge-node frequency). Nothing is invented,
//!   so nothing needs a citation and nothing can hallucinate.
//! - **Determinism.** Every collection is a `BTreeMap`/`BTreeSet` and every
//!   page list is sorted by `(title, slug)`. Identical input must produce a
//!   byte-identical page.
//! - **Identity is assigned once (PRD §45).** Aggregate ids are *not* derived
//!   from names — an existing slug reuses the previous generation's id, and
//!   only a brand-new aggregate page calls [`WikiPageId::generate`].
//!
//! Aggregate pages are recomputed on EVERY build (incremental or not): their
//! content depends on the whole page set, so carrying one over would be stale
//! by construction. The cost is O(pages · log pages) in memory with zero
//! network round-trips.

use std::collections::{BTreeMap, BTreeSet};

use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::{BuildId, KnowledgeNodeId, WikiPageId};
use llm_wiki_core::plan::slugify;
use llm_wiki_storage::{PageLinkRecord, WikiPageRecord};

use crate::compile::{render_frontmatter, FrontmatterInput};

/// Reserved `category` of the single overview page.
pub const CATEGORY_OVERVIEW: &str = "overview";
/// Reserved `category` of the single global index page.
pub const CATEGORY_INDEX: &str = "index";
/// Reserved `category` of every domain page (one per source category).
///
/// Re-exported from storage so the graph's `aggregates` edge selection and this
/// pass can never disagree on the literal.
pub const CATEGORY_DOMAIN: &str = llm_wiki_storage::graph::CATEGORY_DOMAIN;

/// The domain vocabulary the planner may assign.
///
/// Re-exported from [`llm_wiki_core::plan::WIKI_CATEGORIES`] so the planner's
/// `UNKNOWN_CATEGORY` validation and this pass can never disagree. These
/// topics are registered BEFORE planning (see `build::register_domain_topics`):
/// the plan cache key includes the registry revision (PRD §14), so creating a
/// topic after planning would invalidate the very next build's plan and
/// regenerate every page id — breaking §4.8 rebuild determinism.
pub const PLANNER_CATEGORIES: &[&str] = llm_wiki_core::plan::WIKI_CATEGORIES;

/// Slug of the overview page (fixed: it is the entry point of the wiki).
pub const SLUG_OVERVIEW: &str = "_agg-overview";
/// Slug of the global index page.
pub const SLUG_INDEX: &str = "_agg-index";

/// True when a page was produced by this pass rather than by the
/// planner/compiler.
///
/// This is the single source of truth for the predicate: replan's plan diff
/// (which must NOT see aggregate pages as retired plan pages), the topology
/// guard and lint all call it instead of repeating a `matches!` on the
/// category strings and drifting apart.
pub fn is_aggregate_category(category: &str) -> bool {
    matches!(
        category,
        CATEGORY_OVERVIEW | CATEGORY_INDEX | CATEGORY_DOMAIN
    )
}

/// Tuning for one aggregation pass.
#[derive(Debug, Clone)]
pub struct AggregateOptions {
    /// How many most-cited pages the overview lists.
    pub top_hub_pages: usize,
    /// How many key concepts a domain page lists.
    pub top_concepts: usize,
    /// Frontmatter `schema_version`, mirroring the compiler's.
    pub schema_version: u32,
    /// category → 2-5 sentence summary (V0.6 Phase 3, opt-in).
    ///
    /// Empty means "deterministic only" — this pass never calls a model on its
    /// own. A missing key omits that domain's paragraph, so a partial map (a
    /// domain skipped for size) degrades gracefully instead of failing.
    pub summaries: BTreeMap<String, String>,
}

impl Default for AggregateOptions {
    fn default() -> Self {
        Self {
            top_hub_pages: 10,
            top_concepts: 12,
            schema_version: 1,
            summaries: BTreeMap::new(),
        }
    }
}

/// One domain's summary request (V0.6 Phase 3).
///
/// Carries ONLY titles and concept labels: the model never sees page bodies, so
/// it cannot invent specifics and cannot contradict what the pages say.
#[derive(Debug, Clone)]
pub struct DomainSummaryRequest {
    pub category: String,
    pub title: String,
    pub page_titles: Vec<String>,
    pub concepts: Vec<String>,
}

/// Result of one aggregation pass.
#[derive(Debug, Clone, Default)]
pub struct AggregateOutcome {
    /// The aggregate pages to append to the generation.
    pub pages: Vec<WikiPageRecord>,
    /// Number of domain pages produced (exposed for logs and tests).
    pub domains: usize,
}

impl AggregateOutcome {
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

/// One domain's key concept: the knowledge node, its display label and how
/// many pages of the domain reference it.
struct Concept {
    id: String,
    label: String,
    pages: usize,
}

/// Builds the aggregate pages for one generation.
///
/// * `pages` — the final page set of this build (recompiled + carried). Any
///   aggregate page already present is dropped first: they are always
///   recomputed.
/// * `labels` — display names for ACTIVE knowledge nodes
///   ([`llm_wiki_storage::load_active_labels`]); a node without a label is
///   skipped rather than shown as an opaque id.
/// * `previous` — slug → page id of the PREVIOUS generation's aggregate
///   pages, so identity survives a rebuild.
pub fn aggregate_pages(
    pages: &[WikiPageRecord],
    labels: &BTreeMap<String, String>,
    topics: &BTreeMap<String, KnowledgeNodeId>,
    previous: &BTreeMap<String, WikiPageId>,
    build_id: &BuildId,
    options: &AggregateOptions,
) -> AggregateOutcome {
    let groups = group_by_category(pages);
    if groups.is_empty() {
        return AggregateOutcome::default();
    }
    let language = dominant_language(pages);
    let layout = AggregateLayout::resolve(groups, previous, &language);
    let inbound = inbound_degrees(pages);

    let mut out = domain_pages(&layout, labels, topics, build_id, options);
    out.push(overview_page(&layout, &inbound, build_id, options));
    out.push(index_page(&layout, build_id, options));

    AggregateOutcome {
        domains: layout.groups.len(),
        pages: out,
    }
}

/// Groups the ordinary pages by `category`, members sorted by `(title, slug)`.
///
/// The single source of that ordering: the rendered domain page and the summary
/// request must list pages identically, otherwise a cached summary would
/// describe a different page set than the body shows.
fn group_by_category(pages: &[WikiPageRecord]) -> BTreeMap<String, Vec<&WikiPageRecord>> {
    let mut groups: BTreeMap<String, Vec<&WikiPageRecord>> = BTreeMap::new();
    for page in pages
        .iter()
        .filter(|page| !is_aggregate_category(&page.category))
    {
        groups.entry(page.category.clone()).or_default().push(page);
    }
    for members in groups.values_mut() {
        members.sort_by(|a, b| a.title.cmp(&b.title).then_with(|| a.slug.cmp(&b.slug)));
    }
    groups
}

/// Inbound degree over the ordinary pages only: an aggregate page citing a
/// page is not evidence that the page is a hub.
fn inbound_degrees(pages: &[WikiPageRecord]) -> BTreeMap<String, usize> {
    let mut inbound: BTreeMap<String, usize> = BTreeMap::new();
    for page in pages
        .iter()
        .filter(|page| !is_aggregate_category(&page.category))
    {
        for link in &page.links {
            if link.to_page_id == page.page_id {
                continue;
            }
            *inbound.entry(link.to_page_id.as_str().to_owned()).or_insert(0) += 1;
        }
    }
    inbound
}

/// Ids and titles shared by every aggregate page of one pass.
///
/// Resolved once because the three page kinds cross-link each other: the
/// overview needs the index id, the index needs the overview id, and both need
/// every domain id.
struct AggregateLayout<'a> {
    groups: BTreeMap<String, Vec<&'a WikiPageRecord>>,
    language: String,
    overview_id: WikiPageId,
    index_id: WikiPageId,
    overview_title: String,
    index_title: String,
    domain_ids: BTreeMap<String, WikiPageId>,
    domain_titles: BTreeMap<String, String>,
}

impl<'a> AggregateLayout<'a> {
    fn resolve(
        groups: BTreeMap<String, Vec<&'a WikiPageRecord>>,
        previous: &BTreeMap<String, WikiPageId>,
        language: &str,
    ) -> Self {
        let zh = is_chinese(language);
        let domain_titles = groups
            .keys()
            .map(|category| (category.clone(), domain_title(language, category)))
            .collect();
        let domain_ids = groups
            .keys()
            .map(|category| (category.clone(), id_for(previous, &domain_slug(category))))
            .collect();
        Self {
            groups,
            language: language.to_owned(),
            overview_id: id_for(previous, SLUG_OVERVIEW),
            index_id: id_for(previous, SLUG_INDEX),
            overview_title: if zh { "知识全景" } else { "Overview" }.to_owned(),
            index_title: if zh { "索引" } else { "Index" }.to_owned(),
            domain_ids,
            domain_titles,
        }
    }

    fn domain_title_of(&self, category: &str) -> String {
        self.domain_titles
            .get(category)
            .cloned()
            .unwrap_or_else(|| domain_title(&self.language, category))
    }

    fn domain_id_of(&self, category: &str) -> WikiPageId {
        self.domain_ids
            .get(category)
            .cloned()
            .expect("layout resolved an id for every group")
    }

    fn domain_links(&self) -> Vec<PageLinkRecord> {
        self.groups
            .keys()
            .map(|category| PageLinkRecord {
                to_page_id: self.domain_id_of(category),
                target_title: self.domain_title_of(category),
            })
            .collect()
    }
}

/// One page per category: its members plus the knowledge they share, led by the
/// domain's own `topic` node.
fn domain_pages(
    layout: &AggregateLayout<'_>,
    labels: &BTreeMap<String, String>,
    topics: &BTreeMap<String, KnowledgeNodeId>,
    build_id: &BuildId,
    options: &AggregateOptions,
) -> Vec<WikiPageRecord> {
    let mut out = Vec::with_capacity(layout.groups.len());
    for (category, members) in &layout.groups {
        let concepts = key_concepts(members, labels, options.top_concepts);
        let body = render_domain_body(
            members,
            &concepts,
            &layout.language,
            options.summaries.get(category).map(String::as_str),
        );
        let links: Vec<PageLinkRecord> = members
            .iter()
            .map(|page| PageLinkRecord {
                to_page_id: page.page_id.clone(),
                target_title: page.title.clone(),
            })
            .collect();
        let knowledge_refs = domain_refs(topics.get(category.as_str()), &concepts);
        out.push(finish_page(PageParts {
            id: layout.domain_id_of(category),
            slug: domain_slug(category),
            title: layout.domain_title_of(category),
            category: CATEGORY_DOMAIN,
            language: &layout.language,
            body,
            links,
            knowledge_refs,
            build_id,
            schema_version: options.schema_version,
        }));
    }
    out
}

/// The domain page's knowledge refs: its own `topic` node first, then the
/// shared concepts.
///
/// The topic leads because the domain IS a knowledge unit (filled by the
/// caller) — referencing it is what makes it first-class rather than a label
/// on pages, and leading the list means a reader (and the graph's `contains`
/// scan) meets the aggregate before its members.
fn domain_refs(topic: Option<&KnowledgeNodeId>, concepts: &[Concept]) -> Vec<KnowledgeNodeId> {
    let mut refs: Vec<KnowledgeNodeId> = Vec::with_capacity(concepts.len() + 1);
    if let Some(topic) = topic {
        refs.push(topic.clone());
    }
    refs.extend(
        concepts
            .iter()
            .map(|concept| KnowledgeNodeId::from_validated(concept.id.clone())),
    );
    refs
}

fn overview_page(
    layout: &AggregateLayout<'_>,
    inbound: &BTreeMap<String, usize>,
    build_id: &BuildId,
    options: &AggregateOptions,
) -> WikiPageRecord {
    let ordinary: Vec<&WikiPageRecord> = layout.groups.values().flatten().copied().collect();
    let hubs = hub_pages(&ordinary, inbound, options.top_hub_pages);
    let body = render_overview_body(
        &layout.groups,
        &layout.domain_titles,
        &hubs,
        &layout.index_title,
        &layout.language,
    );
    let mut links = layout.domain_links();
    links.push(PageLinkRecord {
        to_page_id: layout.index_id.clone(),
        target_title: layout.index_title.clone(),
    });
    finish_page(PageParts {
        id: layout.overview_id.clone(),
        slug: SLUG_OVERVIEW.to_owned(),
        title: layout.overview_title.clone(),
        category: CATEGORY_OVERVIEW,
        language: &layout.language,
        body,
        links,
        knowledge_refs: Vec::new(),
        build_id,
        schema_version: options.schema_version,
    })
}

fn index_page(
    layout: &AggregateLayout<'_>,
    build_id: &BuildId,
    options: &AggregateOptions,
) -> WikiPageRecord {
    let body = render_index_body(
        &layout.groups,
        &layout.domain_titles,
        &layout.overview_title,
        &layout.language,
    );
    let member_count: usize = layout.groups.values().map(Vec::len).sum();
    let mut links: Vec<PageLinkRecord> =
        Vec::with_capacity(member_count + layout.groups.len() + 1);
    links.push(PageLinkRecord {
        to_page_id: layout.overview_id.clone(),
        target_title: layout.overview_title.clone(),
    });
    for (category, members) in &layout.groups {
        links.push(PageLinkRecord {
            to_page_id: layout.domain_id_of(category),
            target_title: layout.domain_title_of(category),
        });
        for page in members {
            links.push(PageLinkRecord {
                to_page_id: page.page_id.clone(),
                target_title: page.title.clone(),
            });
        }
    }
    finish_page(PageParts {
        id: layout.index_id.clone(),
        slug: SLUG_INDEX.to_owned(),
        title: layout.index_title.clone(),
        category: CATEGORY_INDEX,
        language: &layout.language,
        body,
        links,
        knowledge_refs: Vec::new(),
        build_id,
        schema_version: options.schema_version,
    })
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

struct PageParts<'a> {
    id: WikiPageId,
    slug: String,
    title: String,
    category: &'static str,
    language: &'a str,
    body: String,
    links: Vec<PageLinkRecord>,
    knowledge_refs: Vec<KnowledgeNodeId>,
    build_id: &'a BuildId,
    schema_version: u32,
}

/// Renders frontmatter + title + body and hashes the result — the exact same
/// contract the compiler uses, so a page is a page regardless of who wrote it.
fn finish_page(parts: PageParts<'_>) -> WikiPageRecord {
    // Aggregate pages cite nothing: every statement is a count over the page
    // set, which is machine-derived rather than sourced prose. `sources: []`
    // is the legal empty form (lint's missing-source check iterates citations
    // and therefore skips a page with none).
    let no_sources: [String; 0] = [];
    let mut content = render_frontmatter(&FrontmatterInput {
        page_id: parts.id.as_str(),
        slug: &parts.slug,
        title: &parts.title,
        category: parts.category,
        schema_version: parts.schema_version,
        language: parts.language,
        build_id: parts.build_id.as_str(),
        sources: &no_sources,
    });
    content.push_str(&format!("\n# {}\n\n", parts.title));
    content.push_str(parts.body.trim());
    content.push('\n');
    WikiPageRecord {
        page_id: parts.id,
        slug: parts.slug,
        title: parts.title,
        category: parts.category.to_owned(),
        language: parts.language.to_owned(),
        body_hash: sha256_hex(content.as_bytes()),
        content,
        knowledge_refs: parts.knowledge_refs,
        citations: Vec::new(),
        links: parts.links,
    }
}

fn render_domain_body(
    members: &[&WikiPageRecord],
    concepts: &[Concept],
    language: &str,
    summary: Option<&str>,
) -> String {
    let zh = is_chinese(language);
    let mut out = String::new();
    // The summary (Phase 3, opt-in) leads the body: it orients the reader
    // before the page list. Prose without citations by design — the prompt
    // only receives titles and concept labels, so it cannot assert anything
    // the pages do not already carry.
    if let Some(summary) = summary.map(str::trim).filter(|text| !text.is_empty()) {
        out.push_str(summary);
        out.push_str("\n\n");
    }
    out.push_str(if zh {
        "## 页面\n\n"
    } else {
        "## Pages\n\n"
    });
    for page in members {
        out.push_str(&format!("- [[{}]]\n", page.title));
    }
    if !concepts.is_empty() {
        out.push_str(if zh {
            "\n## 关键概念\n\n"
        } else {
            "\n## Key concepts\n\n"
        });
        for concept in concepts {
            out.push_str(&format!(
                "- {} — {}\n",
                concept.label,
                if zh {
                    format!("{} 个页面涉及", concept.pages)
                } else {
                    format!("referenced by {} page(s)", concept.pages)
                }
            ));
        }
    }
    out
}

fn render_overview_body(
    groups: &BTreeMap<String, Vec<&WikiPageRecord>>,
    domain_titles: &BTreeMap<String, String>,
    hubs: &[(String, usize)],
    index_title: &str,
    language: &str,
) -> String {
    let zh = is_chinese(language);
    let page_count: usize = groups.values().map(Vec::len).sum();
    let mut out = String::new();
    out.push_str(&if zh {
        format!("共 {page_count} 个知识页面，分为 {} 个域。\n", groups.len())
    } else {
        format!(
            "{page_count} knowledge page(s) across {} domain(s).\n",
            groups.len()
        )
    });
    out.push_str(if zh {
        "\n## 域\n\n"
    } else {
        "\n## Domains\n\n"
    });
    for (category, members) in groups {
        let title = domain_titles
            .get(category.as_str())
            .map(String::as_str)
            .unwrap_or(category.as_str());
        out.push_str(&format!(
            "- [[{}]] — {} {}\n",
            title,
            members.len(),
            if zh { "页" } else { "page(s)" }
        ));
    }
    if !hubs.is_empty() {
        out.push_str(if zh {
            "\n## 枢纽页面\n\n"
        } else {
            "\n## Hub pages\n\n"
        });
        for (title, count) in hubs {
            out.push_str(&format!(
                "- [[{}]] — {}\n",
                title,
                if zh {
                    format!("被 {count} 页引用")
                } else {
                    format!("referenced by {count} page(s)")
                }
            ));
        }
    }
    out.push_str(if zh {
        "\n## 索引\n\n"
    } else {
        "\n## Index\n\n"
    });
    out.push_str(&format!("- [[{index_title}]]\n"));
    out
}

fn render_index_body(
    groups: &BTreeMap<String, Vec<&WikiPageRecord>>,
    domain_titles: &BTreeMap<String, String>,
    overview_title: &str,
    language: &str,
) -> String {
    let zh = is_chinese(language);
    let mut out = String::new();
    out.push_str(if zh {
        "## 概览\n\n"
    } else {
        "## Overview\n\n"
    });
    out.push_str(&format!("- [[{overview_title}]]\n"));
    out.push_str(if zh {
        "\n## 域\n\n"
    } else {
        "\n## Domains\n\n"
    });
    for (category, members) in groups {
        let title = domain_titles
            .get(category.as_str())
            .map(String::as_str)
            .unwrap_or(category.as_str());
        out.push_str(&format!(
            "- [[{}]] — {} {}\n",
            title,
            members.len(),
            if zh { "页" } else { "page(s)" }
        ));
    }
    out.push_str(if zh {
        "\n## 全部页面\n"
    } else {
        "\n## All pages\n"
    });
    for (category, members) in groups {
        let title = domain_titles
            .get(category.as_str())
            .map(String::as_str)
            .unwrap_or(category.as_str());
        out.push_str(&format!("\n### {title}\n\n"));
        for page in members {
            out.push_str(&format!("- [[{}]]\n", page.title));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Derivation helpers
// ---------------------------------------------------------------------------

/// Reuses the previous generation's id for this slug, or assigns a fresh one.
///
/// PRD §45 forbids deriving an id from a name; "keep the id the previous
/// generation used" is the sanctioned way to stay stable across rebuilds.
fn id_for(previous: &BTreeMap<String, WikiPageId>, slug: &str) -> WikiPageId {
    previous
        .get(slug)
        .cloned()
        .unwrap_or_else(WikiPageId::generate)
}

fn domain_slug(category: &str) -> String {
    let slug = slugify(category);
    if slug.is_empty() {
        format!("{CATEGORY_DOMAIN}-other")
    } else {
        format!("_agg-{CATEGORY_DOMAIN}-{slug}")
    }
}

/// The language tag most ordinary pages carry, ties broken by the
/// lexicographically smallest tag so the choice is deterministic.
fn majority_language(pages: &[&WikiPageRecord]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for page in pages {
        *counts.entry(page.language.as_str()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map(|(language, _)| language.to_owned())
        .unwrap_or_else(|| "und".to_owned())
}

fn is_chinese(language: &str) -> bool {
    language.to_lowercase().starts_with("zh")
}

fn category_label_zh(category: &str) -> &str {
    match category {
        "concepts" => "概念",
        "architecture" => "架构",
        "security" => "安全",
        "guides" => "指南",
        "reference" => "参考",
        other => other,
    }
}

fn domain_title(language: &str, category: &str) -> String {
    if is_chinese(language) {
        format!("域：{}", category_label_zh(category))
    } else {
        format!("Domain: {category}")
    }
}

/// Language tag most ordinary pages carry, ties broken lexicographically.
pub fn dominant_language(pages: &[WikiPageRecord]) -> String {
    let refs: Vec<&WikiPageRecord> = pages
        .iter()
        .filter(|page| !is_aggregate_category(&page.category))
        .collect();
    majority_language(&refs)
}

/// Builds one summary request per domain (V0.6 Phase 3).
///
/// Ordering matches `aggregate_pages` (BTreeMap by category, members sorted by
/// title then slug), so the same page set always yields the same request —
/// which is what makes the stage cache effective.
pub fn domain_summary_requests(
    pages: &[WikiPageRecord],
    labels: &BTreeMap<String, String>,
    language: &str,
    top_concepts: usize,
) -> Vec<DomainSummaryRequest> {
    group_by_category(pages)
        .into_iter()
        .map(|(category, members)| {
            let concepts = key_concepts(&members, labels, top_concepts);
            DomainSummaryRequest {
                title: domain_title(language, &category),
                category,
                page_titles: members.iter().map(|page| page.title.clone()).collect(),
                concepts: concepts.into_iter().map(|concept| concept.label).collect(),
            }
        })
        .collect()
}

/// The domain's shared knowledge nodes, ranked by how many member pages
/// reference them (each page counts once per node).
fn key_concepts(
    members: &[&WikiPageRecord],
    labels: &BTreeMap<String, String>,
    limit: usize,
) -> Vec<Concept> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for page in members {
        let unique: BTreeSet<&str> = page
            .knowledge_refs
            .iter()
            .map(|id| id.as_str())
            .collect();
        for id in unique {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<(&str, usize)> = counts
        .into_iter()
        .filter(|(id, _)| labels.contains_key(*id))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked.truncate(limit);
    ranked
        .into_iter()
        .map(|(id, pages)| Concept {
            id: id.to_owned(),
            label: labels
                .get(id)
                .cloned()
                .unwrap_or_else(|| id.to_owned()),
            pages,
        })
        .collect()
}

/// Ordinary pages ranked by inbound links (hubs = shared dependencies).
fn hub_pages(
    pages: &[&WikiPageRecord],
    inbound: &BTreeMap<String, usize>,
    limit: usize,
) -> Vec<(String, usize)> {
    let mut ranked: Vec<(&str, usize)> = pages
        .iter()
        .filter_map(|page| {
            inbound
                .get(page.page_id.as_str())
                .filter(|count| **count > 0)
                .map(|count| (page.title.as_str(), *count))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ranked.truncate(limit);
    ranked
        .into_iter()
        .map(|(title, count)| (title.to_owned(), count))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_core::ids::WikiPageId;

    fn page(slug: &str, title: &str, category: &str, refs: &[&str]) -> WikiPageRecord {
        WikiPageRecord {
            page_id: WikiPageId::generate(),
            slug: slug.to_owned(),
            title: title.to_owned(),
            category: category.to_owned(),
            language: "zh-CN".to_owned(),
            body_hash: format!("hash-{slug}"),
            content: format!("# {title}\n"),
            knowledge_refs: refs
                .iter()
                .map(|id| KnowledgeNodeId::from_validated((*id).to_owned()))
                .collect(),
            citations: Vec::new(),
            links: Vec::new(),
        }
    }

    fn labels(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(id, label)| ((*id).to_owned(), (*label).to_owned()))
            .collect()
    }

    #[test]
    fn grouping_sorts_members_and_ignores_aggregate_pages() {
        let pages = vec![
            page("b", "Beta", "concepts", &[]),
            page("a", "Alpha", "concepts", &[]),
            page(SLUG_OVERVIEW, "Overview", CATEGORY_OVERVIEW, &[]),
            page("c", "Gamma", "security", &[]),
        ];
        let groups = group_by_category(&pages);
        assert_eq!(groups.len(), 2, "only ordinary categories are grouped");
        let titles: Vec<&str> = groups["concepts"]
            .iter()
            .map(|page| page.title.as_str())
            .collect();
        assert_eq!(titles, ["Alpha", "Beta"], "members sort by title");

        // The same ordering feeds the rendered page AND the summary request, so
        // a cached summary always describes the page list the body shows.
        let requests = domain_summary_requests(&pages, &BTreeMap::new(), "zh-CN", 5);
        let concepts = requests
            .iter()
            .find(|request| request.category == "concepts")
            .expect("the concepts domain is requested");
        assert_eq!(concepts.page_titles, ["Alpha", "Beta"]);
    }

    #[test]
    fn aggregate_is_deterministic_for_identical_input() {
        let pages = vec![
            page("auth", "认证流程", "architecture", &["kn_a"]),
            page("perm", "权限模型", "architecture", &["kn_a", "kn_b"]),
            page("secret", "密钥轮换", "security", &["kn_b"]),
        ];
        let labels = labels(&[("kn_a", "Plugin System"), ("kn_b", "Permission Model")]);
        let previous = BTreeMap::new();
        let build_id = BuildId::generate();
        let options = AggregateOptions::default();

        let no_topics: BTreeMap<String, KnowledgeNodeId> = BTreeMap::new();
        let first = aggregate_pages(
            &pages, &labels, &no_topics, &previous, &build_id, &options,
        );
        // Same inputs, same previous (now including what the first pass
        // produced) — the ids must be reused, not regenerated.
        let previous_after: BTreeMap<String, WikiPageId> = first
            .pages
            .iter()
            .map(|page| (page.slug.clone(), page.page_id.clone()))
            .collect();
        let second = aggregate_pages(
            &pages,
            &labels,
            &no_topics,
            &previous_after,
            &build_id,
            &options,
        );

        assert_eq!(first.pages.len(), second.pages.len());
        for (a, b) in first.pages.iter().zip(second.pages.iter()) {
            assert_eq!(a.page_id, b.page_id, "aggregate ids must be stable");
            assert_eq!(a.slug, b.slug);
            assert_eq!(a.content, b.content);
            assert_eq!(a.body_hash, b.body_hash);
        }
    }

    #[test]
    fn one_domain_page_per_category_plus_overview_and_index() {
        let pages = vec![
            page("auth", "认证流程", "architecture", &["kn_a"]),
            page("secret", "密钥轮换", "security", &["kn_b"]),
        ];
        let labels = labels(&[("kn_a", "Plugin System"), ("kn_b", "Permission Model")]);
        let outcome = aggregate_pages(
            &pages,
            &labels,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BuildId::generate(),
            &AggregateOptions::default(),
        );

        assert_eq!(outcome.domains, 2);
        assert_eq!(outcome.pages.len(), 4); // 2 domains + overview + index
        let categories: Vec<&str> = outcome
            .pages
            .iter()
            .map(|page| page.category.as_str())
            .collect();
        assert_eq!(
            categories,
            [CATEGORY_DOMAIN, CATEGORY_DOMAIN, CATEGORY_OVERVIEW, CATEGORY_INDEX]
        );
    }

    #[test]
    fn every_ordinary_page_gains_an_inbound_link() {
        let pages = vec![
            page("a", "A", "architecture", &[]),
            page("b", "B", "security", &[]),
        ];
        let outcome = aggregate_pages(
            &pages,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BuildId::generate(),
            &AggregateOptions::default(),
        );
        let linked: BTreeSet<&str> = outcome
            .pages
            .iter()
            .flat_map(|page| page.links.iter())
            .map(|link| link.to_page_id.as_str())
            .collect();
        for ordinary in &pages {
            assert!(
                linked.contains(ordinary.page_id.as_str()),
                "page {} must be reachable from the aggregate layer",
                ordinary.slug
            );
        }
    }

    #[test]
    fn aggregate_pages_are_dropped_before_regrouping() {
        let pages = vec![
            page("auth", "认证流程", "architecture", &[]),
            page(SLUG_OVERVIEW, "旧全景", CATEGORY_OVERVIEW, &[]),
            page("_agg-domain-old", "旧域", CATEGORY_DOMAIN, &[]),
        ];
        let outcome = aggregate_pages(
            &pages,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BuildId::generate(),
            &AggregateOptions::default(),
        );
        // Exactly one domain (architecture) — the stale domain page must not
        // be treated as a category.
        assert_eq!(outcome.domains, 1);
        assert!(outcome
            .pages
            .iter()
            .all(|page| page.slug != "_agg-domain-old"));
    }

    #[test]
    fn empty_page_set_yields_nothing() {
        let outcome = aggregate_pages(
            &[],
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BuildId::generate(),
            &AggregateOptions::default(),
        );
        assert!(outcome.is_empty());
    }

    #[test]
    fn key_concepts_rank_by_page_spread() {
        let pages = [
            page("a", "A", "architecture", &["kn_a", "kn_b"]),
            page("b", "B", "architecture", &["kn_a"]),
            page("c", "C", "architecture", &["kn_a"]),
        ];
        let labels = labels(&[("kn_a", "Core"), ("kn_b", "Minor")]);
        let concepts = key_concepts(&pages.iter().collect::<Vec<_>>(), &labels, 10);
        assert_eq!(concepts[0].id, "kn_a");
        assert_eq!(concepts[0].pages, 3);
        assert_eq!(concepts[1].id, "kn_b");
    }
}
