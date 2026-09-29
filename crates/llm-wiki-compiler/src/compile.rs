//! Wiki page compiler (PRD §15/§16).
//!
//! Per page: the LLM writes the body strictly from the page's knowledge nodes
//! and cites claims by registry id (`<!-- llm-wiki:cite claim="kn_..." -->`).
//! Application code then validates the response (shape → referential →
//! semantic, one repair), **expands** every citation from database-side
//! anchors (source path, heading path, range, evidence digest — never taken
//! from the model), resolves WikiLinks to `WikiPageId`s and renders the
//! app-owned frontmatter (`generated`, `schema_version`, `language`, build id,
//! sources).

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use serde::Deserialize;

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::{BuildId, KnowledgeNodeId, WikiPageId};
use llm_wiki_core::model::WikiPlan;
use llm_wiki_core::plan::{estimate_tokens, KnowledgeBase, PlanAnchor};
use llm_wiki_llm::structured;
use llm_wiki_llm::{LlmProvider, LlmRequest, LlmResponse};
use llm_wiki_storage::{PageCitationRecord, PageLinkRecord, WikiPageRecord};

use crate::cache::{generate_cached, remember_validated, repair_request, StageCache};
use crate::prompt::PromptDocument;

/// Upper bound of the per-page output budget (`build_page_request` clamps
/// every page to this ceiling); `replan --dry-run` uses it for the estimated
/// compile cost upper bound (PRD §19.2).
pub const MAX_PAGE_OUTPUT_TOKENS: u32 = 16_384;

#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Instruction for the writing language (goes into the prompt).
    pub language: String,
    /// BCP-47-ish tag persisted in the page frontmatter (`und` when unknown;
    /// source-language persistence lands with the V0.2 CJK work).
    pub language_tag: String,
    /// Frontmatter `schema_version` (PRD §15.2).
    pub schema_version: u32,
    /// Input budget for one page-compilation prompt (PRD §14: over-budget
    /// payloads must fail closed — never truncate, never widen the window).
    pub max_input_tokens: u64,
    /// FLOOR for the per-page output ceiling: the compile stage scales its
    /// own estimate with the page's knowledge, but thinking models spend
    /// chain-of-thought from the same budget, so the estimate never goes
    /// below the configured `[llm] max_output_tokens`.
    pub min_output_tokens: u32,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            language: "the sources' language".to_owned(),
            language_tag: "und".to_owned(),
            schema_version: 1,
            max_input_tokens: 32_000,
            min_output_tokens: 4096,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompiledGeneration {
    pub pages: Vec<WikiPageRecord>,
    pub llm_request_count: u32,
}

pub struct WikiCompiler {
    provider: Arc<dyn LlmProvider>,
    prompt: PromptDocument,
    config: CompilerConfig,
    /// §28 stage cache; only grounded, citation-valid responses are stored.
    cache: Option<Arc<dyn StageCache>>,
}

impl WikiCompiler {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        prompt: PromptDocument,
        config: CompilerConfig,
    ) -> Self {
        Self {
            provider,
            prompt,
            config,
            cache: None,
        }
    }

    /// Wires the §28 cache: lookups short-circuit identical page requests,
    /// writes happen only after the body validated (grounded + citations
    /// resolvable + links resolvable, PRD §15/§28).
    pub fn with_cache(mut self, cache: Arc<dyn StageCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Compiles every page of the plan. Compilation failure aborts the whole
    /// generation — no partial wiki is produced (PRD §34).
    pub async fn compile_plan(
        &self,
        plan: &WikiPlan,
        base: &KnowledgeBase,
        build_id: &BuildId,
    ) -> Result<CompiledGeneration> {
        let all: BTreeSet<WikiPageId> = plan.pages.iter().map(|page| page.id.clone()).collect();
        self.compile_plan_subset(plan, base, build_id, &all).await
    }

    /// Compiles ONLY the pages in `page_ids` — the incremental pipeline
    /// (PRD §19.2) recompiles the affected set and carries every other page
    /// over verbatim. `plan` must still contain ALL surviving pages:
    /// WikiLink/Related resolution needs the full title map of the
    /// generation, while only the subset consumes LLM requests.
    pub async fn compile_plan_subset(
        &self,
        plan: &WikiPlan,
        base: &KnowledgeBase,
        build_id: &BuildId,
        page_ids: &BTreeSet<WikiPageId>,
    ) -> Result<CompiledGeneration> {
        let mut title_to_id: BTreeMap<String, WikiPageId> = BTreeMap::new();
        for page in &plan.pages {
            let folded = page.title.trim().to_lowercase();
            if title_to_id
                .insert(folded.clone(), page.id.clone())
                .is_some()
            {
                return Err(WikiError::Compilation(format!(
                    "plan contains duplicate page title '{}'; titles must be unique for WikiLink resolution",
                    folded
                )));
            }
        }

        let mut pages = Vec::new();
        let mut llm_request_count = 0u32;
        for page in &plan.pages {
            if !page_ids.contains(&page.id) {
                continue; // carried over verbatim by the caller (PRD §19)
            }
            let (record, requests) = self
                .compile_page(page, plan, base, build_id, &title_to_id)
                .await?;
            llm_request_count += requests;
            pages.push(record);
        }
        Ok(CompiledGeneration {
            pages,
            llm_request_count,
        })
    }

    /// Length note: ~108 lines — the single-page pipeline (validate → repair → expand citations → resolve links); stages share local state and run in a fixed order.
    async fn compile_page(
        &self,
        page: &llm_wiki_core::model::WikiPagePlan,
        plan: &WikiPlan,
        base: &KnowledgeBase,
        build_id: &BuildId,
        title_to_id: &BTreeMap<String, WikiPageId>,
    ) -> Result<(WikiPageRecord, u32)> {
        for node_id in &page.knowledge_refs {
            if !base.nodes.contains_key(node_id) {
                return Err(WikiError::Compilation(format!(
                    "page '{}' references unknown knowledge node '{}'",
                    page.title, node_id
                )));
            }
        }

        let claim_ids: BTreeSet<String> = page
            .knowledge_refs
            .iter()
            .filter(|node_id| base.nodes[*node_id].kind == "claim")
            .map(|node_id| node_id.as_str().to_owned())
            .collect();
        let related: Vec<(String, String)> = page
            .related_pages
            .iter()
            .filter_map(|related_id| {
                plan.pages
                    .iter()
                    .find(|candidate| &candidate.id == related_id)
                    .map(|candidate| (related_id.as_str().to_owned(), candidate.title.clone()))
            })
            .collect();
        let anchors_by_claim = claim_anchors(base, page);
        // Claims the page is allowed to cite but that have no stored evidence
        // anchors: citing them can never expand, so it must fail validation.
        let unanchored: BTreeSet<String> = claim_ids
            .iter()
            .filter(|claim_id| !anchors_by_claim.contains_key(*claim_id))
            .cloned()
            .collect();
        let context = CompileContext {
            claim_ids: &claim_ids,
            unanchored: &unanchored,
            link_targets: title_to_id,
            page_claim_count: claim_ids.len(),
        };

        let bundle = self.build_page_request(page, base, &related)?;

        // ---- Stage 1 (+ one repair, PRD §11/§15/§28) ----
        // Hits from the §28 cache consume no LLM request (§37.3).
        let mut llm_request_count = 0u32;
        let (first_response, added) =
            generate_cached(&self.provider, self.cache.as_ref(), bundle.request.clone()).await?;
        llm_request_count += added;

        // Held until the body fully validates; only then enters the cache.
        let first = structured::parse_json::<RawCompiledPage>(&first_response.text).map(|raw| {
            let issues = validate_body(&raw.markdown, &context);
            (raw, issues)
        });
        let (raw, validated) = match first {
            Ok((raw, issues)) if issues.is_empty() => {
                (raw, Some((bundle.request.clone(), first_response)))
            }
            Ok((_, issues)) => {
                tracing::warn!(
                    issues = ?issues,
                    page = %page.title,
                    "compile validation failed, repairing once"
                );
                let (raw, request, response, added) =
                    self.repair_round(&bundle, &issues, &context).await?;
                llm_request_count += added;
                (raw, Some((request, response)))
            }
            Err(stage1) => {
                tracing::warn!(
                    reason = %stage1,
                    page = %page.title,
                    "compile stage-1 failed, repairing once"
                );
                let (raw, request, response, added) = self
                    .repair_round(&bundle, &[stage1.machine_reason()], &context)
                    .await?;
                llm_request_count += added;
                (raw, Some((request, response)))
            }
        };
        // The body passed grounding/citation/link validation: cache it.
        if let Some((request, response)) = validated {
            remember_validated(self.cache.as_ref(), &request, &response);
        }

        // ---- Citation expansion from DB-side anchors (PRD §16) ----
        let (body, citations, sources) = expand_citations(&raw.markdown, &anchors_by_claim);
        let record = self.render_page(PageRender {
            page,
            related: &related,
            body: &body,
            sources: &sources,
            build_id,
            title_to_id,
            citations,
        });
        Ok((record, llm_request_count))
    }

    /// Builds the compilation prompt for one page: payloads, the input budget
    /// gate (PRD §14 — fail closed, never truncate) and the rendered request.
    fn build_page_request(
        &self,
        page: &llm_wiki_core::model::WikiPagePlan,
        base: &KnowledgeBase,
        related: &[(String, String)],
    ) -> Result<PageRequestBundle> {
        let page_json = serde_json::json!({
            "title": page.title,
            "category": page.category,
            "purpose": page.purpose,
        })
        .to_string();
        let knowledge_json = page_knowledge_payload(base, page);
        let related_json = serde_json::json!(related
            .iter()
            .map(|(related_id, title)| serde_json::json!({ "id": related_id, "title": title }))
            .collect::<Vec<_>>())
        .to_string();

        let input_estimate = estimate_tokens(&format!(
            "{page_json}\u{1f}{knowledge_json}\u{1f}{related_json}"
        ));
        if input_estimate > self.config.max_input_tokens {
            return Err(WikiError::Compilation(format!(
                "page '{}' compilation input is ~{input_estimate} tokens, above max_input_tokens = {}; the plan must split this page (PRD §14 forbids exceeding input budgets)",
                page.title, self.config.max_input_tokens
            )));
        }

        // Output scales with the knowledge the page may render.
        let max_output_tokens = (estimate_tokens(&knowledge_json) * 6 + 2_000)
            .clamp(2_048, u64::from(MAX_PAGE_OUTPUT_TOKENS))
            .max(u64::from(self.config.min_output_tokens)) as u32;

        let template = self.prompt.render(&[
            ("LANGUAGE", &self.config.language),
            ("PAGE", &page_json),
            ("KNOWLEDGE", &knowledge_json),
            ("RELATED", &related_json),
        ]);
        Ok(PageRequestBundle {
            request: LlmRequest {
                task_tag: self.prompt.name.clone(),
                system: None,
                prompt: template.replace("{{REPAIR_NOTES}}", ""),
                temperature: 0.0,
                max_output_tokens,
                json_mode: true,
            },
            template,
        })
    }

    /// Renders the final page record: app-owned frontmatter (PRD §15.2),
    /// body, Related section and link resolution over the FINAL content
    /// (PRD §15.2: app-generated links persist to ids too).
    fn render_page(&self, render: PageRender<'_>) -> WikiPageRecord {
        let page = render.page;
        let title = page.title.clone();
        let mut content = render_frontmatter(&FrontmatterInput {
            page_id: page.id.as_str(),
            slug: &page.slug,
            title: &title,
            category: &page.category,
            schema_version: self.config.schema_version,
            language: &self.config.language_tag,
            build_id: render.build_id.as_str(),
            sources: render.sources,
        });
        content.push_str(&format!("\n# {title}\n\n"));
        content.push_str(render.body.trim());
        content.push('\n');
        if !page.related_pages.is_empty() {
            content.push_str("\n## Related\n\n");
            for (_, related_title) in render.related {
                content.push_str(&format!("- [[{related_title}]]\n"));
            }
        }
        let links = resolve_links(&content, render.title_to_id);
        WikiPageRecord {
            page_id: page.id.clone(),
            slug: page.slug.clone(),
            title,
            category: page.category.clone(),
            language: self.config.language_tag.clone(),
            body_hash: sha256_hex(content.as_bytes()),
            content,
            knowledge_refs: page.knowledge_refs.clone(),
            citations: render.citations,
            links,
        }
    }

    /// One repair round (PRD §11/§15: exactly one): injects machine-readable
    /// reasons into the untouched `{{REPAIR_NOTES}}` slot and validates the
    /// result, failing closed on residual issues. Returns the parsed page plus
    /// the (request, response) pair so the caller can cache the response that
    /// finally validated (PRD §28).
    async fn repair_round(
        &self,
        bundle: &PageRequestBundle,
        reasons: &[String],
        context: &CompileContext<'_>,
    ) -> Result<(RawCompiledPage, LlmRequest, LlmResponse, u32)> {
        let repair = repair_request(&bundle.request, &bundle.template, reasons);
        let (response, added) =
            generate_cached(&self.provider, self.cache.as_ref(), repair.clone()).await?;
        let raw: RawCompiledPage = structured::parse_json(&response.text).map_err(|stage1| {
            WikiError::Compilation(format!(
                "compilation failed after repair: {}",
                stage1.machine_reason()
            ))
        })?;
        let issues = validate_body(&raw.markdown, context);
        if !issues.is_empty() {
            return Err(WikiError::Compilation(format!(
                "page compilation failed validation after repair: {}",
                issues.join("; ")
            )));
        }
        Ok((raw, repair, response, added))
    }
}

/// Rendered prompt plus the template, kept so a repair can inject
/// machine-readable reasons into the untouched `{{REPAIR_NOTES}}` slot.
struct PageRequestBundle {
    request: LlmRequest,
    template: String,
}

/// Inputs for rendering one page record from its validated body.
struct PageRender<'a> {
    page: &'a llm_wiki_core::model::WikiPagePlan,
    related: &'a [(String, String)],
    body: &'a str,
    sources: &'a [String],
    build_id: &'a BuildId,
    title_to_id: &'a BTreeMap<String, WikiPageId>,
    citations: Vec<PageCitationRecord>,
}

/// Everything validation needs to know about one page.
struct CompileContext<'a> {
    claim_ids: &'a BTreeSet<String>,
    /// Claim ids of this page that have NO stored anchors: citing them can
    /// never expand to a citation, so it must be repaired or fail (PRD §34).
    unanchored: &'a BTreeSet<String>,
    link_targets: &'a BTreeMap<String, WikiPageId>,
    page_claim_count: usize,
}

#[derive(Debug, Deserialize)]
struct RawCompiledPage {
    #[serde(default)]
    markdown: String,
}

/// claim id → its stored anchors, for expansion.
fn claim_anchors<'a>(
    base: &'a KnowledgeBase,
    page: &llm_wiki_core::model::WikiPagePlan,
) -> BTreeMap<String, Vec<&'a PlanAnchor>> {
    page.knowledge_refs
        .iter()
        .filter(|node_id| base.nodes[*node_id].kind == "claim")
        .flat_map(|node_id| {
            base.nodes[node_id]
                .anchors
                .iter()
                .map(move |anchor| (node_id.as_str().to_owned(), anchor))
        })
        .fold(BTreeMap::new(), |mut map, (claim_id, anchor)| {
            map.entry(claim_id).or_default().push(anchor);
            map
        })
}

fn validate_body(markdown: &str, context: &CompileContext<'_>) -> Vec<String> {
    let mut issues = Vec::new();
    let cites = scan_citations(markdown);
    if cites.is_empty() && context.page_claim_count > 0 {
        issues.push(
            "UNGROUNDED_BODY: the page cites no claims although claim knowledge was provided"
                .to_owned(),
        );
    }
    for cite in &cites {
        let claim = cite.claim.as_deref().unwrap_or("");
        if claim.is_empty() {
            issues.push("EMPTY_CLAIM_REF: a citation comment has no claim id".to_owned());
        } else if context.unanchored.contains(claim) {
            issues.push(format!(
                "UNANCHORED_CLAIM: '{claim}' has no stored evidence anchors and cannot be cited"
            ));
        } else if !context.claim_ids.contains(claim) {
            issues.push(format!(
                "UNKNOWN_CLAIM_REF: '{claim}' is not a claim of this page"
            ));
        }
    }
    for link in scan_wikilinks(markdown) {
        let folded = link.target.trim().to_lowercase();
        if !context.link_targets.contains_key(&folded) {
            issues.push(format!(
                "UNKNOWN_LINK_TARGET: [[{}]] does not match any planned page title",
                link.target
            ));
        }
    }
    if markdown.matches("[[").count() != markdown.matches("]]").count() {
        issues.push(
            "UNCLOSED_WIKILINK: `[[` and `]]` counts differ; a WikiLink is malformed".to_owned(),
        );
    }
    if markdown.matches("<!--").count() != markdown.matches("-->").count() {
        issues.push("UNCLOSED_COMMENT: an HTML comment is not closed".to_owned());
    }
    let stripped = strip_citations(markdown);
    if stripped.trim().is_empty() {
        issues.push("EMPTY_BODY: page body is empty".to_owned());
    } else if !markdown.contains("## ") && stripped.trim().chars().count() > 500 {
        issues.push(
            "NO_SECTIONS: long page body has no `##` sections; structure it for readability"
                .to_owned(),
        );
    }
    issues
}

pub(crate) struct ScannedCitation {
    pub(crate) span: Range<usize>,
    pub(crate) claim: Option<String>,
}

pub(crate) struct ScannedLink {
    pub(crate) target: String,
}

/// Finds `<!-- llm-wiki:cite ... -->` comments with their `claim` attribute.
pub(crate) fn scan_citations(body: &str) -> Vec<ScannedCitation> {
    scan_delimited(body, "<!--", "-->")
        .into_iter()
        .filter_map(|span| {
            let inner = &body[span.start + "<!--".len()..span.end - "-->".len()];
            let rest = inner.trim().strip_prefix("llm-wiki:cite")?;
            if !rest.starts_with(char::is_whitespace) && !rest.is_empty() {
                return None;
            }
            let claim = parse_attr(rest, "claim");
            Some(ScannedCitation { span, claim })
        })
        .collect()
}

/// Finds `[[Target]]` / `[[Target|display]]` links (PRD §15.2).
pub(crate) fn scan_wikilinks(body: &str) -> Vec<ScannedLink> {
    scan_delimited(body, "[[", "]]")
        .into_iter()
        .map(|span| {
            let inner = &body[span.start + 2..span.end - 2];
            ScannedLink {
                target: inner.split('|').next().unwrap_or("").trim().to_owned(),
            }
        })
        .collect()
}

/// Byte spans of every `open...close` pair, markers included, ordered by
/// position. Unmatched openers stop the scan.
fn scan_delimited(body: &str, open: &str, close: &str) -> Vec<Range<usize>> {
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

fn parse_attr(text: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = text.find(&needle)? + needle.len();
    let rest = &text[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

/// Replaces each `<!-- llm-wiki:cite claim="kn_X" -->` with one full comment
/// per stored evidence anchor (PRD §16). Returns the expanded body, the
/// citation records (body order) and the distinct source paths (first-use
/// order) for the frontmatter.
fn expand_citations(
    body: &str,
    anchors_by_claim: &BTreeMap<String, Vec<&PlanAnchor>>,
) -> (String, Vec<PageCitationRecord>, Vec<String>) {
    let mut out = String::with_capacity(body.len());
    let mut records = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    let mut seen_sources = BTreeSet::new();
    let mut cursor = 0;
    for cite in scan_citations(body) {
        out.push_str(&body[cursor..cite.span.start]);
        cursor = cite.span.end;
        let Some(claim) = cite.claim.as_deref() else {
            continue;
        };
        let Some(anchors) = anchors_by_claim.get(claim) else {
            continue;
        };
        let mut expanded = Vec::new();
        for anchor in anchors {
            expanded.push(format!(
                "<!-- llm-wiki:cite claim=\"{claim}\" source=\"{}\" section=\"{}\" range=\"{}-{}\" digest=\"{}\" -->",
                anchor.rel_path,
                anchor.heading_path.join(" > "),
                anchor.range.start,
                anchor.range.end,
                anchor.evidence_digest,
            ));
            if seen_sources.insert(anchor.rel_path.clone()) {
                sources.push(anchor.rel_path.clone());
            }
            records.push(PageCitationRecord {
                claim_node_id: KnowledgeNodeId::from_validated(claim.to_owned()),
                source_id: anchor.source_id.clone(),
                section_id: anchor.section_id.clone(),
                range: anchor.range,
                source_hash: anchor.source_hash.clone(),
                evidence_digest: anchor.evidence_digest.clone(),
                heading_path: anchor.heading_path.clone(),
            });
        }
        out.push_str(&expanded.join(" "));
    }
    out.push_str(&body[cursor..]);
    (out, records, sources)
}

/// Removes citation comments so body-length/section checks measure prose.
pub(crate) fn strip_citations(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let mut cursor = 0;
    for cite in scan_citations(body) {
        out.push_str(&body[cursor..cite.span.start]);
        cursor = cite.span.end;
    }
    out.push_str(&body[cursor..]);
    out
}

/// Resolves WikiLink targets to page ids; dedupes per page keeping first
/// occurrence order (resolvability was validated already).
fn resolve_links(body: &str, title_to_id: &BTreeMap<String, WikiPageId>) -> Vec<PageLinkRecord> {
    let mut records = Vec::new();
    let mut seen = BTreeSet::new();
    for link in scan_wikilinks(body) {
        let folded = link.target.trim().to_lowercase();
        if let Some(page_id) = title_to_id.get(&folded) {
            if seen.insert(page_id.clone()) {
                records.push(PageLinkRecord {
                    to_page_id: page_id.clone(),
                    target_title: link.target,
                });
            }
        }
    }
    records
}

struct FrontmatterInput<'a> {
    page_id: &'a str,
    slug: &'a str,
    title: &'a str,
    category: &'a str,
    schema_version: u32,
    language: &'a str,
    build_id: &'a str,
    sources: &'a [String],
}

/// Renders app-owned frontmatter (PRD §15.2).
fn render_frontmatter(input: &FrontmatterInput<'_>) -> String {
    let mut out = String::from("---\n");
    out.push_str(&format!("schema_version: {}\n", input.schema_version));
    out.push_str("generated: true\n");
    out.push_str(&format!("build: {}\n", yaml_scalar(input.build_id)));
    out.push_str(&format!("id: {}\n", yaml_scalar(input.page_id)));
    out.push_str(&format!("slug: {}\n", yaml_scalar(input.slug)));
    out.push_str(&format!("title: {}\n", yaml_scalar(input.title)));
    out.push_str(&format!("category: {}\n", yaml_scalar(input.category)));
    out.push_str(&format!("language: {}\n", yaml_scalar(input.language)));
    if input.sources.is_empty() {
        out.push_str("sources: []\n");
    } else {
        out.push_str("sources:\n");
        for source in input.sources {
            out.push_str(&format!("  - {}\n", yaml_scalar(source)));
        }
    }
    out.push_str("---\n");
    out
}

/// Double-quotes scalars unless they are plainly safe YAML plain scalars.
fn yaml_scalar(value: &str) -> String {
    let safe = !value.is_empty()
        && value
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphanumeric())
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/'))
        && !value.ends_with(':');
    if safe {
        value.to_owned()
    } else {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

/// Builds the KNOWLEDGE payload: full detail for the page's nodes; claims
/// carry their anchors so the model can name sections in prose.
fn page_knowledge_payload(
    base: &KnowledgeBase,
    page: &llm_wiki_core::model::WikiPagePlan,
) -> String {
    let nodes: Vec<serde_json::Value> = page
        .knowledge_refs
        .iter()
        .map(|node_id| {
            let node = &base.nodes[node_id];
            if node.kind == "claim" {
                serde_json::json!({
                    "id": node.id.as_str(),
                    "kind": "claim",
                    "statement": node.statement,
                    "evidence": node
                        .anchors
                        .iter()
                        .map(|anchor| {
                            serde_json::json!({
                                "source": anchor.rel_path,
                                "section": anchor.heading_path.join(" > "),
                                "range": format!("{}-{}", anchor.range.start, anchor.range.end),
                            })
                        })
                        .collect::<Vec<_>>(),
                })
            } else {
                serde_json::json!({
                    "id": node.id.as_str(),
                    "kind": node.kind,
                    "name": node.name,
                    "type": node.entity_type,
                    "description": node.description,
                })
            }
        })
        .collect();
    serde_json::json!({ "nodes": nodes }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cite_scanning_is_tolerant_to_spacing() {
        let body = "text\n<!-- llm-wiki:cite claim=\"kn_01ARZ3NDEKTSV4RRFFQ69G5FAV\" -->\nmore";
        let cites = scan_citations(body);
        assert_eq!(cites.len(), 1);
        assert_eq!(
            cites[0].claim.as_deref(),
            Some("kn_01ARZ3NDEKTSV4RRFFQ69G5FAV")
        );
    }

    #[test]
    fn non_cite_comments_are_ignored() {
        let body = "<!-- a normal html comment --> and <!-- llm-wiki:citation-unknown x -->";
        assert!(scan_citations(body).is_empty());
    }

    #[test]
    fn wikilink_display_text_is_split_off() {
        let body = "see [[Plugin Runtime|the runtime]] and [[Solo]]";
        let links = scan_wikilinks(body);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].target, "Plugin Runtime");
        assert_eq!(links[1].target, "Solo");
    }

    #[test]
    fn yaml_scalar_quotes_only_when_needed() {
        assert_eq!(yaml_scalar("plugin-system"), "plugin-system");
        assert_eq!(yaml_scalar("docs/a.md"), "docs/a.md");
        assert_eq!(yaml_scalar("Plugin: System"), "\"Plugin: System\"");
        assert_eq!(yaml_scalar(""), "\"\"");
        assert_eq!(yaml_scalar("标题"), "\"标题\"");
    }
}
