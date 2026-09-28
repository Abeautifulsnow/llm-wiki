//! Hierarchical wiki planner (PRD §14).
//!
//! Stage flow: deterministic clustering (core::plan) → per-cluster summary →
//! per-cluster local plan → global reconciliation consuming only sorted
//! summary/local-plan keys and node ids — never the full claim corpus. Every
//! stage validates its response in three steps (shape → referential →
//! semantic, PRD §28) and repairs **once** with machine-readable reasons.
//!
//! The LLM only *references* node ids; page identity (`WikiPageId`), slugs,
//! title de-duplication and `related_pages` are decided here, deterministically.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{KnowledgeNodeId, SourceId, WikiPageId};
use llm_wiki_core::model::{WikiPagePlan, WikiPlan};
use llm_wiki_core::plan::{
    cluster_knowledge, cluster_summary_key, estimate_tokens, local_plan_key, planning_config_tag,
    reconciliation_key, slugify, KnowledgeBase,
};
use llm_wiki_llm::structured;
use llm_wiki_llm::{LlmProvider, LlmRequest};

use crate::prompt::PromptDocument;

#[derive(Debug, Clone)]
pub struct PlannerConfig {
    pub hierarchical: bool,
    pub max_cluster_nodes: usize,
    pub max_plan_input_tokens: u64,
    /// Instruction for output language; matches the analysis prompt contract.
    pub language: String,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            hierarchical: true,
            max_cluster_nodes: 24,
            max_plan_input_tokens: 32_000,
            language: "the sources' language".to_owned(),
        }
    }
}

/// Layer cache keys, exposed so the build/cache layer (§28/§44) can persist
/// and reuse stage outputs without recomputation.
#[derive(Debug, Clone)]
pub struct PlanCacheKeys {
    pub cluster_summary_keys: Vec<String>,
    pub local_plan_keys: Vec<String>,
    pub reconciliation_key: String,
}

#[derive(Debug, Clone)]
pub struct PlanOutcome {
    pub plan: WikiPlan,
    pub cache: PlanCacheKeys,
    pub llm_request_count: u32,
    pub flat_mode: bool,
}

pub struct WikiPlanner {
    provider: Arc<dyn LlmProvider>,
    prompt: PromptDocument,
    config: PlannerConfig,
}

impl WikiPlanner {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        prompt: PromptDocument,
        config: PlannerConfig,
    ) -> Self {
        Self {
            provider,
            prompt,
            config,
        }
    }

    pub async fn plan(&self, base: &KnowledgeBase, registry_revision: u64) -> Result<PlanOutcome> {
        let config_tag =
            planning_config_tag(self.config.hierarchical, self.config.max_cluster_nodes);
        let planner_version = self.prompt.fingerprint_tag();
        if base.nodes.is_empty() {
            return Ok(PlanOutcome {
                plan: WikiPlan::default(),
                cache: PlanCacheKeys {
                    cluster_summary_keys: Vec::new(),
                    local_plan_keys: Vec::new(),
                    reconciliation_key: reconciliation_key(&[], registry_revision, &config_tag),
                },
                llm_request_count: 0,
                flat_mode: !self.config.hierarchical,
            });
        }
        if self.config.hierarchical {
            self.plan_hierarchical(base, registry_revision, &config_tag, &planner_version)
                .await
        } else {
            self.plan_flat(base, registry_revision, &config_tag, &planner_version)
                .await
        }
    }

    async fn plan_hierarchical(
        &self,
        base: &KnowledgeBase,
        registry_revision: u64,
        config_tag: &str,
        planner_version: &str,
    ) -> Result<PlanOutcome> {
        let clusters = cluster_knowledge(
            base,
            self.config.max_cluster_nodes,
            self.config.max_plan_input_tokens,
        );
        let summary_stage = self.prompt.stage_block("cluster-summary")?;
        let local_stage = self.prompt.stage_block("local-plan")?;

        let mut summary_keys = Vec::new();
        let mut local_keys = Vec::new();
        let mut proposals: Vec<RawPage> = Vec::new();
        let mut llm_request_count = 0u32;

        for cluster in &clusters {
            let summary_key = cluster_summary_key(base, cluster);
            summary_keys.push(summary_key.clone());

            let payload = node_payload(base, &cluster.nodes);
            self.ensure_stage_budget(&payload, "cluster summary")?;
            let (raw_summary, requests) = self
                .stage_round::<RawSummary>(&summary_stage, &payload)
                .await?;
            llm_request_count += requests;

            let compact = compact_payload(base, &cluster.nodes, &raw_summary.summary);
            self.ensure_stage_budget(&compact, "local plan")?;
            let allowed: BTreeSet<String> = cluster
                .nodes
                .iter()
                .map(|node_id| node_id.as_str().to_owned())
                .collect();
            let (raw_local, requests) = self
                .stage_round_validated::<RawPlanResponse, _>(&local_stage, &compact, |plan| {
                    validate_proposals(&plan.pages, &allowed)
                })
                .await?;
            llm_request_count += requests;
            local_keys.push(local_plan_key(&summary_key, planner_version, config_tag));
            proposals.extend(raw_local.pages);
        }

        let reconcile_stage = self.prompt.stage_block("reconcile")?;
        let reconcile_json = reconcile_payload(&proposals, &summary_keys, &local_keys);
        self.ensure_stage_budget(&reconcile_json, "reconciliation")?;
        let proposed_union: BTreeSet<String> = proposals
            .iter()
            .flat_map(|proposal| proposal.knowledge_refs.iter().cloned())
            .collect();
        let (merged, requests) = self
            .reconcile_stage(&reconcile_stage, &reconcile_json, &proposed_union)
            .await?;
        llm_request_count += requests;

        let plan = self.finalize(merged, base)?;
        let reconcile_key = reconciliation_key(&local_keys, registry_revision, config_tag);
        Ok(PlanOutcome {
            plan,
            cache: PlanCacheKeys {
                cluster_summary_keys: summary_keys,
                local_plan_keys: local_keys,
                reconciliation_key: reconcile_key,
            },
            llm_request_count,
            flat_mode: false,
        })
    }

    /// Flat (non-hierarchical) mode: allowed only while the whole knowledge
    /// payload fits one budget, otherwise it must fail with an actionable
    /// message instead of producing an incomplete global plan (PRD §14).
    async fn plan_flat(
        &self,
        base: &KnowledgeBase,
        registry_revision: u64,
        config_tag: &str,
        planner_version: &str,
    ) -> Result<PlanOutcome> {
        let all: Vec<KnowledgeNodeId> = base.nodes.keys().cloned().collect();
        let payload = node_payload(base, &all);
        let estimated = estimate_tokens(&payload);
        if estimated > self.config.max_plan_input_tokens {
            return Err(WikiError::Planning(format!(
                "planning input is ~{estimated} tokens, above max_plan_input_tokens = {}; enable hierarchical planning ([planning] hierarchical = true) instead of running an incomplete global plan",
                self.config.max_plan_input_tokens
            )));
        }
        let local_stage = self.prompt.stage_block("local-plan")?;
        let allowed: BTreeSet<String> = all
            .iter()
            .map(|node_id| node_id.as_str().to_owned())
            .collect();
        let (raw_local, requests) = self
            .stage_round_validated::<RawPlanResponse, _>(&local_stage, &payload, |plan| {
                validate_proposals(&plan.pages, &allowed)
            })
            .await?;
        let key = local_plan_key(
            &llm_wiki_core::hash::sha256_hex(payload.as_bytes()),
            planner_version,
            config_tag,
        );
        let plan = self.finalize(raw_local.pages, base)?;
        Ok(PlanOutcome {
            plan,
            cache: PlanCacheKeys {
                cluster_summary_keys: Vec::new(),
                local_plan_keys: vec![key.clone()],
                reconciliation_key: reconciliation_key(
                    std::slice::from_ref(&key),
                    registry_revision,
                    config_tag,
                ),
            },
            llm_request_count: requests,
            flat_mode: true,
        })
    }

    /// Global reconciliation consumes sorted summary/local-plan keys and node
    /// id references only (PRD §14: never the claim corpus).
    async fn reconcile_stage(
        &self,
        stage: &PromptDocument,
        payload: &str,
        proposed: &BTreeSet<String>,
    ) -> Result<(Vec<RawPage>, u32)> {
        self.stage_round_validated::<RawPlanResponse, _>(stage, payload, |plan| {
            validate_merged(&plan.pages, proposed)
        })
        .await
        .map(|(response, count)| (response.pages, count))
    }

    fn ensure_stage_budget(&self, payload: &str, stage: &str) -> Result<()> {
        let estimated = estimate_tokens(payload);
        if estimated > self.config.max_plan_input_tokens {
            return Err(WikiError::Planning(format!(
                "{stage} payload is ~{estimated} tokens, above max_plan_input_tokens = {}; clusters must be subdivided further (PRD §14 forbids truncation)",
                self.config.max_plan_input_tokens
            )));
        }
        Ok(())
    }

    /// Turns validated merged pages into `WikiPagePlan`s: app-assigned
    /// `WikiPageId`s, unique titles/slugs, source refs from claim anchors and
    /// overlap-based `related_pages`.
    fn finalize(&self, merged: Vec<RawPage>, base: &KnowledgeBase) -> Result<WikiPlan> {
        let node_index: BTreeMap<&str, &KnowledgeNodeId> = base
            .nodes
            .keys()
            .map(|node_id| (node_id.as_str(), node_id))
            .collect();
        let mut pages: Vec<WikiPagePlan> = Vec::new();
        let mut seen_titles: BTreeMap<String, usize> = BTreeMap::new();
        let mut seen_slugs: BTreeSet<String> = BTreeSet::new();

        for page in merged {
            let title = dedupe_title(&page.title, &mut seen_titles);
            let slug = dedupe_slug(&slugify(&title), &mut seen_slugs);
            let knowledge_refs: Vec<KnowledgeNodeId> = page
                .knowledge_refs
                .iter()
                .filter_map(|ref_id| node_index.get(ref_id.as_str()).copied().cloned())
                .collect();
            let mut source_refs: Vec<SourceId> = knowledge_refs
                .iter()
                .flat_map(|node_id| base.nodes[node_id].anchors.iter())
                .map(|anchor| anchor.source_id.clone())
                .collect();
            source_refs.sort();
            source_refs.dedup();
            pages.push(WikiPagePlan {
                id: WikiPageId::generate(),
                slug,
                title,
                category: page.category,
                purpose: page.purpose,
                knowledge_refs,
                source_refs,
                related_pages: Vec::new(),
            });
        }

        attach_related_pages(&mut pages);
        Ok(WikiPlan { pages })
    }

    // -- LLM plumbing -------------------------------------------------------

    /// One shape-only round: request → parse; a shape failure repairs once.
    async fn stage_round<T>(&self, stage: &PromptDocument, payload: &str) -> Result<(T, u32)>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.stage_round_validated::<T, _>(stage, payload, |_| Vec::new())
            .await
    }

    /// Shape → referential/semantic round with exactly one repair carrying
    /// machine-readable reasons (PRD §11/§28). Whichever stage fails first
    /// consumes the single repair budget; the repaired response is validated
    /// again and any residual issue fails the stage closed — a repaired
    /// response never skips validation (PRD §28: hallucinated refs are
    /// rejected, never written into the plan).
    async fn stage_round_validated<T, F>(
        &self,
        stage: &PromptDocument,
        payload: &str,
        validate: F,
    ) -> Result<(T, u32)>
    where
        T: for<'de> Deserialize<'de>,
        F: Fn(&T) -> Vec<String>,
    {
        let template = stage.render(&[("LANGUAGE", &self.config.language), ("PAYLOAD", payload)]);
        let base_request = LlmRequest {
            task_tag: self.prompt.name.clone(),
            system: None,
            prompt: template.replace("{{REPAIR_NOTES}}", ""),
            temperature: 0.0,
            max_output_tokens: 4096,
            json_mode: true,
        };

        let mut llm_request_count = 1u32;
        let response = self
            .provider
            .generate(base_request.clone())
            .await
            .map_err(WikiError::from)?;
        let (parsed, issues) = match structured::parse_json::<T>(&response.text) {
            Ok(parsed) => {
                let issues = validate(&parsed);
                (parsed, issues)
            }
            Err(stage1) => {
                tracing::warn!(reason = %stage1, "planning stage shape failure, repairing once");
                let parsed = self
                    .repaired_request::<T>(&base_request, &template, &[stage1.machine_reason()])
                    .await?;
                llm_request_count += 1;
                // The repair budget is spent: validate here and fail closed.
                let issues = validate(&parsed);
                if !issues.is_empty() {
                    return Err(WikiError::Planning(format!(
                        "planning stage '{}' failed validation after repair: {}",
                        self.prompt.name,
                        issues.join("; ")
                    )));
                }
                (parsed, issues)
            }
        };

        if issues.is_empty() {
            return Ok((parsed, llm_request_count));
        }
        tracing::warn!(issues = ?issues, "planning stage validation failed, repairing once");
        let parsed = self
            .repaired_request::<T>(&base_request, &template, &issues)
            .await?;
        llm_request_count += 1;
        let issues = validate(&parsed);
        if !issues.is_empty() {
            return Err(WikiError::Planning(format!(
                "planning stage '{}' failed validation after repair: {}",
                self.prompt.name,
                issues.join("; ")
            )));
        }
        Ok((parsed, llm_request_count))
    }

    async fn repaired_request<T>(
        &self,
        base: &LlmRequest,
        template: &str,
        reasons: &[String],
    ) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut repair = base.clone();
        let notes = format!(
            "## Previous attempt rejected\nYour previous reply failed validation:\n{}\n\nFix every issue and resend the COMPLETE JSON object.",
            reasons
                .iter()
                .map(|reason| format!("- {reason}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        repair.prompt = template.replace("{{REPAIR_NOTES}}", &notes);
        let response = self
            .provider
            .generate(repair)
            .await
            .map_err(WikiError::from)?;
        structured::parse_json(&response.text).map_err(|stage1| {
            WikiError::Planning(format!(
                "planning failed after repair: {}",
                stage1.machine_reason()
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Payloads and validation helpers
// ---------------------------------------------------------------------------

/// Full node payload for cluster summaries: descriptions and claim statements
/// (the content being organized).
fn node_payload(base: &KnowledgeBase, ids: &[KnowledgeNodeId]) -> String {
    let nodes: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            let node = &base.nodes[id];
            serde_json::json!({
                "id": node.id.as_str(),
                "kind": node.kind,
                "name": node.name,
                "type": node.entity_type,
                "description": node.description,
                "statement": node.statement,
            })
        })
        .collect();
    serde_json::json!({ "nodes": nodes }).to_string()
}

/// Compact payload for local planning: summary + id/name/kind list.
fn compact_payload(base: &KnowledgeBase, ids: &[KnowledgeNodeId], summary: &str) -> String {
    let nodes: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            let node = &base.nodes[id];
            serde_json::json!({
                "id": node.id.as_str(),
                "kind": node.kind,
                "name": node.name,
            })
        })
        .collect();
    serde_json::json!({ "summary": summary, "nodes": nodes }).to_string()
}

/// The single reconcile payload shared by the budget check and the actual
/// request (PRD §14: estimate what you send, send what you estimated).
fn reconcile_payload(
    proposals: &[RawPage],
    summary_keys: &[String],
    local_keys: &[String],
) -> String {
    let clusters: Vec<serde_json::Value> = local_keys
        .iter()
        .zip(summary_keys.iter())
        .map(|(local, summary)| {
            serde_json::json!({
                "local_plan_key": local,
                "summary_key": summary,
            })
        })
        .collect();
    let pages: Vec<serde_json::Value> = proposals
        .iter()
        .enumerate()
        .map(|(index, proposal)| {
            serde_json::json!({
                "index": index,
                "title": proposal.title,
                "category": proposal.category,
                "purpose": proposal.purpose,
                "knowledge_refs": proposal.knowledge_refs,
            })
        })
        .collect();
    serde_json::json!({ "clusters": clusters, "proposals": pages }).to_string()
}

fn validate_proposals(pages: &[RawPage], allowed: &BTreeSet<String>) -> Vec<String> {
    let mut issues = Vec::new();
    if pages.is_empty() {
        issues.push("EMPTY_PLAN: no pages proposed".to_owned());
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for page in pages {
        if page.title.trim().is_empty() {
            issues.push("EMPTY_TITLE: page title is empty".to_owned());
        }
        if page.category.trim().is_empty() {
            issues.push("EMPTY_CATEGORY: page category is empty".to_owned());
        }
        if page.knowledge_refs.is_empty() {
            issues.push(format!(
                "EMPTY_REFS: page '{}' has no knowledge_refs",
                page.title
            ));
        }
        for ref_id in &page.knowledge_refs {
            if !allowed.contains(ref_id) {
                issues.push(format!(
                    "UNKNOWN_NODE_REF: '{}' is not a node of this cluster",
                    ref_id
                ));
            }
            if !seen.insert(ref_id.as_str()) {
                issues.push(format!(
                    "DUPLICATE_NODE_ASSIGNMENT: '{}' assigned to multiple pages",
                    ref_id
                ));
            }
        }
    }
    issues
}

fn validate_merged(pages: &[RawPage], proposed: &BTreeSet<String>) -> Vec<String> {
    let mut issues = Vec::new();
    if pages.is_empty() {
        issues.push("EMPTY_PLAN: reconciliation produced no pages".to_owned());
    }
    let mut covered: BTreeSet<String> = BTreeSet::new();
    for page in pages {
        if page.title.trim().is_empty() {
            issues.push("EMPTY_TITLE: page title is empty".to_owned());
        }
        if page.category.trim().is_empty() {
            issues.push("EMPTY_CATEGORY: page category is empty".to_owned());
        }
        if page.purpose.trim().is_empty() {
            issues.push(format!(
                "EMPTY_PURPOSE: page '{}' has no purpose",
                page.title
            ));
        }
        if page.knowledge_refs.is_empty() {
            issues.push(format!(
                "EMPTY_REFS: page '{}' has no knowledge_refs",
                page.title
            ));
        }
        for ref_id in &page.knowledge_refs {
            if !proposed.contains(ref_id) {
                issues.push(format!(
                    "UNKNOWN_NODE_REF: '{}' was not proposed by any cluster",
                    ref_id
                ));
            }
            if !covered.insert(ref_id.clone()) {
                issues.push(format!(
                    "DUPLICATE_NODE_ASSIGNMENT: '{}' appears in multiple final pages",
                    ref_id
                ));
            }
        }
    }
    // Coverage (PRD §14): knowledge proposed by any cluster must survive
    // reconciliation — a dropped node would silently vanish from the wiki.
    for ref_id in proposed {
        if !covered.contains(ref_id) {
            issues.push(format!(
                "MISSING_NODE_COVERAGE: '{ref_id}' was proposed by a cluster but missing from the final plan"
            ));
        }
    }
    issues
}

fn dedupe_title(title: &str, seen: &mut BTreeMap<String, usize>) -> String {
    let folded = title.trim().to_lowercase();
    match seen.get(&folded) {
        None => {
            seen.insert(folded, 1);
            title.trim().to_owned()
        }
        Some(count) => {
            let next = count + 1;
            seen.insert(folded, next);
            format!("{} ({next})", title.trim())
        }
    }
}

fn dedupe_slug(slug: &str, seen: &mut BTreeSet<String>) -> String {
    let mut candidate = slug.to_owned();
    let mut counter = 2;
    while !seen.insert(candidate.clone()) {
        candidate = format!("{slug}-{counter}");
        counter += 1;
    }
    candidate
}

/// Pages sharing at least one knowledge node become related (deterministic:
/// overlap count desc, then page id asc), capped to keep pages tidy.
fn attach_related_pages(pages: &mut [WikiPagePlan]) {
    const MAX_RELATED: usize = 8;
    let refs: Vec<BTreeSet<KnowledgeNodeId>> = pages
        .iter()
        .map(|page| page.knowledge_refs.iter().cloned().collect())
        .collect();
    let ids: Vec<WikiPageId> = pages.iter().map(|page| page.id.clone()).collect();
    for (index, page) in pages.iter_mut().enumerate() {
        let mut related: Vec<(usize, &WikiPageId)> = ids
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(other, page_id)| {
                let overlap = refs[index].intersection(&refs[other]).count();
                (overlap, page_id)
            })
            .filter(|(overlap, _)| *overlap > 0)
            .collect();
        related.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        page.related_pages = related
            .into_iter()
            .take(MAX_RELATED)
            .map(|(_, page_id)| page_id.clone())
            .collect();
    }
}

// ---------------------------------------------------------------------------
// LLM-facing raw shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawSummary {
    #[serde(default)]
    summary: String,
}

/// One proposed page; used by both the local-plan and reconcile stages.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RawPage {
    #[serde(default)]
    title: String,
    #[serde(default)]
    category: String,
    #[serde(default)]
    purpose: String,
    #[serde(default)]
    knowledge_refs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawPlanResponse {
    #[serde(default)]
    pages: Vec<RawPage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn related_pages_follow_overlap_then_id() {
        let a = KnowledgeNodeId::parse("kn_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let b = KnowledgeNodeId::parse("kn_01BX5ZZKBKACTAV9WEVGEMMVRZ").unwrap();
        let shared = KnowledgeNodeId::parse("kn_01CZZZZZZZZZZZZZZZZZZZZZZZ").unwrap();
        let mk = |id: &str, title: &str, refs: Vec<KnowledgeNodeId>| WikiPagePlan {
            id: WikiPageId::parse(id).unwrap(),
            slug: title.to_lowercase(),
            title: title.to_owned(),
            category: "concepts".into(),
            purpose: "p".into(),
            knowledge_refs: refs,
            source_refs: vec![],
            related_pages: vec![],
        };
        let mut pages = vec![
            mk(
                "wp_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "One",
                vec![shared.clone(), a.clone()],
            ),
            mk("wp_01BX5ZZKBKACTAV9WEVGEMMVRZ", "Two", vec![shared.clone()]),
            mk("wp_01CZZZZZZZZZZZZZZZZZZZZZZA", "Three", vec![a]),
        ];
        attach_related_pages(&mut pages);
        // One: shares `shared` with Two and `a` with Three (tie → id order).
        assert_eq!(
            pages[0].related_pages,
            vec![pages[1].id.clone(), pages[2].id.clone()]
        );
        assert_eq!(pages[1].related_pages, vec![pages[0].id.clone()]);
        // Three shares `a` with One only.
        assert_eq!(pages[2].related_pages, vec![pages[0].id.clone()]);
        let _ = b;
    }

    #[test]
    fn title_and_slug_deduplication_is_deterministic() {
        let mut titles = BTreeMap::new();
        assert_eq!(dedupe_title("Runtime", &mut titles), "Runtime");
        assert_eq!(dedupe_title("runtime", &mut titles), "runtime (2)");

        let mut slugs = BTreeSet::new();
        assert_eq!(dedupe_slug("runtime", &mut slugs), "runtime");
        assert_eq!(dedupe_slug("runtime", &mut slugs), "runtime-2");
    }
}
