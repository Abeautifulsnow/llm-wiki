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

use crate::cache::{
    generate_cached, plan_cache_key, remember_validated, repair_request, StageCache,
};
use crate::prompt::PromptDocument;

#[derive(Debug, Clone)]
pub struct PlannerConfig {
    pub hierarchical: bool,
    pub max_cluster_nodes: usize,
    pub max_plan_input_tokens: u64,
    /// Instruction for output language; matches the analysis prompt contract.
    pub language: String,
    /// Per-request output ceiling (config `[llm] max_output_tokens`): thinking
    /// models spend chain-of-thought from this same budget.
    pub max_output_tokens: u32,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            hierarchical: true,
            max_cluster_nodes: 24,
            max_plan_input_tokens: 32_000,
            language: "the sources' language".to_owned(),
            max_output_tokens: 4096,
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
    /// True when the whole plan came from the §45 plan-identity cache (page
    /// IDs preserved, zero LLM requests).
    pub plan_cache_hit: bool,
}

pub struct WikiPlanner {
    provider: Arc<dyn LlmProvider>,
    prompt: PromptDocument,
    config: PlannerConfig,
    /// §28 request-level cache for summary/local/reconcile responses.
    cache: Option<Arc<dyn StageCache>>,
    /// §45 plan-identity: `plan_cache_identity(...)` over config/model/schema;
    /// `Some` enables plan persistence under the reconciliation key.
    plan_identity: Option<String>,
    /// §19.2 explicit re-plan: when `true` the plan-identity short-circuit is
    /// DISABLED — planning always runs fresh so the diff against the current
    /// generation reflects the planner's actual output. The request-level §28
    /// cache and plan persistence stay intact, so a `replan --dry-run` warms
    /// the stage cache and the follow-up execute pays for planning once.
    force_fresh: bool,
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
            cache: None,
            plan_identity: None,
            force_fresh: false,
        }
    }

    /// Wires the §28 stage cache and the §45 plan-identity cache. `identity`
    /// is `plan_cache_identity(config_hash, model, schema_version)` — it must
    /// change when model, schema or the effective config change.
    pub fn with_plan_cache(mut self, cache: Arc<dyn StageCache>, identity: String) -> Self {
        self.cache = Some(cache);
        self.plan_identity = Some(identity);
        self
    }

    /// Forces FRESH planning by skipping the plan-identity short-circuit
    /// (§19.2: `llm-wiki replan` must re-derive the plan even when the
    /// reconciliation key still matches). Idempotent; default is `false`.
    pub fn with_force_fresh(mut self, force_fresh: bool) -> Self {
        self.force_fresh = force_fresh;
        self
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
                plan_cache_hit: false,
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

    /// Length note: ~94 lines — the three-layer stage scheduler (summary → local → reconcile), symmetric with plan_flat and driven by PlanCacheKeys.
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

        // Layer keys are LLM-free (PRD §14): computable BEFORE any request,
        // which is what makes the plan-identity short-circuit possible.
        let summary_keys: Vec<String> = clusters
            .iter()
            .map(|cluster| cluster_summary_key(base, cluster))
            .collect();
        let local_keys: Vec<String> = summary_keys
            .iter()
            .map(|summary_key| local_plan_key(summary_key, planner_version, config_tag))
            .collect();
        let reconcile_key = reconciliation_key(&local_keys, registry_revision, config_tag);

        if let Some(plan) = self.cached_plan(&reconcile_key)? {
            return Ok(PlanOutcome {
                plan,
                cache: PlanCacheKeys {
                    cluster_summary_keys: summary_keys,
                    local_plan_keys: local_keys,
                    reconciliation_key: reconcile_key,
                },
                llm_request_count: 0,
                flat_mode: false,
                plan_cache_hit: true,
            });
        }

        let mut proposals: Vec<RawPage> = Vec::new();
        let mut llm_request_count = 0u32;
        let mut orphan_refs: Vec<String> = Vec::new();

        for (cluster, _summary_key) in clusters.iter().zip(summary_keys.iter()) {
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
            // Cross-cluster salvage (T1 Run 8/11-12): models attribute
            // nodes to a neighboring cluster's id space. The validator closure
            // strips refs that are REAL library nodes from other clusters into
            // the orphan pool (they re-enter as a dedicated page after
            // reconciliation); refs that are not library nodes still count as
            // UNKNOWN_NODE_REF and go through the normal repair path.
            let orphans: Arc<std::sync::Mutex<Vec<String>>> =
                Arc::new(std::sync::Mutex::new(Vec::new()));
            let orphans_v = Arc::clone(&orphans);
            let validate = |plan: &RawPlanResponse| -> Vec<String> {
                let mut issues = Vec::new();
                let mut salvaged: Vec<String> = Vec::new();
                let mut seen: BTreeSet<&str> = BTreeSet::new();
                if plan.pages.is_empty() {
                    issues.push("EMPTY_PLAN: no pages proposed".to_owned());
                }
                for page in &plan.pages {
                    if page.title.trim().is_empty() {
                        issues.push("EMPTY_TITLE: page title is empty".to_owned());
                    }
                    if page.knowledge_refs.is_empty() {
                        issues.push(format!(
                            "EMPTY_REFS: page '{}' has no knowledge_refs",
                            page.title
                        ));
                    }
                    for ref_id in &page.knowledge_refs {
                        if allowed.contains(ref_id) {
                            if !seen.insert(ref_id.as_str()) {
                                issues.push(format!(
                                    "DUPLICATE_NODE_ASSIGNMENT: '{ref_id}' assigned to multiple pages"
                                ));
                            }
                        } else if base.nodes.keys().any(|k| k.as_str() == ref_id.as_str()) {
                            salvaged.push(ref_id.clone());
                        } else {
                            issues.push(format!(
                                "UNKNOWN_NODE_REF: '{ref_id}' is not a node of this cluster"
                            ));
                        }
                    }
                }
                orphans_v.lock().unwrap().extend(salvaged);
                issues
            };
            let (raw_local, requests) = self
                .stage_round_validated::<RawPlanResponse, _>(&local_stage, &compact, validate)
                .await?;
            llm_request_count += requests;
            orphan_refs.extend(orphans.lock().unwrap().drain(..));
            proposals.extend(raw_local.pages);
        }

        // Salvaged cross-cluster nodes BYPASS reconciliation: they form one
        // deterministic page appended AFTER the merged plan (the model never
        // sees them, so it cannot invent coverage errors around them). The
        // proposed_union handed to the validator is widened accordingly.
        let salvage_page = if orphan_refs.is_empty() {
            None
        } else {
            let mut refs: Vec<String> = std::mem::take(&mut orphan_refs);
            refs.sort();
            refs.dedup();
            tracing::warn!(
                count = refs.len(),
                "salvaging cross-cluster node refs into a dedicated page"
            );
            Some(RawPage {
                title: "Miscellaneous Knowledge".to_owned(),
                category: "concepts".to_owned(),
                purpose: "Knowledge nodes whose assigning cluster could not be determined during planning.".to_owned(),
                knowledge_refs: refs,
            })
        };

        let reconcile_stage = self.prompt.stage_block("reconcile")?;
        let mut proposed_union: BTreeSet<String> = proposals
            .iter()
            .flat_map(|proposal| proposal.knowledge_refs.iter().cloned())
            .collect();
        if let Some(salvage) = &salvage_page {
            proposed_union.extend(salvage.knowledge_refs.iter().cloned());
        }

        // Batched reconciliation (T1 Run 13: a 60-doc corpus produced a
        // ~506K-token reconcile payload against a 32K budget — 16x over, and
        // PRD §14 forbids truncation). When the single-shot payload exceeds
        // the budget, proposals are split into budget-sized batches; each
        // batch is reconciled independently (its output pages become the
        // proposals of the next round), and a final round merges the batch
        // results. Guarantees preserved per round: coverage validated against
        // that batch's proposed set; DUPLICATE across batches is impossible
        // because batches partition the proposals.
        let mut merged: Vec<RawPage>;
        let mut round_proposals: Vec<RawPage> = proposals;
        let mut round_keys_summary: Vec<String> = summary_keys.clone();
        let mut round_keys_local: Vec<String> = local_keys.clone();
        let mut reconcile_requests = 0u32;
        loop {
            let reconcile_json =
                reconcile_payload(&round_proposals, &round_keys_summary, &round_keys_local);
            let single_round = reconcile_json.len() <= MAX_RECONCILE_PAYLOAD_BYTES;
            self.ensure_stage_budget(&reconcile_json, "reconciliation")?;
            let (round_merged, requests) = self
                .reconcile_stage(&reconcile_stage, &reconcile_json, &proposed_union)
                .await?;
            reconcile_requests += requests;
            if single_round {
                merged = round_merged;
                break;
            }
            // Batch the PROPOSALS (the dominant payload term) into
            // budget-sized slices; the next round reconciles the merged pages
            // of each batch. Summaries/keys shrink with each round because
            // each batch yields fewer pages than it consumed proposals.
            let batch_count = reconcile_json.len().div_ceil(MAX_RECONCILE_PAYLOAD_BYTES) + 1;
            let per_batch = round_proposals.len().div_ceil(batch_count).max(1);
            let mut next_round: Vec<RawPage> = Vec::new();
            for batch in round_proposals.chunks(per_batch) {
                let batch_json = reconcile_payload(batch, &round_keys_summary, &round_keys_local);
                self.ensure_stage_budget(&batch_json, "reconciliation batch")?;
                let batch_union: BTreeSet<String> = batch
                    .iter()
                    .flat_map(|proposal| proposal.knowledge_refs.iter().cloned())
                    .collect();
                let (batch_merged, requests) = self
                    .reconcile_stage(&reconcile_stage, &batch_json, &batch_union)
                    .await?;
                reconcile_requests += requests;
                next_round.extend(batch_merged);
            }
            round_proposals = next_round;
            round_keys_summary = round_proposals
                .iter()
                .map(|_| format!("batch-{}", uuid_batch()))
                .collect();
            round_keys_local = round_proposals
                .iter()
                .map(|_| format!("batch-{}", uuid_batch()))
                .collect();
            if round_proposals.len() <= 1 {
                // A single (merged) proposal cannot be subdivided further;
                // the next loop iteration fits by construction.
                merged = round_proposals;
                break;
            }
        }
        llm_request_count += reconcile_requests;

        if let Some(mut salvage) = salvage_page {
            // Drop refs the model already merged into topical pages — a
            // duplicate assignment would fail validate_merged below.
            salvage.knowledge_refs.retain(|ref_id| {
                !merged
                    .iter()
                    .any(|page| page.knowledge_refs.contains(ref_id))
            });
            if !salvage.knowledge_refs.is_empty() {
                merged.push(salvage);
            }
        }

        // Orphan sweep: refs the model invented into the final plan that are
        // REAL library nodes but were never proposed become one deterministic
        // page instead of failing reconciliation. Refs that are NOT library
        // nodes stay rejected by validate_merged.
        let mut unknown_in_merged: Vec<String> = Vec::new();
        for page in &merged {
            for ref_id in &page.knowledge_refs {
                if !proposed_union.contains(ref_id)
                    && base.nodes.keys().any(|k| k.as_str() == ref_id.as_str())
                {
                    unknown_in_merged.push(ref_id.clone());
                }
            }
        }
        let mut orphan_all: BTreeSet<String> = BTreeSet::from_iter(unknown_in_merged);
        // Salvaged refs are already covered by their appended page; nodes the
        // model merged into real pages are no longer orphans.
        for page in &merged {
            for ref_id in &page.knowledge_refs {
                orphan_all.remove(ref_id);
            }
        }
        if !orphan_all.is_empty() {
            tracing::warn!(
                count = orphan_all.len(),
                "planner referenced nodes outside their cluster; salvaging them into a dedicated page"
            );
            let refs: Vec<String> = orphan_all.into_iter().collect();
            let title = "Miscellaneous Knowledge".to_owned();
            merged.push(RawPage {
                title,
                category: "concepts".to_owned(),
                purpose: "Knowledge nodes whose assigning cluster could not be determined during planning.".to_owned(),
                knowledge_refs: refs,
            });
        }

        let plan = self.finalize(merged, base)?;
        self.store_plan(&reconcile_key, &plan);
        Ok(PlanOutcome {
            plan,
            cache: PlanCacheKeys {
                cluster_summary_keys: summary_keys,
                local_plan_keys: local_keys,
                reconciliation_key: reconcile_key,
            },
            llm_request_count,
            flat_mode: false,
            plan_cache_hit: false,
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
        let key = local_plan_key(
            &llm_wiki_core::hash::sha256_hex(payload.as_bytes()),
            planner_version,
            config_tag,
        );
        let reconcile_key =
            reconciliation_key(std::slice::from_ref(&key), registry_revision, config_tag);

        if let Some(plan) = self.cached_plan(&reconcile_key)? {
            return Ok(PlanOutcome {
                plan,
                cache: PlanCacheKeys {
                    cluster_summary_keys: Vec::new(),
                    local_plan_keys: vec![key],
                    reconciliation_key: reconcile_key,
                },
                llm_request_count: 0,
                flat_mode: true,
                plan_cache_hit: true,
            });
        }

        let allowed: BTreeSet<String> = all
            .iter()
            .map(|node_id| node_id.as_str().to_owned())
            .collect();
        let (raw_local, requests) = self
            .stage_round_validated::<RawPlanResponse, _>(&local_stage, &payload, |plan| {
                validate_proposals(&plan.pages, &allowed)
            })
            .await?;
        let plan = self.finalize(raw_local.pages, base)?;
        self.store_plan(&reconcile_key, &plan);
        Ok(PlanOutcome {
            plan,
            cache: PlanCacheKeys {
                cluster_summary_keys: Vec::new(),
                local_plan_keys: vec![key],
                reconciliation_key: reconcile_key,
            },
            llm_request_count: requests,
            flat_mode: true,
            plan_cache_hit: false,
        })
    }

    // -- Plan identity persistence (PRD §45) --------------------------------

    /// On a reconciliation-key hit, returns the stored validated plan — page
    /// IDs included — so rebuilds reuse planner-assigned identity (§45) with
    /// zero LLM requests. Skipped entirely when force-fresh planning was
    /// requested (§19.2 explicit re-plan).
    fn cached_plan(&self, reconcile_key: &str) -> Result<Option<WikiPlan>> {
        if self.force_fresh {
            tracing::info!("force-fresh planning: skipping the plan-identity short-circuit");
            return Ok(None);
        }
        let (cache, identity) = match (&self.cache, &self.plan_identity) {
            (Some(cache), Some(identity)) => (cache, identity),
            _ => return Ok(None),
        };
        let key = plan_cache_key(reconcile_key, identity);
        let Some(raw) = cache.lookup_raw(&key) else {
            return Ok(None);
        };
        match serde_json::from_str::<WikiPlan>(&raw) {
            Ok(plan) => {
                tracing::info!(
                    plan_pages = plan.pages.len(),
                    "plan cache hit; skipping planning stages"
                );
                Ok(Some(plan))
            }
            Err(err) => {
                // A corrupt entry is a miss, never a hard failure: the plan is
                // recomputed deterministically.
                tracing::warn!(error = %err, "stored plan is unreadable; recomputing");
                Ok(None)
            }
        }
    }

    fn store_plan(&self, reconcile_key: &str, plan: &WikiPlan) {
        let (Some(cache), Some(identity)) = (&self.cache, &self.plan_identity) else {
            return;
        };
        let key = plan_cache_key(reconcile_key, identity);
        match serde_json::to_string(plan) {
            Ok(json) => cache.remember_raw(&key, &json),
            Err(err) => tracing::warn!(error = %err, "could not serialize plan for the cache"),
        }
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
            let issues = validate_merged(&plan.pages, proposed);
            if !issues.is_empty() {
                eprintln!("DEBUG reconcile issues: {issues:?}");
            }
            issues
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

    /// Length note: ~85 lines — one shape→repair→re-validate round with its failure accounting; the repair call must stay adjacent to the validation that judged it.
    /// Shape → referential/semantic round with exactly one repair carrying
    /// machine-readable reasons (PRD §11/§28). Whichever stage fails first
    /// consumes the single repair budget; the repaired response is validated
    /// again and any residual issue fails the stage closed — a repaired
    /// response never skips validation (PRD §28: hallucinated refs are
    /// rejected, never written into the plan). Only the response that finally
    /// validates enters the §28 cache.
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
            max_output_tokens: self.config.max_output_tokens,
            json_mode: true,
        };

        let mut llm_request_count = 0u32;
        let (first_response, added) =
            generate_cached(&self.provider, self.cache.as_ref(), base_request.clone()).await?;
        llm_request_count += added;

        // Stage-1 shape check (one repair). `budget_spent` marks a response
        // that already consumed the repair budget: any residual referential /
        // semantic issue must then fail the stage closed — a repaired response
        // never skips validation (PRD §28: hallucinated refs are rejected,
        // never written into the plan).
        let stage1 = structured::parse_json::<T>(&first_response.text);
        let (parsed, cacheable, budget_spent) = match stage1 {
            Ok(parsed) => (parsed, (base_request.clone(), first_response), false),
            Err(stage1) => {
                tracing::warn!(reason = %stage1, "planning stage shape failure, repairing once");
                let repair = repair_request(&base_request, &template, &[stage1.machine_reason()]);
                let (repair_response, added) =
                    generate_cached(&self.provider, self.cache.as_ref(), repair.clone()).await?;
                llm_request_count += added;
                let parsed =
                    structured::parse_json::<T>(&repair_response.text).map_err(|shape| {
                        WikiError::Planning(format!(
                            "planning failed after repair: {}",
                            shape.machine_reason()
                        ))
                    })?;
                (parsed, (repair, repair_response), true)
            }
        };

        let issues = validate(&parsed);
        if issues.is_empty() {
            // Only the validated response is cached (PRD §28).
            remember_validated(self.cache.as_ref(), &cacheable.0, &cacheable.1);
            return Ok((parsed, llm_request_count));
        }
        if budget_spent {
            return Err(WikiError::Planning(format!(
                "planning stage '{}' failed validation after repair: {}",
                self.prompt.name,
                issues.join("; ")
            )));
        }
        tracing::warn!(issues = ?issues, "planning stage validation failed, repairing once");
        let repair = repair_request(&base_request, &template, &issues);
        let (repair_response, added) =
            generate_cached(&self.provider, self.cache.as_ref(), repair.clone()).await?;
        llm_request_count += added;
        let parsed = structured::parse_json::<T>(&repair_response.text).map_err(|shape| {
            WikiError::Planning(format!(
                "planning failed after repair: {}",
                shape.machine_reason()
            ))
        })?;
        let issues = validate(&parsed);
        if !issues.is_empty() {
            return Err(WikiError::Planning(format!(
                "planning stage '{}' failed validation after repair: {}",
                self.prompt.name,
                issues.join("; ")
            )));
        }
        remember_validated(self.cache.as_ref(), &repair, &repair_response);
        Ok((parsed, llm_request_count))
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
/// Reconcile payload budget (bytes): payload length above this triggers
/// batched reconciliation (T1 Run 13: a 60-doc corpus produced ~506K chars —
/// ~126K tokens — against a 32K-token budget).
const MAX_RECONCILE_PAYLOAD_BYTES: usize = 96_000;

/// Stable unique tag for synthetic batch keys (avoids pulling a uuid dep).
fn uuid_batch() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::SeqCst) + 1;
    let pid = std::process::id();
    format!("{pid:08x}{n:016x}")
}

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
