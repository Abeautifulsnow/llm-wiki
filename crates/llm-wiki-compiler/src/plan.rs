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

use llm_wiki_core::cancel::CancelFlag;
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
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
    /// In-flight LLM call cap for the cluster window (PRD §27
    /// `llm.max_concurrency`): independent clusters run their summary→local
    /// chains concurrently; cluster order is restored deterministically.
    pub max_concurrency: usize,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            hierarchical: true,
            max_cluster_nodes: 24,
            max_plan_input_tokens: 32_000,
            language: "the sources' language".to_owned(),
            max_output_tokens: 4096,
            max_concurrency: 4,
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

/// One cluster's concurrent staging result: (cluster index, page proposals,
/// salvaged cross-cluster refs, cluster summary text, LLM request count).
type ClusterOutcome = (usize, Vec<RawPage>, Vec<String>, String, u32);

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
    /// §31 cooperative cancellation: checked before each cluster spawn and
    /// reconcile round.
    cancel: Option<CancelFlag>,
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
            cancel: None,
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

    /// §31 cooperative cancellation (see [`CancelFlag`]).
    pub fn with_cancel(mut self, cancel: Option<CancelFlag>) -> Self {
        self.cancel = cancel;
        self
    }

    fn checkpoint(&self) -> Result<()> {
        if let Some(cancel) = &self.cancel {
            cancel.check()?;
        }
        Ok(())
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

    /// Length note: ~185 lines — the three-layer stage scheduler (summary →
    /// local → reconcile): cluster staging is delegated to
    /// [`PlannerTask::run_cluster`], the reconcile loop to
    /// [`Self::reconcile_rounds`]; what remains is orchestration, salvage
    /// bookkeeping and the orphan sweep.
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

        // Cluster-parallel staging (T1 perf / audit FIX-007): clusters are
        // independent; each task runs its summary→local chain while a
        // JoinSet window bounds in-flight LLM calls (`max_concurrency`,
        // PRD §27). JoinSet yields COMPLETION order, so every task carries
        // its cluster index and outcomes are sorted back into cluster
        // order — proposals and salvaged refs stay deterministic.
        let task = PlannerTask::from(self);
        let base = Arc::new(base.clone());
        let concurrency = self.config.max_concurrency.max(1);
        let mut cluster_outcomes: Vec<ClusterOutcome> = Vec::with_capacity(clusters.len());
        let mut next = 0usize;
        let mut join_set = tokio::task::JoinSet::new();
        while next < clusters.len() || !join_set.is_empty() {
            while next < clusters.len() && join_set.len() < concurrency {
                // §31 cooperative cancellation: no NEW cluster work.
                self.checkpoint()?;
                let index = next;
                let task = task.clone();
                let base = Arc::clone(&base);
                let nodes = clusters[next].nodes.clone();
                let summary_stage = summary_stage.clone();
                let local_stage = local_stage.clone();
                join_set.spawn(async move {
                    let (pages, orphans, summary, requests) = task
                        .run_cluster(&base, &nodes, &summary_stage, &local_stage)
                        .await?;
                    Ok::<_, WikiError>((index, pages, orphans, summary, requests))
                });
                next += 1;
            }
            if let Some(joined) = join_set.join_next().await {
                let (index, pages, orphans, summary, requests) = joined.map_err(|e| {
                    WikiError::Planning(format!("planner cluster task panicked: {e}"))
                })??;
                cluster_outcomes.push((index, pages, orphans, summary, requests));
            }
        }
        cluster_outcomes.sort_by_key(|(index, ..)| *index);

        let mut proposals: Vec<RawPage> = Vec::new();
        let mut llm_request_count = 0u32;
        let mut orphan_refs: Vec<String> = Vec::new();
        // Cluster origin per proposal (proposals extend in cluster order) and
        // the cluster summary texts — the reconcile context (FIX-015).
        let mut proposal_clusters: Vec<usize> = Vec::new();
        let mut cluster_summaries: Vec<String> = vec![String::new(); clusters.len()];
        for (index, pages, orphans, summary, requests) in cluster_outcomes {
            llm_request_count += requests;
            orphan_refs.extend(orphans);
            proposal_clusters.extend(std::iter::repeat_n(index, pages.len()));
            cluster_summaries[index] = summary;
            proposals.extend(pages);
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

        let mut proposed_union: BTreeSet<String> = proposals
            .iter()
            .flat_map(|proposal| proposal.knowledge_refs.iter().cloned())
            .collect();
        if let Some(salvage) = &salvage_page {
            proposed_union.extend(salvage.knowledge_refs.iter().cloned());
        }

        // Each proposal carries its originating cluster's summary into the
        // reconcile context (audit FIX-015: the global merge sees cluster-level
        // semantics, not just keys).
        let proposal_summaries: Vec<String> = proposal_clusters
            .iter()
            .map(|&cluster| cluster_summaries[cluster].clone())
            .collect();
        let (mut merged, reconcile_requests) = self
            .reconcile_rounds(
                proposals,
                &summary_keys,
                &local_keys,
                &proposed_union,
                &proposal_summaries,
            )
            .await?;
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

        // NOTE: no orphan sweep here (audit FIX-008 follow-up) — the sweep
        // was dead code by construction: validate_merged already rejects any
        // ref outside `proposed_union`, so a REAL library node in `merged`
        // that was never proposed cannot exist. Cross-cluster salvage above
        // is the only path for unproposed-but-real refs.

        let plan = self.finalize(merged, &base)?;
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

    /// Batched reconciliation (T1 Run 13: a 60-doc corpus produced a
    /// ~506K-token reconcile payload against a 32K budget — 16x over, and
    /// PRD §14 forbids truncation). When the single-shot payload exceeds
    /// the budget, proposals are split into budget-sized batches; each
    /// batch is reconciled independently (its output pages become the
    /// proposals of the next round), and a final round merges the batch
    /// results. The full payload is NEVER estimated against the stage
    /// budget nor sent — only what a request will actually carry is
    /// validated (estimate what you send, send what you estimated).
    /// Guarantees preserved per round: coverage validated against that
    /// batch's proposed set; DUPLICATE across batches is impossible
    /// because batches partition the proposals.
    /// Length note: ~60 lines — the multi-round loop, symmetric with the
    /// cluster staging inside plan_hierarchical.
    async fn reconcile_rounds(
        &self,
        proposals: Vec<RawPage>,
        summary_keys: &[String],
        local_keys: &[String],
        proposed_union: &BTreeSet<String>,
        proposal_summaries: &[String],
    ) -> Result<(Vec<RawPage>, u32)> {
        let task = PlannerTask::from(self);
        let mut round_proposals = proposals;
        let mut round_keys_summary: Vec<String> = summary_keys.to_vec();
        let mut round_keys_local: Vec<String> = local_keys.to_vec();
        // Summaries exist only for the FIRST round's cluster-originated
        // proposals; batch-merged pages carry no origin cluster.
        let mut round_summaries: Option<&[String]> = Some(proposal_summaries);
        let mut reconcile_requests = 0u32;
        let merged: Vec<RawPage>;
        let mut rounds = 0usize;
        loop {
            rounds += 1;
            if rounds > MAX_RECONCILE_ROUNDS {
                // #C02 fail-closed: convergence relies on every batched round
                // merging its batches into fewer pages. A degenerate model
                // that echoes one page per proposal would loop forever.
                return Err(WikiError::Planning(format!(
                    "reconciliation did not converge after {MAX_RECONCILE_ROUNDS} rounds; the model keeps returning one page per proposal — narrow the corpus or raise max_plan_input_tokens (PRD §14 forbids truncation)"
                )));
            }
            // §31 cooperative cancellation: stop before the next round's
            // reconcile request(s).
            self.checkpoint()?;
            let reconcile_json = reconcile_payload(
                &round_proposals,
                &round_keys_summary,
                &round_keys_local,
                round_summaries,
            );
            let single_round = reconcile_json.len() <= MAX_RECONCILE_PAYLOAD_BYTES;
            if single_round {
                task.ensure_stage_budget(&reconcile_json, "reconciliation")?;
                let (round_merged, requests) = task
                    .reconcile_stage(&reconcile_json, proposed_union)
                    .await?;
                reconcile_requests += requests;
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
            let mut offset = 0usize;
            for batch in round_proposals.chunks(per_batch) {
                // Summaries slice in lockstep with the proposal chunk.
                let end = offset + batch.len();
                let batch_summaries = round_summaries.map(|all| &all[offset..end]);
                offset = end;
                let batch_json = reconcile_payload(
                    batch,
                    &round_keys_summary,
                    &round_keys_local,
                    batch_summaries,
                );
                task.ensure_stage_budget(&batch_json, "reconciliation batch")?;
                let batch_union: BTreeSet<String> = batch
                    .iter()
                    .flat_map(|proposal| proposal.knowledge_refs.iter().cloned())
                    .collect();
                let (batch_merged, requests) =
                    task.reconcile_stage(&batch_json, &batch_union).await?;
                reconcile_requests += requests;
                next_round.extend(batch_merged);
            }
            round_proposals = next_round;
            round_summaries = None;
            round_keys_summary = round_proposals
                .iter()
                .map(|page| batch_key("batch-s", page))
                .collect();
            round_keys_local = round_proposals
                .iter()
                .map(|page| batch_key("batch-l", page))
                .collect();
            if round_proposals.len() <= 1 {
                // A single (merged) proposal cannot be subdivided further;
                // the next loop iteration fits by construction.
                merged = round_proposals;
                break;
            }
        }
        Ok((merged, reconcile_requests))
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
        let task = PlannerTask::from(self);
        let (raw_local, requests) = task
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

        attach_related_pages(&mut pages, base);
        Ok(WikiPlan { pages })
    }
}

/// Owned snapshot of the planner dependencies for one spawned cluster task
/// (everything Arc or cheap clone; no SQLite touches). The request plumbing —
/// budget gate, stage rounds, reconciliation — lives here so the serial paths
/// (flat plan, reconcile rounds) and the spawned cluster tasks share one
/// implementation.
#[derive(Clone)]
struct PlannerTask {
    provider: Arc<dyn LlmProvider>,
    prompt: PromptDocument,
    config: PlannerConfig,
    /// §28 request-level cache for summary/local/reconcile responses.
    cache: Option<Arc<dyn StageCache>>,
}

impl From<&WikiPlanner> for PlannerTask {
    fn from(planner: &WikiPlanner) -> Self {
        Self {
            provider: Arc::clone(&planner.provider),
            prompt: planner.prompt.clone(),
            config: planner.config.clone(),
            cache: planner.cache.clone(),
        }
    }
}

impl PlannerTask {
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

    /// Global reconciliation consumes sorted summary/local-plan keys and node
    /// id references only (PRD §14: never the claim corpus).
    async fn reconcile_stage(
        &self,
        payload: &str,
        proposed: &BTreeSet<String>,
    ) -> Result<(Vec<RawPage>, u32)> {
        let stage = self.prompt.stage_block("reconcile")?;
        self.stage_round_validated::<RawPlanResponse, _>(&stage, payload, |plan| {
            let issues = validate_merged(&plan.pages, proposed);
            if !issues.is_empty() {
                eprintln!("DEBUG reconcile issues: {issues:?}");
            }
            issues
        })
        .await
        .map(|(response, count)| (response.pages, count))
    }

    /// One cluster of the hierarchical plan: summary → local plan (PRD §14).
    /// Returns the cluster's page proposals, the salvaged cross-cluster node
    /// refs, the cluster summary text (reconcile context, audit FIX-015) and
    /// the LLM request count.
    async fn run_cluster(
        &self,
        base: &KnowledgeBase,
        nodes: &[KnowledgeNodeId],
        summary_stage: &PromptDocument,
        local_stage: &PromptDocument,
    ) -> Result<(Vec<RawPage>, Vec<String>, String, u32)> {
        let payload = node_payload(base, nodes);
        self.ensure_stage_budget(&payload, "cluster summary")?;
        let (raw_summary, requests) = self
            .stage_round::<RawSummary>(summary_stage, &payload)
            .await?;

        let compact = compact_payload(base, nodes, &raw_summary.summary);
        self.ensure_stage_budget(&compact, "local plan")?;
        let allowed: BTreeSet<String> = nodes
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
        let (raw_local, repair_requests) = self
            .stage_round_validated::<RawPlanResponse, _>(local_stage, &compact, validate)
            .await?;
        // Bind before the tail expression: a MutexGuard temporary in tail
        // position would outlive `orphans` (E0597).
        let salvaged: Vec<String> = orphans.lock().unwrap().drain(..).collect();
        Ok((
            raw_local.pages,
            salvaged,
            raw_summary.summary,
            requests + repair_requests,
        ))
    }

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

/// Reconciliation must converge: every batched round merges its batches into
/// strictly fewer pages (each batch yields fewer pages than it consumed
/// proposals). PRD §14 fail-closed: a degenerate model that keeps echoing one
/// page per proposal stops here instead of looping forever.
const MAX_RECONCILE_ROUNDS: usize = 8;

/// Deterministic synthetic key for a batch-round page: derived from the
/// page's own content, so identical proposals produce the identical prompt —
/// and therefore the identical §28 cache key — in every process. Never built
/// from PID or a process-local counter.
fn batch_key(prefix: &str, page: &RawPage) -> String {
    let canonical = serde_json::json!({
        "title": page.title,
        "category": page.category,
        "purpose": page.purpose,
        "knowledge_refs": page.knowledge_refs,
    });
    format!("{prefix}-{}", sha256_hex(canonical.to_string().as_bytes()))
}

fn reconcile_payload(
    proposals: &[RawPage],
    summary_keys: &[String],
    local_keys: &[String],
    proposal_summaries: Option<&[String]>,
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
            let mut value = serde_json::json!({
                "index": index,
                "title": proposal.title,
                "category": proposal.category,
                "purpose": proposal.purpose,
                "knowledge_refs": proposal.knowledge_refs,
            });
            // Cluster-level context for the global merge (audit FIX-015);
            // batch-round pages have no origin cluster and carry none.
            if let Some(summary) = proposal_summaries
                .and_then(|summaries| summaries.get(index))
                .filter(|summary| !summary.is_empty())
            {
                value["cluster_summary"] = serde_json::json!(summary);
            }
            value
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

/// Pages become related through the knowledge graph, never through shared
/// nodes: the planner's DUPLICATE_NODE_ASSIGNMENT rule keeps page ref sets
/// disjoint, so node overlap can never fire (audit FIX-008). Deterministic
/// signals: (1) knowledge relation adjacency between the pages' nodes
/// (undirected), (2) shared sources among the pages' node anchors. Score =
/// 2 × adjacency edges + 1 × shared sources, score desc then page id asc,
/// capped to keep pages tidy.
fn attach_related_pages(pages: &mut [WikiPagePlan], base: &KnowledgeBase) {
    const MAX_RELATED: usize = 8;
    const ADJACENCY_WEIGHT: usize = 2;
    let ids: Vec<WikiPageId> = pages.iter().map(|page| page.id.clone()).collect();
    let sources: Vec<BTreeSet<SourceId>> = pages
        .iter()
        .map(|page| page.source_refs.iter().cloned().collect())
        .collect();
    // Undirected relation-edge count between page pairs, via node → page
    // ownership (ref sets are disjoint across pages by validation).
    let adjacency: BTreeMap<(usize, usize), usize> = {
        let node_page: BTreeMap<&str, usize> = pages
            .iter()
            .enumerate()
            .flat_map(|(index, page)| {
                page.knowledge_refs
                    .iter()
                    .map(move |node_id| (node_id.as_str(), index))
            })
            .collect();
        let mut adjacency: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for relation in &base.relations {
            let (Some(a), Some(b)) = (
                node_page.get(relation.source.as_str()).copied(),
                node_page.get(relation.target.as_str()).copied(),
            ) else {
                continue;
            };
            if a == b {
                continue; // intra-page: not a relation BETWEEN pages
            }
            *adjacency.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
        adjacency
    };

    for (index, page) in pages.iter_mut().enumerate() {
        let mut related: Vec<(usize, &WikiPageId)> = ids
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .filter_map(|(other, page_id)| {
                let pair = (index.min(other), index.max(other));
                let adjacency_edges = adjacency.get(&pair).copied().unwrap_or(0);
                let shared_sources = sources[index].intersection(&sources[other]).count();
                let score = ADJACENCY_WEIGHT * adjacency_edges + shared_sources;
                (score > 0).then_some((score, page_id))
            })
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
    use llm_wiki_core::plan::PlanRelation;

    #[test]
    fn related_pages_follow_relation_adjacency_then_id() {
        let a = KnowledgeNodeId::parse("kn_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let b = KnowledgeNodeId::parse("kn_01BX5ZZKBKACTAV9WEVGEMMVRZ").unwrap();
        let c = KnowledgeNodeId::parse("kn_01CZZZZZZZZZZZZZZZZZZZZZZZ").unwrap();
        let isolated = KnowledgeNodeId::parse("kn_01DZZZZZZZZZZZZZZZZZZZZZZZ").unwrap();
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
            mk("wp_01ARZ3NDEKTSV4RRFFQ69G5FAV", "One", vec![a.clone()]),
            mk("wp_01BX5ZZKBKACTAV9WEVGEMMVRZ", "Two", vec![b.clone()]),
            mk(
                "wp_01CZZZZZZZZZZZZZZZZZZZZZZA",
                "Three",
                vec![c.clone(), isolated],
            ),
        ];
        // Audit FIX-008: page ref sets are disjoint by validation, so
        // relatedness comes from knowledge relation adjacency. One→Two and
        // One→Three are adjacent; Two and Three are not related to each other.
        let base = KnowledgeBase {
            nodes: BTreeMap::new(),
            relations: vec![
                PlanRelation {
                    source: a.clone(),
                    relation_type: "uses".into(),
                    target: b.clone(),
                },
                PlanRelation {
                    source: a,
                    relation_type: "uses".into(),
                    target: c,
                },
            ],
        };
        attach_related_pages(&mut pages, &base);
        // One: adjacent to Two and Three (1 edge each → tie → id order).
        assert_eq!(
            pages[0].related_pages,
            vec![pages[1].id.clone(), pages[2].id.clone()]
        );
        assert_eq!(pages[1].related_pages, vec![pages[0].id.clone()]);
        assert_eq!(pages[2].related_pages, vec![pages[0].id.clone()]);
    }

    #[test]
    fn related_pages_follow_shared_sources_without_relations() {
        let a = KnowledgeNodeId::parse("kn_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let b = KnowledgeNodeId::parse("kn_01BX5ZZKBKACTAV9WEVGEMMVRZ").unwrap();
        let source = SourceId::generate();
        let mk = |id: &str, title: &str, refs: Vec<KnowledgeNodeId>| WikiPagePlan {
            id: WikiPageId::parse(id).unwrap(),
            slug: title.to_lowercase(),
            title: title.to_owned(),
            category: "concepts".into(),
            purpose: "p".into(),
            knowledge_refs: refs,
            source_refs: vec![source.clone()],
            related_pages: vec![],
        };
        let mut pages = vec![
            mk("wp_01ARZ3NDEKTSV4RRFFQ69G5FAV", "One", vec![a]),
            mk("wp_01BX5ZZKBKACTAV9WEVGEMMVRZ", "Two", vec![b]),
        ];
        let base = KnowledgeBase {
            nodes: BTreeMap::new(),
            relations: vec![],
        };
        // No relation edges — the shared source anchor still relates them.
        attach_related_pages(&mut pages, &base);
        assert_eq!(pages[0].related_pages, vec![pages[1].id.clone()]);
        assert_eq!(pages[1].related_pages, vec![pages[0].id.clone()]);
    }

    #[test]
    fn reconcile_payload_carries_cluster_summaries() {
        let proposal = |title: &str| RawPage {
            title: title.to_owned(),
            category: "concepts".to_owned(),
            purpose: "p".to_owned(),
            knowledge_refs: vec!["kn_01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned()],
        };
        let proposals = vec![proposal("One"), proposal("Two")];
        let keys = vec!["s1".to_owned(), "s2".to_owned()];

        // First round: each proposal carries its cluster's summary.
        let payload = reconcile_payload(
            &proposals,
            &keys,
            &keys,
            Some(&["summary one".into(), "".into()]),
        );
        let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(
            value["proposals"][0]["cluster_summary"], "summary one",
            "the proposal's cluster summary rides into the reconcile context"
        );
        // An empty summary (unknown origin) adds no field.
        assert!(value["proposals"][1].get("cluster_summary").is_none());

        // Batch rounds: no summaries at all.
        let payload = reconcile_payload(&proposals, &keys, &keys, None);
        let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(value["proposals"][0].get("cluster_summary").is_none());
    }

    #[test]
    fn batch_keys_derive_from_content_not_process_state() {
        let page = RawPage {
            title: "Merged".to_owned(),
            category: "concepts".to_owned(),
            purpose: "merged".to_owned(),
            knowledge_refs: vec!["kn_01ARZ3NDEKTSV4RRFFQ69G5FAV".to_owned()],
        };
        // Same content → same key (identical prompts and cache keys across
        // processes); the summary/local prefixes must differ.
        assert_eq!(batch_key("batch-s", &page), batch_key("batch-s", &page));
        assert_ne!(batch_key("batch-s", &page), batch_key("batch-l", &page));
        let mut other = page.clone();
        other.title = "Different".to_owned();
        assert_ne!(batch_key("batch-s", &page), batch_key("batch-s", &other));
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
