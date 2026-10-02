//! Integration tests for the hierarchical planner and page compiler
//! (PRD §14/§15/§16), driven by FakeLlmProvider — no real model in CI
//! (PRD §54).
//!
//! Node ids are opaque registry-assigned ULIDs, so every scripted response
//! references the ids returned by the registry in this run — never literals.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use rusqlite::{params, Connection};

use llm_wiki_compiler::{load_prompt, CompilerConfig, PlannerConfig, WikiCompiler, WikiPlanner};
use llm_wiki_core::ids::{BuildId, KnowledgeNodeId, SourceLocatorKey};
use llm_wiki_core::model::WikiPagePlan;
use llm_wiki_core::plan::KnowledgeBase;
use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmProvider, LlmRequest};
use llm_wiki_storage::{
    get_or_create_batch, load_knowledge_base, open_in_memory, persist_generation, upsert_source,
    NodeDraft, NodeKind,
};

const SECTION_A: &str = "sec_01ARZ3NDEKTSV4RRFFQ69G5FAV";
const SECTION_B: &str = "sec_01BX5ZZKBKACTAV9WEVGEMMVRZ";

/// Fake provider that replays scripted responses.
struct ScriptedLlm {
    inner: FakeLlmProvider,
}

impl ScriptedLlm {
    fn new(responses: Vec<String>) -> Self {
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let inner = FakeLlmProvider::new(
            "fake-planner",
            Arc::new(move |_request: &LlmRequest| {
                queue
                    .lock()
                    .unwrap()
                    .pop_front()
                    .ok_or_else(|| LlmError::Api {
                        code: 500,
                        message: "script exhausted".into(),
                    })
            }),
        );
        Self { inner }
    }
}

#[async_trait::async_trait]
impl LlmProvider for ScriptedLlm {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    async fn generate(&self, request: LlmRequest) -> Result<llm_wiki_llm::LlmResponse, LlmError> {
        self.inner.generate(request).await
    }
}

/// Routes responses by prompt content instead of call order: cluster order
/// depends on registry-assigned ULIDs, so planner tests must not depend on
/// it. Each route is `(matcher, responses)`; the first matching route pops
/// its next response (repairs pop from the same route).
type Matcher = Box<dyn Fn(&LlmRequest) -> bool + Send + Sync>;

struct RouterLlm {
    inner: FakeLlmProvider,
}

fn router(routes: Vec<(Matcher, Vec<String>)>) -> RouterLlm {
    let mut matchers = Vec::with_capacity(routes.len());
    let mut queues = Vec::with_capacity(routes.len());
    for (matcher, responses) in routes {
        matchers.push(matcher);
        queues.push(Mutex::new(VecDeque::from(responses)));
    }
    let state = Arc::new((matchers, queues));
    let state_for_closure = state.clone();
    let inner = FakeLlmProvider::new(
        "fake-router",
        Arc::new(move |request: &LlmRequest| {
            let (matchers, queues) = &*state_for_closure;
            for (matcher, queue) in matchers.iter().zip(queues.iter()) {
                if matcher(request) {
                    return queue
                        .lock()
                        .unwrap()
                        .pop_front()
                        .ok_or_else(|| LlmError::Api {
                            code: 500,
                            message: "route script exhausted".into(),
                        });
                }
            }
            Err(LlmError::Api {
                code: 500,
                message: "no route matched".into(),
            })
        }),
    );
    RouterLlm { inner }
}

fn route(marker: &'static str, ids: &[String], responses: Vec<String>) -> (Matcher, Vec<String>) {
    let ids = ids.to_vec();
    let matcher: Matcher = Box::new(move |request: &LlmRequest| {
        request.prompt.contains(marker) && ids.iter().all(|id| request.prompt.contains(id.as_str()))
    });
    (matcher, responses)
}

const SUMMARY_MARK: &str = "Summarize the following cluster";
const LOCAL_MARK: &str = "THIS cluster's knowledge";
const RECONCILE_MARK: &str = "final global wiki plan";

#[async_trait::async_trait]
impl LlmProvider for RouterLlm {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    async fn generate(&self, request: LlmRequest) -> Result<llm_wiki_llm::LlmResponse, LlmError> {
        self.inner.generate(request).await
    }
}

// ---------------------------------------------------------------------------
// Fixture: two plugin sources, one relation, two grounded claims.
// ---------------------------------------------------------------------------

struct Seeded {
    base: KnowledgeBase,
    entity: String,
    concept: String,
    claim_a: String,
    claim_b: String,
}

fn seed_knowledge(conn: &mut Connection) -> Seeded {
    let (arch_id, _) = upsert_source(
        conn,
        &SourceLocatorKey::compute("ws", "plugin/architecture.md"),
        "plugin/architecture.md",
        "hash-arch",
        10,
        None,
    )
    .unwrap();
    let (sec_id, _) = upsert_source(
        conn,
        &SourceLocatorKey::compute("ws", "plugin/security.md"),
        "plugin/security.md",
        "hash-sec",
        10,
        None,
    )
    .unwrap();
    for (section_id, source, heading, range_end) in [
        (
            SECTION_A,
            arch_id.clone(),
            "Plugin Architecture > Lifecycle",
            100,
        ),
        (
            SECTION_B,
            sec_id.clone(),
            "Plugin Security > Permissions",
            80,
        ),
    ] {
        conn.execute(
            "INSERT INTO source_sections (section_id, source_id, heading_path_json, heading_path_key, content_fingerprint, range_start, range_end, status)
             VALUES (?1, ?2, ?3, ?3, 'fp', 0, ?4, 'active')",
            params![section_id, source.as_str(), heading, range_end],
        )
        .unwrap();
    }

    let drafts = vec![
        NodeDraft {
            kind: NodeKind::Entity,
            canonical_key: "plugin runtime".into(),
            canonical_name: "Plugin Runtime".into(),
            entity_type: Some("component".into()),
            description: Some("hosts plugins".into()),
        },
        NodeDraft {
            kind: NodeKind::Concept,
            canonical_key: "at-least-once delivery".into(),
            canonical_name: "At-Least-Once Delivery".into(),
            entity_type: None,
            description: Some("delivery guarantee".into()),
        },
        NodeDraft {
            kind: NodeKind::Claim,
            canonical_key: "stmt-runtime-retries".into(),
            canonical_name: "claim runtime retries".into(),
            entity_type: None,
            description: None,
        },
        NodeDraft {
            kind: NodeKind::Claim,
            canonical_key: "stmt-bus-permissions".into(),
            canonical_name: "claim bus permissions".into(),
            entity_type: None,
            description: None,
        },
    ];
    let ids = get_or_create_batch(conn, &drafts, None).unwrap();
    let entity = ids[0].as_str().to_owned();
    let concept = ids[1].as_str().to_owned();
    let claim_a = ids[2].as_str().to_owned();
    let claim_b = ids[3].as_str().to_owned();

    conn.execute(
        "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
         VALUES ('an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'completed', '2026-01-01'),
                ('an_01BX5ZZKBKACTAV9WEVGEMMVRZ', ?2, 'completed', '2026-01-01')",
        params![arch_id.as_str(), sec_id.as_str()],
    )
    .unwrap();
    for (claim_row, node, source, section, statement) in [
        (
            "cl_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            claim_a.clone(),
            arch_id.clone(),
            SECTION_A,
            "The runtime retries the transition up to three times.",
        ),
        (
            "cl_01BX5ZZKBKACTAV9WEVGEMMVRZ",
            claim_b.clone(),
            sec_id.clone(),
            SECTION_B,
            "Permissions are enforced at the message bus boundary.",
        ),
    ] {
        conn.execute(
            "INSERT INTO claims (claim_id, node_id, source_id, section_id, analysis_id, statement, evidence_digest, status)
             VALUES (?1, ?2, ?3, ?4, 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?5, 'digest-1', 'active')",
            params![claim_row, node, source.as_str(), section, statement],
        )
        .unwrap();
    }
    for (citation_row, claim_row, source, section, range_end, heading) in [
        (
            "cit_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "cl_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            arch_id.clone(),
            SECTION_A,
            64i64,
            "[\"Plugin Architecture\",\"Lifecycle\"]",
        ),
        (
            "cit_01BX5ZZKBKACTAV9WEVGEMMVRZ",
            "cl_01BX5ZZKBKACTAV9WEVGEMMVRZ",
            sec_id.clone(),
            SECTION_B,
            55i64,
            "[\"Plugin Security\",\"Permissions\"]",
        ),
    ] {
        conn.execute(
            "INSERT INTO citations (citation_id, owner_kind, owner_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json)
             VALUES (?1, 'claim', ?2, ?3, ?4, 4, ?5, 'hash-arch', 'digest-1', ?6)",
            params![citation_row, claim_row, source.as_str(), section, range_end, heading],
        )
        .unwrap();
    }
    conn.execute(
        "INSERT INTO relations (relation_id, analysis_id, source_node_id, relation_type, target_node_id, status)
         VALUES ('rel_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, 'uses', ?2, 'active')",
        params![entity, concept],
    )
    .unwrap();

    let base = load_knowledge_base(conn).unwrap();
    Seeded {
        base,
        entity,
        concept,
        claim_a,
        claim_b,
    }
}

fn summary_response(text: &str) -> String {
    serde_json::json!({ "summary": text }).to_string()
}

fn plan_response(pages: Vec<serde_json::Value>) -> String {
    serde_json::json!({ "pages": pages }).to_string()
}

fn proposal(title: &str, category: &str, refs: &[String]) -> serde_json::Value {
    serde_json::json!({
        "title": title,
        "category": category,
        "purpose": format!("cover {title}"),
        "knowledge_refs": refs,
    })
}

fn compile_response(markdown: &str) -> String {
    serde_json::json!({ "markdown": markdown }).to_string()
}

fn make_planner(config: PlannerConfig, responses: Vec<String>) -> (WikiPlanner, Arc<ScriptedLlm>) {
    let llm = Arc::new(ScriptedLlm::new(responses));
    let prompt = load_prompt("wiki-planning", None).unwrap();
    (WikiPlanner::new(llm.clone(), prompt, config), llm)
}

fn make_compiler(config: CompilerConfig, responses: Vec<String>) -> WikiCompiler {
    let llm = Arc::new(ScriptedLlm::new(responses));
    let prompt = load_prompt("wiki-compilation", None).unwrap();
    WikiCompiler::new(llm.clone(), prompt, config, 4)
}

fn planner_with(provider: Arc<dyn LlmProvider>, config: PlannerConfig) -> WikiPlanner {
    let prompt = load_prompt("wiki-planning", None).unwrap();
    WikiPlanner::new(provider, prompt, config)
}

// ---------------------------------------------------------------------------
// Planner tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn planner_full_pipeline_produces_pages_with_cache_keys() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);

    // Two clusters: [entity, concept] (relation) then [claim A, claim B]
    // (shared `plugin/` source dir). Routes select responses by stage + node
    // ids, so cluster iteration order (registry ULIDs) does not matter.
    let relation_pair = vec![seeded.entity.clone(), seeded.concept.clone()];
    let claims_pair = vec![seeded.claim_a.clone(), seeded.claim_b.clone()];
    let responses = vec![
        route(
            SUMMARY_MARK,
            &relation_pair,
            vec![summary_response("runtime and delivery")],
        ),
        route(
            SUMMARY_MARK,
            &claims_pair,
            vec![summary_response("guarantees and permissions")],
        ),
        route(
            LOCAL_MARK,
            &relation_pair,
            vec![plan_response(vec![proposal(
                "Plugin System",
                "concepts",
                &relation_pair,
            )])],
        ),
        route(
            LOCAL_MARK,
            &claims_pair,
            vec![plan_response(vec![proposal(
                "Plugin Guarantees",
                "concepts",
                &claims_pair,
            )])],
        ),
        route(
            RECONCILE_MARK,
            &[],
            vec![plan_response(vec![
                proposal("Plugin System", "concepts", &relation_pair),
                proposal("Plugin Guarantees", "concepts", &claims_pair),
            ])],
        ),
    ];
    let wiki_planner = planner_with(Arc::new(router(responses)), PlannerConfig::default());
    let outcome = wiki_planner.plan(&seeded.base, 4).await.unwrap();

    assert_eq!(outcome.llm_request_count, 5);
    assert_eq!(outcome.plan.pages.len(), 2);
    assert!(!outcome.flat_mode);
    for page in &outcome.plan.pages {
        assert!(page.id.as_str().starts_with("wp_"));
        assert!(!page.slug.is_empty());
        assert!(!page.purpose.is_empty());
    }
    let page_one = &outcome.plan.pages[0];
    assert_eq!(page_one.title, "Plugin System");
    assert_eq!(page_one.knowledge_refs.len(), 2);
    assert_eq!(
        page_one.source_refs.len(),
        0,
        "pure concept pages carry no claim anchors; sources come from claims"
    );
    assert_eq!(
        outcome.plan.pages[1].source_refs.len(),
        2,
        "claim page cites both plugin sources"
    );
    // Disjoint pages are not related.
    assert!(page_one.related_pages.is_empty());
    assert_eq!(outcome.plan.pages[1].title, "Plugin Guarantees");

    assert_eq!(outcome.cache.cluster_summary_keys.len(), 2);
    assert_eq!(outcome.cache.local_plan_keys.len(), 2);
    assert!(!outcome.cache.reconciliation_key.is_empty());

    // Determinism: identical inputs yield identical cache keys.
    let responses_again = vec![
        route(
            SUMMARY_MARK,
            &relation_pair,
            vec![summary_response("runtime and delivery")],
        ),
        route(
            SUMMARY_MARK,
            &claims_pair,
            vec![summary_response("guarantees and permissions")],
        ),
        route(
            LOCAL_MARK,
            &relation_pair,
            vec![plan_response(vec![proposal(
                "Plugin System",
                "concepts",
                &relation_pair,
            )])],
        ),
        route(
            LOCAL_MARK,
            &claims_pair,
            vec![plan_response(vec![proposal(
                "Plugin Guarantees",
                "concepts",
                &claims_pair,
            )])],
        ),
        route(
            RECONCILE_MARK,
            &[],
            vec![plan_response(vec![
                proposal("Plugin System", "concepts", &relation_pair),
                proposal("Plugin Guarantees", "concepts", &claims_pair),
            ])],
        ),
    ];
    let again_planner = planner_with(Arc::new(router(responses_again)), PlannerConfig::default());
    let again = again_planner.plan(&seeded.base, 4).await.unwrap();
    assert_eq!(
        again.cache.cluster_summary_keys,
        outcome.cache.cluster_summary_keys
    );
    assert_eq!(again.cache.local_plan_keys, outcome.cache.local_plan_keys);
    assert_eq!(
        again.cache.reconciliation_key,
        outcome.cache.reconciliation_key
    );
}

#[tokio::test]
async fn planner_repairs_unknown_node_ref_once() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);

    let relation_pair = vec![seeded.entity.clone(), seeded.concept.clone()];
    let claims_pair = vec![seeded.claim_a.clone(), seeded.claim_b.clone()];
    let responses = vec![
        route(
            SUMMARY_MARK,
            &relation_pair,
            vec![summary_response("runtime and delivery")],
        ),
        route(
            SUMMARY_MARK,
            &claims_pair,
            vec![summary_response("guarantees")],
        ),
        // First local plan hallucinates a node id; the repair pops the same
        // route and must fix the referential issue.
        route(
            LOCAL_MARK,
            &relation_pair,
            vec![
                plan_response(vec![serde_json::json!({
                    "title": "Plugin System",
                    "category": "concepts",
                    "purpose": "cover",
                    "knowledge_refs": ["kn_UNKNOWNNODE000000000000000"],
                })]),
                plan_response(vec![proposal("Plugin System", "concepts", &relation_pair)]),
            ],
        ),
        route(
            LOCAL_MARK,
            &claims_pair,
            vec![plan_response(vec![proposal(
                "Plugin Guarantees",
                "concepts",
                &claims_pair,
            )])],
        ),
        route(
            RECONCILE_MARK,
            &[],
            vec![plan_response(vec![
                proposal("Plugin System", "concepts", &relation_pair),
                proposal("Plugin Guarantees", "concepts", &claims_pair),
            ])],
        ),
    ];
    let wiki_planner = planner_with(Arc::new(router(responses)), PlannerConfig::default());
    let outcome = wiki_planner.plan(&seeded.base, 4).await.unwrap();
    assert_eq!(outcome.llm_request_count, 6, "one extra repair request");
    assert_eq!(outcome.plan.pages.len(), 2);
}

/// T1 Run 8/11-12 finding: thinking models attribute nodes to a NEIGHBORING
/// cluster's id space. A cross-cluster ref to a REAL library node must be
/// salvaged (its own page before finalize), not fail the stage; a ref that is
/// not a library node at all still repairs/fails as before.
#[tokio::test]
async fn cross_cluster_refs_are_salvaged_into_a_dedicated_page() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);

    let relation_pair = vec![seeded.entity.clone(), seeded.concept.clone()];
    let claims_pair = vec![seeded.claim_a.clone(), seeded.claim_b.clone()];
    let responses = vec![
        route(
            SUMMARY_MARK,
            &relation_pair,
            vec![summary_response("runtime and delivery")],
        ),
        route(
            SUMMARY_MARK,
            &claims_pair,
            vec![summary_response("guarantees")],
        ),
        // Cluster 1's local plan cites one of ITS OWN nodes correctly — plus
        // one node from cluster 2 (cross-cluster). The cross-cluster ref must
        // be salvaged, not trigger a repair.
        route(
            LOCAL_MARK,
            &relation_pair,
            vec![plan_response(vec![serde_json::json!({
                "title": "Plugin System",
                "category": "concepts",
                "purpose": "cover",
                "knowledge_refs": [seeded.entity, seeded.concept, seeded.claim_a],
            })])],
        ),
        route(
            LOCAL_MARK,
            &claims_pair,
            vec![plan_response(vec![proposal(
                "Plugin Guarantees",
                "concepts",
                &claims_pair,
            )])],
        ),
        route(
            RECONCILE_MARK,
            &[],
            // With bypass semantics the salvage page never enters the
            // reconcile input — the model sees only genuine cluster
            // proposals and echoes them (coverage-complete by construction).
            // The salvaged claim lands in its own appended page afterwards.
            vec![plan_response(vec![
                proposal("Plugin System", "concepts", &relation_pair),
                proposal("Plugin Guarantees", "concepts", &claims_pair),
            ])],
        ),
    ];
    let wiki_planner = planner_with(Arc::new(router(responses)), PlannerConfig::default());
    let outcome = wiki_planner.plan(&seeded.base, 4).await.unwrap();
    // No repair request: 2 summaries + 2 locals + 1 reconcile = 5 — the
    // cross-cluster ref did NOT trip the validator (salvage ran first).
    assert_eq!(outcome.llm_request_count, 5);
    // The salvage page appended nothing here: the model's own "Plugin
    // Guarantees" already covers the salvaged node (dedup retain emptied it).
    // What matters: coverage is complete and no repair was needed.
    assert_eq!(outcome.plan.pages.len(), 2);
    // Every node — including the cross-cluster one — is covered.
    let refs: Vec<String> = outcome
        .plan
        .pages
        .iter()
        .flat_map(|page| page.knowledge_refs.iter().map(|n| n.as_str().to_owned()))
        .collect();
    assert!(refs.contains(&seeded.claim_a.as_str().to_owned()));
}

#[tokio::test]
async fn planner_subdivides_clusters_over_node_budget() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    // max_cluster_nodes = 1 splits the 2-node relation cluster and the
    // 2-claim dir group into four singletons: 4 summaries + 4 locals + 1
    // reconcile merging everything into one page.
    let merge_all = plan_response(vec![proposal(
        "Plugin Platform",
        "concepts",
        &[
            seeded.entity.clone(),
            seeded.concept.clone(),
            seeded.claim_a.clone(),
            seeded.claim_b.clone(),
        ],
    )]);
    let singleton = |id: &str, page: &str| {
        route(
            LOCAL_MARK,
            std::slice::from_ref(&id.to_owned()),
            vec![plan_response(vec![proposal(
                page,
                "concepts",
                &[id.to_owned()],
            )])],
        )
    };
    let responses = vec![
        route(
            SUMMARY_MARK,
            std::slice::from_ref(&seeded.entity),
            vec![summary_response("s1")],
        ),
        singleton(&seeded.entity, "Page A"),
        route(
            SUMMARY_MARK,
            std::slice::from_ref(&seeded.concept),
            vec![summary_response("s2")],
        ),
        singleton(&seeded.concept, "Page B"),
        route(
            SUMMARY_MARK,
            std::slice::from_ref(&seeded.claim_a),
            vec![summary_response("s3")],
        ),
        singleton(&seeded.claim_a, "Page C"),
        route(
            SUMMARY_MARK,
            std::slice::from_ref(&seeded.claim_b),
            vec![summary_response("s4")],
        ),
        singleton(&seeded.claim_b, "Page D"),
        route(RECONCILE_MARK, &[], vec![merge_all]),
    ];
    let config = PlannerConfig {
        max_cluster_nodes: 1,
        ..PlannerConfig::default()
    };
    let wiki_planner = planner_with(Arc::new(router(responses)), config);
    let outcome = wiki_planner.plan(&seeded.base, 4).await.unwrap();
    assert_eq!(outcome.cache.cluster_summary_keys.len(), 4, "subdivided");
    assert_eq!(outcome.plan.pages.len(), 1);
    assert_eq!(
        outcome.plan.pages[0].knowledge_refs.len(),
        4,
        "no knowledge dropped"
    );
    assert_eq!(outcome.llm_request_count, 9);
}

/// T1 Run 13 regression (FIX-001): a reconcile payload that exceeds
/// `max_plan_input_tokens` (~40K estimated tokens against the 32K budget
/// here) must be BATCHED, never budget-checked as a whole — the old order
/// validated the full payload first and failed the build before the batch
/// path could run. The fake provider derives every response from the node
/// ids present in the prompt, so cluster/batch composition (registry ULIDs)
/// cannot break the script; it also records each reconcile prompt to prove
/// the full payload is never estimated or sent.
#[tokio::test]
async fn oversized_reconcile_payload_batches_before_budget_validation() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let known = [
        seeded.entity.clone(),
        seeded.concept.clone(),
        seeded.claim_a.clone(),
        seeded.claim_b.clone(),
    ];

    // (prompt chars, huge-purpose markers present, node ids present)
    let reconcile_prompts: Arc<Mutex<Vec<(usize, usize, usize)>>> = Arc::default();
    let capture = reconcile_prompts.clone();
    let ids_for_handler = known.clone();
    let handler = Arc::new(move |request: &LlmRequest| {
        let prompt = &request.prompt;
        if prompt.contains(SUMMARY_MARK) {
            return Ok(summary_response("s"));
        }
        if prompt.contains(LOCAL_MARK) {
            // One singleton cluster per node: propose one page whose huge
            // purpose (40K chars × 4 ≈ 160K chars ≈ 40K estimated tokens)
            // pushes the combined reconcile payload far past the 32K budget
            // and past the 96KB single-round byte ceiling.
            let refs: Vec<String> = ids_for_handler
                .iter()
                .filter(|id| prompt.contains(id.as_str()))
                .cloned()
                .collect();
            let node = refs[0].clone();
            let purpose = format!("BIGPURPOSE-{node} {}", "p".repeat(40_000));
            return Ok(plan_response(vec![serde_json::json!({
                "title": format!("Page {node}"),
                "category": "concepts",
                "purpose": purpose,
                "knowledge_refs": refs,
            })]));
        }
        if prompt.contains(RECONCILE_MARK) {
            let refs: Vec<String> = ids_for_handler
                .iter()
                .filter(|id| prompt.contains(id.as_str()))
                .cloned()
                .collect();
            let markers = ids_for_handler
                .iter()
                .filter(|id| prompt.contains(&format!("BIGPURPOSE-{id}")))
                .count();
            capture
                .lock()
                .unwrap()
                .push((prompt.chars().count(), markers, refs.len()));
            // Merge whatever THIS request proposed into one page covering
            // exactly those refs (echo ⇒ coverage holds per batch).
            return Ok(plan_response(vec![serde_json::json!({
                "title": "Merged",
                "category": "concepts",
                "purpose": "merged",
                "knowledge_refs": refs,
            })]));
        }
        Err(LlmError::Api {
            code: 500,
            message: "no stage matched".into(),
        })
    });
    let provider = FakeLlmProvider::new("fake-batched", handler);

    // max_cluster_nodes = 1 → four singleton clusters, mirroring
    // planner_subdivides_clusters_over_node_budget.
    let config = PlannerConfig {
        max_cluster_nodes: 1,
        ..PlannerConfig::default()
    };
    let prompt = load_prompt("wiki-planning", None).unwrap();
    let wiki_planner = WikiPlanner::new(Arc::new(provider), prompt, config);
    let outcome = wiki_planner.plan(&seeded.base, 4).await.unwrap();

    // 4 summaries + 4 locals + 2 batches + 1 final single-round reconcile,
    // no repairs anywhere.
    assert_eq!(outcome.llm_request_count, 11);
    assert_eq!(outcome.plan.pages.len(), 1);
    assert_eq!(
        outcome.plan.pages[0].knowledge_refs.len(),
        4,
        "full coverage survives batching"
    );

    let captured = reconcile_prompts.lock().unwrap().clone();
    assert_eq!(captured.len(), 3, "two batches + one final round");
    // The full 4-proposal payload (~160K chars) was never estimated or sent:
    // every reconcile request carried at most one batch (2 proposals).
    for (chars, markers, _) in &captured {
        assert!(*markers <= 2, "a request carried {markers} proposals");
        assert!(
            *chars < 110_000,
            "request of {chars} chars was not batch-sized"
        );
    }
    // The two batch rounds partitioned the four singleton proposals 2 + 2.
    assert_eq!(captured[0].1, 2);
    assert_eq!(captured[1].1, 2);
    assert_eq!(
        captured[2].1, 0,
        "final round sees the merged pages, not the huge purposes"
    );
}

/// #C02 regression: a degenerate reconcile that echoes one page per proposal
/// (huge purposes keep every round over the single-round ceiling) must fail
/// closed after MAX_RECONCILE_ROUNDS instead of looping forever.
#[tokio::test]
async fn non_converging_reconcile_fails_closed_after_the_round_cap() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let known = [
        seeded.entity.clone(),
        seeded.concept.clone(),
        seeded.claim_a.clone(),
        seeded.claim_b.clone(),
    ];

    let known_for_handler = known.clone();
    let handler = Arc::new(move |request: &LlmRequest| {
        let prompt = &request.prompt;
        if prompt.contains(SUMMARY_MARK) {
            return Ok(summary_response("s"));
        }
        if prompt.contains(LOCAL_MARK) {
            let refs: Vec<String> = known_for_handler
                .iter()
                .filter(|id| prompt.contains(id.as_str()))
                .cloned()
                .collect();
            let node = refs[0].clone();
            return Ok(plan_response(vec![serde_json::json!({
                "title": format!("Page {node}"),
                "category": "concepts",
                "purpose": format!("BIGPURPOSE-{node} {}", "p".repeat(40_000)),
                "knowledge_refs": refs,
            })]));
        }
        if prompt.contains(RECONCILE_MARK) {
            // Degenerate echo: one page per BIGPURPOSE marker, purposes kept
            // huge so every round stays over the single-round ceiling and the
            // proposal count never shrinks.
            let pages: Vec<serde_json::Value> = known_for_handler
                .iter()
                .filter(|id| prompt.contains(&format!("BIGPURPOSE-{id}")))
                .map(|id| {
                    serde_json::json!({
                        "title": format!("Page {id}"),
                        "category": "concepts",
                        "purpose": format!("BIGPURPOSE-{id} {}", "p".repeat(40_000)),
                        "knowledge_refs": [id.clone()],
                    })
                })
                .collect();
            return Ok(plan_response(pages));
        }
        Err(LlmError::Api {
            code: 500,
            message: "no stage matched".into(),
        })
    });
    let provider = Arc::new(FakeLlmProvider::new("fake-echo", handler));

    // max_cluster_nodes = 1 → four singleton clusters, each proposing one
    // huge-purpose page.
    let config = PlannerConfig {
        max_cluster_nodes: 1,
        ..PlannerConfig::default()
    };
    let prompt = load_prompt("wiki-planning", None).unwrap();
    let planner = WikiPlanner::new(provider.clone(), prompt, config);
    let error = planner.plan(&seeded.base, 4).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("did not converge"),
        "actionable: {message}"
    );

    // Bounded work: 4 summaries + 4 locals + at most MAX_RECONCILE_ROUNDS
    // reconcile rounds of 2 batched requests — then it stops.
    let requests = provider.request_count();
    assert!(
        requests <= 4 + 4 + 8 * 2,
        "the reconcile loop ran away: {requests} requests"
    );
}

/// FIX-007 regression: cluster staging must run CONCURRENTLY. Cluster-stage
/// requests pair up on a tokio Barrier — a serial planner deadlocks the pair
/// and plan() times out. The single reconcile request passes through
/// unpaired.
struct BarrierLlm {
    barrier: Arc<tokio::sync::Barrier>,
}

/// Every `kn_…` id occurring in the prompt, first-occurrence order.
fn kn_ids_in(prompt: &str) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    let mut rest = prompt;
    while let Some(pos) = rest.find("kn_") {
        let tail = &rest[pos..];
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        let id = tail[..end].to_owned();
        if !ids.contains(&id) {
            ids.push(id);
        }
        rest = &tail[end..];
    }
    ids
}

#[async_trait::async_trait]
impl LlmProvider for BarrierLlm {
    fn model(&self) -> &str {
        "fake-barrier"
    }

    fn provider_name(&self) -> &str {
        "fake-barrier"
    }

    async fn generate(&self, request: LlmRequest) -> Result<llm_wiki_llm::LlmResponse, LlmError> {
        let prompt = request.prompt;
        let text = if prompt.contains(RECONCILE_MARK) {
            plan_response(vec![proposal("Merged", "concepts", &kn_ids_in(&prompt))])
        } else if prompt.contains(SUMMARY_MARK) {
            self.barrier.wait().await;
            summary_response("s")
        } else if prompt.contains(LOCAL_MARK) {
            self.barrier.wait().await;
            plan_response(vec![proposal(
                "Cluster Page",
                "concepts",
                &kn_ids_in(&prompt),
            )])
        } else {
            return Err(LlmError::Api {
                code: 500,
                message: "no stage matched".into(),
            });
        };
        Ok(llm_wiki_llm::LlmResponse {
            text,
            model: "fake-barrier".to_owned(),
            input_tokens: 0,
            output_tokens: 0,
            finish_reason: Some("stop".to_owned()),
        })
    }
}

#[tokio::test]
async fn planner_clusters_stage_concurrently() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);

    // Two clusters (relation pair + claims pair): both summary→local chains
    // must overlap. A serial planner deadlocks the barrier pair → timeout.
    let config = PlannerConfig {
        max_concurrency: 2,
        ..PlannerConfig::default()
    };
    let planner = WikiPlanner::new(
        Arc::new(BarrierLlm {
            barrier: Arc::new(tokio::sync::Barrier::new(2)),
        }),
        load_prompt("wiki-planning", None).unwrap(),
        config,
    );
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        planner.plan(&seeded.base, 4),
    )
    .await
    .expect("planner deadlocked: cluster staging did not run concurrently")
    .unwrap();

    assert_eq!(
        outcome.llm_request_count, 5,
        "2 summaries + 2 locals + 1 reconcile"
    );
    assert_eq!(
        outcome.plan.pages.len(),
        1,
        "reconcile merged both proposals"
    );
    assert_eq!(
        outcome.plan.pages[0].knowledge_refs.len(),
        4,
        "full node coverage survives the concurrent staging"
    );
}

#[tokio::test]
async fn flat_mode_fails_with_actionable_error_over_budget() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let config = PlannerConfig {
        hierarchical: false,
        max_plan_input_tokens: 5,
        ..PlannerConfig::default()
    };
    let (wiki_planner, _llm) = make_planner(config, vec![]);
    let error = wiki_planner.plan(&seeded.base, 4).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("hierarchical"),
        "actionable message: {message}"
    );
    assert!(message.contains("max_plan_input_tokens"));
}

#[tokio::test]
async fn reconcile_repair_fails_closed_after_second_bad_response() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let relation_pair = vec![seeded.entity.clone(), seeded.concept.clone()];
    let claims_pair = vec![seeded.claim_a.clone(), seeded.claim_b.clone()];
    let hallucinated = plan_response(vec![proposal(
        "X",
        "concepts",
        &["kn_NOPE0000000000000000000000".to_owned()],
    )]);
    let responses = vec![
        route(SUMMARY_MARK, &relation_pair, vec![summary_response("s1")]),
        route(SUMMARY_MARK, &claims_pair, vec![summary_response("s2")]),
        route(
            LOCAL_MARK,
            &relation_pair,
            vec![plan_response(vec![proposal(
                "Plugin System",
                "concepts",
                &relation_pair,
            )])],
        ),
        route(
            LOCAL_MARK,
            &claims_pair,
            vec![plan_response(vec![proposal(
                "Plugin Guarantees",
                "concepts",
                &claims_pair,
            )])],
        ),
        // Reconciliation hallucinates a node id twice → the build fails.
        route(
            RECONCILE_MARK,
            &[],
            vec![hallucinated.clone(), hallucinated],
        ),
    ];
    let wiki_planner = planner_with(Arc::new(router(responses)), PlannerConfig::default());
    let error = wiki_planner.plan(&seeded.base, 4).await.unwrap_err();
    assert!(error.to_string().contains("UNKNOWN_NODE_REF"));
}

// ---------------------------------------------------------------------------
// Compiler tests
// ---------------------------------------------------------------------------

fn two_page_plan(seeded: &Seeded) -> llm_wiki_core::model::WikiPlan {
    use llm_wiki_core::ids::{SourceId, WikiPageId};
    let mut page_one = WikiPagePlan {
        id: WikiPageId::generate(),
        slug: "plugin-system".into(),
        title: "Plugin System".into(),
        category: "concepts".into(),
        purpose: "overview".into(),
        knowledge_refs: vec![
            KnowledgeNodeId::parse(seeded.entity.clone()).unwrap(),
            KnowledgeNodeId::parse(seeded.concept.clone()).unwrap(),
            KnowledgeNodeId::parse(seeded.claim_a.clone()).unwrap(),
        ],
        source_refs: vec![SourceId::generate()],
        related_pages: vec![],
    };
    let page_two = WikiPagePlan {
        id: WikiPageId::generate(),
        slug: "plugin-guarantees".into(),
        title: "Plugin Guarantees".into(),
        category: "concepts".into(),
        purpose: "guarantees".into(),
        knowledge_refs: vec![KnowledgeNodeId::parse(seeded.claim_b.clone()).unwrap()],
        source_refs: vec![],
        related_pages: vec![page_one.id.clone()],
    };
    page_one.related_pages = vec![page_two.id.clone()];
    llm_wiki_core::model::WikiPlan {
        pages: vec![page_one, page_two],
    }
}

#[tokio::test]
async fn compiler_expands_citations_resolves_links_and_persists() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let plan = two_page_plan(&seeded);

    let body_one = format!(
        "## Overview\n\nThe runtime hosts plugins and retries transitions. <!-- llm-wiki:cite claim=\"{}\" -->\n\nGuarantees live in [[Plugin Guarantees]].",
        seeded.claim_a
    );
    let body_two = format!(
        "## Guarantees\n\nDelivery is at-least-once. <!-- llm-wiki:cite claim=\"{}\" -->",
        seeded.claim_b
    );
    let responses = vec![compile_response(&body_one), compile_response(&body_two)];
    let compiler = make_compiler(
        CompilerConfig {
            language_tag: "en".into(),
            ..CompilerConfig::default()
        },
        responses,
    );

    let build_id = BuildId::generate();
    let generation = compiler
        .compile_plan(&plan, &seeded.base, &build_id)
        .await
        .unwrap();
    assert_eq!(generation.llm_request_count, 2);
    assert_eq!(generation.pages.len(), 2);

    let page_one = &generation.pages[0];
    assert!(page_one.content.contains("generated: true"));
    assert!(page_one.content.contains("schema_version: 1"));
    assert!(page_one.content.contains("language: en"));
    assert!(page_one.content.starts_with("---\n"));
    assert!(page_one
        .content
        .contains(&format!("id: {}", page_one.page_id.as_str())));
    assert!(page_one.content.contains("  - plugin/architecture.md"));
    // Citation expanded from stored anchors: path, heading path, range, digest.
    assert!(page_one
        .content
        .contains("source=\"plugin/architecture.md\" section=\"Plugin Architecture > Lifecycle\""));
    assert!(page_one.content.contains("range=\"4-64\""));
    assert!(page_one.content.contains("digest=\"digest-1\""));
    assert!(page_one.content.contains("## Related"));
    assert!(page_one.content.contains("- [[Plugin Guarantees]]"));
    assert!(!page_one.body_hash.is_empty());
    assert_eq!(page_one.citations.len(), 1);
    assert_eq!(page_one.links.len(), 1);
    assert_eq!(page_one.links[0].target_title, "Plugin Guarantees");
    // The app-generated Related link on page two resolves as well.
    assert_eq!(generation.pages[1].links.len(), 1);
    assert_eq!(
        generation.pages[1].links[0].to_page_id,
        generation.pages[0].page_id
    );

    // Persisted machine state (PRD §16: Markdown + DB mapping).
    let stats = persist_generation(&mut conn, &build_id, &generation.pages).unwrap();
    assert_eq!(
        stats,
        llm_wiki_storage::GenerationStats {
            pages: 2,
            citations: 2,
            links: 2
        }
    );
}

#[tokio::test]
async fn compiler_repairs_hallucinated_claim_once() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let plan = two_page_plan(&seeded);

    let bad = compile_response(
        "## Overview\n\nText. <!-- llm-wiki:cite claim=\"kn_HALLUCINATED000000000000000\" -->",
    );
    let good = compile_response(&format!(
        "## Overview\n\nText. <!-- llm-wiki:cite claim=\"{}\" -->",
        seeded.claim_a
    ));
    let page_two_ok = compile_response(&format!(
        "## Guarantees\n\nFine. <!-- llm-wiki:cite claim=\"{}\" -->",
        seeded.claim_b
    ));
    let responses = vec![bad, good, page_two_ok];
    let compiler = make_compiler(CompilerConfig::default(), responses);

    let generation = compiler
        .compile_plan(&plan, &seeded.base, &BuildId::generate())
        .await
        .unwrap();
    assert_eq!(
        generation.llm_request_count, 3,
        "page 1: repair; page 2: one"
    );
    assert_eq!(generation.pages[0].citations.len(), 1);
}

#[tokio::test]
async fn compiler_fails_closed_when_repair_still_hallucinates() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let plan = two_page_plan(&seeded);

    let bad = compile_response(
        "## Overview\n\nText. <!-- llm-wiki:cite claim=\"kn_HALLUCINATED000000000000000\" -->",
    );
    let compiler = make_compiler(CompilerConfig::default(), vec![bad.clone(), bad]);

    let error = compiler
        .compile_plan(&plan, &seeded.base, &BuildId::generate())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("UNKNOWN_CLAIM_REF"));
}

#[tokio::test]
async fn compiler_ungrounded_body_triggers_single_repair() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let plan = two_page_plan(&seeded);

    // Page 1 has claim knowledge but cites nothing → UNGROUNDED_BODY.
    let ungrounded = compile_response("## Overview\n\nNo citations here at all.");
    let good = compile_response(&format!(
        "## Overview\n\nCited text. <!-- llm-wiki:cite claim=\"{}\" -->",
        seeded.claim_a
    ));
    let page_two_ok = compile_response(&format!(
        "## Guarantees\n\nFine. <!-- llm-wiki:cite claim=\"{}\" -->",
        seeded.claim_b
    ));
    let responses = vec![ungrounded, good, page_two_ok];
    let compiler = make_compiler(CompilerConfig::default(), responses);

    let generation = compiler
        .compile_plan(&plan, &seeded.base, &BuildId::generate())
        .await
        .unwrap();
    assert_eq!(generation.llm_request_count, 3);
}

/// FIX-004 regression: JoinSet yields COMPLETION order, so a page whose LLM
/// call is slow must not reorder the generation — pages come back in plan
/// order regardless of when their tasks finish.
struct DelayingLlm {
    inner: FakeLlmProvider,
}

#[async_trait::async_trait]
impl LlmProvider for DelayingLlm {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    async fn generate(&self, request: LlmRequest) -> Result<llm_wiki_llm::LlmResponse, LlmError> {
        if request.prompt.contains("Slow Page") {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        self.inner.generate(request).await
    }
}

#[tokio::test]
async fn compiler_returns_pages_in_plan_order_despite_completion_order() {
    use llm_wiki_core::ids::WikiPageId;
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let mk = |title: &str| WikiPagePlan {
        id: WikiPageId::generate(),
        slug: title.to_lowercase().replace(' ', "-"),
        title: title.to_owned(),
        category: "concepts".into(),
        purpose: "overview".into(),
        knowledge_refs: vec![KnowledgeNodeId::parse(seeded.entity.clone()).unwrap()],
        source_refs: vec![],
        related_pages: vec![],
    };
    let plan = llm_wiki_core::model::WikiPlan {
        pages: vec![mk("Slow Page"), mk("Fast B"), mk("Fast C")],
    };

    let inner = FakeLlmProvider::fixed(
        "fake-slow",
        compile_response("## Overview\n\nGrounded overview text."),
    );
    let prompt = load_prompt("wiki-compilation", None).unwrap();
    let compiler = WikiCompiler::new(
        Arc::new(DelayingLlm { inner }),
        prompt,
        CompilerConfig::default(),
        4,
    );

    let generation = compiler
        .compile_plan(&plan, &seeded.base, &BuildId::generate())
        .await
        .unwrap();
    assert_eq!(generation.llm_request_count, 3);
    let titles: Vec<&str> = generation.pages.iter().map(|p| p.title.as_str()).collect();
    assert_eq!(titles, vec!["Slow Page", "Fast B", "Fast C"]);
}

// ---------------------------------------------------------------------------
// Review-fix regression tests (#I01–#I04)
// ---------------------------------------------------------------------------

/// Inserts an active claim with NO citation rows (no anchors), so citing it
/// can never expand (review #I04).
fn seed_unanchored_claim(conn: &mut Connection) -> String {
    let drafts = vec![NodeDraft {
        kind: NodeKind::Claim,
        canonical_key: "stmt-orphan".into(),
        canonical_name: "claim orphan".into(),
        entity_type: None,
        description: None,
    }];
    let ids = get_or_create_batch(conn, &drafts, None).unwrap();
    let node = ids[0].as_str().to_owned();
    conn.execute(
        "INSERT INTO claims (claim_id, node_id, source_id, section_id, analysis_id, statement, evidence_digest, status)
         VALUES ('cl_01CZZZZZZZZZZZZZZZZZZZZZZA', ?1,
                 (SELECT source_id FROM sources ORDER BY source_id LIMIT 1),
                 NULL, 'an_01ARZ3NDEKTSV4RRFFQ69G5FAV', 'Orphan statement with no stored anchor.', 'digest-orphan', 'active')",
        params![node],
    )
    .unwrap();
    node
}

#[tokio::test]
async fn planner_shape_repair_with_hallucinated_refs_fails_closed() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    // Review #I01: the FIRST local-plan response is unparseable, the repair
    // parses but hallucinates a node id. The repaired response must still go
    // through validation and fail the stage closed.
    let relation_pair = vec![seeded.entity.clone(), seeded.concept.clone()];
    let claims_pair = vec![seeded.claim_a.clone(), seeded.claim_b.clone()];
    let responses = vec![
        route(SUMMARY_MARK, &relation_pair, vec![summary_response("s1")]),
        route(SUMMARY_MARK, &claims_pair, vec![summary_response("s2")]),
        route(
            LOCAL_MARK,
            &relation_pair,
            vec![
                "this is not json at all".to_owned(),
                plan_response(vec![proposal(
                    "Bad",
                    "concepts",
                    &["kn_NOPE0000000000000000000000".to_owned()],
                )]),
            ],
        ),
        route(
            LOCAL_MARK,
            &claims_pair,
            vec![plan_response(vec![proposal(
                "Plugin Guarantees",
                "concepts",
                &claims_pair,
            )])],
        ),
    ];
    let wiki_planner = planner_with(Arc::new(router(responses)), PlannerConfig::default());
    let error = wiki_planner.plan(&seeded.base, 4).await.unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("UNKNOWN_NODE_REF") && message.contains("after repair"),
        "repaired response must be validated: {message}"
    );
}

#[tokio::test]
async fn reconcile_missing_coverage_fails_closed() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    // Review #I02: reconciliation drops the claims cluster → coverage check
    // must reject the plan instead of silently losing knowledge.
    let relation_pair = vec![seeded.entity.clone(), seeded.concept.clone()];
    let claims_pair = vec![seeded.claim_a.clone(), seeded.claim_b.clone()];
    let responses = vec![
        route(SUMMARY_MARK, &relation_pair, vec![summary_response("s1")]),
        route(SUMMARY_MARK, &claims_pair, vec![summary_response("s2")]),
        route(
            LOCAL_MARK,
            &relation_pair,
            vec![plan_response(vec![proposal(
                "Plugin System",
                "concepts",
                &relation_pair,
            )])],
        ),
        route(
            LOCAL_MARK,
            &claims_pair,
            vec![plan_response(vec![proposal(
                "Plugin Guarantees",
                "concepts",
                &claims_pair,
            )])],
        ),
        route(
            RECONCILE_MARK,
            &[],
            // Both the first response and its repair drop the claims cluster:
            // after the single repair the plan must fail closed.
            vec![
                plan_response(vec![proposal("Plugin System", "concepts", &relation_pair)]),
                plan_response(vec![proposal("Plugin System", "concepts", &relation_pair)]),
            ],
        ),
    ];
    let wiki_planner = planner_with(Arc::new(router(responses)), PlannerConfig::default());
    let error = wiki_planner.plan(&seeded.base, 4).await.unwrap_err();
    assert!(
        error.to_string().contains("MISSING_NODE_COVERAGE"),
        "dropped knowledge must fail reconciliation: {error}"
    );
}

#[tokio::test]
async fn compiler_fails_closed_when_page_input_exceeds_budget() {
    let mut conn = open_in_memory().unwrap();
    let seeded = seed_knowledge(&mut conn);
    let plan = two_page_plan(&seeded);
    // Review #I03: an over-budget page payload must fail before any request.
    let compiler = make_compiler(
        CompilerConfig {
            max_input_tokens: 10,
            ..CompilerConfig::default()
        },
        vec![],
    );
    let error = compiler
        .compile_plan(&plan, &seeded.base, &BuildId::generate())
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("max_input_tokens") && message.contains("split"),
        "actionable budget failure expected: {message}"
    );
}

#[tokio::test]
async fn compiler_repairs_unanchored_claim_citation() {
    let mut conn = open_in_memory().unwrap();
    let mut seeded = seed_knowledge(&mut conn);
    let orphan = seed_unanchored_claim(&mut conn);
    seeded.base = load_knowledge_base(&conn).unwrap();
    let mut plan = two_page_plan(&seeded);
    plan.pages[0]
        .knowledge_refs
        .push(KnowledgeNodeId::parse(orphan.clone()).unwrap());

    // Page 1 cites the anchor-less claim → UNANCHORED_CLAIM; the repair must
    // fall back to the anchored claim.
    let bad = compile_response(&format!(
        "## Overview\n\nText. <!-- llm-wiki:cite claim=\"{orphan}\" -->"
    ));
    let good = compile_response(&format!(
        "## Overview\n\nText. <!-- llm-wiki:cite claim=\"{}\" -->",
        seeded.claim_a
    ));
    let page_two_ok = compile_response(&format!(
        "## Guarantees\n\nFine. <!-- llm-wiki:cite claim=\"{}\" -->",
        seeded.claim_b
    ));
    let compiler = make_compiler(CompilerConfig::default(), vec![bad, good, page_two_ok]);

    let generation = compiler
        .compile_plan(&plan, &seeded.base, &BuildId::generate())
        .await
        .unwrap();
    assert_eq!(generation.llm_request_count, 3, "page 1 repairs once");
    assert_eq!(generation.pages[0].citations.len(), 1);
    assert!(generation.pages[0].content.contains("digest=\"digest-1\""));
    assert!(
        !generation.pages[0].content.contains(&orphan),
        "the unexpandable citation must not survive"
    );
}

#[tokio::test]
async fn compiler_fails_closed_when_repair_still_cites_unanchored() {
    let mut conn = open_in_memory().unwrap();
    let mut seeded = seed_knowledge(&mut conn);
    let orphan = seed_unanchored_claim(&mut conn);
    seeded.base = load_knowledge_base(&conn).unwrap();
    let mut plan = two_page_plan(&seeded);
    plan.pages[0]
        .knowledge_refs
        .push(KnowledgeNodeId::parse(orphan.clone()).unwrap());

    let bad = compile_response(&format!(
        "## Overview\n\nText. <!-- llm-wiki:cite claim=\"{orphan}\" -->"
    ));
    let compiler = make_compiler(CompilerConfig::default(), vec![bad.clone(), bad]);

    let error = compiler
        .compile_plan(&plan, &seeded.base, &BuildId::generate())
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("UNANCHORED_CLAIM"),
        "persistent unanchored citations must fail the build: {error}"
    );
}
