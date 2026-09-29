//! V0.2 retrieval gates (PRD §20, §53 DoD #6/#9): build the fixture corpus
//! wiki with the stage-routing FakeLlmProvider, then run retrieval queries
//! against the REAL FTS index of the published generation and assert Top-K
//! for every §20 required category — 2–4 character CJK words, a CJK sentence,
//! English questions and mixed 中文/English terms.
//!
//! Queries are the verbatim question strings from `evals/questions.yaml`
//! (asserted against the file below, so fixture and gate cannot drift); the
//! expected target of every query is the wiki page whose citations cover the
//! question's `expected_sources` corpus doc.

use std::collections::BTreeMap;

use llm_wiki_compiler::run_build;
use llm_wiki_core::config::Config;
use llm_wiki_evals::{eval_llm, eval_workspace, evals_dir, load_fixtures};
use llm_wiki_search::{FullTextSearch, SqliteFullTextSearch};
use llm_wiki_storage::{list_sources, load_generation_view, open};

/// (query, expected corpus doc, Top-K bound, category label).
/// The strings are verbatim from `evals/questions.yaml` (Q-001, Q-006, Q-013,
/// Q-017, Q-022) or dedicated §20 term queries.
const GATE_QUERIES: &[(&str, &str, usize, &str)] = &[
    // §20 category: 2–4 character CJK words (dedicated term queries).
    ("检查点", "streams/checkpoints.cn.md", 3, "CJK word"),
    ("单点登录", "auth/sso.cn.md", 3, "CJK word"),
    // §20 category: CJK sentence (Q-013).
    (
        "流任务失败后从哪里恢复处理？",
        "streams/checkpoints.cn.md",
        5,
        "CJK sentence",
    ),
    // §20 category: English questions.
    (
        "How long do access tokens issued by Nimbus last?",
        "auth/authentication.md",
        5,
        "EN question",
    ),
    (
        "How many messages per second can a project publish to topics?",
        "messaging/topics.md",
        5,
        "EN question",
    ),
    (
        "How long can a single function invocation run?",
        "compute/functions.md",
        5,
        "EN question",
    ),
    (
        "What HTTP status do clients get when they exceed the API rate limit?",
        "api/api-rate-limits.md",
        5,
        "EN question",
    ),
    // §20 category: mixed 中文/English terms (title token + body terms).
    ("SSO 登录", "auth/sso.cn.md", 3, "mixed term"),
    (
        "HMAC webhook 签名",
        "messaging/webhooks.mdx",
        3,
        "mixed term",
    ),
];

/// Slugs of the ACTIVE generation's pages, keyed by the corpus doc path they
/// cite (citation → source registry rel_path).
fn pages_by_source(
    db_path: &std::path::Path,
    build_id: &llm_wiki_core::ids::BuildId,
) -> BTreeMap<String, Vec<String>> {
    let conn = open(db_path).unwrap();
    let mut source_rel: BTreeMap<String, String> = BTreeMap::new();
    for source in list_sources(&conn).unwrap() {
        source_rel.insert(
            source.source_id.as_str().to_owned(),
            source.rel_path.clone(),
        );
    }
    let mut mapping: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for page in load_generation_view(&conn, build_id).unwrap() {
        for citation in &page.citations {
            let rel = source_rel
                .get(citation.source_id.as_str())
                .cloned()
                .unwrap_or_default();
            let slugs = mapping.entry(rel).or_default();
            let slug = page.slug.clone();
            if !slugs.contains(&slug) {
                slugs.push(slug);
            }
        }
    }
    mapping
}

#[tokio::test]
async fn v02_search_gates_reach_top_k_for_cjk_en_and_mixed_queries() {
    let evals = evals_dir();
    // The gate queries must stay verbatim questions.yaml entries.
    let questions = std::fs::read_to_string(evals.join("questions.yaml")).unwrap();
    for (query, _, _, label) in GATE_QUERIES {
        if *label != "CJK word" && *label != "mixed term" {
            assert!(
                questions.contains(query),
                "{label} query {query:?} drifted from evals/questions.yaml"
            );
        }
    }

    let (dataset, expected_pages) =
        load_fixtures(&evals).expect("eval fixtures load and meet §37.3 minimums");
    let workspace = eval_workspace("search-gates");
    let config = Config::load(&workspace).unwrap();
    let db_path = workspace.join(".llm-wiki").join("state.db");

    // §53 #6: FTS searches the GENERATED wiki — built by the real pipeline.
    let provider = eval_llm(&dataset, &expected_pages);
    let report = run_build(&workspace, &config, provider)
        .await
        .expect("eval build succeeds");
    let by_source = pages_by_source(&db_path, &report.build_id);

    let conn = open(&db_path).unwrap();
    let search = SqliteFullTextSearch::new(conn, config.search.full_text);
    assert!(
        search.has_published().unwrap(),
        "the generation is active after publish"
    );

    let mut failures = Vec::new();
    for (query, expected_doc, k, label) in GATE_QUERIES {
        let hits = search.search(query, 10).await.unwrap();
        let expected_pages: Vec<&String> = by_source
            .get(*expected_doc)
            .map(|slugs| slugs.iter().collect())
            .unwrap_or_default();
        assert!(
            !expected_pages.is_empty(),
            "fixture bug: no wiki page cites {expected_doc}"
        );
        let hit_slugs: Vec<String> = hits.iter().map(|hit| hit.slug.clone()).collect();
        let found = hits
            .iter()
            .take(*k)
            .any(|hit| expected_pages.contains(&&hit.slug));
        if !found {
            failures.push(format!(
                "[{label}] {query:?} missed top-{k}: expected page(s) {expected_pages:?} citing {expected_doc}, got {hit_slugs:?}"
            ));
        }
        println!(
            "[{label}] {query:?} -> top{}: {}",
            k,
            hit_slugs
                .iter()
                .take(*k)
                .map(|slug| slug.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    assert!(
        failures.is_empty(),
        "§53 #9 Top-K retrieval gates failed:\n{}",
        failures.join("\n")
    );
}

#[tokio::test]
async fn search_refuses_to_run_before_any_build() {
    let workspace = eval_workspace("search-empty");
    let config = Config::load(&workspace).unwrap();
    let db_path = workspace.join(".llm-wiki").join("state.db");
    let conn = open(&db_path).unwrap();
    let search = SqliteFullTextSearch::new(conn, config.search.full_text);

    assert!(
        !search.has_published().unwrap(),
        "a fresh workspace has no active generation"
    );
    let err = search.search("检查点", 10).await.unwrap_err();
    assert!(
        err.to_string().contains("nothing published"),
        "search before any build must fail loudly, got: {err}"
    );
}
