//! V0.5 retrieval eval gate (PRD §50): build the fixture wiki, run EVERY
//! `evals/questions.yaml` query against the real FTS index and assert
//! aggregate Recall@5 / MRR thresholds. This is the ranked-retrieval
//! baseline any future rerank strategy must beat (see `llm-wiki-search`
//! rerank abstraction).

use std::collections::BTreeMap;

use llm_wiki_compiler::run_build;
use llm_wiki_core::config::Config;
use llm_wiki_evals::{
    aggregate, eval_llm, eval_workspace, evals_dir, load_fixtures, load_questions, score_query,
    RetrievalQuestionResult,
};
use llm_wiki_search::{FullTextSearch, SqliteFullTextSearch};
use llm_wiki_storage::{list_sources, load_generation_view, open};

/// Page slugs keyed by the corpus doc path they cite.
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

/// Candidate depth per query (recall@5 over the top 5 hits).
const K: usize = 5;
/// §37-style release gates. Calibrated against the fixture corpus: the
/// lexical-only baseline must stay honest — raise these only together with
/// a rerank/retrieval improvement, never to make a failing build pass.
const MIN_RECALL_AT_K: f64 = 0.75;
const MIN_MRR: f64 = 0.5;

#[tokio::test]
async fn retrieval_eval_recall_and_mrr_over_the_full_question_set() {
    let evals = evals_dir();
    let (dataset, expected_pages) =
        load_fixtures(&evals).expect("eval fixtures load and meet §37.3 minimums");
    let queries = load_questions(&evals).expect("questions.yaml parses");

    let workspace = eval_workspace("retrieval-eval");
    let config = Config::load(&workspace).unwrap();
    let db_path = workspace.join(".llm-wiki").join("state.db");
    let provider = eval_llm(&dataset, &expected_pages);
    let report = run_build(&workspace, &config, provider)
        .await
        .expect("eval build succeeds");
    let by_source = pages_by_source(&db_path, &report.build_id);

    let conn = open(&db_path).unwrap();
    let search = SqliteFullTextSearch::new(conn, config.search.full_text);
    assert!(search.has_published().unwrap());

    let mut results = Vec::new();
    for query in &queries {
        let hits = search.search(&query.question, 25).await.unwrap();
        let hit_slugs: Vec<String> = hits.iter().map(|hit| hit.slug.clone()).collect();
        let rank = score_query(&hit_slugs, &query.expected_sources, &by_source);
        results.push(RetrievalQuestionResult {
            id: query.id.clone(),
            first_relevant_rank: rank,
        });
    }
    let eval = aggregate(results, K);

    println!(
        "retrieval eval: {} questions, recall@{K} = {:.3}, mrr = {:.3}",
        eval.questions, eval.recall_at_k, eval.mrr
    );
    for question in &eval.per_question {
        if question.first_relevant_rank == 0 {
            println!("  MISSED: {}", question.id);
        }
    }

    assert_eq!(eval.questions, queries.len(), "every question is scored");
    assert!(
        eval.recall_at_k >= MIN_RECALL_AT_K,
        "recall@{K} {:.3} < {MIN_RECALL_AT_K} — lexical retrieval regressed",
        eval.recall_at_k
    );
    assert!(
        eval.mrr >= MIN_MRR,
        "MRR {:.3} < {MIN_MRR} — lexical retrieval regressed",
        eval.mrr
    );
}
