//! §37.3 V0.1 release gates over the checked-in fixture corpus, driven by
//! FakeLlmProvider (CI-safe, PRD §54). Thresholds live in `gates::thresholds`
//! and MUST match `evals/README.md`.

use std::collections::BTreeMap;

use llm_wiki_compiler::{run_build, BuildReport};
use llm_wiki_core::config::Config;
use llm_wiki_evals::{
    eval_llm, eval_workspace, evals_dir, evaluate_gates, generation_manifest, load_fixtures,
};
use llm_wiki_storage::{list_sources, load_generation_view, open};

type View = Vec<llm_wiki_storage::GenerationPageView>;

fn manifest_of(db_path: &std::path::Path, report: &BuildReport) -> View {
    let conn = open(db_path).unwrap();
    let mut view = load_generation_view(&conn, &report.build_id).unwrap();
    view.sort_by(|a, b| a.slug.cmp(&b.slug));
    view
}

#[tokio::test]
async fn v01_release_gates_pass_over_the_fixture_corpus() {
    let evals = evals_dir();
    let (dataset, expected_pages) =
        load_fixtures(&evals).expect("eval fixtures load and meet §37.3 minimums");

    let workspace = eval_workspace("gates");
    let config = Config::load(&workspace).unwrap();
    let db_path = workspace.join(".llm-wiki").join("state.db");

    let provider = eval_llm(&dataset, &expected_pages);
    let first = run_build(&workspace, &config, provider.clone())
        .await
        .expect("first eval build succeeds");
    assert!(first.llm_request_count > 0, "first build calls the model");

    let second = run_build(&workspace, &config, provider.clone())
        .await
        .expect("second eval build succeeds");

    // Machine state of the two generations + the knowledge base + sources.
    let view_first = manifest_of(&db_path, &first);
    let view_second = manifest_of(&db_path, &second);
    let manifest_first = generation_manifest(&mut view_first.clone());
    let manifest_second = generation_manifest(&mut view_second.clone());

    let conn = open(&db_path).unwrap();
    let knowledge = llm_wiki_storage::load_knowledge_base(&conn).unwrap();
    let sources: BTreeMap<String, llm_wiki_storage::SourceRecord> = list_sources(&conn)
        .unwrap()
        .into_iter()
        .map(|s| (s.source_id.as_str().to_owned(), s))
        .collect();
    let statements: Vec<String> = knowledge
        .nodes
        .values()
        .filter(|n| n.kind == "claim")
        .filter_map(|n| n.statement.clone())
        .collect();

    let (report, metrics) = evaluate_gates(
        &dataset,
        &expected_pages,
        &mut view_second.clone(),
        &sources,
        &statements,
        &knowledge,
        second.llm_request_count,
        &manifest_first,
        &manifest_second,
    );

    println!(
        "eval metrics: coverage {}/{} high facts ({:.2}), citations {}/{} valid, claims {} ({} unbacked), expected pages {}",
        metrics.coverage.covered.len(),
        metrics.coverage.total,
        metrics.coverage.ratio(),
        metrics.citations.checked - metrics.citations.invalid.len(),
        metrics.citations.checked,
        metrics.hallucination.total_claims,
        metrics.hallucination.unbacked.len(),
        metrics.synthesis.len(),
    );

    assert!(
        report.passed(),
        "§37.3 gates failed: {:#?}",
        report.failures
    );
}
