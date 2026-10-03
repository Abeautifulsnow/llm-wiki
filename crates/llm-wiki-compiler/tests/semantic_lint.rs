//! Integration tests for the semantic lint (audit FIX-019), driven by
//! FakeLlmProvider: a real generation is published first, then the semantic
//! review runs over it — findings validated against the page's real claims,
//! cached across runs, with the fail-open page-skip contract.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use llm_wiki_compiler::run_build;
use llm_wiki_core::config::Config;
use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmRequest};
use std::path::PathBuf;

const PAGE_REVIEW_MARK: &str = "Review ONE page";
const GAP_MARK: &str = "coverage gaps";

/// Stage-routing fake: handles the full build pipeline (so a generation is
/// published) plus the semantic-lint stages. The page-review handler can be
/// scripted with a queue of responses (repairs pop from the same queue).
fn semantic_llm(page_review_responses: Arc<Mutex<VecDeque<String>>>) -> Arc<FakeLlmProvider> {
    let proposed: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
    let handler = Arc::new(move |request: &LlmRequest| -> Result<String, LlmError> {
        let prompt = &request.prompt;
        match request.task_tag.as_str() {
            "document-analysis" => {
                let section_ids = json_string_values(prompt, "section_id");
                let contents = json_string_values(prompt, "content");
                let claims: Vec<serde_json::Value> = section_ids
                    .iter()
                    .zip(contents.iter())
                    .filter_map(|(section_id, content)| {
                        let quote = content
                            .lines()
                            .map(str::trim)
                            .find(|line| !line.is_empty())?;
                        Some(serde_json::json!({
                            "text": quote,
                            "section_id": section_id,
                            "evidence_text": quote,
                            "evidence_start": content.find(quote).unwrap_or(0),
                            "confidence": 0.9,
                        }))
                    })
                    .collect();
                Ok(serde_json::json!({
                    "summary": "semantic fixture analysis.",
                    "topics": ["fixture"],
                    "entities": [],
                    "concepts": [],
                    "claims": claims,
                    "relations": [],
                })
                .to_string())
            }
            "wiki-planning" => {
                if prompt.contains("Summarize the following cluster") {
                    return Ok(serde_json::json!({"summary": "cluster summary"}).to_string());
                }
                let ids = kn_ids_in(prompt);
                if prompt.contains("THIS cluster's knowledge") {
                    proposed.lock().unwrap().extend(ids.iter().cloned());
                    return Ok(serde_json::json!({
                        "pages": [{
                            "title": "Semantic Fixture",
                            "category": "concepts",
                            "purpose": "cover the fixture",
                            "knowledge_refs": ids,
                        }]
                    })
                    .to_string());
                }
                let ids = proposed.lock().unwrap().clone();
                Ok(serde_json::json!({
                    "pages": [{
                        "title": "Semantic Fixture",
                        "category": "concepts",
                        "purpose": "cover the fixture",
                        "knowledge_refs": ids,
                    }]
                })
                .to_string())
            }
            "wiki-compilation" => {
                let claims = claim_ids_in(prompt);
                let body = if claims.is_empty() {
                    "## Overview\n\nA plain page.".to_owned()
                } else {
                    format!(
                        "## Overview\n\nThe fixture page cites its stored claims.\n\n<!-- llm-wiki:cite claim=\"{}\" -->",
                        claims[0]
                    )
                };
                Ok(serde_json::json!({ "markdown": body }).to_string())
            }
            "wiki-semantic-lint" => {
                let mut queue = page_review_responses.lock().unwrap();
                let response = if prompt.contains(PAGE_REVIEW_MARK) {
                    queue.pop_front().ok_or_else(|| LlmError::Api {
                        code: 500,
                        message: "page-review script exhausted".into(),
                    })?
                } else if prompt.contains(GAP_MARK) {
                    serde_json::json!({"gaps": [{
                        "topic": "retry budgets",
                        "reason": "the fixture page leans on retry semantics with no page of its own",
                    }]})
                    .to_string()
                } else {
                    return Err(LlmError::Api {
                        code: 500,
                        message: "no semantic stage matched".into(),
                    });
                };
                drop(queue);
                Ok(response)
            }
            other => Err(LlmError::Api {
                code: 500,
                message: format!("unexpected task tag {other}"),
            }),
        }
    });
    Arc::new(FakeLlmProvider::new("fake-semantic", handler))
}

fn kn_ids_in(prompt: &str) -> Vec<String> {
    let mut ids = Vec::new();
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

/// Claim node ids from a compilation prompt: each `"kind":"claim"` node's
/// `"id"` (the payload is serialized without spaces by serde_json).
fn claim_ids_in(prompt: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = prompt[search_from..].find("\"kind\":\"claim\"") {
        let before = &prompt[..search_from + offset];
        if let Some(id_offset) = before.rfind("\"id\":\"kn_") {
            let start = id_offset + "\"id\":\"".len();
            let rest = &prompt[start..];
            let end = rest.find('"').unwrap_or(rest.len());
            let id = rest[..end].to_owned();
            if !out.contains(&id) {
                out.push(id);
            }
        }
        search_from += offset + "\"kind\":\"claim\"".len();
    }
    out
}

fn json_string_values(text: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\":\"");
    let mut values = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find(&needle) {
        let tail = &rest[pos + needle.len()..];
        let end = tail.find('"').unwrap_or(tail.len());
        values.push(tail[..end].to_owned());
        rest = &tail[end..];
    }
    values
}

fn fixture_workspace(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-semantic-{tag}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    std::fs::create_dir_all(dir.join("docs")).unwrap();
    std::fs::write(
        dir.join("docs").join("runtime.md"),
        "# Runtime\n\nThe scheduler retries failed tasks up to three times before giving up.\n",
    )
    .unwrap();
    std::fs::create_dir_all(dir.join(".llm-wiki")).unwrap();
    std::fs::write(
        dir.join(".llm-wiki").join("config.toml"),
        "[project]\nname = \"fixture\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-semantic\"\n",
    )
    .unwrap();
    dir
}

/// End-to-end: build a generation, then `run_semantic_lint` reports the
/// scripted contradiction (validated against the page's REAL claim id and a
/// verbatim excerpt) and the corpus gap — and the second run pays zero model
/// calls (§28 cache).
#[tokio::test]
async fn semantic_lint_reports_validated_findings_and_caches() {
    let workspace = fixture_workspace("e2e");
    let config = Config::load(&workspace).unwrap();

    // Publish one generation.
    let build_provider = semantic_llm(Arc::new(Mutex::new(VecDeque::new())));
    let report = run_build(&workspace, &config, build_provider)
        .await
        .unwrap();
    assert!(report.citations >= 1, "the fixture page cites its claim");

    // The page-review response cites the fixture's real claim id (extracted
    // from the prompt at request time) with a verbatim body excerpt.
    let page_response = Arc::new(Mutex::new(VecDeque::new()));
    let scripted = Arc::clone(&page_response);
    let provider = semantic_llm(Arc::clone(&page_response));
    // The claim id is not known until the build ran — capture it by letting
    // the FIRST response be produced lazily: instead, read it from the KB
    // through the built report? The build provider consumed the claim id;
    // simplest robust approach: the queue response uses a wildcard resolved
    // by the fake (kn_ids_in). So the scripted response is built by the fake
    // itself — here we can just return a template the fake filled in. For
    // the queue-based fake the response must already contain the id; pull it
    // from the DB.
    {
        let conn = llm_wiki_storage::open(&workspace.join(".llm-wiki").join("state.db")).unwrap();
        let base = llm_wiki_storage::load_knowledge_base(&conn).unwrap();
        let claim_id = base
            .nodes
            .keys()
            .find(|id| base.nodes[*id].kind == "claim")
            .expect("fixture claim exists")
            .as_str()
            .to_owned();
        scripted.lock().unwrap().push_back(
            serde_json::json!({"findings": [{
                "kind": "contradiction",
                "claim_ids": [claim_id],
                "reason": "the two halves of the statement cannot both hold",
                "excerpt": "The fixture page cites its stored claims.",
            }]})
            .to_string(),
        );
    }

    let first = llm_wiki_compiler::run_semantic_lint(&workspace, &config, provider.clone())
        .await
        .unwrap()
        .expect("generation is published");
    assert_eq!(first.pages_reviewed, 1);
    let contradictions: Vec<_> = first
        .findings
        .iter()
        .filter(|f| f.kind == llm_wiki_compiler::SemanticFindingKind::Contradiction)
        .collect();
    assert_eq!(contradictions.len(), 1, "{:?}", first.findings);
    assert_eq!(contradictions[0].page_slug, "semantic-fixture");
    assert!(
        contradictions[0].message.contains("cannot both hold"),
        "{:?}",
        contradictions[0].message
    );
    assert!(first.findings.iter().any(|f| f.kind
        == llm_wiki_compiler::SemanticFindingKind::KnowledgeGap
        && f.message.contains("retry budgets")));
    assert!(first.skipped_pages.is_empty(), "{:?}", first.skipped_pages);

    // Second run: everything comes from the §28 cache — zero new requests.
    let requests_before = provider.request_count();
    let second = llm_wiki_compiler::run_semantic_lint(&workspace, &config, provider.clone())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        provider.request_count() - requests_before,
        0,
        "a repeated semantic lint must be fully cache-served"
    );
    assert_eq!(second.findings.len(), first.findings.len());
}

/// A page-review response citing an UNKNOWN claim is dropped (advisory
/// fail-open), and a malformed first response repairs once.
#[tokio::test]
async fn semantic_lint_repairs_once_and_drops_hallucinated_claim_references() {
    let workspace = fixture_workspace("repair");
    let config = Config::load(&workspace).unwrap();

    run_build(
        &workspace,
        &config,
        semantic_llm(Arc::new(Mutex::new(VecDeque::new()))),
    )
    .await
    .unwrap();

    let queue = Arc::new(Mutex::new(VecDeque::new()));
    // 1) shape failure: not JSON → repair 2) valid JSON but hallucinated
    // claim → dropped; the gap stage still runs.
    queue
        .lock()
        .unwrap()
        .push_back("this is not json".to_owned());
    queue.lock().unwrap().push_back(
        serde_json::json!({"findings": [{
            "kind": "superseded",
            "claim_ids": ["kn_HALLUCINATED00000000000000"],
            "reason": "should be dropped",
        }]})
        .to_string(),
    );
    let provider = semantic_llm(Arc::clone(&queue));
    let report = llm_wiki_compiler::run_semantic_lint(&workspace, &config, provider)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(report.pages_reviewed, 1);
    assert!(
        !report
            .findings
            .iter()
            .any(|f| f.kind == llm_wiki_compiler::SemanticFindingKind::Superseded),
        "the hallucinated claim reference must be dropped: {:?}",
        report.findings
    );
    assert!(
        report.skipped_pages.is_empty(),
        "{:?}",
        report.skipped_pages
    );
}
