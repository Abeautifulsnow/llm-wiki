//! Integration tests for `llm-wiki ask` (audit FIX-020): retrieval-grounded
//! synthesis with verified citations and the insight write-back layer,
//! driven by FakeLlmProvider over a REAL published generation.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llm_wiki_compiler::run_build;
use llm_wiki_core::config::Config;
use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmRequest};
use llm_wiki_storage::list_insights;

/// Stage-routing fake handling the full build pipeline plus the `wiki-ask`
/// synthesis stage (scripted queue; repairs pop from the same queue).
fn ask_llm(synthesis_responses: Arc<Mutex<VecDeque<String>>>) -> Arc<FakeLlmProvider> {
    let proposed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
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
                    "summary": "ask fixture analysis.",
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
                            "title": "Ask Fixture",
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
                        "title": "Ask Fixture",
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
                        "## Overview\n\nThe scheduler retries failed tasks three times.\n\n<!-- llm-wiki:cite claim=\"{}\" -->",
                        claims[0]
                    )
                };
                Ok(serde_json::json!({ "markdown": body }).to_string())
            }
            "wiki-ask" => {
                let mut queue = synthesis_responses.lock().unwrap();
                let response = queue.pop_front().ok_or_else(|| LlmError::Api {
                    code: 500,
                    message: "synthesis script exhausted".into(),
                })?;
                drop(queue);
                Ok(response)
            }
            other => Err(LlmError::Api {
                code: 500,
                message: format!("unexpected task tag {other}"),
            }),
        }
    });
    Arc::new(FakeLlmProvider::new("fake-ask", handler))
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

/// Claim node ids from a compilation prompt (`"kind":"claim"` nodes' ids).
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
        "llm-wiki-ask-{tag}-{}-{}",
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
        "[project]\nname = \"fixture\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-ask\"\n",
    )
    .unwrap();
    dir
}

/// A cited, grounded answer: the claim id is resolved from the REAL
/// knowledge base at test time and passed through the script.
fn cited_answer(claim_id: &str) -> String {
    serde_json::json!({"answer": format!(
        "## Retries\n\nThe scheduler retries failed tasks up to three times. <!-- llm-wiki:cite claim=\"{claim_id}\" -->\n"
    )})
    .to_string()
}

/// End-to-end: build → ask (cited answer, expanded anchors, dry run) →
/// `--write-back` persists an insight with provenance → repeat ask is fully
/// cache-served.
#[tokio::test]
async fn ask_grounds_cites_persists_and_caches() {
    let workspace = fixture_workspace("e2e");
    let config = Config::load(&workspace).unwrap();

    run_build(
        &workspace,
        &config,
        ask_llm(Arc::new(Mutex::new(VecDeque::new()))),
    )
    .await
    .unwrap();

    let conn = llm_wiki_storage::open(&workspace.join(".llm-wiki").join("state.db")).unwrap();
    let base = llm_wiki_storage::load_knowledge_base(&conn).unwrap();
    let claim_id = base
        .nodes
        .keys()
        .find(|id| base.nodes[*id].kind == "claim")
        .expect("fixture claim exists")
        .as_str()
        .to_owned();
    drop(conn);

    // Dry ask: verified answer, no persistence.
    let provider = ask_llm(Arc::new(Mutex::new(VecDeque::from([cited_answer(
        &claim_id,
    )]))));
    let dry = llm_wiki_compiler::run_ask(
        &workspace,
        &config,
        provider.clone(),
        "how do retries work",
        false,
        None,
    )
    .await
    .unwrap()
    .expect("generation is published");
    assert_eq!(dry.llm_request_count, 1);
    assert!(dry.insight_id.is_none());
    // Citation expanded from the stored anchor: source path rides along.
    assert!(
        dry.answer.contains("source=\"runtime.md\""),
        "anchors expand into the answer: {}",
        dry.answer
    );
    assert!(dry.answer.contains(&format!("claim=\"{claim_id}\"")));
    assert_eq!(dry.sources, vec!["runtime.md".to_owned()]);
    assert!(list_insights(
        &llm_wiki_storage::open(&workspace.join(".llm-wiki").join("state.db")).unwrap()
    )
    .unwrap()
    .is_empty());

    // Write-back ask: the insight lands with provenance.
    let provider = ask_llm(Arc::new(Mutex::new(VecDeque::from([cited_answer(
        &claim_id,
    )]))));
    let writeback = llm_wiki_compiler::run_ask(
        &workspace,
        &config,
        provider,
        "how do retries work",
        true,
        None,
    )
    .await
    .unwrap()
    .unwrap();
    let insight_id = writeback.insight_id.expect("write-back ran");
    let insights = list_insights(
        &llm_wiki_storage::open(&workspace.join(".llm-wiki").join("state.db")).unwrap(),
    )
    .unwrap();
    assert_eq!(insights.len(), 1);
    assert_eq!(insights[0].insight_id.as_str(), insight_id);
    assert_eq!(insights[0].query, "how do retries work");
    assert_eq!(
        insights[0].citations[0].claim_node_id, claim_id,
        "provenance records the cited claim"
    );
    assert_eq!(insights[0].citations[0].source, "runtime.md");

    // Repeat dry ask over the same query: fully cache-served.
    let provider = ask_llm(Arc::new(Mutex::new(VecDeque::new())));
    let cached = llm_wiki_compiler::run_ask(
        &workspace,
        &config,
        provider.clone(),
        "how do retries work",
        false,
        None,
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        provider.request_count(),
        0,
        "the synthesis must come from the §28 cache"
    );
    assert_eq!(cached.answer, dry.answer);
}

/// Validation is fail-closed: an answer citing an UNKNOWN claim repairs
/// once, then the command fails with the machine-readable issue; nothing is
/// persisted.
#[tokio::test]
async fn ask_fails_closed_on_hallucinated_citations() {
    let workspace = fixture_workspace("failclosed");
    let config = Config::load(&workspace).unwrap();

    run_build(
        &workspace,
        &config,
        ask_llm(Arc::new(Mutex::new(VecDeque::new()))),
    )
    .await
    .unwrap();

    let hallucinated = serde_json::json!({"answer": format!(
        "## No\n\nUnverifiable claim. <!-- llm-wiki:cite claim=\"{}\" -->\n",
        "kn_NOPE0000000000000000000000"
    )})
    .to_string();
    let provider = ask_llm(Arc::new(Mutex::new(VecDeque::from([
        hallucinated.clone(),
        hallucinated,
    ]))));
    let error = llm_wiki_compiler::run_ask(&workspace, &config, provider, "retries", true, None)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("UNKNOWN_CLAIM_REF"),
        "fail-closed with the machine-readable issue: {error}"
    );
    assert!(
        list_insights(
            &llm_wiki_storage::open(&workspace.join(".llm-wiki").join("state.db")).unwrap()
        )
        .unwrap()
        .is_empty(),
        "a failed verification never writes back"
    );
}
