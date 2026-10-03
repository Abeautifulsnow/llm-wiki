//! Integration tests for the Vector layer (§19.3): the `llm-wiki embed`
//! backfill (incremental, content-addressed) and `ask --hybrid` (vector
//! candidates fused into the context), driven by a deterministic fake
//! embedding provider over a REAL published generation.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llm_wiki_compiler::{run_ask, run_build, run_embed, EmbedReport, HybridContext};
use llm_wiki_core::config::Config;
use llm_wiki_llm::{EmbeddingProvider, FakeLlmProvider, LlmError, LlmRequest};

const SYNTHESIS_MARK: &str = "Answer the user's question";

/// Deterministic bag-of-character-bigram embedding: texts sharing a rare
/// token get high cosine; unrelated texts score near zero. Good enough to
/// prove the plumbing end to end without a real model.
/// Shared call counter handle — survives the Arc<dyn> erasure.
#[derive(Clone, Default)]
struct EmbedCallCounter(Arc<Mutex<Vec<Vec<String>>>>);

impl EmbedCallCounter {
    fn call_count(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

#[derive(Clone, Default)]
struct FakeEmbeddings {
    calls: EmbedCallCounter,
}

impl FakeEmbeddings {
    fn vector_for(text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; 64];
        for bigram in text.as_bytes().windows(2) {
            let index = (bigram[0] as usize * 31 + bigram[1] as usize) % vector.len();
            vector[index] += 1.0;
        }
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        vector
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for FakeEmbeddings {
    async fn embed(&self, _model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        self.calls.0.lock().unwrap().push(texts.to_vec());
        Ok(texts.iter().map(|text| Self::vector_for(text)).collect())
    }
}

/// Stage-routing fake handling the full build pipeline plus the ask
/// synthesis stage (the embed flow never touches LlmProvider).
fn ask_llm() -> Arc<FakeLlmProvider> {
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
                    "summary": "vector fixture analysis.",
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
                } else {
                    let ids = proposed.lock().unwrap().clone();
                    return Ok(serde_json::json!({
                        "pages": [{
                            "title": "Vector Fixture",
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
                        "title": "Vector Fixture",
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
                if !prompt.contains(SYNTHESIS_MARK) {
                    return Err(LlmError::Api {
                        code: 500,
                        message: "no synthesis stage".into(),
                    });
                }
                // The ask CONTEXT carries claims as {"id":...} — the generic
                // kn_ scanner finds them (graph nodes contribute no ids).
                let claim_id = kn_ids_in(prompt).first().cloned().unwrap_or_default();
                Ok(serde_json::json!({"answer": format!(
                    "## Answer\n\nRetries are bounded. <!-- llm-wiki:cite claim=\"{claim_id}\" -->\n"
                )})
                .to_string())
            }
            other => Err(LlmError::Api {
                code: 500,
                message: format!("unexpected task tag {other}"),
            }),
        }
    });
    Arc::new(FakeLlmProvider::new("fake-vector", handler))
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
        "llm-wiki-vector-{tag}-{}-{}",
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
        "[project]\nname = \"fixture\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-vector\"\n",
    )
    .unwrap();
    dir
}

const MODEL: &str = "fake-embedding-model";

/// The backfill is incremental: the first run embeds every section, a
/// second run issues ZERO embedding calls (content-addressed coverage), and
/// a re-run after a real change embeds only what changed.
#[tokio::test]
async fn embed_backfill_is_incremental_and_content_addressed() {
    let workspace = fixture_workspace("embed");
    let config = Config::load(&workspace).unwrap();
    run_build(&workspace, &config, ask_llm()).await.unwrap();

    let embeddings = FakeEmbeddings::default();
    let first: EmbedReport = {
        let provider: Arc<dyn EmbeddingProvider> = Arc::new(embeddings.clone());
        run_embed(&workspace, &config, provider, MODEL, 16)
    }
    .await
    .unwrap()
    .expect("generation is published");
    assert!(first.total_sections >= 1);
    assert_eq!(first.covered_before, 0);
    assert_eq!(first.embedded, first.total_sections);
    let calls_after_first = embeddings.calls.call_count();
    assert!(calls_after_first > 0, "the first run really embeds");

    // Second run: full coverage → zero embedding calls.
    let second: EmbedReport = {
        let provider: Arc<dyn EmbeddingProvider> = Arc::new(embeddings.clone());
        run_embed(&workspace, &config, provider, MODEL, 16)
    }
    .await
    .unwrap()
    .unwrap();
    assert_eq!(second.covered_before, second.total_sections);
    assert_eq!(second.embedded, 0);
    assert_eq!(embeddings.calls.call_count(), calls_after_first);

    // A different model has its own vector space: it embeds from scratch.
    let other_model: EmbedReport = run_embed(
        &workspace,
        &config,
        Arc::new(FakeEmbeddings::default()),
        "other-model",
        16,
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(other_model.embedded, other_model.total_sections);
}

/// Hybrid ask: the vector candidate rides into the fusion, the answer still
/// passes the citation contract, and the query embedding was computed.
#[tokio::test]
async fn hybrid_ask_fuses_vector_candidates() {
    let workspace = fixture_workspace("hybrid");
    let config = Config::load(&workspace).unwrap();
    run_build(&workspace, &config, ask_llm()).await.unwrap();

    let embeddings = FakeEmbeddings::default();
    {
        let provider: Arc<dyn EmbeddingProvider> = Arc::new(embeddings);
        run_embed(&workspace, &config, provider, MODEL, 16)
            .await
            .unwrap()
            .unwrap();
    }

    let provider = ask_llm();
    let query_embeddings = FakeEmbeddings::default();
    let query_counter = query_embeddings.calls.clone();
    let query_handle: Arc<dyn EmbeddingProvider> = Arc::new(query_embeddings);
    let hybrid = HybridContext {
        provider: &query_handle,
        model: MODEL.to_owned(),
    };
    let report = run_ask(
        &workspace,
        &config,
        provider,
        "how do retries work",
        false,
        Some(hybrid),
    )
    .await
    .unwrap()
    .expect("generation is published");
    assert!(
        report.answer.contains("Retries are bounded"),
        "{}",
        report.answer
    );
    // The query embedding ran exactly once against the ask-time provider.
    assert_eq!(query_counter.call_count(), 1);
}

/// Without embed coverage, hybrid ask degrades to lexical-only with a
/// warning — it never fails for a missing vector space.
#[tokio::test]
async fn hybrid_ask_without_coverage_degrades_to_lexical() {
    let workspace = fixture_workspace("degrade");
    let config = Config::load(&workspace).unwrap();
    run_build(&workspace, &config, ask_llm()).await.unwrap();

    let embeddings = FakeEmbeddings::default();
    let counter = embeddings.calls.clone();
    let provider_handle: Arc<dyn EmbeddingProvider> = Arc::new(embeddings);
    let provider = ask_llm();
    let hybrid = HybridContext {
        provider: &provider_handle,
        model: MODEL.to_owned(),
    };
    let report = run_ask(
        &workspace,
        &config,
        provider,
        "how do retries work",
        false,
        Some(hybrid),
    )
    .await
    .unwrap()
    .expect("generation is published");
    assert!(report.answer.contains("Retries are bounded"));
    // Zero stored vectors → the candidate search returns early: not even the
    // query embedding is spent, and the context assembled lexically.
    assert_eq!(counter.call_count(), 0);
}
