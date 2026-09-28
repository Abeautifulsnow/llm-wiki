//! Golden test (PRD §54): run the full pipeline over the small
//! `test-data/docs` fixture and assert the STRUCTURE of the built wiki —
//! page set, source mapping, citation mapping and WikiLink structure. Body
//! prose is never compared byte-for-byte; the fake provider's wording is
//! irrelevant, the pipeline's machine state is the contract.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use llm_wiki_compiler::{run_build, BuildReport};
use llm_wiki_core::config::Config;
use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmProvider, LlmRequest};
use llm_wiki_storage::{get_active_build_id, list_sources, load_generation_view, open};

// ---------------------------------------------------------------------------
// A deterministic stage-routing FakeLlmProvider for the test-data corpus.
// ---------------------------------------------------------------------------

fn json_string_values(text: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\":\"");
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = text[search_from..].find(&needle) {
        let start = search_from + offset + needle.len();
        let rest = &text[start..];
        let end = rest.find('"').unwrap_or(rest.len());
        out.push(unescape_json(&rest[..end]));
        search_from = start + end;
    }
    out
}

fn unescape_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('"') => out.push('"'),
            Some('/') => out.push('/'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                for _ in 0..4 {
                    chars.next();
                }
                out.push('?');
            }
            _ => {}
        }
    }
    out
}

fn kn_ids_in(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = text[search_from..].find("\"id\":\"kn_") {
        let start = search_from + offset + "\"id\":\"".len();
        let rest = &text[start..];
        let end = rest.find('"').unwrap_or(rest.len());
        let id = &rest[..end];
        if !out.iter().any(|known| known == id) {
            out.push(id.to_owned());
        }
        search_from = start + end;
    }
    out
}

fn claim_ids_in(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = text[search_from..].find("\"kind\":\"claim\"") {
        let before = &text[..search_from + offset];
        if let Some(id_offset) = before.rfind("\"id\":\"kn_") {
            let start = id_offset + "\"id\":\"".len();
            let rest = &text[start..];
            let end = rest.find('"').unwrap_or(rest.len());
            let id = rest[..end].to_owned();
            if !out.iter().any(|known| known == &id) {
                out.push(id);
            }
        }
        search_from += offset + "\"kind\":\"claim\"".len();
    }
    out
}

fn golden_llm() -> Arc<dyn LlmProvider> {
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
                    "summary": "Golden fixture analysis.",
                    "topics": ["golden"],
                    "entities": [],
                    "concepts": [],
                    "claims": claims,
                    "relations": [],
                })
                .to_string())
            }
            "wiki-planning" => {
                if prompt.contains("Summarize the following cluster") {
                    return Ok(
                        serde_json::json!({"summary": "cluster over the golden corpus"})
                            .to_string(),
                    );
                }
                if prompt.contains("THIS cluster's knowledge") {
                    let ids = kn_ids_in(prompt);
                    proposed.lock().unwrap().extend(ids.iter().cloned());
                    return Ok(serde_json::json!({
                        "pages": [{
                            "title": "Golden Platform",
                            "category": "concepts",
                            "purpose": "cover the golden corpus",
                            "knowledge_refs": ids,
                        }]
                    })
                    .to_string());
                }
                if prompt.contains("final global wiki plan") {
                    let ids = proposed.lock().unwrap().clone();
                    return Ok(serde_json::json!({
                        "pages": [{
                            "title": "Golden Platform",
                            "category": "concepts",
                            "purpose": "cover the golden corpus",
                            "knowledge_refs": ids,
                        }]
                    })
                    .to_string());
                }
                Err(LlmError::Api {
                    code: 500,
                    message: "unrecognized planning stage".into(),
                })
            }
            "wiki-compilation" => {
                let claims = claim_ids_in(prompt);
                let mut body = String::from("## Overview\n\nThe golden page cites its claims.\n");
                for id in &claims {
                    body.push_str(&format!("\n<!-- llm-wiki:cite claim=\"{id}\" -->\n"));
                }
                Ok(serde_json::json!({ "markdown": body }).to_string())
            }
            other => Err(LlmError::Api {
                code: 500,
                message: format!("unexpected task tag {other}"),
            }),
        }
    });
    Arc::new(FakeLlmProvider::new("fake-golden", handler))
}

// ---------------------------------------------------------------------------
// Golden assertions
// ---------------------------------------------------------------------------

fn workspace_for(tag: &str, docs_src: &Path) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-golden-{tag}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    let docs = dir.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    // Copy the checked-in fixture corpus (5 Markdown/MDX docs).
    copy_tree(docs_src, &docs);
    let state = dir.join(".llm-wiki");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("config.toml"),
        "[project]\nname = \"golden\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-golden\"\n",
    )
    .unwrap();
    dir
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap().flatten() {
        let target = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives at <root>/crates/<name>")
        .to_path_buf()
}

#[tokio::test]
async fn golden_test_data_docs_structure() {
    let workspace = workspace_for("golden", &repo_root().join("test-data").join("docs"));
    let config = Config::load(&workspace).unwrap();
    let db_path = workspace.join(".llm-wiki").join("state.db");

    let report: BuildReport = run_build(&workspace, &config, golden_llm())
        .await
        .expect("golden build succeeds");

    // The visible wiki is the published generation.
    let conn = open(&db_path).unwrap();
    let build_id = get_active_build_id(&conn).unwrap().expect("published");
    assert_eq!(build_id, report.build_id);
    let view = load_generation_view(&conn, &build_id).unwrap();

    // Page set: the fake plans exactly one page over the corpus knowledge.
    assert_eq!(
        view.len(),
        1,
        "page set: {:?}",
        view.iter().map(|p| &p.title).collect::<Vec<_>>()
    );
    let page = &view[0];
    assert_eq!(page.title, "Golden Platform");
    assert_eq!(page.slug, "golden-platform");
    assert_eq!(page.category, "concepts");

    // Source mapping: the page's citations resolve into the corpus sources.
    let sources: std::collections::BTreeMap<String, llm_wiki_storage::SourceRecord> =
        list_sources(&conn)
            .unwrap()
            .into_iter()
            .map(|s| (s.source_id.as_str().to_owned(), s))
            .collect();
    assert_eq!(sources.len(), 5, "all corpus docs are registered");
    let cited: std::collections::BTreeSet<String> = page
        .citations
        .iter()
        .map(|c| c.source_id.as_str().to_owned())
        .collect();
    assert!(
        !cited.is_empty() && cited.len() <= sources.len(),
        "citations resolve to registered sources"
    );

    // Citation mapping: every citation has a range inside its source, a
    // source hash that matches the registry, a heading path and a digest.
    for citation in &page.citations {
        let source = &sources[citation.source_id.as_str()];
        assert_eq!(citation.source_hash, source.content_hash);
        assert!(citation.range.start <= citation.range.end);
        assert!(citation.range.end as i64 <= source.size.max(0));
        assert!(!citation.heading_path.is_empty());
        assert_eq!(citation.evidence_digest.len(), 64);
    }

    // Knowledge refs: the page covers claim nodes with grounding anchors.
    let knowledge = llm_wiki_storage::load_knowledge_base(&conn).unwrap();
    let claim_count = knowledge
        .nodes
        .values()
        .filter(|n| n.kind == "claim")
        .count();
    assert!(claim_count >= 5, "corpus claims registered: {claim_count}");
    assert!(!page.knowledge_refs.is_empty());

    // WikiLink structure: one page → no outbound links; every link row the
    // pipeline DID create would resolve inside the generation (none here).
    assert!(page.links.is_empty());
    assert_eq!(page.inbound_links, 0);

    // The published file on disk carries app-owned frontmatter and the
    // expanded citation comments (never bare LLM output).
    let generation_dir = workspace
        .join("wiki")
        .join("generations")
        .join(report.build_id.as_str());
    let files: Vec<_> = std::fs::read_dir(&generation_dir)
        .unwrap()
        .flatten()
        .collect();
    assert_eq!(files.len(), 1);
    let body = std::fs::read_to_string(files[0].path()).unwrap();
    assert!(body.contains("generated: true"));
    assert!(body.contains("schema_version: 1"));
    assert!(body.contains("llm-wiki:cite"), "citations are expanded");
}

/// Real-LLM E2E (PRD §54): `#[ignore]`-gated; only runs when
/// `LLM_WIKI_E2E_API_KEY` and `LLM_WIKI_E2E_BASE_URL` are set. Never in CI.
#[tokio::test]
#[ignore = "requires a real LLM endpoint (LLM_WIKI_E2E_API_KEY + LLM_WIKI_E2E_BASE_URL)"]
async fn real_llm_e2e_over_test_data_docs() {
    let api_key = std::env::var("LLM_WIKI_E2E_API_KEY").expect("LLM_WIKI_E2E_API_KEY set");
    let base_url = std::env::var("LLM_WIKI_E2E_BASE_URL").expect("LLM_WIKI_E2E_BASE_URL set");

    let workspace = workspace_for("reale2e", &repo_root().join("test-data").join("docs"));
    let mut config = Config::load(&workspace).unwrap();
    config.llm.base_url = base_url;
    config.llm.model = std::env::var("LLM_WIKI_E2E_MODEL").unwrap_or_default();
    config.llm.api_key_env = "LLM_WIKI_E2E_API_KEY".to_owned();
    std::env::set_var("LLM_WIKI_E2E_API_KEY", api_key);

    let provider: Arc<dyn LlmProvider> = Arc::new(
        llm_wiki_llm::OpenAiCompatibleProvider::new(
            &config.llm.base_url,
            &config.llm.model,
            &config.llm.api_key_env,
            config.llm.timeout_seconds,
            2,
        )
        .expect("provider constructs"),
    );

    let report = run_build(&workspace, &config, provider)
        .await
        .expect("real-LLM build succeeds");
    assert!(report.pages >= 1);
    assert!(report.citations >= 1);
    assert!(report.recovery.is_none());

    let conn = open(&workspace.join(".llm-wiki").join("state.db")).unwrap();
    let build_id = get_active_build_id(&conn).unwrap().expect("published");
    let view = load_generation_view(&conn, &build_id).unwrap();
    assert!(!view.is_empty());
    for page in &view {
        assert!(!page.content.is_empty());
        assert!(page.content.contains("generated: true"));
    }
}
