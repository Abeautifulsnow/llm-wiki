//! Eval pipeline plumbing: the fixture FakeLlmProvider (stage-routing, like
//! `publish_build.rs`) wired for the `evals/corpus/` dataset so analysis
//! claims quote annotated spans, the planner groups claims into the
//! `expected/pages.yaml` pages, and compilation cites every claim — making
//! evidence validation and the §37.3 gates pass by construction.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmProvider, LlmRequest};

use crate::fixtures::{Dataset, ExpectedPages};

/// Repo-root `evals/` directory, resolved from this crate's location
/// (`<root>/crates/llm-wiki-evals` → `<root>/evals`).
pub fn evals_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives at <root>/crates/<name>")
        .join("evals")
}

/// A temporary eval workspace with `source.root` pointing at the fixture
/// corpus. Callers own the directory (and should clean it up).
pub fn eval_workspace(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-evals-{tag}-{}-{}",
        std::process::id(),
        chrono_utc_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let state = dir.join(".llm-wiki");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("config.toml"),
        format!(
            "[project]\nname = \"evals\"\nwiki_dir = \"{}\"\n\n[source]\nroot = \"{}\"\n\n[llm]\nmodel = \"fake-evals\"\n",
            toml_path(&dir.join("wiki")),
            toml_path(&evals_dir().join("corpus")),
        ),
    )
    .unwrap();
    dir
}

fn toml_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn chrono_utc_nanos() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// span → page titles from the dataset and expected pages: the grouping
/// signal the fake planner needs (claims quote spans verbatim, so the span
/// resolves each claim to its doc, and each doc to EVERY expected page that
/// lists it — shared docs feed multiple pages, which is what cross-document
/// synthesis requires). Docs listed in no expected page still need claim
/// coverage (the §37.3 Source Coverage denominator spans the WHOLE dataset),
/// so their facts group under a catch-all page.
pub fn span_grouping(dataset: &Dataset, expected: &ExpectedPages) -> BTreeMap<String, Vec<String>> {
    const CATCH_ALL: &str = "Eval Platform";
    let mut doc_to_pages: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for page in &expected.pages {
        for source in &page.sources {
            let titles = doc_to_pages.entry(source.clone()).or_default();
            if !titles.contains(&page.title) {
                titles.push(page.title.clone());
            }
        }
    }
    let mut span_to_pages: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for fact in &dataset.facts {
        let titles = doc_to_pages
            .get(&fact.doc_path)
            .cloned()
            .unwrap_or_else(|| vec![CATCH_ALL.to_owned()]);
        span_to_pages.insert(fact.span.clone(), titles);
    }
    span_to_pages
}

/// Extracts `\u{1f}`-free JSON string values for `key` from compact JSON —
/// the same scanner used by the compiler integration tests.
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

const SUMMARY_MARK: &str = "Summarize the following cluster";
const LOCAL_MARK: &str = "THIS cluster's knowledge";
const RECONCILE_MARK: &str = "final global wiki plan";

/// Builds the eval FakeLlmProvider. Grouping requires the fixture data; the
/// provider records claim id → page title at the SUMMARY stage (statements
/// are spans) and reuses that map at LOCAL/RECONCILE.
pub fn eval_llm(dataset: &Dataset, expected: &ExpectedPages) -> Arc<dyn LlmProvider> {
    let span_to_page = span_grouping(dataset, expected);
    // claim id → page titles, filled during summaries and consumed later.
    let id_to_page: Arc<Mutex<BTreeMap<String, Vec<String>>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
    // Node ids proposed by LOCAL stages; the RECONCILE payload carries no
    // node ids, so reconciliation groups this accumulated set.
    let proposed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    // statement → id observed at summary time is unnecessary: the summary
    // payload carries node id + statement, so resolve directly.
    let handler = Arc::new(move |request: &LlmRequest| -> Result<String, LlmError> {
        let prompt = &request.prompt;
        match request.task_tag.as_str() {
            "document-analysis" => {
                let section_ids = json_string_values(prompt, "section_id");
                let contents = json_string_values(prompt, "content");
                let mut claims: Vec<serde_json::Value> = Vec::new();
                for (section_id, content) in section_ids.iter().zip(contents.iter()) {
                    // Annotated spans in this section become claims quoting
                    // them verbatim; sections without any fall back to their
                    // first non-empty line (still source-backed).
                    let mut quoted = false;
                    for line in content.lines().map(str::trim).filter(|l| !l.is_empty()) {
                        if span_to_page.contains_key(line) {
                            claims.push(serde_json::json!({
                                "text": line,
                                "section_id": section_id,
                                "evidence_text": line,
                                "evidence_start": content.find(line).unwrap_or(0),
                                "confidence": 0.95,
                            }));
                            quoted = true;
                        }
                    }
                    if !quoted {
                        if let Some(first) =
                            content.lines().map(str::trim).find(|line| !line.is_empty())
                        {
                            claims.push(serde_json::json!({
                                "text": first,
                                "section_id": section_id,
                                "evidence_text": first,
                                "evidence_start": content.find(first).unwrap_or(0),
                                "confidence": 0.9,
                            }));
                        }
                    }
                }
                Ok(serde_json::json!({
                    "summary": "Deterministic eval analysis.",
                    "topics": ["eval"],
                    "entities": [],
                    "concepts": [],
                    "claims": claims,
                    "relations": [],
                })
                .to_string())
            }
            "wiki-planning" => {
                if prompt.contains(SUMMARY_MARK) {
                    // Summary payload: nodes with id/kind/name/statement —
                    // record the page grouping for every claim span.
                    let ids = json_string_values(prompt, "id");
                    let statements = json_string_values(prompt, "statement");
                    let mut map = id_to_page.lock().unwrap();
                    for (id, statement) in ids.into_iter().zip(statements) {
                        if let Some(titles) = span_to_page.get(&statement) {
                            for title in titles {
                                map.entry(id.clone()).or_default().push(title.clone());
                            }
                        }
                    }
                    return Ok(
                        serde_json::json!({"summary": "cluster of eval corpus knowledge"})
                            .to_string(),
                    );
                }
                if prompt.contains(LOCAL_MARK) {
                    let ids = kn_ids_in(prompt);
                    proposed.lock().unwrap().extend(ids.iter().cloned());
                    let map = id_to_page.lock().unwrap();
                    let mut by_page: BTreeMap<String, Vec<String>> = BTreeMap::new();
                    for id in &ids {
                        let titles = map
                            .get(id)
                            .cloned()
                            .unwrap_or_else(|| vec!["Eval Platform".to_owned()]);
                        for title in titles {
                            let refs = by_page.entry(title).or_default();
                            if !refs.contains(id) {
                                refs.push(id.clone());
                            }
                        }
                    }
                    let pages: Vec<serde_json::Value> = by_page
                        .into_iter()
                        .map(|(title, refs)| {
                            serde_json::json!({
                                "title": title,
                                "category": "concepts",
                                "purpose": "cover the eval corpus knowledge",
                                "knowledge_refs": refs,
                            })
                        })
                        .collect();
                    return Ok(serde_json::json!({ "pages": pages }).to_string());
                }
                if prompt.contains(RECONCILE_MARK) {
                    let ids = proposed.lock().unwrap().clone();
                    let map = id_to_page.lock().unwrap();
                    let mut by_page: BTreeMap<String, Vec<String>> = BTreeMap::new();
                    for id in &ids {
                        let titles = map
                            .get(id)
                            .cloned()
                            .unwrap_or_else(|| vec!["Eval Platform".to_owned()]);
                        for title in titles {
                            let refs = by_page.entry(title).or_default();
                            if !refs.contains(id) {
                                refs.push(id.clone());
                            }
                        }
                    }
                    let pages: Vec<serde_json::Value> = by_page
                        .into_iter()
                        .map(|(title, refs)| {
                            serde_json::json!({
                                "title": title,
                                "category": "concepts",
                                "purpose": "cover the eval corpus knowledge",
                                "knowledge_refs": refs,
                            })
                        })
                        .collect();
                    return Ok(serde_json::json!({ "pages": pages }).to_string());
                }
                Err(LlmError::Api {
                    code: 500,
                    message: "unrecognized planning stage".into(),
                })
            }
            "wiki-compilation" => {
                let claims = claim_ids_in(prompt);
                // Full-text baseline (PRD §20/§37.4): the body carries every
                // claim's VERBATIM statement — the annotated corpus spans —
                // so the V0.2 FTS gates retrieve real content (CJK spans
                // included) rather than page titles alone.
                let statements = json_string_values(prompt, "statement");
                let mut body = String::from("## Overview\n\nThe eval page cites its claims.\n");
                for statement in &statements {
                    body.push_str(&format!("\n{statement}\n"));
                }
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
    Arc::new(FakeLlmProvider::new("fake-evals", handler))
}
