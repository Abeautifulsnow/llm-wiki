//! Integration tests for the atomic publisher (PRD §35) and the end-to-end
//! `run_build` pipeline (PRD §29/§31), driven by FakeLlmProvider — no real
//! model in CI (PRD §54).
//!
//! The crash-recovery matrix simulates a crash after EVERY publish step by
//! writing the corresponding journal/filesystem/DB states directly and then
//! running `recover_if_needed`; every outcome must be a consistent old-or-new
//! state, never a mix.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rusqlite::params;

use llm_wiki_compiler::{
    journal_exists, publish, read_current_pointer, recover_if_needed, write_current_pointer,
    write_generation, write_journal, PublishPaths,
};
use llm_wiki_core::error::WikiError;
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::BuildId;
use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmProvider, LlmRequest};
use llm_wiki_storage::{
    activate_build, get_active_build_id, load_generation_view, open, open_in_memory,
    persist_generation, start_build, WikiPageRecord,
};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-pubbuild-{tag}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn page(slug: &str, content: &str) -> WikiPageRecord {
    WikiPageRecord {
        page_id: llm_wiki_core::ids::WikiPageId::generate(),
        slug: slug.to_owned(),
        title: slug.to_owned(),
        category: "concepts".into(),
        language: "en".into(),
        body_hash: sha256_hex(content.as_bytes()),
        content: content.to_owned(),
        knowledge_refs: Vec::new(),
        citations: Vec::new(),
        links: Vec::new(),
    }
}

fn new_build(conn: &mut rusqlite::Connection) -> BuildId {
    start_build(conn, &llm_wiki_storage::BuildDraft::default()).unwrap()
}

fn build_status(conn: &rusqlite::Connection, build: &BuildId) -> String {
    conn.query_row(
        "SELECT status FROM builds WHERE build_id = ?1",
        params![build.as_str()],
        |r| r.get(0),
    )
    .unwrap()
}

fn pointer_of(wiki_dir: &Path) -> Option<String> {
    read_current_pointer(&PublishPaths::new(wiki_dir))
        .unwrap()
        .map(|pointer| pointer.build_id)
}

/// Publishes the first generation so the matrix cases have a "previous good
/// generation" to fall back to.
fn seed_published_generation(
    conn: &mut rusqlite::Connection,
    wiki_dir: &Path,
    pages: &[WikiPageRecord],
) -> BuildId {
    let build = new_build(conn);
    publish(conn, wiki_dir, &build, pages, 3).unwrap();
    build
}

// ---------------------------------------------------------------------------
// Crash-recovery matrix (PRD §35)
// ---------------------------------------------------------------------------

#[test]
fn crash_after_generation_write_without_journal_is_a_noop() {
    let wiki_dir = temp_dir("crash-step1");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];
    let first = seed_published_generation(&mut conn, &wiki_dir, &pages);

    // A new build wrote its generation files, then crashed before recording
    // the intent: pointer and DB still name the old build — consistent.
    let second = new_build(&mut conn);
    write_generation(
        &PublishPaths::new(&wiki_dir),
        &second,
        &pages,
        &BTreeMap::new(),
    )
    .unwrap();

    let recovered = recover_if_needed(&mut conn, &wiki_dir).unwrap();
    assert!(recovered.is_none(), "consistent state needs no recovery");
    assert_eq!(pointer_of(&wiki_dir).as_deref(), Some(first.as_str()));
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(first));
}

#[test]
fn crash_after_journal_written_rolls_back_to_old() {
    let wiki_dir = temp_dir("crash-step4");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];
    let first = seed_published_generation(&mut conn, &wiki_dir, &pages);

    // Steps 1–4 done (files + intent), crash before the pointer move.
    let second = new_build(&mut conn);
    let paths = PublishPaths::new(&wiki_dir);
    write_generation(&paths, &second, &pages, &BTreeMap::new()).unwrap();
    write_journal(&paths, Some(&first), &second).unwrap();

    let recovered = recover_if_needed(&mut conn, &wiki_dir).unwrap().unwrap();
    assert_eq!(
        recovered.action,
        llm_wiki_compiler::RecoveryAction::RolledBack
    );

    assert!(!journal_exists(&paths), "journal must be cleared");
    assert_eq!(pointer_of(&wiki_dir).as_deref(), Some(first.as_str()));
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(first.clone()));
    assert_eq!(build_status(&conn, &second), "INTERRUPTED");
    assert_eq!(build_status(&conn, &first), "COMPLETED");
}

#[test]
fn crash_after_pointer_move_completes_the_verified_new_version() {
    let wiki_dir = temp_dir("crash-step5");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];
    let first = seed_published_generation(&mut conn, &wiki_dir, &pages);

    // Steps 1–5 done, crash before the DB transaction: the pointer already
    // names the new build. The generation re-validates against wiki_pages
    // rows, so recovery completes the new version.
    let second = new_build(&mut conn);
    let paths = PublishPaths::new(&wiki_dir);
    write_generation(&paths, &second, &pages, &BTreeMap::new()).unwrap();
    persist_generation(&mut conn, &second, &pages).unwrap();
    write_journal(&paths, Some(&first), &second).unwrap();
    write_current_pointer(&paths, &second).unwrap();

    let recovered = recover_if_needed(&mut conn, &wiki_dir).unwrap().unwrap();
    assert_eq!(
        recovered.action,
        llm_wiki_compiler::RecoveryAction::CompletedNew
    );

    assert!(!journal_exists(&paths));
    assert_eq!(pointer_of(&wiki_dir).as_deref(), Some(second.as_str()));
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(second.clone()));
    assert_eq!(build_status(&conn, &second), "COMPLETED");
    let _ = first;
}

#[test]
fn crash_after_pointer_move_with_invalid_generation_rolls_back() {
    let wiki_dir = temp_dir("crash-step5-bad");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];
    let first = seed_published_generation(&mut conn, &wiki_dir, &pages);

    // Pointer moved but the generation cannot be verified (no wiki_pages
    // rows → nothing to validate against): recovery must roll back instead
    // of completing something unverifiable.
    let second = new_build(&mut conn);
    let paths = PublishPaths::new(&wiki_dir);
    write_journal(&paths, Some(&first), &second).unwrap();
    write_current_pointer(&paths, &second).unwrap();

    let recovered = recover_if_needed(&mut conn, &wiki_dir).unwrap().unwrap();
    assert_eq!(
        recovered.action,
        llm_wiki_compiler::RecoveryAction::RolledBack
    );

    assert!(!journal_exists(&paths));
    assert_eq!(pointer_of(&wiki_dir).as_deref(), Some(first.as_str()));
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(first));
    assert_eq!(build_status(&conn, &second), "INTERRUPTED");
}

#[test]
fn crash_after_db_commit_only_clears_the_journal() {
    let wiki_dir = temp_dir("crash-step6");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];
    let first = seed_published_generation(&mut conn, &wiki_dir, &pages);
    let _ = first;

    // Steps 1–6 done, crash before step 7: everything committed, the journal
    // file is a leftover.
    let second = new_build(&mut conn);
    let paths = PublishPaths::new(&wiki_dir);
    write_generation(&paths, &second, &pages, &BTreeMap::new()).unwrap();
    write_journal(&paths, None, &second).unwrap();
    write_current_pointer(&paths, &second).unwrap();
    activate_build(&mut conn, &second).unwrap();

    let recovered = recover_if_needed(&mut conn, &wiki_dir).unwrap().unwrap();
    assert_eq!(
        recovered.action,
        llm_wiki_compiler::RecoveryAction::CompletedNew
    );
    assert!(!journal_exists(&paths));
    assert_eq!(pointer_of(&wiki_dir).as_deref(), Some(second.as_str()));
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(second));
}

#[test]
fn crash_of_first_publish_without_pointer_rolls_back_to_empty() {
    let wiki_dir = temp_dir("crash-first");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];

    // Very first publish crashed after recording the intent: there is no old
    // version, so recovery rolls back to an empty wiki.
    let build = new_build(&mut conn);
    let paths = PublishPaths::new(&wiki_dir);
    write_generation(&paths, &build, &pages, &BTreeMap::new()).unwrap();
    write_journal(&paths, None, &build).unwrap();

    let recovered = recover_if_needed(&mut conn, &wiki_dir).unwrap().unwrap();
    assert_eq!(
        recovered.action,
        llm_wiki_compiler::RecoveryAction::RolledBack
    );
    assert!(!journal_exists(&paths));
    assert!(pointer_of(&wiki_dir).is_none());
    assert!(get_active_build_id(&conn).unwrap().is_none());
    assert_eq!(build_status(&conn, &build), "INTERRUPTED");
}

#[test]
fn pointer_db_mismatch_without_journal_is_publish_recovery() {
    let wiki_dir = temp_dir("mismatch");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];
    let first = seed_published_generation(&mut conn, &wiki_dir, &pages);

    // Someone/something moved the pointer without a journal: recovery must
    // refuse to guess (PRD §35).
    let phantom = new_build(&mut conn);
    write_current_pointer(&PublishPaths::new(&wiki_dir), &phantom).unwrap();

    let err = recover_if_needed(&mut conn, &wiki_dir).unwrap_err();
    assert!(matches!(err, WikiError::PublishRecovery(_)), "{err}");
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(first));
}

#[test]
fn missing_pointer_with_db_active_without_journal_is_publish_recovery() {
    let wiki_dir = temp_dir("mismatch-missing-pointer");
    let mut conn = open_in_memory().unwrap();
    let pages = vec![page("runtime", "# Runtime")];
    let build = new_build(&mut conn);
    publish(&mut conn, &wiki_dir, &build, &pages, 3).unwrap();

    std::fs::remove_file(PublishPaths::new(&wiki_dir).pointer_path()).unwrap();
    let err = recover_if_needed(&mut conn, &wiki_dir).unwrap_err();
    assert!(matches!(err, WikiError::PublishRecovery(_)), "{err}");
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(build));
}

// ---------------------------------------------------------------------------
// Wiki Graph (PRD §17): rebuilt inside the §35 activate transaction, verified
// alongside the FTS index on recovery
// ---------------------------------------------------------------------------

#[test]
fn publish_populates_the_graph_and_the_next_publish_flips_it_atomically() {
    let wiki_dir = temp_dir("graph-publish");
    let mut conn = open_in_memory().unwrap();

    let target = page("sso", "# SSO");
    let target_id = target.page_id.clone();
    let mut overview = page("overview", "# Overview");
    overview.links = vec![llm_wiki_storage::PageLinkRecord {
        to_page_id: target_id.clone(),
        target_title: "sso".into(),
    }];
    let overview_id = overview.page_id.clone();
    // §35.3: generation rows are persisted BEFORE publish (the publish step-6
    // index/graph rebuilds read them).
    let first = new_build(&mut conn);
    persist_generation(&mut conn, &first, &[target.clone(), overview.clone()]).unwrap();
    publish(&mut conn, &wiki_dir, &first, &[target, overview], 3).unwrap();

    let graph_counts = |conn: &rusqlite::Connection| -> (i64, i64) {
        conn.query_row(
            "SELECT (SELECT COUNT(*) FROM graph_nodes), (SELECT COUNT(*) FROM graph_edges)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap()
    };
    assert_eq!(graph_counts(&conn), (2, 1), "two pages, one links_to edge");

    // Expansion reaches the linked page in both directions (§22 1-hop).
    let out = llm_wiki_storage::graph_expand_from_page(&conn, &overview_id, 10).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].label, "sso");
    let back = llm_wiki_storage::graph_expand_from_page(&conn, &target_id, 10).unwrap();
    assert_eq!(back.len(), 1);
    assert_eq!(
        back[0].direction,
        llm_wiki_storage::NeighborDirection::Incoming
    );

    // A graph drift (page node lost — its edges first, they FK to it) is
    // healed back to the active generation (the recovery path calls the same
    // storage verification).
    conn.execute("DELETE FROM graph_edges", []).unwrap();
    conn.execute(
        "DELETE FROM graph_nodes WHERE id = ?1",
        params![llm_wiki_storage::page_node_id(&target_id)],
    )
    .unwrap();
    let healed = llm_wiki_storage::ensure_graph_matches_active(&mut conn).unwrap();
    assert!(healed.is_some(), "drift triggers a rebuild");
    assert_eq!(graph_counts(&conn), (2, 1));

    // A second publish with a disjoint page set flips the graph atomically —
    // no node of the old generation survives.
    let second_pages = vec![page("solo", "# Solo")];
    let second = new_build(&mut conn);
    persist_generation(&mut conn, &second, &second_pages).unwrap();
    publish(&mut conn, &wiki_dir, &second, &second_pages, 3).unwrap();
    assert_eq!(graph_counts(&conn), (1, 0));
    let stale_page: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM graph_nodes WHERE id = ?1",
            params![llm_wiki_storage::page_node_id(&overview_id)],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(stale_page, 0, "old generation page nodes are gone");
}

// ---------------------------------------------------------------------------
// End-to-end run_build with FakeLlmProvider
// ---------------------------------------------------------------------------

const SUMMARY_MARK: &str = "Summarize the following cluster";
const LOCAL_MARK: &str = "THIS cluster's knowledge";
const RECONCILE_MARK: &str = "final global wiki plan";

/// Extracts every JSON-string value for `key` (compact serialization only —
/// the prompt templates' schema examples use spaces after the colon and are
/// never matched).
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

/// All `kn_…` node ids serialized in the payload, deduplicated in order.
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

/// Claim ids in the compilation KNOWLEDGE payload: the id immediately before
/// each compact `"kind":"claim"` marker.
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

/// A deterministic FakeLlmProvider that answers every pipeline stage from the
/// request itself: analysis claims quote real section content from the
/// manifest, planning proposes pages over the ids actually present, and
/// compilation cites only claims of the page.
fn pipeline_llm() -> Arc<dyn LlmProvider> {
    let proposed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(move |request: &LlmRequest| -> Result<String, LlmError> {
        let prompt = &request.prompt;
        if request.task_tag == "document-analysis" {
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
            return Ok(serde_json::json!({
                "summary": "Deterministic fixture analysis.",
                "topics": ["fixture"],
                "entities": [],
                "concepts": [],
                "claims": claims,
                "relations": [],
            })
            .to_string());
        }
        if request.task_tag == "wiki-planning" {
            if prompt.contains(SUMMARY_MARK) {
                return Ok(
                    serde_json::json!({"summary": "cluster about the fixture sources"}).to_string(),
                );
            }
            if prompt.contains(LOCAL_MARK) {
                let ids = kn_ids_in(prompt);
                proposed.lock().unwrap().extend(ids.iter().cloned());
                return Ok(serde_json::json!({
                    "pages": [{
                        "title": "Fixture Platform",
                        "category": "concepts",
                        "purpose": "cover the fixture knowledge",
                        "knowledge_refs": ids,
                    }]
                })
                .to_string());
            }
            if prompt.contains(RECONCILE_MARK) {
                let ids = proposed.lock().unwrap().clone();
                return Ok(serde_json::json!({
                    "pages": [{
                        "title": "Fixture Platform",
                        "category": "concepts",
                        "purpose": "cover the fixture knowledge",
                        "knowledge_refs": ids,
                    }]
                })
                .to_string());
            }
        }
        if request.task_tag == "wiki-compilation" {
            let claims = claim_ids_in(prompt);
            let body = if claims.is_empty() {
                "## Overview\n\nA plain page without claim citations.".to_owned()
            } else {
                format!(
                    "## Overview\n\nThe fixture page cites its stored claims.\n\n<!-- llm-wiki:cite claim=\"{}\" -->",
                    claims[0]
                )
            };
            return Ok(serde_json::json!({ "markdown": body }).to_string());
        }
        Err(LlmError::Api {
            code: 500,
            message: format!("unexpected task tag {}", request.task_tag),
        })
    });
    Arc::new(FakeLlmProvider::new("fake-pipeline", handler))
}

/// Creates a workspace with `docs/guide/*.md` sources and default config
/// (`source.root = ./docs`, `wiki_dir = ./wiki`).
fn fixture_workspace(tag: &str) -> PathBuf {
    let workspace = temp_dir(tag);
    let docs = workspace.join("docs").join("guide");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("runtime.md"),
        "# Runtime\n\nThe scheduler retries failed tasks up to three times before giving up.\n\n## Delivery\n\nEvents are delivered at least once and handlers must stay idempotent.\n",
    )
    .unwrap();
    std::fs::write(
        docs.join("security.md"),
        "# Security\n\nPermissions are enforced at the message bus boundary for every request.\n\n## Audit\n\nEvery permission change lands in the immutable audit log.\n",
    )
    .unwrap();
    let state = workspace.join(".llm-wiki");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("config.toml"),
        "[project]\nname = \"fixture\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-pipeline\"\n",
    )
    .unwrap();
    workspace
}

fn read_visible_wiki(wiki_dir: &Path) -> Vec<(String, String)> {
    let paths = PublishPaths::new(wiki_dir);
    let pointer = read_current_pointer(&paths).expect("pointer readable");
    let generation = pointer
        .map(|p| paths.generation_dir(&BuildId::parse(p.build_id).unwrap()))
        .expect("visible generation");
    let mut files: Vec<(String, String)> = std::fs::read_dir(generation)
        .expect("generation dir readable")
        .flatten()
        .map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let content = std::fs::read_to_string(entry.path()).unwrap();
            (name, content)
        })
        .collect();
    files.sort();
    files
}

#[tokio::test]
async fn run_build_publishes_a_readable_generation() {
    let workspace = fixture_workspace("e2e");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");

    let report = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    assert!(report.pages >= 1);
    assert!(report.citations >= 1);
    assert!(report.llm_request_count > 0);
    assert!(report.recovery.is_none());

    // The wiki is readable at wiki_dir via current.json → generation.
    let visible = read_visible_wiki(&wiki_dir);
    assert!(!visible.is_empty());
    let (_, content) = &visible[0];
    assert!(content.contains("generated: true"));
    assert!(content.contains("llm-wiki:cite"), "citations are expanded");

    // DB agrees with the pointer and the build is COMPLETED.
    let conn = open(&workspace.join(".llm-wiki").join("state.db")).unwrap();
    assert_eq!(
        get_active_build_id(&conn)
            .unwrap()
            .as_ref()
            .map(|b| b.as_str()),
        Some(report.build_id.as_str())
    );
    assert_eq!(build_status(&conn, &report.build_id), "COMPLETED");
    assert!(!journal_exists(&PublishPaths::new(&wiki_dir)));
}

#[tokio::test]
async fn second_build_keeps_both_generations_and_advances_the_pointer() {
    let workspace = fixture_workspace("e2e-two");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");

    let first = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    // ULID ids order chronologically at millisecond resolution.
    std::thread::sleep(std::time::Duration::from_millis(5));
    // FIX-006: an unchanged rebuild is a true no-op (the active generation
    // stays published), so the pointer only advances via a real change.
    std::fs::write(
        workspace.join("docs").join("guide").join("runtime.md"),
        "# Runtime\n\nThe scheduler retries failed tasks up to three times before giving up.\n\n## Delivery\n\nEvents are delivered exactly once in v2 and handlers must stay idempotent.\n",
    )
    .unwrap();
    let second = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    assert_ne!(first.build_id, second.build_id);

    let paths = PublishPaths::new(&wiki_dir);
    assert_eq!(
        pointer_of(&wiki_dir).as_deref(),
        Some(second.build_id.as_str())
    );
    assert!(
        paths.generation_dir(&first.build_id).is_dir(),
        "old generation retained"
    );
    assert!(paths.generation_dir(&second.build_id).is_dir());
    let visible = read_visible_wiki(&wiki_dir);
    assert!(!visible.is_empty());
}

#[tokio::test]
async fn llm_failure_marks_build_failed_and_keeps_previous_generation_visible() {
    let workspace = fixture_workspace("e2e-fail");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");

    let first = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    let visible_before = read_visible_wiki(&wiki_dir);

    // Second build over a REAL change (FIX-006: an unchanged rebuild is a
    // no-op that never reaches the provider): the provider explodes on the
    // first analysis request.
    std::fs::write(
        workspace.join("docs").join("guide").join("runtime.md"),
        "# Runtime\n\nThe scheduler retries failed tasks up to three times before giving up.\n\n## Delivery\n\nEvents are delivered exactly once in v2 and handlers must stay idempotent.\n",
    )
    .unwrap();
    let failing: Arc<dyn LlmProvider> = Arc::new(FakeLlmProvider::new(
        "fake-failing",
        Arc::new(|_request: &LlmRequest| {
            Err(LlmError::Api {
                code: 500,
                message: "provider down".into(),
            })
        }),
    ));
    let err = llm_wiki_compiler::run_build(&workspace, &config, failing)
        .await
        .unwrap_err();
    assert!(matches!(err, WikiError::Llm(_)), "{err}");

    // The previously published wiki is intact and readable.
    let visible_after = read_visible_wiki(&wiki_dir);
    assert_eq!(visible_after, visible_before);
    let conn = open(&workspace.join(".llm-wiki").join("state.db")).unwrap();
    assert_eq!(
        get_active_build_id(&conn)
            .unwrap()
            .as_ref()
            .map(|b| b.as_str()),
        Some(first.build_id.as_str())
    );
    assert_eq!(build_status(&conn, &first.build_id), "COMPLETED");

    // The failed build itself: a row marked FAILED exists.
    let failed_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM builds WHERE status = 'FAILED'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(failed_count, 1);
}

#[test]
fn replan_required_maps_to_the_replan_terminal_status() {
    // ReplanRequired (V0.2 trigger) must mark the build REPLAN_REQUIRED, not
    // FAILED — the CLI exits with code 7 and the pointer stays untouched.
    let err = WikiError::ReplanRequired {
        reason: "sources restructured".into(),
    };
    assert_eq!(err.exit_code(), 7);
    assert!(err.to_string().contains("sources restructured"));
    assert!(err.to_string().contains("replan --dry-run"));
}

/// §37.3 Rebuild Determinism (V0.1 DoD #16) under the FIX-006 fast path: a
/// second build over completely unchanged sources is a TRUE no-op — it
/// returns the ACTIVE generation with zero new LLM requests, zero cache
/// traffic (the pipeline never runs), no new generation and the pointer
/// unmoved.
#[tokio::test]
async fn second_identical_build_consumes_zero_new_llm_requests_and_keeps_the_manifest() {
    let workspace = fixture_workspace("determinism");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let db_path = workspace.join(".llm-wiki").join("state.db");

    let first = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    assert!(
        first.llm_request_count > 0,
        "first build really calls the model"
    );
    assert!(first.cache.misses > 0, "first build populates the cache");

    let conn = conn_of(&db_path);
    let manifest_of = |build: &BuildId| -> Vec<String> {
        let mut view = load_generation_view(&conn, build).unwrap();
        view.sort_by(|a, b| a.slug.cmp(&b.slug));
        view.into_iter()
            .map(|page| {
                let citations: Vec<String> = page
                    .citations
                    .iter()
                    .map(|c| {
                        format!(
                            "{}/{}/{}/{}/{}",
                            c.claim_node_id,
                            c.source_id,
                            c.range.start,
                            c.range.end,
                            c.evidence_digest
                        )
                    })
                    .collect();
                let links: Vec<String> = page
                    .links
                    .iter()
                    .map(|l| format!("{}/{}", l.to_page_id, l.target_title))
                    .collect();
                format!(
                    "{}|{}|{}|{:?}|{:?}|{:?}",
                    page.page_id, page.slug, page.title, page.knowledge_refs, citations, links
                )
            })
            .collect()
    };
    let manifest_first = manifest_of(&first.build_id);

    let second = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    assert_eq!(
        second.llm_request_count, 0,
        "the fast path issues no LLM requests"
    );
    // FIX-006: the pipeline never ran — not even §28 cache lookups.
    assert_eq!(second.cache.hits, 0);
    assert_eq!(second.cache.misses, 0);
    // The ACTIVE generation is returned unchanged.
    assert_eq!(second.published_path, first.published_path);
    assert_eq!(
        pointer_of(&wiki_dir).as_deref(),
        Some(first.build_id.as_str())
    );
    assert_eq!(build_status(&conn, &second.build_id), "COMPLETED");
    assert_eq!(
        manifest_of(&first.build_id),
        manifest_first,
        "the published manifest is untouched"
    );
}

/// §37.3 cache absorption on the FULL pipeline (`build.incremental = false`):
/// an identical rebuild re-uses every §28 cached response — zero new LLM
/// requests — and the plan-identity cache preserves page IDs, so the
/// structured manifest is identical.
#[tokio::test]
async fn full_rebuild_absorbs_every_request_in_the_cache_and_keeps_the_manifest() {
    let workspace = fixture_workspace("determinism-full");
    let mut config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    config.build.incremental = false;
    let db_path = workspace.join(".llm-wiki").join("state.db");

    let first = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    assert!(first.llm_request_count > 0);

    let second = llm_wiki_compiler::run_build(&workspace, &config, pipeline_llm())
        .await
        .unwrap();
    assert_eq!(
        second.llm_request_count, 0,
        "every request must be answered from the §28 cache"
    );
    assert!(second.cache.hits > 0, "the cache must actually serve");
    assert_eq!(second.cache.misses, 0, "nothing may recompute");
    assert_eq!(
        build_status(&conn_of(&db_path), &second.build_id),
        "COMPLETED"
    );

    // Structured manifest identity across the two generations.
    let conn = conn_of(&db_path);
    let manifest_of = |build: &BuildId| -> Vec<String> {
        let mut view = load_generation_view(&conn, build).unwrap();
        view.sort_by(|a, b| a.slug.cmp(&b.slug));
        view.into_iter()
            .map(|page| {
                let citations: Vec<String> = page
                    .citations
                    .iter()
                    .map(|c| {
                        format!(
                            "{}/{}/{}/{}/{}",
                            c.claim_node_id,
                            c.source_id,
                            c.range.start,
                            c.range.end,
                            c.evidence_digest
                        )
                    })
                    .collect();
                let links: Vec<String> = page
                    .links
                    .iter()
                    .map(|l| format!("{}/{}", l.to_page_id, l.target_title))
                    .collect();
                // Note: page content legitimately differs between generations
                // because §15.2 frontmatter carries the owning build id; the
                // §37.3 manifest covers ids, refs, citations and links only.
                format!(
                    "{}|{}|{}|{:?}|{:?}|{:?}",
                    page.page_id, page.slug, page.title, page.knowledge_refs, citations, links
                )
            })
            .collect()
    };
    assert_eq!(
        manifest_of(&first.build_id),
        manifest_of(&second.build_id),
        "page IDs, citation mapping and links must be identical"
    );
}

fn conn_of(path: &Path) -> rusqlite::Connection {
    open(path).unwrap()
}

// ---------------------------------------------------------------------------
// Publish-step search index (PRD §20/§35: FTS joins the §35 step-6 transaction)
// ---------------------------------------------------------------------------

/// The tokenizer's OR-of-quoted-terms MATCH expression, mirroring the search
/// crate's `fts_query` — keeps these tests independent of llm-wiki-search.
fn fts_expression(tokenizer: &dyn llm_wiki_storage::SearchTokenizer, text: &str) -> String {
    let mut seen = std::collections::BTreeSet::new();
    let mut terms = Vec::new();
    for token in tokenizer.analyze(text) {
        if seen.insert(token.clone()) {
            terms.push(format!("\"{}\"", token.replace('"', "\"\"")));
        }
    }
    terms.join(" OR ")
}

#[test]
fn publish_flips_pointer_and_search_index_atomically() {
    let wiki_dir = temp_dir("fts-publish");
    let mut conn = open_in_memory().unwrap();
    let tokenizer = llm_wiki_storage::default_tokenizer();

    let first_pages = vec![page(
        "streaming",
        "# Streaming Processing\n\n## Checkpoints\n\n检查点默认每 30 秒持久化一次。\n",
    )];
    let first = new_build(&mut conn);
    persist_generation(&mut conn, &first, &first_pages).unwrap();
    publish(&mut conn, &wiki_dir, &first, &first_pages, 3).unwrap();

    // CJK bigram + single-char queries hit the published page's section.
    for query in ["检查点", "检查", "点"] {
        let hits =
            llm_wiki_storage::search_index(&conn, &fts_expression(tokenizer, query), 10).unwrap();
        assert!(
            hits.iter().any(|hit| hit.slug == "streaming"),
            "query {query:?} must find the published page, got {:?}",
            hits.iter().map(|h| h.slug.clone()).collect::<Vec<_>>()
        );
    }

    // A second publish replaces the index contents in the same transaction.
    let second_pages = vec![page(
        "sso",
        "# Identity & Access\n\n配置单点登录（SSO）。\n",
    )];
    let second = new_build(&mut conn);
    persist_generation(&mut conn, &second, &second_pages).unwrap();
    publish(&mut conn, &wiki_dir, &second, &second_pages, 3).unwrap();

    let hits =
        llm_wiki_storage::search_index(&conn, &fts_expression(tokenizer, "检查点"), 10).unwrap();
    assert!(
        hits.iter().all(|hit| hit.slug != "streaming"),
        "the superseded generation leaves the index"
    );
    let hits =
        llm_wiki_storage::search_index(&conn, &fts_expression(tokenizer, "SSO 登录"), 10).unwrap();
    assert_eq!(hits[0].slug, "sso");
}

#[test]
fn recovery_rollback_rebuilds_the_previous_generation_index() {
    let wiki_dir = temp_dir("fts-rollback");
    let mut conn = open_in_memory().unwrap();
    let tokenizer = llm_wiki_storage::default_tokenizer();

    let old_pages = vec![page("old-doc", "# Old Doc\n\nrollback 检查点 content")];
    let old_build = seed_published_generation(&mut conn, &wiki_dir, &old_pages);
    // seed_published_generation writes only the filesystem side; the FTS
    // rebuild reads wiki_pages rows, so persist the rolled-back generation.
    persist_generation(&mut conn, &old_build, &old_pages).unwrap();
    let new_pages = vec![page("new-doc", "# New Doc\n\nfresh 单点登录 content")];
    let new_build = new_build(&mut conn);
    persist_generation(&mut conn, &new_build, &new_pages).unwrap();

    // Simulated crash: the database committed the new generation (index and
    // all) but the pointer rename was lost — recovery must roll back to the
    // explicit old build and the index must follow.
    write_journal(&PublishPaths::new(&wiki_dir), Some(&old_build), &new_build).unwrap();
    llm_wiki_storage::activate_build_with_search_index(&mut conn, &new_build, tokenizer).unwrap();
    write_current_pointer(&PublishPaths::new(&wiki_dir), &old_build).unwrap();
    assert_eq!(pointer_of(&wiki_dir).as_deref(), Some(old_build.as_str()));

    let report = recover_if_needed(&mut conn, &wiki_dir).unwrap().unwrap();
    assert!(matches!(
        report.action,
        llm_wiki_compiler::RecoveryAction::RolledBack
    ));
    assert_eq!(get_active_build_id(&conn).unwrap(), Some(old_build));

    let hits =
        llm_wiki_storage::search_index(&conn, &fts_expression(tokenizer, "检查点"), 10).unwrap();
    assert!(hits.iter().any(|hit| hit.slug == "old-doc"), "{hits:?}");
    let hits =
        llm_wiki_storage::search_index(&conn, &fts_expression(tokenizer, "单点登录"), 10).unwrap();
    assert!(
        hits.iter().all(|hit| hit.slug != "new-doc"),
        "the rolled-back generation never serves search: {hits:?}"
    );
}

// ---------------------------------------------------------------------------
// Document-parallel analysis (T1 finding #12): `llm.max_concurrency` must
// bound the in-flight analysis requests (D1 red-test — dropping the window
// must turn this red), and results must stay deterministic per document.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn analysis_respects_max_concurrency_and_stays_deterministic() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let workspace = fixture_workspace("parallel-analysis");
    // Four independent docs; each unit's fake sleeps briefly so overlaps
    // are observable.
    let docs_dir = workspace.join("docs");
    for name in ["alpha", "beta", "gamma", "delta"] {
        std::fs::write(
            docs_dir.join(format!("{name}.md")),
            format!(
                "# {name}

The {name} scheduler retries failed tasks up to three times before giving up.

## Delivery

Events for {name} are delivered at least once and handlers stay idempotent.
"
            ),
        )
        .unwrap();
    }

    // Stage-routing provider (same shape as pipeline_llm) with an in-flight
    // gauge wrapped around the ANALYSIS stage only — planning/compilation
    // stages run after analysis, so their requests must never inflate the
    // concurrency peak.
    let inflight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let inflight_h = Arc::clone(&inflight);
    let peak_h = Arc::clone(&peak);
    let proposed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let proposed_h = Arc::clone(&proposed);
    let handler = Arc::new(move |request: &LlmRequest| -> Result<String, LlmError> {
        if request.task_tag == "document-analysis" {
            let now = inflight_h.fetch_add(1, Ordering::SeqCst) + 1;
            peak_h.fetch_max(now, Ordering::SeqCst);
            // Simulate network latency WITHOUT blocking the tokio worker
            // thread (a sync sleep would serialize the join window and hide
            // any concurrency): spin until a sibling is also in flight,
            // bounded so a serial implementation still terminates (its peak
            // stays 1 and the overlap assertion below fails — the red test).
            let mut spins = 0u32;
            while inflight_h.load(Ordering::SeqCst) < 2 && spins < 2_000_000 {
                std::hint::spin_loop();
                spins += 1;
            }

            // Extract a verbatim line from the PROMPT itself (the prompt
            // embeds the section content), so the evidence quote can never
            // drift from the document — the exact failure the locator
            // guards against.
            let prompt = &request.prompt;
            // Manifest embeds content as a JSON string ("content":"..."): take
            // the first JSON-escaped line containing the marker phrase, then
            // cut at the JSON string terminator. Unescaping is unnecessary —
            // the locator folds punctuation, and the fixture body has none.
            let quote = prompt
                .split("\"content\":\"")
                .skip(1)
                .find_map(|seg| {
                    seg.split("\n")
                        .map(str::trim)
                        .find(|line| line.contains("retries failed tasks"))
                        .and_then(|line| line.split("\",\"").next().map(str::to_owned))
                })
                .unwrap_or_else(|| {
                    // Fallback for sections without the marker phrase: any
                    // non-empty content line works (it is verbatim by
                    // construction).
                    prompt
                        .split("\"content\":\"")
                        .skip(1)
                        .find_map(|seg| {
                            seg.split("\n")
                                .map(str::trim)
                                .find(|line| !line.is_empty() && !line.starts_with('\"'))
                                .and_then(|line| line.split("\",\"").next().map(str::to_owned))
                        })
                        .unwrap_or_default()
                });
            eprintln!("DEBUG tag={} quote={quote:?}", request.task_tag);
            let out = Ok(serde_json::json!({
                "summary": "parallel fixture",
                "topics": ["parallel"],
                "entities": [],
                "concepts": [],
                "claims": [{
                    "text": quote,
                    "section_id": "",
                    "evidence_text": quote,
                    "evidence_start": 0,
                    "confidence": 0.9
                }],
                "relations": []
            })
            .to_string());
            inflight_h.fetch_sub(1, Ordering::SeqCst);
            return out;
        }
        if request.task_tag == "wiki-planning" {
            if request.prompt.contains(SUMMARY_MARK) {
                return Ok(
                    serde_json::json!({"summary": "cluster about the parallel fixture"})
                        .to_string(),
                );
            }
            // LOCAL plan proposes; RECONCILE replays everything proposed.
            let ids = kn_ids_in(&request.prompt);
            proposed_h.lock().unwrap().extend(ids.iter().cloned());
            let ids = if request.prompt.contains(RECONCILE_MARK) {
                proposed_h.lock().unwrap().clone()
            } else {
                ids
            };
            return Ok(serde_json::json!({
                "pages": [{"title": "Parallel Fixture", "category": "concepts",
                           "purpose": "cover the fixture", "knowledge_refs": ids}]
            })
            .to_string());
        }
        if request.task_tag == "wiki-compilation" {
            let claims = claim_ids_in(&request.prompt);
            let body = if claims.is_empty() {
                "## Overview\n\nA plain page without claim citations.".to_owned()
            } else {
                format!(
                    "## Overview\n\nThe fixture page cites its stored claims.\n\n<!-- llm-wiki:cite claim=\"{}\" -->",
                    claims[0]
                )
            };
            return Ok(serde_json::json!({ "markdown": body }).to_string());
        }
        Err(LlmError::Api {
            code: 500,
            message: format!("unexpected task tag {}", request.task_tag),
        })
    });
    let provider: Arc<dyn LlmProvider> = Arc::new(FakeLlmProvider::new("fake-parallel", handler));

    let mut config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    config.llm.max_concurrency = 2;

    let report = llm_wiki_compiler::run_build(&workspace, &config, provider)
        .await
        .unwrap();
    assert!(
        report.pages >= 1 && report.sources == 6,
        "the single-plan fake merges all knowledge into one page:          pages={}, sources={} — both matter",
        report.pages,
        report.sources
    );
    assert!(report.citations >= 1, "claims made it through to citations");

    let observed_peak = peak.load(Ordering::SeqCst);
    assert!(
        observed_peak <= 2,
        "in-flight analysis peaked at {observed_peak}, config caps at 2"
    );
    assert!(
        observed_peak >= 2,
        "no overlap observed (peak {observed_peak}) — the test cannot prove concurrency is bounded, only serial"
    );
}
