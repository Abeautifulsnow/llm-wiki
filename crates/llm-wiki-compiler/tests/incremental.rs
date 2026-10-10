//! Integration tests for the §19 incremental build pipeline (PRD §19, V0.2
//! DoD §53 #1–#5 and #7), driven by FakeLlmProvider — no real model in CI
//! (PRD §54).
//!
//! The fixture fake plans TWO pages (runtime knowledge vs security knowledge)
//! so deletion can empty a page into `obsolete` and modification can be
//! localized onto exactly one page.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rusqlite::params;

use llm_wiki_compiler::{read_current_pointer, PublishPaths};
use llm_wiki_core::error::WikiError;
use llm_wiki_core::ids::BuildId;
use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmProvider, LlmRequest};
use llm_wiki_storage::{
    get_active_build_id, list_plan_decisions, list_sources, load_generation_view, open,
    OUTCOME_FAST_PATH, OUTCOME_LOCAL_UPDATE, OUTCOME_REPLAN_REQUIRED, TRIGGER_FINGERPRINT_CHANGED,
    TRIGGER_UNMAPPABLE_NODE,
};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-incremental-{tag}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_workspace_config(workspace: &Path, extra: &str) {
    let state = workspace.join(".llm-wiki");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("config.toml"),
        format!(
            "[project]\nname = \"incremental\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-incremental\"\n{extra}"
        ),
    )
    .unwrap();
}

fn seed_runtime_doc(workspace: &Path, delivery_text: &str) {
    let docs = workspace.join("docs").join("guide");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("runtime.md"),
        format!(
            "# Runtime\n\nThe scheduler retries failed tasks up to three times before giving up.\n\n## Delivery\n\n{delivery_text}\n"
        ),
    )
    .unwrap();
}

fn fixture_workspace(tag: &str) -> PathBuf {
    let workspace = temp_dir(tag);
    seed_runtime_doc(
        &workspace,
        "Events are delivered at least once and handlers must stay idempotent.",
    );
    std::fs::write(
        workspace.join("docs").join("guide").join("security.md"),
        "# Security\n\nPermissions are enforced at the message bus boundary for every request.\n\n## Audit\n\nEvery permission change lands in the immutable audit log.\n",
    )
    .unwrap();
    write_workspace_config(&workspace, "");
    workspace
}

/// Extracts every JSON-string value for `key` (compact serialization only).
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

/// (node id, node name) pairs from a compact planning payload.
fn id_name_pairs(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = text[search_from..].find("\"id\":\"kn_") {
        let id_start = search_from + offset + "\"id\":\"".len();
        let rest = &text[id_start..];
        let id_end = rest.find('"').unwrap_or(rest.len());
        let id = rest[..id_end].to_owned();
        let after = &rest[id_end..];
        let name = after
            .find("\"name\":\"")
            .map(|name_offset| {
                let name_rest = &after[name_offset + "\"name\":\"".len()..];
                let name_end = name_rest.find('"').unwrap_or(name_rest.len());
                unescape_json(&name_rest[..name_end])
            })
            .unwrap_or_default();
        if !out.iter().any(|(known, _)| *known == id) {
            out.push((id, name));
        }
        search_from = id_start + id_end;
    }
    out
}

/// Claim ids in the compilation KNOWLEDGE payload.
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

const RUNTIME_TITLE: &str = "Runtime Platform";
const SECURITY_TITLE: &str = "Security Platform";

/// Security-flavored claim names route to the security page; everything else
/// to the runtime page. Deterministic two-page plan over the fixture corpus.
fn route_title(name: &str) -> &'static str {
    let security_flavored = ["ermission", "audit", "bus bound"]
        .iter()
        .any(|needle| name.contains(needle));
    if security_flavored {
        SECURITY_TITLE
    } else {
        RUNTIME_TITLE
    }
}

/// A deterministic FakeLlmProvider that answers every pipeline stage; the
/// planner proposes one RUNTIME page and one SECURITY page.
fn incremental_llm() -> Arc<dyn LlmProvider> {
    let proposed: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
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
                "summary": "Incremental fixture analysis.",
                "topics": ["fixture"],
                "entities": [],
                "concepts": [],
                "claims": claims,
                "relations": [],
            })
            .to_string());
        }
        if request.task_tag == "wiki-planning" {
            if prompt.contains("Summarize the following cluster") {
                return Ok(
                    serde_json::json!({"summary": "cluster about the incremental fixtures"})
                        .to_string(),
                );
            }
            if prompt.contains("THIS cluster's knowledge") {
                let pairs = id_name_pairs(prompt);
                let mut pages: Vec<serde_json::Value> = Vec::new();
                // The fixture has a single cluster, so each local-plan round
                // REPLACES the proposal set (a second build's planning round
                // must not inherit the previous build's node ids).
                let mut entries: Vec<(String, String)> = Vec::new();
                for title in [RUNTIME_TITLE, SECURITY_TITLE] {
                    let refs: Vec<String> = pairs
                        .iter()
                        .filter(|(_, name)| route_title(name) == title)
                        .map(|(id, _)| id.clone())
                        .collect();
                    if !refs.is_empty() {
                        pages.push(serde_json::json!({
                            "title": title,
                            "category": if title == RUNTIME_TITLE { "concepts" } else { "architecture" },
                            "purpose": format!("cover the {title} knowledge"),
                            "knowledge_refs": refs,
                        }));
                    }
                    for (id, name) in &pairs {
                        if route_title(name) == title {
                            entries.push((title.to_owned(), id.clone()));
                        }
                    }
                }
                *proposed.lock().unwrap() = entries;
                return Ok(serde_json::json!({ "pages": pages }).to_string());
            }
            if prompt.contains("final global wiki plan") {
                // Group the locally proposed ids by page title (deterministic
                // order) so reconciliation keeps the two-page split.
                let taken = proposed.lock().unwrap().clone();
                let mut grouped: std::collections::BTreeMap<String, Vec<String>> =
                    std::collections::BTreeMap::new();
                for (title, id) in taken {
                    grouped.entry(title).or_default().push(id);
                }
                let pages: Vec<serde_json::Value> = grouped
                    .into_iter()
                    .map(|(title, refs)| {
                        serde_json::json!({
                            "title": title,
                            "category": if title == RUNTIME_TITLE { "concepts" } else { "architecture" },
                            "purpose": format!("cover the {title} knowledge"),
                            "knowledge_refs": refs,
                        })
                    })
                    .collect();
                return Ok(serde_json::json!({ "pages": pages }).to_string());
            }
        }
        if request.task_tag == "wiki-compilation" {
            let claims = claim_ids_in(prompt);
            let body = if claims.is_empty() {
                "## Overview\n\nA plain page without claim citations.".to_owned()
            } else {
                format!(
                    "## Overview\n\nThe incremental page cites its stored claims.\n\n<!-- llm-wiki:cite claim=\"{}\" -->",
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
    Arc::new(FakeLlmProvider::new("fake-incremental", handler))
}

fn conn_of(workspace: &Path) -> rusqlite::Connection {
    open(&workspace.join(".llm-wiki").join("state.db")).unwrap()
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

fn visible_wiki(wiki_dir: &Path) -> Vec<(String, String)> {
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

/// page_id per slug of one generation.
fn page_ids_by_slug(
    conn: &rusqlite::Connection,
    build: &BuildId,
) -> std::collections::BTreeMap<String, String> {
    load_generation_view(conn, build)
        .unwrap()
        .into_iter()
        .map(|page| (page.slug, page.page_id.as_str().to_owned()))
        .collect()
}

fn content_by_slug(
    conn: &rusqlite::Connection,
    build: &BuildId,
) -> std::collections::BTreeMap<String, String> {
    load_generation_view(conn, build)
        .unwrap()
        .into_iter()
        .map(|page| (page.slug, page.content))
        .collect()
}

// ---------------------------------------------------------------------------
// §19.1/§19.2 happy path: one modified doc, one recompiled page
// ---------------------------------------------------------------------------

#[tokio::test]
async fn modified_source_recompiles_only_its_page_and_carries_the_rest() {
    let workspace = fixture_workspace("happy");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    assert!(first.incremental.is_none(), "first build is a full build");
    assert_eq!(first.pages, 2, "runtime + security pages");
    let ids_first = page_ids_by_slug(&conn_of(&workspace), &first.build_id);
    let content_first = content_by_slug(&conn_of(&workspace), &first.build_id);

    // §19.2: modify ONE section of ONE source.
    seed_runtime_doc(
        &workspace,
        "Events are delivered exactly once in v2 and handlers must stay idempotent.",
    );
    let config2 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let second = llm_wiki_compiler::run_build(&workspace, &config2, provider.clone())
        .await
        .unwrap();

    // DoD #1–#3: changed source localized onto exactly one page.
    let summary = second.incremental.expect("incremental summary present");
    assert_eq!(
        summary,
        llm_wiki_compiler::IncrementalSummary {
            changed: 1,
            deleted: 0,
            recompiled: 1,
            carried: 1,
            obsolete: 0,
        }
    );
    // Request accounting: ONE analysis unit + ONE page compile (zero planner
    // requests — the plan identity is reused).
    assert_eq!(
        second.llm_request_count, 2,
        "only the changed source's analysis and the affected page's compile may run"
    );

    // Page IDs are stable (§45: carried identity).
    let ids_second = page_ids_by_slug(&conn_of(&workspace), &second.build_id);
    assert_eq!(ids_first, ids_second, "page identity survives the rebuild");

    // The carried page is byte-identical (frontmatter keeps the ORIGINAL
    // build id — never rewritten).
    let content_second = content_by_slug(&conn_of(&workspace), &second.build_id);
    assert_eq!(
        content_first["security-platform"], content_second["security-platform"],
        "unrelated pages must be carried over verbatim"
    );
    assert!(content_second["security-platform"].contains(&format!("build: {}", first.build_id)));
    assert_ne!(
        content_first["runtime-platform"], content_second["runtime-platform"],
        "the affected page was recompiled into the new build"
    );

    // DoD #4: the replaced claim's stale citations are gone.
    let conn = conn_of(&workspace);
    let stale: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM page_citations WHERE build_id = ?1
             AND claim_node_id NOT IN (SELECT node_id FROM claims WHERE status = 'active')",
            params![second.build_id.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stale, 0, "no citation may reference retired claims");

    // Publish advanced; both generations remain on disk.
    assert_eq!(
        pointer_of(&wiki_dir).as_deref(),
        Some(second.build_id.as_str())
    );

    // §19.2: the judgment is recorded.
    let decisions = list_plan_decisions(&conn, &second.build_id).unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].outcome, OUTCOME_LOCAL_UPDATE);
    assert_eq!(decisions[0].trigger, None);
    assert_eq!(decisions[0].affected_pages, 1);
    assert_eq!(build_status(&conn, &second.build_id), "COMPLETED");
}

// ---------------------------------------------------------------------------
// Fast path: unchanged rebuild
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unchanged_rebuild_takes_the_fast_path_and_records_it() {
    let workspace = fixture_workspace("fast-path");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    let fts_before: i64 = conn_of(&workspace)
        .query_row("SELECT COUNT(*) FROM wiki_fts", [], |r| r.get(0))
        .unwrap();
    let second = llm_wiki_compiler::run_build(&workspace, &config, provider)
        .await
        .unwrap();

    assert_eq!(
        second.llm_request_count, 0,
        "the fast path issues no requests"
    );
    // FIX-006: the pipeline never ran — not even §28 cache lookups.
    assert_eq!(second.cache.hits, 0);
    assert_eq!(second.cache.misses, 0);
    assert!(second.incremental.is_none(), "fast path is not incremental");
    // The ACTIVE generation is returned: pointer unmoved, no new generation
    // directory, and the report points at the still-published generation.
    assert_eq!(
        pointer_of(&wiki_dir).as_deref(),
        Some(first.build_id.as_str())
    );
    assert_eq!(second.published_path, first.published_path);
    assert!(
        !PublishPaths::new(&wiki_dir)
            .generation_dir(&second.build_id)
            .exists(),
        "the no-op build must not create a generation"
    );
    // FTS was not rebuilt.
    let fts_after: i64 = conn_of(&workspace)
        .query_row("SELECT COUNT(*) FROM wiki_fts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(fts_after, fts_before, "the index is untouched");
    let conn = conn_of(&workspace);
    let decisions = list_plan_decisions(&conn, &second.build_id).unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].outcome, OUTCOME_FAST_PATH);
    assert_eq!(decisions[0].trigger, None);
    assert_eq!(build_status(&conn, &second.build_id), "COMPLETED");
}

/// Audit FIX-013 acceptance: a carried page (byte-identical) is hardlinked
/// into the new generation instead of rewritten — same inode, zero content
/// copy — and the recompiled page is a fresh file.
#[cfg(unix)]
#[tokio::test]
async fn carried_pages_reuse_the_previous_generation_file_via_hardlink() {
    use std::os::unix::fs::MetadataExt;

    let workspace = fixture_workspace("hardlink-reuse");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    seed_runtime_doc(
        &workspace,
        "Events are delivered exactly once in v2 and handlers must stay idempotent.",
    );
    let second = llm_wiki_compiler::run_build(&workspace, &config, provider)
        .await
        .unwrap();

    let paths = PublishPaths::new(&wiki_dir);
    let file_of = |report: &llm_wiki_compiler::BuildReport, slug: &str| {
        let view = load_generation_view(&conn_of(&workspace), &report.build_id).unwrap();
        let page = view.iter().find(|page| page.slug == slug).unwrap().clone();
        paths
            .generation_dir(&report.build_id)
            .join(llm_wiki_compiler::page_file_name(
                &page.slug,
                page.page_id.as_str(),
            ))
    };
    let carried_first = file_of(&first, "security-platform");
    let carried_second = file_of(&second, "security-platform");
    let recompiled_second = file_of(&second, "runtime-platform");

    let first_meta = std::fs::metadata(&carried_first).unwrap();
    let second_meta = std::fs::metadata(&carried_second).unwrap();
    assert_eq!(
        (first_meta.dev(), first_meta.ino()),
        (second_meta.dev(), second_meta.ino()),
        "the carried page shares the previous generation's inode (hardlink reuse)"
    );
    assert_ne!(
        std::fs::metadata(&recompiled_second).unwrap().ino(),
        first_meta.ino(),
        "the recompiled page is a fresh file"
    );
    // The carried content still validates byte-for-byte after the reuse.
    assert_eq!(
        std::fs::read_to_string(&carried_first).unwrap(),
        std::fs::read_to_string(&carried_second).unwrap()
    );
}

/// FIX-006 health condition: when the active generation's files are gone
/// (pointer intact, directory deleted), the fast path must NOT return the
/// missing generation — the full pipeline republishes from scratch.
#[tokio::test]
async fn missing_active_generation_directory_falls_back_to_the_full_pipeline() {
    let workspace = fixture_workspace("fast-path-selfheal");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let paths = PublishPaths::new(&wiki_dir);
    std::fs::remove_dir_all(paths.generation_dir(&first.build_id)).unwrap();

    let second = llm_wiki_compiler::run_build(&workspace, &config, provider)
        .await
        .unwrap();
    assert!(
        paths.generation_dir(&second.build_id).is_dir(),
        "the full pipeline republished the generation"
    );
    assert_eq!(
        pointer_of(&wiki_dir).as_deref(),
        Some(second.build_id.as_str())
    );
    assert_eq!(
        build_status(&conn_of(&workspace), &second.build_id),
        "COMPLETED"
    );
}

// ---------------------------------------------------------------------------
// §19.3 deletion: ghost-claim-free retirement + obsolete page
// ---------------------------------------------------------------------------

#[tokio::test]
async fn deleted_source_leaves_no_ghost_claims_and_drops_the_obsolete_page() {
    let workspace = fixture_workspace("deletion");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let content_first = content_by_slug(&conn_of(&workspace), &first.build_id);

    // Delete the whole security source: its page loses ALL knowledge.
    std::fs::remove_file(workspace.join("docs").join("guide").join("security.md")).unwrap();
    let config2 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let second = llm_wiki_compiler::run_build(&workspace, &config2, provider)
        .await
        .unwrap();

    // Zero LLM requests: nothing is analyzed or compiled — the affected page
    // is obsolete, the untouched page carries over.
    assert_eq!(second.llm_request_count, 0);
    let summary = second.incremental.expect("incremental summary present");
    assert_eq!(
        summary,
        llm_wiki_compiler::IncrementalSummary {
            changed: 0,
            deleted: 1,
            recompiled: 0,
            carried: 1,
            obsolete: 1,
        }
    );

    let conn = conn_of(&workspace);
    // The new generation has exactly the surviving page, byte-identical.
    let view = load_generation_view(&conn, &second.build_id).unwrap();
    assert_eq!(view.len(), 1, "the emptied page is excluded (§19.3.5)");
    assert_eq!(view[0].slug, "runtime-platform");
    let content_second = content_by_slug(&conn, &second.build_id);
    assert_eq!(
        content_first["runtime-platform"], content_second["runtime-platform"],
        "carried page stays byte-identical"
    );

    // §53 DoD #5: no ghost claims — nothing active may reference the removed
    // source, and its unsupported knowledge nodes are retired.
    let removed_claims: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE status = 'active'
             AND source_id NOT IN (SELECT source_id FROM sources WHERE status = 'active')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(removed_claims, 0, "no active claim of a removed source");
    let retired_nodes: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM knowledge_registry WHERE status = 'retired'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        retired_nodes, 2,
        "both security-only knowledge nodes are retired"
    );
    assert_eq!(
        list_sources(&conn).unwrap().len(),
        1,
        "only the runtime source stays active"
    );
    // No citation of the new generation references the removed source.
    let stale_citations: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM page_citations WHERE build_id = ?1
             AND source_id NOT IN (SELECT source_id FROM sources WHERE status = 'active')",
            params![second.build_id.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stale_citations, 0);
    assert_eq!(
        pointer_of(&wiki_dir).as_deref(),
        Some(second.build_id.as_str())
    );

    let decisions = list_plan_decisions(&conn, &second.build_id).unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].outcome, OUTCOME_LOCAL_UPDATE);
    assert_eq!(decisions[0].affected_pages, 0);
}

// ---------------------------------------------------------------------------
// §19.2 REPLAN_REQUIRED: brand-new source (unmappable nodes)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn added_source_with_new_nodes_stops_at_replan_required() {
    let workspace = fixture_workspace("replan-unmappable");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let visible_before = visible_wiki(&wiki_dir);

    // A brand-new source creates brand-new knowledge nodes: they have NO
    // previous page ownership, so the change is not localizable (§19.2).
    std::fs::write(
        workspace.join("docs").join("guide").join("search.md"),
        "# Search\n\nThe search index refreshes within five seconds after every write.\n",
    )
    .unwrap();
    let config2 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let err = llm_wiki_compiler::run_build(&workspace, &config2, provider)
        .await
        .unwrap_err();

    assert!(matches!(err, WikiError::ReplanRequired { .. }), "{err}");
    assert_eq!(err.exit_code(), 7);
    let message = err.to_string();
    assert!(message.contains(TRIGGER_UNMAPPABLE_NODE), "{message}");

    // The previous generation stays visible and untouched.
    assert_eq!(visible_wiki(&wiki_dir), visible_before);
    let conn = conn_of(&workspace);
    assert_eq!(
        get_active_build_id(&conn)
            .unwrap()
            .as_ref()
            .map(|b| b.as_str()),
        Some(first.build_id.as_str())
    );
    assert_eq!(build_status(&conn, &first.build_id), "COMPLETED");
    assert_eq!(
        build_status(&conn_of(&workspace), &latest_build_id(&conn)),
        "REPLAN_REQUIRED"
    );

    // The recorded trigger is auditable (§19.2: every judgment is recorded).
    let decisions = list_plan_decisions(&conn, &latest_build_id(&conn)).unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].outcome, OUTCOME_REPLAN_REQUIRED);
    assert_eq!(
        decisions[0].trigger.as_deref(),
        Some(TRIGGER_UNMAPPABLE_NODE)
    );
}

// ---------------------------------------------------------------------------
// §19.2 REPLAN_REQUIRED: BuildFingerprint drift
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fingerprint_drift_stops_at_replan_required() {
    let workspace = fixture_workspace("replan-fingerprint");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let visible_before = visible_wiki(&wiki_dir);

    // Planning-relevant config change → the effective config hash (part of
    // the BuildFingerprint) drifts.
    write_workspace_config(&workspace, "\n[planning]\nmax_cluster_nodes = 12\n");
    let config2 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let err = llm_wiki_compiler::run_build(&workspace, &config2, provider)
        .await
        .unwrap_err();

    assert!(matches!(err, WikiError::ReplanRequired { .. }), "{err}");
    assert_eq!(err.exit_code(), 7);
    let message = err.to_string();
    assert!(message.contains(TRIGGER_FINGERPRINT_CHANGED), "{message}");

    // Previous generation intact.
    assert_eq!(visible_wiki(&wiki_dir), visible_before);
    let conn = conn_of(&workspace);
    assert_eq!(
        get_active_build_id(&conn)
            .unwrap()
            .as_ref()
            .map(|b| b.as_str()),
        Some(first.build_id.as_str())
    );
    assert_eq!(
        build_status(&conn, &latest_build_id(&conn)),
        "REPLAN_REQUIRED"
    );
    let decisions = list_plan_decisions(&conn, &latest_build_id(&conn)).unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].outcome, OUTCOME_REPLAN_REQUIRED);
    assert_eq!(
        decisions[0].trigger.as_deref(),
        Some(TRIGGER_FINGERPRINT_CHANGED)
    );
}

// ---------------------------------------------------------------------------
// build.incremental = false → full V0.1 path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn incremental_disabled_runs_the_full_pipeline() {
    let workspace = fixture_workspace("full-path");
    write_workspace_config(&workspace, "\n[build]\nincremental = false\n");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let ids_first = page_ids_by_slug(&conn_of(&workspace), &first.build_id);

    seed_runtime_doc(
        &workspace,
        "Events are delivered exactly once in v2 and handlers must stay idempotent.",
    );
    let config2 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let second = llm_wiki_compiler::run_build(&workspace, &config2, provider)
        .await
        .unwrap();

    // Full path: the planner re-runs and re-mints page identity.
    assert!(second.incremental.is_none(), "no incremental summary");
    assert!(
        second.llm_request_count >= 4,
        "analysis + planning + compiles all ran ({} requests)",
        second.llm_request_count
    );
    let ids_second = page_ids_by_slug(&conn_of(&workspace), &second.build_id);
    assert_eq!(
        ids_first.len(),
        ids_second.len(),
        "same page set (two pages)"
    );
    assert_ne!(
        ids_first, ids_second,
        "full path re-plans and re-mints page identity"
    );
}

/// The most recent build row (any status) — for asserting the failed build.
fn latest_build_id(conn: &rusqlite::Connection) -> BuildId {
    let id: String = conn
        .query_row(
            "SELECT build_id FROM builds ORDER BY started_at DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    BuildId::from_validated(id)
}

// ---------------------------------------------------------------------------
// §19.2 guard: a COMPLETED build without a fingerprint cannot prove plan
// compatibility → REPLAN_REQUIRED (conservative, §18.1)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completed_build_without_fingerprint_is_treated_as_drift() {
    let workspace = fixture_workspace("null-fingerprint");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let visible_before = visible_wiki(&wiki_dir);

    // Simulate a pre-§19 build row: COMPLETED but never fingerprinted.
    let conn = conn_of(&workspace);
    conn.execute(
        "UPDATE builds SET build_fingerprint = NULL WHERE build_id = ?1",
        params![first.build_id.as_str()],
    )
    .unwrap();

    // The guard runs even with an EMPTY ChangeSet: the fast path must not
    // silently extend a plan whose inputs cannot be proven compatible.
    let err = llm_wiki_compiler::run_build(&workspace, &config, provider)
        .await
        .unwrap_err();
    assert!(matches!(err, WikiError::ReplanRequired { .. }), "{err}");
    assert!(err.to_string().contains(TRIGGER_FINGERPRINT_CHANGED));
    assert_eq!(visible_wiki(&wiki_dir), visible_before);
    let decisions = list_plan_decisions(&conn, &latest_build_id(&conn)).unwrap();
    assert_eq!(decisions.len(), 1);
    assert_eq!(
        decisions[0].trigger.as_deref(),
        Some(TRIGGER_FINGERPRINT_CHANGED)
    );
}

// ---------------------------------------------------------------------------
// §19.3 + §19.2: a restored (re-added) source reactivates its knowledge; the
// missing page makes the change structural → REPLAN_REQUIRED, and a full
// rebuild regains everything (no permanent knowledge loss)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn restored_source_reactivates_knowledge_and_replans_for_the_lost_page() {
    let workspace = fixture_workspace("restore");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let provider = incremental_llm();

    let _first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();

    // Delete the security source: its nodes retire, its page goes obsolete.
    std::fs::remove_file(workspace.join("docs").join("guide").join("security.md")).unwrap();
    let config2 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let second = llm_wiki_compiler::run_build(&workspace, &config2, provider.clone())
        .await
        .unwrap();
    assert_eq!(second.incremental.expect("incremental").obsolete, 1);

    // Restore the file with identical content: the diff classifies it as
    // ADDED (changeset.rs locator semantics), re-analysis re-derives the
    // claims — the registry must reactivate the retired nodes under their
    // stable ids instead of writing claims onto invisible retired nodes.
    std::fs::write(
        workspace.join("docs").join("guide").join("security.md"),
        "# Security\n\nPermissions are enforced at the message bus boundary for every request.\n\n## Audit\n\nEvery permission change lands in the immutable audit log.\n",
    )
    .unwrap();
    let config3 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let err = llm_wiki_compiler::run_build(&workspace, &config3, provider)
        .await
        .unwrap_err();
    assert!(
        matches!(err, WikiError::ReplanRequired { .. }),
        "the security page must be re-created structurally: {err}"
    );

    // The re-derived knowledge is NOT lost: the retired claim nodes came
    // back active with their original ids, and the active claims are
    // visible to the planner again.
    let conn = conn_of(&workspace);
    let invisible: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE status = 'active'
             AND node_id IN (SELECT id FROM knowledge_registry WHERE status = 'retired')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        invisible, 0,
        "no active claim may hang off a retired registry node"
    );
    assert_eq!(
        build_status(&conn, &latest_build_id(&conn)),
        "REPLAN_REQUIRED"
    );
    assert_eq!(
        pointer_of(&wiki_dir).as_deref(),
        Some(second.build_id.as_str())
    );

    // Full rebuild (the documented escape hatch until `replan` lands): the
    // planner sees the reactivated knowledge and regains the security page.
    let state = workspace.join(".llm-wiki");
    let mut config_text = std::fs::read_to_string(state.join("config.toml")).unwrap();
    config_text.push_str("\n[build]\nincremental = false\n");
    std::fs::write(state.join("config.toml"), config_text).unwrap();
    let config_full = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let full = llm_wiki_compiler::run_build(&workspace, &config_full, incremental_llm())
        .await
        .unwrap();
    assert_eq!(full.pages, 2, "the security page is rebuilt");
    let restored_view = load_generation_view(&conn_of(&workspace), &full.build_id).unwrap();
    assert!(
        restored_view
            .iter()
            .any(|page| page.slug == "security-platform"),
        "the restored source's page is back in the visible generation"
    );
}

// ---------------------------------------------------------------------------
// EPIC A PR2: the raw-source index survives incremental publishes
// ---------------------------------------------------------------------------

/// The PR2 gap regression (EPIC A PRD §2.1): an incremental build that changes
/// exactly ONE source must leave EVERY unchanged source searchable through
/// `search_source_fts` after publish. The publish-time activate transaction
/// rebuilds `source_fts` from the ACTIVE build's staging rows only, so without
/// the carry-forward of the unchanged sources' chunk rows this test is RED —
/// the untouched content would silently vanish from the index.
#[tokio::test]
async fn incremental_publish_keeps_unchanged_sources_searchable_in_source_fts() {
    let workspace = fixture_workspace("source-fts-carry");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let provider = incremental_llm();

    let first = llm_wiki_compiler::run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    assert_eq!(first.pages, 2, "runtime + security pages");

    // Modify ONE source (runtime.md); security.md stays byte-identical.
    seed_runtime_doc(
        &workspace,
        "Events are delivered exactly once in v2 and handlers must stay idempotent.",
    );
    let config2 = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let second = llm_wiki_compiler::run_build(&workspace, &config2, provider)
        .await
        .unwrap();
    let summary = second.incremental.expect("incremental summary present");
    assert_eq!(summary.changed, 1, "exactly one source changed");
    assert_eq!(summary.deleted, 0);

    let conn = conn_of(&workspace);
    assert_eq!(
        get_active_build_id(&conn)
            .unwrap()
            .map(|b| b.as_str().to_owned()),
        Some(second.build_id.as_str().to_owned()),
        "the second build is active"
    );

    // THE invariant: the UNCHANGED source's content is still searchable, and
    // every hit serves the ACTIVE build only.
    let hits = llm_wiki_storage::search_source_fts(&conn, "immutable audit log", 10).unwrap();
    assert!(
        !hits.is_empty(),
        "unchanged security.md content must remain searchable after an incremental publish"
    );
    let security_id: String = conn
        .query_row(
            "SELECT source_id FROM sources WHERE rel_path = 'guide/security.md'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        hits.iter()
            .all(|hit| hit.build_id == second.build_id.as_str()),
        "the source index serves the ACTIVE build only"
    );
    assert!(
        hits.iter().any(|hit| hit.source_id.as_str() == security_id),
        "the hit comes from the unchanged source itself"
    );

    // The modified source's NEW content is searchable through the same index.
    let hits = llm_wiki_storage::search_source_fts(&conn, "exactly once in v2", 10).unwrap();
    assert!(
        !hits.is_empty(),
        "the re-staged source's fresh content must be searchable"
    );

    // Staging completeness: the active build holds chunks for BOTH live
    // sources (1 re-staged + 1 carried), and the previous build's staging is
    // untouched (its lifecycle is EPIC H's concern).
    let distinct: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT source_id) FROM source_chunk_text WHERE build_id = ?1",
            params![second.build_id.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        distinct, 2,
        "the active build's staging covers every live source"
    );
    let stale: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM source_chunk_text WHERE build_id = ?1
             AND source_id IN (SELECT source_id FROM sources WHERE rel_path = 'guide/security.md')",
            params![first.build_id.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        stale > 0,
        "the previous build's staging rows persist for EPIC H"
    );
}
