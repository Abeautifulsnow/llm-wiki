//! Integration tests for `llm-wiki replan` (PRD §19.2/§29, V0.2 DoD §53 #8):
//! dry-run audits without compiling or publishing and warms the planner
//! cache; the executed replan keeps semantic-unchanged `WikiPageId`s,
//! records merge/split/retire relations in `page_id_map` and publishes
//! through the unchanged §35 contract. FakeLlm only (PRD §54).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use llm_wiki_compiler::{read_current_pointer, replan, run_build, PublishPaths};
use llm_wiki_core::config::Config;
use llm_wiki_llm::{FakeLlmProvider, LlmError, LlmProvider, LlmRequest};
use llm_wiki_storage::{
    list_plan_decisions, list_recent_plan_decisions_by_outcome, open, OUTCOME_REPLAN_DRY_RUN,
};

// ---------------------------------------------------------------------------
// Fixture: a two-doc workspace + a stage-routing fake whose PLANNING groups
// claims into pages via a MUTABLE span→title map (the test's steering wheel
// for engineering merges/splits/renames between builds).
// ---------------------------------------------------------------------------

const ALPHA_FACTS: [&str; 2] = [
    "alpha fact one for the fixture.",
    "alpha fact two for the fixture.",
];
const BETA_FACTS: [&str; 2] = [
    "beta fact one for the fixture.",
    "beta fact two for the fixture.",
];
const ALPHA_FACT_REWRITTEN: &str = "alpha fact two rewritten for the fixture.";
const ALPHA_FACT_THIRD: &str = "alpha fact three joins the fixture.";

/// One `## Fact N` section per fact — the fake analysis claims each
/// section's FIRST line, so every fact becomes its own knowledge claim.
fn doc_body(facts: &[&str]) -> String {
    let mut body = String::from("# Document\n\n");
    for (index, fact) in facts.iter().enumerate() {
        body.push_str(&format!("## Fact {}\n\n{}\n\n", index + 1, fact));
    }
    body
}

type Grouping = Arc<Mutex<BTreeMap<String, String>>>;

fn temp_workspace(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-replan-{tag}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    let docs = dir.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(docs.join("alpha.md"), doc_body(&ALPHA_FACTS)).unwrap();
    std::fs::write(docs.join("beta.md"), doc_body(&BETA_FACTS)).unwrap();
    let state = dir.join(".llm-wiki");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("config.toml"),
        "[project]\nname = \"replan\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-replan\"\n",
    )
    .unwrap();
    dir
}

/// span → page title grouping: every doc under its own page by default.
fn one_page_per_doc() -> Grouping {
    let mut map = BTreeMap::new();
    for fact in ALPHA_FACTS {
        map.insert(fact.to_owned(), "Alpha".to_owned());
    }
    for fact in BETA_FACTS {
        map.insert(fact.to_owned(), "Beta".to_owned());
    }
    Arc::new(Mutex::new(map))
}

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

/// Every `kn_…` string value in the payload (works for reconcile proposals,
/// whose refs are serialized as `"knowledge_refs":["kn_…", …]`).
fn all_kn_ids_in(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut search_from = 0;
    while let Some(offset) = text[search_from..].find("kn_") {
        let start = search_from + offset;
        let rest = &text[start..];
        let end = rest
            .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .unwrap_or(rest.len());
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

/// The stage-routing fake: analysis quotes section first-lines as claims;
/// planning records only the STABLE id→statement mapping at summary time and
/// resolves each claim's page from the MUTABLE statement→title grouping at
/// LOCAL/RECONCILE time — so a re-plan regroups under the CURRENT grouping
/// even when some planning requests replay from the §28 cache. Compilation
/// cites every claim of the page.
fn grouped_llm(grouping: Grouping) -> Arc<dyn LlmProvider> {
    let id_to_statement: Arc<Mutex<BTreeMap<String, String>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
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
                    "summary": "Replan fixture analysis.",
                    "topics": ["replan"],
                    "entities": [],
                    "concepts": [],
                    "claims": claims,
                    "relations": [],
                })
                .to_string())
            }
            "wiki-planning" => {
                if prompt.contains("Summarize the following cluster") {
                    // Record the STABLE id→statement mapping (statements never
                    // change for a claim id; titles resolve fresh later).
                    let ids = json_string_values(prompt, "id");
                    let statements = json_string_values(prompt, "statement");
                    let mut map = id_to_statement.lock().unwrap();
                    for (id, statement) in ids.into_iter().zip(statements) {
                        map.insert(id, statement);
                    }
                    return Ok(
                        serde_json::json!({"summary": "cluster over the replan fixture"})
                            .to_string(),
                    );
                }
                if prompt.contains("THIS cluster's knowledge") {
                    let ids = kn_ids_in(prompt);
                    return Ok(group_pages(&grouping, &id_to_statement, &ids).to_string());
                }
                if prompt.contains("final global wiki plan") {
                    let ids = all_kn_ids_in(prompt);
                    return Ok(group_pages(&grouping, &id_to_statement, &ids).to_string());
                }
                Err(LlmError::Api {
                    code: 500,
                    message: "unrecognized planning stage".into(),
                })
            }
            "wiki-compilation" => {
                let claims = claim_ids_in(prompt);
                let mut body = String::from("## Overview\n\nThe page cites its claims.\n");
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
    Arc::new(FakeLlmProvider::new("fake-replan", handler))
}

fn group_pages(
    grouping: &Grouping,
    id_to_statement: &Mutex<BTreeMap<String, String>>,
    ids: &[String],
) -> serde_json::Value {
    let statements = id_to_statement.lock().unwrap();
    let grouping = grouping.lock().unwrap();
    let mut by_title: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for id in ids {
        // Titles resolve from the CURRENT grouping at call time — a re-plan
        // regroups under the fresh grouping even when planning requests
        // replay from the §28 cache.
        let title = statements
            .get(id)
            .and_then(|statement| grouping.get(statement))
            .cloned()
            .unwrap_or_else(|| "Unplaced".into());
        by_title.entry(title).or_default().push(id.clone());
    }
    let pages: Vec<serde_json::Value> = by_title
        .into_iter()
        .map(|(title, refs)| {
            serde_json::json!({
                "title": title,
                "category": "concepts",
                "purpose": "cover the grouped knowledge",
                "knowledge_refs": refs,
            })
        })
        .collect();
    serde_json::json!({ "pages": pages })
}

fn pointer_of(wiki_dir: &Path) -> Option<String> {
    let paths = PublishPaths::new(wiki_dir);
    read_current_pointer(&paths)
        .unwrap()
        .map(|pointer| pointer.build_id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dry_run_audits_without_publishing_and_warms_the_planner_cache() {
    let workspace = temp_workspace("dryrun");
    let config = Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let db_path = workspace.join(".llm-wiki").join("state.db");
    let grouping = one_page_per_doc();
    let provider = grouped_llm(grouping.clone());

    let build = run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let pointer_before = pointer_of(&wiki_dir).unwrap();

    // No source changes: force-fresh planning still re-derives the plan, but
    // every request cache-hits from the build (planning paid once, §28).
    let report = replan(&workspace, &config, provider.clone(), true)
        .await
        .unwrap();

    assert!(report.dry_run);
    assert!(report.no_replan_needed, "identical plan: {report:?}");
    assert_eq!(report.unchanged, 2);
    assert_eq!(
        report.llm_request_count, 0,
        "dry-run replans from the warm §28 cache"
    );
    assert_eq!(
        pointer_of(&wiki_dir).unwrap(),
        pointer_before,
        "dry-run must not move the pointer"
    );
    assert_eq!(report.build_id, None, "dry-run owns no build row");
    assert_eq!(report.published_path, None);

    // The audit decision row is anchored to the CURRENT active build.
    let conn = open(&db_path).unwrap();
    let decisions = list_plan_decisions(&conn, &build.build_id).unwrap();
    assert!(
        decisions
            .iter()
            .any(|row| row.outcome == OUTCOME_REPLAN_DRY_RUN),
        "dry-run decision row recorded: {decisions:?}"
    );

    // The warmed cache carries over to an EXECUTE with an identical plan.
    let execute = replan(&workspace, &config, provider, false).await.unwrap();
    assert!(execute.no_replan_needed);
    assert_eq!(
        execute.llm_request_count, 0,
        "execute replans from the dry-run's warm cache"
    );
}

#[tokio::test]
async fn executed_replan_merges_pages_under_the_dominant_predecessor_id() {
    let workspace = temp_workspace("merge");
    let config = Config::load(&workspace).unwrap();
    let wiki_dir = workspace.join("wiki");
    let db_path = workspace.join(".llm-wiki").join("state.db");
    let grouping = one_page_per_doc();
    let provider = grouped_llm(grouping.clone());

    run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let old_conn = open(&db_path).unwrap();
    let old_pointer = pointer_of(&wiki_dir).unwrap();
    let old_pages = llm_wiki_storage::load_generation_view(
        &old_conn,
        &llm_wiki_storage::get_active_build_id(&old_conn)
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let alpha_id = old_pages
        .iter()
        .find(|page| page.title == "Alpha")
        .unwrap()
        .page_id
        .clone();
    let beta_id = old_pages
        .iter()
        .find(|page| page.title == "Beta")
        .unwrap()
        .page_id
        .clone();
    drop(old_conn);

    // Rewrite the second alpha fact (its own section → its own claim) and
    // regroup everything into ONE "Combined" page. The rewritten statement
    // is grouped too, so it lands on the merged page.
    std::fs::write(
        workspace.join("docs").join("alpha.md"),
        doc_body(&[ALPHA_FACTS[0], ALPHA_FACT_REWRITTEN]),
    )
    .unwrap();
    {
        let mut map = grouping.lock().unwrap();
        for fact in ALPHA_FACTS.iter().chain(BETA_FACTS.iter()) {
            map.insert((*fact).to_owned(), "Combined".to_owned());
        }
        map.insert(ALPHA_FACT_REWRITTEN.to_owned(), "Combined".to_owned());
    }

    let report = replan(&workspace, &config, provider, false).await.unwrap();
    assert!(!report.no_replan_needed);
    assert_eq!(report.merged, 1, "{report:?}");
    assert_eq!(report.split, 0);

    let conn = open(&db_path).unwrap();
    let new_active = llm_wiki_storage::get_active_build_id(&conn)
        .unwrap()
        .unwrap();
    assert_ne!(
        new_active.as_str(),
        old_pointer,
        "the merge published a new generation"
    );
    let pages = llm_wiki_storage::load_generation_view(&conn, &new_active).unwrap();
    assert_eq!(pages.len(), 1, "two pages merged into one: {pages:?}");
    // Dominant predecessor: Beta (both beta facts unchanged → overlap 2 vs
    // Alpha's 1, since the rewritten alpha fact is a NEW node).
    assert_eq!(
        pages[0].page_id, beta_id,
        "§45: the merged page keeps the DOMINANT predecessor's id"
    );

    // page_id_map: both predecessors recorded; Beta's row is the dominant
    // one (predecessor == successor), Alpha's row points at Beta.
    let rows = llm_wiki_storage::list_page_id_maps_for_build(&conn, &new_active).unwrap();
    let beta_row = rows
        .iter()
        .find(|row| row.predecessor_page_id.as_ref() == Some(&beta_id))
        .expect("beta merge row");
    assert_eq!(beta_row.kind, "merge");
    assert_eq!(
        beta_row.successor_page_id,
        Some(beta_id.clone()),
        "dominant row: pred == succ"
    );
    let alpha_row = rows
        .iter()
        .find(|row| row.predecessor_page_id.as_ref() == Some(&alpha_id))
        .expect("alpha merge row");
    assert_eq!(alpha_row.kind, "merge");
    assert_eq!(alpha_row.successor_page_id, Some(beta_id.clone()));
}

#[tokio::test]
async fn executed_replan_splits_pages_with_fresh_ids_and_records_successors() {
    let workspace = temp_workspace("split");
    let config = Config::load(&workspace).unwrap();
    let db_path = workspace.join(".llm-wiki").join("state.db");
    let grouping = one_page_per_doc();
    {
        // Build: everything in ONE page.
        let mut map = grouping.lock().unwrap();
        for fact in BETA_FACTS {
            map.insert(fact.to_owned(), "Alpha".to_owned());
        }
    }
    let provider = grouped_llm(grouping.clone());

    run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let old_conn = open(&db_path).unwrap();
    let old_pages = llm_wiki_storage::load_generation_view(
        &old_conn,
        &llm_wiki_storage::get_active_build_id(&old_conn)
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let alpha_id = old_pages[0].page_id.clone();
    assert_eq!(old_pages.len(), 1);
    drop(old_conn);

    // Append a third alpha fact (planning input changes → cache re-asks)
    // and regroup: Alpha splits into two fresh pages.
    std::fs::write(
        workspace.join("docs").join("alpha.md"),
        doc_body(&[ALPHA_FACTS[0], ALPHA_FACTS[1], ALPHA_FACT_THIRD]),
    )
    .unwrap();
    {
        let mut map = grouping.lock().unwrap();
        for fact in ALPHA_FACTS {
            map.insert(fact.to_owned(), "Alpha One".to_owned());
        }
        map.insert(ALPHA_FACT_THIRD.to_owned(), "Alpha One".to_owned());
        for fact in BETA_FACTS {
            map.insert(fact.to_owned(), "Alpha Two".to_owned());
        }
    }

    let report = replan(&workspace, &config, provider, false).await.unwrap();
    assert_eq!(report.split, 1, "{report:?}");
    // The single old page splits; nothing is plain-retired and no knowledge
    // is lost (every old ref is covered by a successor).
    assert_eq!(report.retired, 0, "{report:?}");
    assert!(report.knowledge_loss_warnings.is_empty(), "{report:?}");

    let conn = open(&db_path).unwrap();
    let new_active = llm_wiki_storage::get_active_build_id(&conn)
        .unwrap()
        .unwrap();
    let pages = llm_wiki_storage::load_generation_view(&conn, &new_active).unwrap();
    assert_eq!(pages.len(), 2);
    assert!(
        pages.iter().all(|page| page.page_id != alpha_id),
        "split successors get FRESH ids, never the old one"
    );

    let rows = llm_wiki_storage::list_page_id_maps_for_build(&conn, &new_active).unwrap();
    let split_rows: Vec<_> = rows
        .iter()
        .filter(|row| row.predecessor_page_id.as_ref() == Some(&alpha_id))
        .collect();
    assert_eq!(split_rows.len(), 2, "both successors recorded");
    assert!(split_rows.iter().all(|row| row.kind == "split"));
    let successors: Vec<_> = split_rows
        .iter()
        .filter_map(|row| row.successor_page_id.clone())
        .collect();
    assert!(pages.iter().all(|page| successors.contains(&page.page_id)));
}

#[tokio::test]
async fn executed_replan_keeps_modified_ids_and_carries_unchanged_pages_verbatim() {
    let workspace = temp_workspace("modified");
    let config = Config::load(&workspace).unwrap();
    let db_path = workspace.join(".llm-wiki").join("state.db");
    let grouping = one_page_per_doc();
    let provider = grouped_llm(grouping.clone());

    run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    let old_conn = open(&db_path).unwrap();
    let old_pages = llm_wiki_storage::load_generation_view(
        &old_conn,
        &llm_wiki_storage::get_active_build_id(&old_conn)
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let alpha_old = old_pages
        .iter()
        .find(|p| p.title == "Alpha")
        .unwrap()
        .clone();
    let beta_old = old_pages
        .iter()
        .find(|p| p.title == "Beta")
        .unwrap()
        .clone();
    drop(old_conn);

    // Rewrite the second alpha fact: Alpha's page content changes (modified,
    // same title → keeps its id); Beta and its knowledge are untouched
    // (unchanged, carried byte-identical). The new statement joins Alpha's
    // grouping so it lands on Alpha, not on an "Unplaced" page.
    std::fs::write(
        workspace.join("docs").join("alpha.md"),
        doc_body(&[ALPHA_FACTS[0], ALPHA_FACT_REWRITTEN]),
    )
    .unwrap();
    grouping
        .lock()
        .unwrap()
        .insert(ALPHA_FACT_REWRITTEN.to_owned(), "Alpha".to_owned());

    let report = replan(&workspace, &config, provider, false).await.unwrap();
    assert_eq!(report.modified, 1, "{report:?}");
    assert_eq!(report.unchanged, 1);
    assert_eq!(report.recompiled, 1);
    assert_eq!(report.carried, 1);

    let conn = open(&db_path).unwrap();
    let new_active = llm_wiki_storage::get_active_build_id(&conn)
        .unwrap()
        .unwrap();
    let pages = llm_wiki_storage::load_generation_view(&conn, &new_active).unwrap();
    assert_eq!(pages.len(), 2);

    // Modified page: SAME WikiPageId, recompiled (title unchanged here, but
    // the body reflects the rewritten fact).
    let alpha_new = pages
        .iter()
        .find(|p| p.page_id == alpha_old.page_id)
        .unwrap();
    assert_eq!(alpha_new.title, "Alpha");
    assert_ne!(
        alpha_new.body_hash, alpha_old.body_hash,
        "modified page was recompiled"
    );

    // Unchanged page: byte-identical carry (content, refs, citations, links).
    let beta_new = pages
        .iter()
        .find(|p| p.page_id == beta_old.page_id)
        .unwrap();
    assert_eq!(beta_new.title, "Beta");
    assert_eq!(
        beta_new.content, beta_old.content,
        "unchanged page carried verbatim"
    );
    assert_eq!(beta_new.body_hash, beta_old.body_hash);
    assert_eq!(beta_new.knowledge_refs, beta_old.knowledge_refs);
    assert_eq!(beta_new.citations.len(), beta_old.citations.len());

    // keep rows recorded for both continued identities.
    let rows = llm_wiki_storage::list_page_id_maps_for_build(&conn, &new_active).unwrap();
    assert!(
        rows.iter()
            .any(|row| row.kind == "keep"
                && row.predecessor_page_id == Some(beta_old.page_id.clone())),
        "keep row for the carried page: {rows:?}"
    );
}

#[tokio::test]
async fn replan_without_a_published_generation_errors() {
    let workspace = temp_workspace("cold");
    let config = Config::load(&workspace).unwrap();
    let provider = grouped_llm(one_page_per_doc());

    let err = replan(&workspace, &config, provider, true)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("nothing to replan"),
        "actionable error: {err}"
    );
}

#[tokio::test]
async fn dry_run_records_pending_replan_triggers_in_the_report() {
    // Seed a replan-required decision (as a failed incremental build would),
    // then assert the dry-run surfaces it as a trigger.
    let workspace = temp_workspace("triggers");
    let config = Config::load(&workspace).unwrap();
    let grouping = one_page_per_doc();
    let provider = grouped_llm(grouping.clone());
    let db_path = workspace.join(".llm-wiki").join("state.db");

    let build = run_build(&workspace, &config, provider.clone())
        .await
        .unwrap();
    {
        let mut conn = open(&db_path).unwrap();
        llm_wiki_storage::insert_plan_decision(
            &mut conn,
            &llm_wiki_storage::PlanDecision {
                build_id: build.build_id.clone(),
                source_id: None,
                outcome: "replan-required".to_owned(),
                trigger: Some("structural-change".to_owned()),
                affected_pages: 1,
                notes: "seeded trigger for the audit test".to_owned(),
            },
        )
        .unwrap();
    }

    let report = replan(&workspace, &config, provider, true).await.unwrap();
    assert!(
        report
            .triggers
            .iter()
            .any(|trigger| trigger.contains("seeded trigger")),
        "pending replan-required decision surfaced as a trigger: {:?}",
        report.triggers
    );

    let conn = open(&db_path).unwrap();
    let decisions =
        list_recent_plan_decisions_by_outcome(&conn, OUTCOME_REPLAN_DRY_RUN, 10).unwrap();
    assert!(!decisions.is_empty(), "dry-run audit row recorded");
}
