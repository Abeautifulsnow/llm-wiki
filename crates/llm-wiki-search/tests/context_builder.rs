//! Integration tests for the Context Builder (§19.3): diversity caps, token
//! budget, graph neighborhood and determinism, over a real published
//! generation assembled through the storage layer (no compiler involved).

use std::collections::BTreeMap;

use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::{BuildId, SourceLocatorKey};
use llm_wiki_search::{build_context, ContextBudget};
use llm_wiki_storage::{
    get_active_build_id, get_or_create_batch, persist_generation, set_active_build, upsert_source,
    BuildDraft, NodeDraft, NodeKind, PageCitationRecord, PageLinkRecord, WikiPageRecord,
};
use rusqlite::params;

/// One published generation: two sources, two claims (one anchored per
/// source), two pages citing one claim each, page A linking to page B, and
/// one analysis relation entity→concept bridged by the pages' refs.
fn published(tag: &str) -> (rusqlite::Connection, BuildId) {
    let mut conn = llm_wiki_storage::open_in_memory().unwrap();
    let (src_a, _) = upsert_source(
        &mut conn,
        &SourceLocatorKey::compute("ws", &format!("docs/{tag}-runtime.md")),
        &format!("docs/{tag}-runtime.md"),
        "hash-a",
        10,
        None,
    )
    .unwrap();
    let (src_b, _) = upsert_source(
        &mut conn,
        &SourceLocatorKey::compute("ws", &format!("docs/{tag}-security.md")),
        &format!("docs/{tag}-security.md"),
        "hash-b",
        10,
        None,
    )
    .unwrap();

    let drafts = vec![
        NodeDraft {
            kind: NodeKind::Claim,
            canonical_key: "retries".into(),
            canonical_name: "claim retries".into(),
            entity_type: None,
            description: None,
        },
        NodeDraft {
            kind: NodeKind::Claim,
            canonical_key: "sso".into(),
            canonical_name: "claim sso".into(),
            entity_type: None,
            description: None,
        },
        NodeDraft {
            kind: NodeKind::Entity,
            canonical_key: "sso entity".into(),
            canonical_name: "SSO".into(),
            entity_type: None,
            description: None,
        },
        NodeDraft {
            kind: NodeKind::Concept,
            canonical_key: "single sign-on".into(),
            canonical_name: "Single Sign-On".into(),
            entity_type: None,
            description: None,
        },
    ];
    let ids = get_or_create_batch(&mut conn, &drafts, None).unwrap();
    let (claim_a, claim_b, entity, concept) = (
        ids[0].clone(),
        ids[1].clone(),
        ids[2].clone(),
        ids[3].clone(),
    );

    // Sections so FTS has something to bite on.
    for (section_id, source, heading) in [
        ("sec_ctx_a", src_a.clone(), "[\"Runtime\",\"Retries\"]"),
        ("sec_ctx_b", src_b.clone(), "[\"Security\",\"SSO\"]"),
    ] {
        conn.execute(
            "INSERT INTO source_sections (section_id, source_id, heading_path_json, heading_path_key, content_fingerprint, range_start, range_end, status)
             VALUES (?1, ?2, ?3, ?3, 'fp', 0, 100, 'active')",
            params![section_id, source.as_str(), heading],
        )
        .unwrap();
    }
    conn.execute(
        "INSERT INTO document_analyses (analysis_id, source_id, status, created_at)
         VALUES ('an_ctx', ?1, 'completed', '2026-01-01')",
        params![src_a.as_str()],
    )
    .unwrap();
    for (claim_id, node, source, section, statement) in [
        (
            "cl_ctx_a",
            claim_a.as_str(),
            src_a.as_str(),
            "sec_ctx_a",
            "The scheduler retries failed tasks three times.",
        ),
        (
            "cl_ctx_b",
            claim_b.as_str(),
            src_b.as_str(),
            "sec_ctx_b",
            "Single sign-on is enforced at the gateway.",
        ),
    ] {
        conn.execute(
            "INSERT INTO claims (claim_id, node_id, source_id, section_id, analysis_id, statement, evidence_digest, status)
             VALUES (?1, ?2, ?3, ?4, 'an_ctx', ?5, 'digest', 'active')",
            params![claim_id, node, source, section, statement],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO citations (citation_id, owner_kind, owner_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json)
             VALUES ('cit_' || ?1, 'claim', ?1, ?2, ?3, 0, 50, 'hash', 'digest', '[\"H\"]')",
            params![claim_id, source, section],
        )
        .unwrap();
    }

    let page = |slug: &str,
                title: &str,
                body: &str,
                refs: &[String],
                links: Vec<&WikiPageRecord>| WikiPageRecord {
        page_id: llm_wiki_core::ids::WikiPageId::generate(),
        slug: slug.to_owned(),
        title: title.to_owned(),
        category: "concepts".into(),
        language: "en".into(),
        body_hash: sha256_hex(slug.as_bytes()),
        content: format!("# {title}\n\n{body}\n"),
        knowledge_refs: refs
            .iter()
            .map(|id| llm_wiki_core::ids::KnowledgeNodeId::from_validated(id.clone()))
            .collect(),
        citations: refs
            .iter()
            .map(|id| PageCitationRecord {
                claim_node_id: llm_wiki_core::ids::KnowledgeNodeId::from_validated(id.clone()),
                source_id: if id == claim_a.as_str() {
                    src_a.clone()
                } else {
                    src_b.clone()
                },
                section_id: None,
                range: llm_wiki_core::model::SourceRange { start: 0, end: 50 },
                source_hash: "hash".into(),
                evidence_digest: "digest".into(),
                heading_path: vec!["H".into()],
            })
            .collect(),
        links: links
            .into_iter()
            .map(|prev| PageLinkRecord {
                to_page_id: prev.page_id.clone(),
                target_title: prev.title.clone(),
            })
            .collect(),
    };
    let page_runtime = page(
        "ctx-runtime",
        "Runtime",
        "retries content",
        &[claim_a.as_str().to_owned()],
        vec![],
    );
    let page_sso = page(
        "ctx-sso",
        "Security",
        "sso content",
        &[
            claim_b.as_str().to_owned(),
            entity.as_str().to_owned(),
            concept.as_str().to_owned(),
        ],
        vec![&page_runtime],
    );
    let build = {
        let build = start_build(&mut conn);
        let pages = vec![page_runtime, page_sso];
        persist_generation(&mut conn, &build, &pages).unwrap();
        // One call = pointer flip + FTS rebuild + graph rebuild (§35 step 6).
        llm_wiki_storage::activate_build_with_search_index(
            &mut conn,
            &build,
            llm_wiki_storage::default_tokenizer(),
        )
        .unwrap();
        build
    };
    (conn, build)
}

fn start_build(conn: &mut rusqlite::Connection) -> BuildId {
    llm_wiki_storage::start_build(conn, &BuildDraft::default()).unwrap()
}

/// Page diversity: both pages match, but max_pages = 1 keeps every chunk on
/// the rank-best page and reports what it dropped.
#[test]
fn page_diversity_cap_confines_selection() {
    let (conn, _) = published("diversity");
    let budget = ContextBudget {
        max_chunks: 8,
        max_pages: 1,
        ..ContextBudget::default()
    };
    let context = build_context(&conn, "retries sso", &budget).unwrap();
    assert!(!context.chunks.is_empty());
    let pages: std::collections::BTreeSet<&str> = context
        .chunks
        .iter()
        .map(|chunk| chunk.slug.as_str())
        .collect();
    assert!(pages.len() <= 1, "page diversity cap violated: {pages:?}");
    assert!(context.dropped > 0, "the cap must drop candidates");
}

/// Token budget: a tiny ceiling shrinks the selection and is never exceeded.
#[test]
fn token_budget_limits_the_selection() {
    let (conn, _) = published("tokens");
    let budget = ContextBudget {
        max_chunks: 8,
        max_tokens: 20,
        ..ContextBudget::default()
    };
    let context = build_context(&conn, "retries sso", &budget).unwrap();
    let selection_tokens = llm_wiki_search::chunks_tokens(&context.chunks);
    assert!(
        selection_tokens <= 20,
        "chunk selection exceeds the budget: {selection_tokens}"
    );
    // The budget demonstrably shrank the selection versus an unbounded one.
    let unbounded = build_context(&conn, "retries sso", &ContextBudget::default()).unwrap();
    assert!(
        selection_tokens < llm_wiki_search::chunks_tokens(&unbounded.chunks),
        "the budget must drop something: {selection_tokens} vs {}",
        llm_wiki_search::chunks_tokens(&unbounded.chunks)
    );
}

/// Source diversity: both pages are capped per source path; with one chunk
/// allowed per source, a two-page context draws at most one chunk from each.
#[test]
fn source_diversity_spreads_across_sources() {
    let (conn, _) = published("sources");
    let budget = ContextBudget {
        max_chunks: 8,
        max_pages: 4,
        max_per_source: 1,
        ..ContextBudget::default()
    };
    let context = build_context(&conn, "retries sso", &budget).unwrap();
    let mut per_source: BTreeMap<String, usize> = BTreeMap::new();
    for chunk in &context.chunks {
        if let Some(source) = chunk.sources.first() {
            *per_source.entry(source.clone()).or_default() += 1;
        }
    }
    assert!(
        per_source.values().all(|count| *count <= 1),
        "source diversity cap violated: {per_source:?}"
    );
}

/// Graph neighborhood: the selected pages reach their link target AND — via
/// the FIX-009 bridge — the contained entity/concept nodes.
#[test]
fn graph_neighborhood_reaches_pages_and_semantics() {
    let (conn, build) = published("graph");
    let _ = build;
    let context = build_context(&conn, "retries sso", &ContextBudget::default()).unwrap();
    let labels: Vec<&str> = context.neighbors.iter().map(|n| n.label.as_str()).collect();
    assert!(
        labels.contains(&"Security"),
        "the links_to neighbor is included: {labels:?}"
    );
    assert!(
        labels.contains(&"SSO") || labels.contains(&"Single Sign-On"),
        "the contains bridge reaches the semantic component: {labels:?}"
    );
    assert!(context
        .neighbors
        .iter()
        .all(|n| matches!(n.relation.as_str(), "links_to" | "contains")));
}

/// Determinism: identical inputs produce identical assemblies.
#[test]
fn assembly_is_deterministic() {
    let (conn, _) = published("determinism");
    let a = build_context(&conn, "retries sso", &ContextBudget::default()).unwrap();
    let b = build_context(&conn, "retries sso", &ContextBudget::default()).unwrap();
    assert_eq!(a, b);
}

/// Nothing published is an honest error.
#[test]
fn without_an_active_build_the_builder_errors() {
    let (mut conn, _) = published("noactive");
    set_active_build(&mut conn, None).unwrap();
    let _ = get_active_build_id(&conn).unwrap();
    let error = build_context(&conn, "retries", &ContextBudget::default()).unwrap_err();
    assert!(error.to_string().contains("nothing published"));
}
