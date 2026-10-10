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
    let context = build_context(&conn, "retries sso", &budget, &[]).unwrap();
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
    let context = build_context(&conn, "retries sso", &budget, &[]).unwrap();
    let selection_tokens = llm_wiki_search::chunks_tokens(&context.chunks);
    assert!(
        selection_tokens <= 20,
        "chunk selection exceeds the budget: {selection_tokens}"
    );
    // The budget demonstrably shrank the selection versus an unbounded one.
    let unbounded = build_context(&conn, "retries sso", &ContextBudget::default(), &[]).unwrap();
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
    let context = build_context(&conn, "retries sso", &budget, &[]).unwrap();
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
    let context = build_context(&conn, "retries sso", &ContextBudget::default(), &[]).unwrap();
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
    let a = build_context(&conn, "retries sso", &ContextBudget::default(), &[]).unwrap();
    let b = build_context(&conn, "retries sso", &ContextBudget::default(), &[]).unwrap();
    assert_eq!(a, b);
}

/// Nothing published is an honest error.
#[test]
fn without_an_active_build_the_builder_errors() {
    let (mut conn, _) = published("noactive");
    set_active_build(&mut conn, None).unwrap();
    let _ = get_active_build_id(&conn).unwrap();
    let error = build_context(&conn, "retries", &ContextBudget::default(), &[]).unwrap_err();
    assert!(error.to_string().contains("nothing published"));
}

/// Vector-layer fusion (§19.3): a vector candidate whose section has NO
/// lexical hit still enters the context; an unknown hash is skipped, never
/// guessed about; lexical-only behavior is unchanged with an empty list.
#[test]
fn vector_candidates_fuse_with_lexical() {
    let (conn, _) = published("fusion");
    // Collect the fixture's real section hashes.
    let sections = llm_wiki_search::active_context_sections(&conn).unwrap();
    assert!(sections.len() >= 2);
    let other = sections
        .iter()
        .find(|section| section.slug == "ctx-sso")
        .expect("fixture sso section");

    // Query only matches the runtime section lexically; the sso section
    // arrives as a VECTOR candidate and must appear in the context.
    let vector = vec![llm_wiki_search::VectorCandidate {
        text_hash: other.text_hash.clone(),
        score: 0.9,
    }];
    let context = build_context(&conn, "retries", &ContextBudget::default(), &vector).unwrap();
    let slugs: Vec<&str> = context
        .chunks
        .iter()
        .map(|chunk| chunk.slug.as_str())
        .collect();
    assert!(
        slugs.contains(&"ctx-sso"),
        "vector candidate included: {slugs:?}"
    );

    // Unknown hashes are skipped without killing the lexical path.
    let stale = vec![llm_wiki_search::VectorCandidate {
        text_hash: "stale-hash".into(),
        score: 1.0,
    }];
    let context = build_context(&conn, "retries", &ContextBudget::default(), &stale).unwrap();
    assert!(
        context
            .chunks
            .iter()
            .any(|chunk| chunk.slug == "ctx-runtime"),
        "lexical retrieval survives stale vector candidates"
    );

    // Empty vector list == pure lexical behavior.
    let lexical_only = build_context(&conn, "retries", &ContextBudget::default(), &[]).unwrap();
    assert!(lexical_only
        .chunks
        .iter()
        .any(|chunk| chunk.slug == "ctx-runtime"));
}

/// The section hash is stable and tied to the ONE text definition.
#[test]
fn context_section_hash_is_content_addressed() {
    let a = llm_wiki_search::context_section_hash("T", &["H".into()], "body");
    let b = llm_wiki_search::context_section_hash("T", &["H".into()], "body");
    let c = llm_wiki_search::context_section_hash("T", &["H".into()], "other");
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ---------------------------------------------------------------------------
// EPIC A PR3 — the fused context builder (source entries + degradation)
// ---------------------------------------------------------------------------

use llm_wiki_search::{
    build_context_with_sources, retrieve, AssembledContext, ContextChunk, EvidenceKind, SourceMode,
};
use llm_wiki_storage::chunks::{rebuild_source_fts, replace_source_chunks, ChunkInput};

/// Stages raw-source chunks for the ACTIVE build and rebuilds the source
/// index (the publish flow does this inside the activate transaction; the
/// fixture stages after activation).
fn stage_chunks(
    conn: &mut rusqlite::Connection,
    rel_path: &str,
    chunks: &[ChunkInput<'_>],
) -> llm_wiki_core::ids::SourceId {
    let (source_id, _) = upsert_source(
        conn,
        &SourceLocatorKey::compute("ws", rel_path),
        rel_path,
        "hash",
        10,
        None,
    )
    .unwrap();
    let build = get_active_build_id(conn).unwrap().unwrap();
    replace_source_chunks(conn, &source_id, &build, rel_path, None, chunks).unwrap();
    rebuild_source_fts(conn, &build).unwrap();
    source_id
}

fn chunk<'a>(
    title: &'a str,
    heading_path: &'a [String],
    body: &'a str,
    ordinal: usize,
) -> ChunkInput<'a> {
    ChunkInput {
        title,
        heading_path,
        ordinal,
        range_start: ordinal * 100,
        range_end: ordinal * 100 + body.len(),
        body,
    }
}

fn hp(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| s.to_string()).collect()
}

/// The published fixture plus raw-source chunks: one source backing the
/// runtime topic and one burst source with three matching chunks (for the
/// per-source cap).
fn published_with_sources(tag: &str) -> (rusqlite::Connection, BuildId) {
    let (mut conn, build) = published(tag);
    stage_chunks(
        &mut conn,
        &format!("docs/{tag}-runtime.md"),
        &[chunk(
            "Retries",
            &hp(&["Runtime"]),
            "The scheduler retries failed tasks three times.",
            0,
        )],
    );
    stage_chunks(
        &mut conn,
        &format!("docs/{tag}-burst.md"),
        &[
            chunk("Burst", &hp(&["Burst"]), "retries one", 0),
            chunk("Burst", &hp(&["Burst"]), "retries two", 1),
            chunk("Burst", &hp(&["Burst"]), "retries three", 2),
        ],
    );
    (conn, build)
}

/// Source entries flow through the budgeted selection as first-class
/// candidates: evidence kind, source ref and the corpus identity on the
/// chunk are all populated.
#[test]
fn source_entries_flow_through_the_context_pipeline() {
    let (conn, _) = published_with_sources("ctxsrc");
    let fused = retrieve(&conn, "retries", SourceMode::Source, 10).unwrap();
    assert_eq!(fused.served.source, llm_wiki_search::SideStatus::Served);

    let context = build_context_with_sources(
        &conn,
        "retries",
        &ContextBudget::default(),
        &[],
        None,
        Some(&fused),
    )
    .unwrap();

    assert!(!context.chunks.is_empty());
    for chunk in &context.chunks {
        assert_eq!(chunk.evidence_kind, EvidenceKind::Source);
        let source_ref = chunk.source_ref.as_ref().expect("source ref populated");
        assert_eq!(chunk.slug, source_ref.file_path);
        assert_eq!(chunk.page_id.as_str(), source_ref.source_id.as_str());
        assert_eq!(chunk.sources, vec![source_ref.file_path.clone()]);
    }
    assert!(context.chunks.iter().any(|chunk| chunk
        .source_ref
        .as_ref()
        .unwrap()
        .file_path
        .ends_with("runtime.md")));
    assert_eq!(context.served_mode, SourceMode::Source);
    let degraded = context
        .degraded
        .expect("fused contexts carry side metadata");
    assert_eq!(degraded.wiki, llm_wiki_search::SideStatus::Disabled);
    assert_eq!(degraded.source, llm_wiki_search::SideStatus::Served);
}

/// mode=Source works when the wiki index is unavailable: the source-only
/// context assembles instead of erroring, with the degradation recorded.
#[test]
fn source_mode_contexts_survive_a_missing_wiki_index() {
    let (conn, _) = published_with_sources("nowiki");
    conn.execute("DROP TABLE wiki_fts", []).unwrap();

    let fused = retrieve(&conn, "retries", SourceMode::Source, 10).unwrap();
    let context = build_context_with_sources(
        &conn,
        "retries",
        &ContextBudget::default(),
        &[],
        None,
        Some(&fused),
    )
    .unwrap();
    assert!(!context.chunks.is_empty());
    assert!(context
        .chunks
        .iter()
        .all(|chunk| chunk.evidence_kind == EvidenceKind::Source));
}

/// The honest-error side of the degradation matrix: nothing matched on ANY
/// side → an error that names each side's status (never a silent empty).
#[test]
fn both_sides_empty_is_an_honest_error_naming_the_sides() {
    let (conn, _) = published_with_sources("empty");
    let fused = retrieve(&conn, "zzqqxx", SourceMode::Fusion, 10).unwrap();
    assert!(fused.entries.is_empty());
    let error = build_context_with_sources(
        &conn,
        "zzqqxx",
        &ContextBudget::default(),
        &[],
        None,
        Some(&fused),
    )
    .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("wiki side"), "{message}");
    assert!(message.contains("source side"), "{message}");
    assert!(message.contains("NoMatches"), "{message}");

    // Nothing published at all: the same honesty, different statuses.
    let (mut conn, _) = published_with_sources("nopub");
    set_active_build(&mut conn, None).unwrap();
    let fused = retrieve(&conn, "retries", SourceMode::Source, 10).unwrap();
    let error = build_context_with_sources(
        &conn,
        "retries",
        &ContextBudget::default(),
        &[],
        None,
        Some(&fused),
    )
    .unwrap_err();
    assert!(error.to_string().contains("NotPublished"), "{error}");
}

/// `max_per_source` counts source chunks by their `source_id` (wiki chunks
/// keep counting by cited source path): a burst source contributes at most
/// the cap even when more of its chunks match.
#[test]
fn max_per_source_caps_source_chunks_by_source_id() {
    let (conn, _) = published_with_sources("cap");
    let fused = retrieve(&conn, "retries", SourceMode::Source, 10).unwrap();
    let budget = ContextBudget {
        max_chunks: 8,
        max_pages: 8,
        max_per_source: 2,
        ..ContextBudget::default()
    };
    let context =
        build_context_with_sources(&conn, "retries", &budget, &[], None, Some(&fused)).unwrap();
    let mut per_source: BTreeMap<String, usize> = BTreeMap::new();
    for chunk in &context.chunks {
        let id = chunk
            .source_ref
            .as_ref()
            .unwrap()
            .source_id
            .as_str()
            .to_owned();
        *per_source.entry(id).or_default() += 1;
    }
    assert_eq!(
        per_source.values().max().copied(),
        Some(2),
        "the burst source must be capped: {per_source:?}"
    );
}

/// The legacy wiki path and the fused mode=Wiki path agree on the selection:
/// same chunks in the same order — and, for a lexical-only query, the same
/// SCORES (level-2 RRF assigns 1/(k + position), and position == lexical
/// rank when no vector side reorders the wiki list, so the numbers match the
/// legacy level-1 RRF exactly, not just the order).
#[test]
fn fused_wiki_path_matches_the_legacy_path() {
    let (conn, _) = published_with_sources("parity");
    let legacy = build_context(&conn, "retries sso", &ContextBudget::default(), &[]).unwrap();
    let fused = retrieve(&conn, "retries sso", SourceMode::Wiki, 36).unwrap();
    let via_fused = build_context_with_sources(
        &conn,
        "retries sso",
        &ContextBudget::default(),
        &[],
        None,
        Some(&fused),
    )
    .unwrap();
    let identity = |context: &AssembledContext| -> Vec<(String, Vec<String>, String)> {
        context
            .chunks
            .iter()
            .map(|chunk: &ContextChunk| {
                (
                    chunk.slug.clone(),
                    chunk.heading_path.clone(),
                    chunk.snippet.clone(),
                )
            })
            .collect()
    };
    assert_eq!(identity(&legacy), identity(&via_fused));
    assert_eq!(legacy.neighbors, via_fused.neighbors);
    assert_eq!(legacy.dropped, via_fused.dropped);
    // Numerical parity, not just order parity: lexical-only queries score
    // identically on both paths (1/(RRF_K + rank) either way).
    let scores = |context: &AssembledContext| -> Vec<f32> {
        context.chunks.iter().map(|chunk| chunk.score).collect()
    };
    assert_eq!(scores(&legacy), scores(&via_fused));
    // All wiki chunks default their PR3 fields on both paths.
    for chunk in legacy.chunks.iter().chain(via_fused.chunks.iter()) {
        assert_eq!(chunk.evidence_kind, EvidenceKind::Wiki);
        assert!(chunk.source_ref.is_none());
    }
    // Legacy transparency: wiki mode, no side metadata.
    assert_eq!(legacy.served_mode, SourceMode::Wiki);
    assert!(legacy.degraded.is_none());
}
