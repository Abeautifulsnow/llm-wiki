//! Top-level retrieval fusion (EPIC A PR3): raw-source chunks and wiki
//! sections as two first-class evidence sides.
//!
//! Two-level design (locked review decision): the wiki side keeps its own
//! lexical+vector RRF inside `context.rs` — untouched, so mode=Wiki behavior
//! stays byte-identical — and this module runs the TOP-LEVEL merge between
//! the source side (`source_fts` via `search_source_fts`) and the wiki side
//! (`wiki_fts` via `search_index`) with the same [`RRF_K`] constant. Dedup
//! keys are evidence-kind-scoped: wiki identity = (page_id, heading_path),
//! source identity = (source_id, range) — the same content hit on both sides
//! is kept as TWO entries, each weighted by its own side's rank, so dual-side
//! content naturally rises above single-side content.
//!
//! Degradation is part of the result, never silent: every side reports WHY it
//! did or did not serve ([`SideStatus`]). Exact-match protection fronts
//! source chunks whose title/heading/body contains every query token
//! verbatim — all normalized by the SAME shared tokenizer (PRD §20: no second
//! normalization), deterministic and testable.
//!
//! Dependency direction unchanged: search → {core, storage}.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use llm_wiki_core::error::Result;
use llm_wiki_core::ids::WikiPageId;
use llm_wiki_storage::chunks::{search_source_fts, source_fts_exists, SourceFtsHit};
use llm_wiki_storage::search_index::{self, SearchIndexRow};
use llm_wiki_storage::{get_active_build_id, Connection};

/// The Reciprocal Rank Fusion constant shared by BOTH fusion levels: the
/// wiki-internal lexical+vector fusion in `context.rs` and this module's
/// top-level source↔wiki merge. Public so callers and future config can tune
/// one knob instead of redefining two constants (PR3 review decision: same
/// k=60 to start, single public constant).
pub const RRF_K: f64 = 60.0;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Which side(s) the top-level retrieval draws from. The default for every
/// pre-existing caller is [`SourceMode::Wiki`] — behavior-identical to the
/// pre-fusion path (PR3 wires no request field or config key; that is PR4).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceMode {
    /// Raw-source chunks only (`source_fts`).
    Source,
    /// Wiki sections only — the historical default.
    #[default]
    Wiki,
    /// Both sides, merged by the top-level RRF.
    Fusion,
}

/// Which corpus side one evidence entry came from. Dedup keys are scoped by
/// this kind, so both kinds coexist in one result by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvidenceKind {
    /// A raw-source chunk (the original documents).
    Source,
    /// A compiled wiki section of the ACTIVE generation.
    Wiki,
}

/// Location of one source-side evidence entry — identity and provenance of a
/// raw-source chunk (identity = `(source_id, range_start, range_end)`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub source_id: llm_wiki_core::ids::SourceId,
    /// Source-relative path.
    pub file_path: String,
    pub heading_path: Vec<String>,
    /// Position of the chunk within its section (0-based).
    pub ordinal: usize,
    /// Absolute offsets into the normalized source text.
    pub range_start: usize,
    pub range_end: usize,
}

/// Why one side did or did not serve. Degradation MUST be visible — the
/// fused result always carries these statuses, never a silent empty side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideStatus {
    /// The side ran and contributed entries.
    Served,
    /// Nothing is published (no ACTIVE build) — the side cannot serve.
    NotPublished,
    /// An ACTIVE build exists but this side's FTS index was never built
    /// (e.g. an FTS5-less runtime, or a pre-index workspace).
    IndexNotBuilt,
    /// The index ran but nothing matched the query.
    NoMatches,
    /// The mode did not ask for this side.
    Disabled,
}

/// Per-side serving metadata carried with every fused retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServedSides {
    pub wiki: SideStatus,
    pub source: SideStatus,
}

impl ServedSides {
    /// True when at least one side degraded (anything but [`SideStatus::Served`]).
    pub fn is_degraded(&self) -> bool {
        self.wiki != SideStatus::Served || self.source != SideStatus::Served
    }
}

/// One wiki-side evidence entry.
#[derive(Debug, Clone, PartialEq)]
pub struct WikiEvidence {
    pub page_id: WikiPageId,
    pub slug: String,
    pub title: String,
    /// The section's heading path, outermost first.
    pub heading_path: Vec<String>,
    /// FTS5 `snippet()` over the section body with `[…]` highlight marks.
    pub snippet: String,
    /// FTS5 `bm25()` rank — numerically SMALLER is a better match.
    pub rank: f64,
}

/// One source-side evidence entry.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceEvidence {
    pub source_ref: SourceRef,
    /// The staged chunk title.
    pub title: String,
    /// FTS5 `snippet()` over the body column with `[…]` highlight marks.
    pub snippet: String,
    /// FTS5 `bm25()` rank — numerically SMALLER is a better match.
    pub rank: f64,
}

/// The unified evidence payload — one entry carries exactly one side.
#[derive(Debug, Clone, PartialEq)]
pub enum Evidence {
    Wiki(WikiEvidence),
    Source(SourceEvidence),
}

impl Evidence {
    /// Which side this entry belongs to.
    pub fn kind(&self) -> EvidenceKind {
        match self {
            Evidence::Wiki(_) => EvidenceKind::Wiki,
            Evidence::Source(_) => EvidenceKind::Source,
        }
    }

    /// The wiki slug of a wiki entry ("" for source entries — only ever read
    /// on the wiki list where the kind is already established).
    pub fn wiki_slug(&self) -> &str {
        match self {
            Evidence::Wiki(wiki) => &wiki.slug,
            Evidence::Source(_) => "",
        }
    }

    /// The wiki heading path of a wiki entry (empty for source entries).
    pub fn wiki_heading(&self) -> &[String] {
        match self {
            Evidence::Wiki(wiki) => &wiki.heading_path,
            Evidence::Source(_) => &[],
        }
    }

    /// The evidence-kind-scoped dedup/tie-break identity: wiki =
    /// `(page_id, heading_path)`, source = `(source_id, range)`. Kinds can
    /// never collide (the `wiki`/`source` prefix is part of the key).
    pub fn identity(&self) -> String {
        match self {
            Evidence::Wiki(wiki) => wiki_identity(wiki.page_id.as_str(), &wiki.heading_path),
            Evidence::Source(source) => format!(
                "source\u{1f}{}\u{1f}{}\u{1f}{}",
                source.source_ref.source_id,
                source.source_ref.range_start,
                source.source_ref.range_end
            ),
        }
    }
}

/// The wiki-side identity key for one section locator — shared with the
/// context builder, whose vector fusion can mint candidates that no fused
/// entry carries yet (same key format as [`Evidence::identity`]).
pub fn wiki_identity(page_id: &str, heading_path: &[String]) -> String {
    format!("wiki\u{1f}{page_id}\u{1f}{}", heading_path.join("\u{1f}"))
}

/// One unified evidence entry: the entry's rank within its own side, the
/// top-level RRF score, the exact-match flag and the side payload.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceEntry {
    /// Position within the entry's OWN side's ranked list (0-based) — the
    /// rank the top-level RRF consumed.
    pub side_rank: usize,
    /// Top-level RRF score (larger is better): `1 / (RRF_K + side_rank)`.
    pub score: f64,
    /// Exact-match protection: every query token appears verbatim in the
    /// chunk's title/heading/body. Always false for wiki entries (their
    /// ordering needs no protection — the wiki fusion handles them).
    pub exact_match: bool,
    pub evidence: Evidence,
}

/// The fused retrieval: the unified evidence list plus the honest per-side
/// metadata. Deterministic for a given database state and query.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedRetrieval {
    pub mode: SourceMode,
    /// Best-first, deterministic ordering: exact-match entries first, then
    /// RRF score, then a stable kind-aware locator.
    pub entries: Vec<EvidenceEntry>,
    /// Which side served and why the other did not. Never silent.
    pub served: ServedSides,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Runs the top-level retrieval for one query in `mode`, taking up to
/// `per_side_limit` hits PER SIDE and merging them (mode=Fusion) by RRF.
/// Deterministic for a given database state; both sides may be empty — the
/// `served` metadata says why, and the CONTEXT layer turns a fully empty
/// result into the honest "no results" error.
pub fn retrieve(
    conn: &Connection,
    query: &str,
    mode: SourceMode,
    per_side_limit: usize,
) -> Result<FusedRetrieval> {
    let analyzer = crate::TextAnalyzer;
    let fts_query = analyzer.fts_query(query);
    // Deduplicated query tokens for the exact-match predicate.
    let query_tokens: Vec<String> = {
        let mut seen = BTreeSet::new();
        analyzer
            .analyze(query)
            .into_iter()
            .filter(|token| seen.insert(token.clone()))
            .collect()
    };

    let (wiki, wiki_status) = if mode == SourceMode::Source {
        (Vec::new(), SideStatus::Disabled)
    } else {
        wiki_side(conn, &fts_query, per_side_limit)?
    };
    let (source, source_status) = if mode == SourceMode::Wiki {
        (Vec::new(), SideStatus::Disabled)
    } else {
        source_side(conn, query, &fts_query, per_side_limit)?
    };

    let mut entries = Vec::with_capacity(wiki.len() + source.len());
    for (side_rank, row) in wiki.into_iter().enumerate() {
        entries.push(EvidenceEntry {
            side_rank,
            score: 1.0 / (RRF_K + side_rank as f64),
            exact_match: false,
            evidence: Evidence::Wiki(WikiEvidence {
                page_id: row.page_id,
                slug: row.slug,
                title: row.title,
                heading_path: row.heading_path,
                snippet: row.snippet,
                rank: row.rank,
            }),
        });
    }
    for (side_rank, hit) in source.into_iter().enumerate() {
        let exact_match = is_exact_match(&query_tokens, &hit.title, &hit.heading_path, &hit.body);
        entries.push(EvidenceEntry {
            side_rank,
            score: 1.0 / (RRF_K + side_rank as f64),
            exact_match,
            evidence: Evidence::Source(SourceEvidence {
                source_ref: SourceRef {
                    source_id: hit.source_id,
                    file_path: hit.file_path,
                    heading_path: hit.heading_path,
                    ordinal: hit.ordinal,
                    range_start: hit.range_start,
                    range_end: hit.range_end,
                },
                title: hit.title,
                snippet: hit.snippet,
                rank: hit.rank,
            }),
        });
    }
    sort_entries(&mut entries);

    Ok(FusedRetrieval {
        mode,
        entries,
        served: ServedSides {
            wiki: wiki_status,
            source: source_status,
        },
    })
}

/// Orders the unified entries deterministically: exact-match protection
/// first (a guaranteed front slot, PR3 review decision), then the RRF score,
/// then a stable kind-aware locator (wiki ties break exactly like the legacy
/// context builder — slug, then heading path; source ties break by file
/// path, then ordinal, then range).
fn sort_entries(entries: &mut [EvidenceEntry]) {
    entries.sort_by(|a, b| {
        b.exact_match
            .cmp(&a.exact_match)
            .then_with(|| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| tie_break(&a.evidence, &b.evidence))
    });
}

/// Deterministic cross-kind tie-break: wiki sorts before source at equal
/// score, and same-kind ties break on the side's natural locator (never on
/// bm25 ranks, which are not comparable across corpora). Shared with the
/// context builder's level-2 ordering so both fusion levels tie-break alike.
pub(crate) fn tie_break(a: &Evidence, b: &Evidence) -> std::cmp::Ordering {
    fn kind_rank(evidence: &Evidence) -> u8 {
        match evidence {
            Evidence::Wiki(_) => 0,
            Evidence::Source(_) => 1,
        }
    }
    fn locator(evidence: &Evidence) -> (String, usize, usize) {
        match evidence {
            Evidence::Wiki(wiki) => (
                format!("{}\u{1f}{}", wiki.slug, wiki.heading_path.join("\u{1f}")),
                0,
                0,
            ),
            Evidence::Source(source) => (
                source.source_ref.file_path.clone(),
                source.source_ref.ordinal,
                source.source_ref.range_start,
            ),
        }
    }
    kind_rank(a)
        .cmp(&kind_rank(b))
        .then_with(|| locator(a).cmp(&locator(b)))
}

// ---------------------------------------------------------------------------
// Per-side execution (status matrix: Served / NotPublished / IndexNotBuilt /
// NoMatches / Disabled — the structural reason wins over the query reason)
// ---------------------------------------------------------------------------

fn wiki_side(
    conn: &Connection,
    fts_query: &str,
    limit: usize,
) -> Result<(Vec<SearchIndexRow>, SideStatus)> {
    if get_active_build_id(conn)?.is_none() {
        return Ok((Vec::new(), SideStatus::NotPublished));
    }
    if !search_index::fts_table_exists(conn)? {
        return Ok((Vec::new(), SideStatus::IndexNotBuilt));
    }
    if fts_query.is_empty() {
        return Ok((Vec::new(), SideStatus::NoMatches));
    }
    let hits = search_index::search_index(conn, fts_query, limit)?;
    let status = if hits.is_empty() {
        SideStatus::NoMatches
    } else {
        SideStatus::Served
    };
    Ok((hits, status))
}

fn source_side(
    conn: &Connection,
    query: &str,
    fts_query: &str,
    limit: usize,
) -> Result<(Vec<SourceFtsHit>, SideStatus)> {
    if get_active_build_id(conn)?.is_none() {
        return Ok((Vec::new(), SideStatus::NotPublished));
    }
    if !source_fts_exists(conn)? {
        return Ok((Vec::new(), SideStatus::IndexNotBuilt));
    }
    if fts_query.is_empty() {
        return Ok((Vec::new(), SideStatus::NoMatches));
    }
    let hits = search_source_fts(conn, query, limit)?;
    let status = if hits.is_empty() {
        SideStatus::NoMatches
    } else {
        SideStatus::Served
    };
    Ok((hits, status))
}

/// Exact-match protection predicate (PR3): true when EVERY deduplicated
/// query token appears verbatim in the chunk's title, heading path or body —
/// all sides normalized by the SAME shared tokenizer, so the check is exact
/// post-normalization and never invents a second one (PRD §20).
fn is_exact_match(
    query_tokens: &[String],
    title: &str,
    heading_path: &[String],
    body: &str,
) -> bool {
    if query_tokens.is_empty() {
        return false;
    }
    let analyzer = crate::TextAnalyzer;
    let mut present = BTreeSet::new();
    for token in analyzer.analyze(title) {
        present.insert(token);
    }
    for token in analyzer.analyze(&heading_path.join(" ")) {
        present.insert(token);
    }
    for token in analyzer.analyze(body) {
        present.insert(token);
    }
    query_tokens.iter().all(|token| present.contains(token))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_core::hash::sha256_hex;
    use llm_wiki_core::ids::{SourceId, SourceLocatorKey};
    use llm_wiki_storage::chunks::{rebuild_source_fts, replace_source_chunks, ChunkInput};
    use llm_wiki_storage::{
        activate_build_with_search_index, default_tokenizer, persist_generation, start_build,
        upsert_source, BuildDraft, WikiPageRecord,
    };

    fn heading_path(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn page(slug: &str, title: &str, content: &str) -> WikiPageRecord {
        WikiPageRecord {
            page_id: WikiPageId::generate(),
            slug: slug.to_owned(),
            title: title.to_owned(),
            category: "concepts".into(),
            language: "en".into(),
            body_hash: sha256_hex(content.as_bytes()),
            content: content.to_owned(),
            knowledge_refs: Vec::new(),
            citations: Vec::new(),
            links: Vec::new(),
        }
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

    fn stage_source(
        conn: &mut rusqlite::Connection,
        build: &llm_wiki_core::ids::BuildId,
        rel_path: &str,
        chunks: &[ChunkInput<'_>],
    ) -> SourceId {
        let (source_id, _) = upsert_source(
            conn,
            &SourceLocatorKey::compute("ws", rel_path),
            rel_path,
            "hash",
            10,
            None,
        )
        .unwrap();
        replace_source_chunks(conn, &source_id, build, rel_path, None, chunks).unwrap();
        source_id
    }

    /// One database with BOTH corpora live: a wiki page and a source chunk
    /// covering the SAME topic (dual-side content), plus wiki-only and
    /// source-only content for the fusion ordering properties.
    fn dual_conn() -> rusqlite::Connection {
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        persist_generation(
            &mut conn,
            &build,
            &[
                page(
                    "streaming",
                    "Alpha Beta Guide",
                    "# Alpha Beta Guide\n\nalpha beta content\n",
                ),
                page(
                    "glossary",
                    "Glossary",
                    "# Glossary\n\nbeta fragment reference\n",
                ),
            ],
        )
        .unwrap();
        activate_build_with_search_index(&mut conn, &build, default_tokenizer()).unwrap();

        let hp = heading_path(&["Guide"]);
        stage_source(
            &mut conn,
            &build,
            "docs/alpha.md",
            &[chunk("Alpha Beta", &hp, "alpha beta guide body", 0)],
        );
        stage_source(
            &mut conn,
            &build,
            "docs/gamma.md",
            &[chunk("Gamma", &hp, "beta fragment only", 0)],
        );
        // Sources staged after the activation need an explicit index rebuild
        // (the publish flow does this inside the activate transaction).
        rebuild_source_fts(&conn, &build).unwrap();
        conn
    }

    /// Marks the entries' side + locator for readable assertions.
    fn labels(entries: &[EvidenceEntry]) -> Vec<String> {
        entries
            .iter()
            .map(|entry| match &entry.evidence {
                Evidence::Wiki(wiki) => format!("wiki:{}", wiki.slug),
                Evidence::Source(source) => format!("source:{}", source.source_ref.file_path),
            })
            .collect()
    }

    // ---- determinism / three modes ----

    #[test]
    fn each_mode_is_independently_deterministic_on_the_same_database() {
        let conn = dual_conn();
        for mode in [SourceMode::Source, SourceMode::Wiki, SourceMode::Fusion] {
            let a = retrieve(&conn, "alpha beta", mode, 10).unwrap();
            let b = retrieve(&conn, "alpha beta", mode, 10).unwrap();
            assert_eq!(a, b, "mode {mode:?} must be deterministic");
        }
        // The three modes each saw the world independently on the SAME db.
        let source_only = retrieve(&conn, "alpha beta", SourceMode::Source, 10).unwrap();
        let wiki_only = retrieve(&conn, "alpha beta", SourceMode::Wiki, 10).unwrap();
        let fusion = retrieve(&conn, "alpha beta", SourceMode::Fusion, 10).unwrap();
        assert_eq!(source_only.entries.len(), 2);
        assert_eq!(wiki_only.entries.len(), 2);
        assert_eq!(fusion.entries.len(), 4);
        assert_eq!(
            source_only.served,
            ServedSides {
                wiki: SideStatus::Disabled,
                source: SideStatus::Served,
            }
        );
        assert_eq!(
            wiki_only.served,
            ServedSides {
                wiki: SideStatus::Served,
                source: SideStatus::Disabled,
            }
        );
        assert_eq!(
            fusion.served,
            ServedSides {
                wiki: SideStatus::Served,
                source: SideStatus::Served,
            }
        );
    }

    // ---- fusion ordering property ----

    #[test]
    fn fusion_ranks_dual_side_hits_above_single_side_hits() {
        let conn = dual_conn();
        let fused = retrieve(&conn, "alpha beta", SourceMode::Fusion, 10).unwrap();
        let labels = labels(&fused.entries);
        // The dual-side content (wiki page "streaming" + source
        // "docs/alpha.md", same topic, rank 0 on BOTH sides) must occupy the
        // top two slots; the single-side hits follow.
        assert_eq!(
            labels[..2],
            ["source:docs/alpha.md", "wiki:streaming"],
            "dual-side entries lead: {labels:?}"
        );
        // The source entry fronts via exact protection (contains every query
        // token verbatim); both dual entries carry the top RRF score.
        assert!(fused.entries[0].exact_match);
        assert!(!fused.entries[1].exact_match);
        assert!((fused.entries[0].score - fused.entries[1].score).abs() < 1e-12);
        assert!(
            fused.entries[0].score > fused.entries[2].score,
            "dual-side content outranks single-side content"
        );
        // RRF scores follow the shared-constant formula per side rank.
        assert!((fused.entries[0].score - 1.0 / RRF_K).abs() < 1e-12);
    }

    // ---- exact-match protection ----

    #[test]
    fn exact_match_protection_fronts_source_entries_above_partial_and_wiki_hits() {
        let mut conn = dual_conn();
        // A source chunk that matches only PART of the query tokens: it rides
        // FTS (OR semantics) but is NOT exact.
        let hp = heading_path(&["Notes"]);
        let build = get_active_build_id(&conn).unwrap().unwrap();
        stage_source(
            &mut conn,
            &build,
            "docs/partial.md",
            &[chunk("Beta", &hp, "beta note", 0)],
        );
        rebuild_source_fts(&conn, &build).unwrap();

        let fused = retrieve(&conn, "alpha beta", SourceMode::Fusion, 10).unwrap();
        let labels = labels(&fused.entries);
        // The full-token source chunk (docs/alpha.md) keeps its guaranteed
        // front slot; the partial source chunk does not jump ahead of it.
        assert_eq!(labels[0], "source:docs/alpha.md", "{labels:?}");
        assert!(fused.entries[0].exact_match);
        let partial = labels
            .iter()
            .position(|label| label == "source:docs/partial.md")
            .unwrap();
        assert!(!fused.entries[partial].exact_match);
        assert!(partial > 0, "non-exact entries never precede exact ones");
    }

    #[test]
    fn exact_predicate_requires_every_query_token_verbatim() {
        let mut seen = BTreeSet::new();
        let tokens: Vec<String> = crate::TextAnalyzer
            .analyze("检查点")
            .into_iter()
            .filter(|token| seen.insert(token.clone()))
            .collect();
        let hp = heading_path(&["流式处理"]);
        // Full content: every token (unigrams + bigrams) round-trips.
        assert!(is_exact_match(
            &tokens,
            "检查点",
            &hp,
            "检查点默认每 30 秒持久化一次。"
        ));
        // Partial content: matches FTS but not every token.
        assert!(!is_exact_match(&tokens, "检查", &hp, "仅仅检查"));
        // No query tokens: never exact.
        assert!(!is_exact_match(&[], "检查点", &hp, "检查点"));
    }

    #[test]
    fn cjk_bigram_query_parity_across_modes_on_the_same_database() {
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        persist_generation(
            &mut conn,
            &build,
            &[page(
                "streaming",
                "Streaming Processing",
                "# Streaming Processing\n\n## Checkpoints\n\n检查点默认每 30 秒持久化一次。\n",
            )],
        )
        .unwrap();
        // The activation rebuilds the source index too, so staging BEFORE it
        // needs no explicit rebuild (the publish-flow shape).
        let hp = heading_path(&["流式处理", "检查点"]);
        stage_source(
            &mut conn,
            &build,
            "docs/stream.md",
            &[chunk("检查点", &hp, "检查点默认每 30 秒持久化一次。", 0)],
        );
        activate_build_with_search_index(&mut conn, &build, default_tokenizer()).unwrap();

        let wiki = retrieve(&conn, "检查点", SourceMode::Wiki, 10).unwrap();
        let source = retrieve(&conn, "检查点", SourceMode::Source, 10).unwrap();
        let fusion = retrieve(&conn, "检查点", SourceMode::Fusion, 10).unwrap();

        // Same shared tokenizer: both sides answer the SAME bigram query.
        assert_eq!(wiki.entries.len(), 1);
        match &wiki.entries[0].evidence {
            Evidence::Wiki(w) => {
                assert_eq!(w.slug, "streaming");
                assert_eq!(w.heading_path, vec!["Streaming Processing", "Checkpoints"]);
            }
            other => panic!("expected a wiki entry, got {other:?}"),
        }
        match &source.entries[0].evidence {
            Evidence::Source(s) => {
                assert_eq!(s.source_ref.file_path, "docs/stream.md");
                assert_eq!(s.source_ref.heading_path, hp);
            }
            other => panic!("expected a source entry, got {other:?}"),
        }
        assert!(
            source.entries[0].exact_match,
            "title+body carry every bigram token"
        );

        // Fusion keeps BOTH kinds for the same query — dedup keys are
        // evidence-kind-scoped, nothing is dropped across sides.
        assert_eq!(
            fusion.entries.len(),
            wiki.entries.len() + source.entries.len()
        );
        assert_eq!(
            fusion.served,
            ServedSides {
                wiki: SideStatus::Served,
                source: SideStatus::Served,
            }
        );
    }

    // ---- degradation matrix ----

    #[test]
    fn source_mode_works_when_the_wiki_side_is_not_available() {
        let conn = dual_conn();
        // Baseline with the wiki index intact, then drop it: the SOURCE side
        // must return identical results — the sides are independent.
        let before = retrieve(&conn, "alpha beta", SourceMode::Source, 10).unwrap();
        conn.execute("DROP TABLE wiki_fts", []).unwrap();
        let after = retrieve(&conn, "alpha beta", SourceMode::Source, 10).unwrap();
        assert_eq!(before, after);
        assert_eq!(after.served.source, SideStatus::Served);

        // Fusion DEGRADES VISIBLY: it still returns the live side plus the
        // reason the wiki side is empty.
        let fused = retrieve(&conn, "alpha beta", SourceMode::Fusion, 10).unwrap();
        assert!(!fused.entries.is_empty());
        assert!(fused
            .entries
            .iter()
            .all(|entry| entry.evidence.kind() == EvidenceKind::Source));
        assert_eq!(fused.served.wiki, SideStatus::IndexNotBuilt);
        assert_eq!(fused.served.source, SideStatus::Served);
        assert!(fused.served.is_degraded());
    }

    #[test]
    fn fusion_reports_index_not_built_for_a_missing_source_index() {
        let conn = dual_conn();
        conn.execute("DROP TABLE source_fts", []).unwrap();

        let fused = retrieve(&conn, "alpha beta", SourceMode::Fusion, 10).unwrap();
        assert!(!fused.entries.is_empty());
        assert!(fused
            .entries
            .iter()
            .all(|entry| entry.evidence.kind() == EvidenceKind::Wiki));
        assert_eq!(fused.served.source, SideStatus::IndexNotBuilt);
        assert_eq!(fused.served.wiki, SideStatus::Served);
        assert!(fused.served.is_degraded());
    }

    #[test]
    fn nothing_published_is_reported_per_side_never_served() {
        let mut conn = dual_conn();
        llm_wiki_storage::set_active_build(&mut conn, None).unwrap();

        for mode in [SourceMode::Source, SourceMode::Wiki, SourceMode::Fusion] {
            let fused = retrieve(&conn, "alpha beta", mode, 10).unwrap();
            assert!(fused.entries.is_empty(), "mode {mode:?} served stale rows");
            if mode != SourceMode::Source {
                assert_eq!(fused.served.wiki, SideStatus::NotPublished);
            }
            if mode != SourceMode::Wiki {
                assert_eq!(fused.served.source, SideStatus::NotPublished);
            }
        }
    }

    #[test]
    fn no_matches_is_reported_when_a_side_has_no_hits() {
        let conn = dual_conn();
        // The fixture has no CJK content: both sides honestly report
        // NoMatches and the entry list is empty — visible, not silent.
        let fused = retrieve(&conn, "检查点", SourceMode::Fusion, 10).unwrap();
        assert!(fused.entries.is_empty());
        assert_eq!(fused.served.wiki, SideStatus::NoMatches);
        assert_eq!(fused.served.source, SideStatus::NoMatches);
        assert!(fused.served.is_degraded());
    }

    // ---- exact protection: deterministic ordering unit ----

    #[test]
    fn sort_entries_gives_exact_entries_a_guaranteed_front_slot() {
        let exact_low = EvidenceEntry {
            side_rank: 9,
            score: 1.0 / (RRF_K + 9.0),
            exact_match: true,
            evidence: Evidence::Source(SourceEvidence {
                source_ref: SourceRef {
                    source_id: SourceId::from_validated("src_z"),
                    file_path: "docs/z.md".into(),
                    heading_path: vec![],
                    ordinal: 0,
                    range_start: 0,
                    range_end: 1,
                },
                title: "z".into(),
                snippet: String::new(),
                rank: -1.0,
            }),
        };
        let plain_high = |index: usize| EvidenceEntry {
            side_rank: index,
            score: 1.0 / (RRF_K + index as f64),
            exact_match: false,
            evidence: Evidence::Wiki(WikiEvidence {
                page_id: WikiPageId::from_validated(format!("wp_{index}")),
                slug: format!("s{index}"),
                title: format!("t{index}"),
                heading_path: vec![],
                snippet: String::new(),
                rank: -(index as f64),
            }),
        };
        let mut entries = vec![plain_high(0), plain_high(1), exact_low, plain_high(2)];
        let snapshot = entries.clone();
        sort_entries(&mut entries);
        assert!(entries[0].exact_match, "exact entry leads: {entries:?}");
        assert!(
            entries[1..].iter().all(|entry| !entry.exact_match),
            "the rest follows by score"
        );
        // Determinism: the same input sorts identically twice.
        let mut again = snapshot;
        sort_entries(&mut again);
        assert_eq!(entries, again);
    }
}
