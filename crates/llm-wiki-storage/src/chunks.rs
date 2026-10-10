//! Raw-source retrieval index (EPIC A PR1): the source-side counterpart of
//! [`crate::search_index`].
//!
//! Storage model: `source_chunk_text` (migration 0013) holds one row per
//! SOURCE SEGMENT — `(source_id, build_id, file_path, title,
//! heading_path_json, locale, ordinal, range_start, range_end, body)` — and
//! the `source_fts` FTS5 virtual table indexes `(title, headings, body)` with
//! the `unicode61` tokenizer. `source_fts` is created LAZILY at index time so
//! an FTS5-less SQLite still opens and migrates the state db (migration 0008
//! rationale); [`probe_fts5`] decides and [`FTS5_UNAVAILABLE`] is the loud
//! degradation.
//!
//! CJK strategy (PRD §20 hard rule, same as `wiki_fts`): rows are
//! pre-tokenized with the shared [`SearchTokenizer`] (NFKC, lowercase,
//! full→half width, Han unigram+bigram) and space-joined on insert; the query
//! side passes through the SAME tokenizer. PR3's fusion layer may also route
//! raw user text through `llm-wiki-search::TextAnalyzer::fts_query`, which is
//! the same tokenizer behind a search-crate façade (dependency direction:
//! search → storage; storage never depends on search).
//!
//! Publish integration (PR2, wired): [`replace_source_chunks`] swaps one
//! (source, build) staging pair atomically during the build; [`carry_source_chunks`]
//! copies the UNCHANGED live sources' rows from the previous active build into
//! the new one (the incremental and replan pipelines only re-stage changed
//! sources — without the carry they would silently vanish from the rebuilt
//! index); [`rebuild_source_fts`] and [`ensure_source_fts_matches_active`]
//! take `&Connection`/own their transaction so the rebuild runs INSIDE the §35
//! activate transaction, where a failure aborts the whole commit point. This
//! module never touches `source_sections` (the Section Matcher registry) —
//! the tables coexist by design (see the coexistence test below).

use std::collections::BTreeSet;

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, SourceId};

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

// ---------------------------------------------------------------------------
// Staging writes (build-time, PR2 caller)
// ---------------------------------------------------------------------------

/// One source segment to stage in [`replace_source_chunks`]: a section
/// segmented at block boundaries, with absolute offsets into the normalized
/// source text (the `source_range` basis of `llm-wiki-markdown::segment`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkInput<'a> {
    pub title: &'a str,
    pub heading_path: &'a [String],
    /// Position of this segment within its section (0-based).
    pub ordinal: usize,
    pub range_start: usize,
    pub range_end: usize,
    pub body: &'a str,
}

/// What one replace wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplaceStats {
    /// Staging rows deleted for the (source, build) pair.
    pub removed: usize,
    /// Staging rows inserted.
    pub written: usize,
}

/// Replaces ALL staging rows of one `(source_id, build_id)` pair with
/// `chunks` in ONE transaction — delete-then-insert, so a failure leaves the
/// previous staging state intact and repeated calls never duplicate rows.
/// `locale` comes from `llm-wiki-markdown` language detection (may be None);
/// `file_path` is the source-relative path, for diagnostics only.
pub fn replace_source_chunks(
    conn: &mut Connection,
    source_id: &SourceId,
    build_id: &BuildId,
    file_path: &str,
    locale: Option<&str>,
    chunks: &[ChunkInput<'_>],
) -> Result<ReplaceStats> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let removed = tx
        .execute(
            "DELETE FROM source_chunk_text WHERE source_id = ?1 AND build_id = ?2",
            params![source_id.as_str(), build_id.as_str()],
        )
        .map_err(db)?;
    {
        let mut insert = tx
            .prepare(
                "INSERT INTO source_chunk_text
                 (source_id, build_id, file_path, title, heading_path_json, locale,
                  ordinal, range_start, range_end, body)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )
            .map_err(|e| WikiError::Storage(format!("prepare chunk insert: {e}")))?;
        for chunk in chunks {
            let heading_json = serde_json::to_string(chunk.heading_path)
                .map_err(|e| WikiError::Storage(format!("serialize heading path: {e}")))?;
            insert
                .execute(params![
                    source_id.as_str(),
                    build_id.as_str(),
                    file_path,
                    chunk.title,
                    heading_json,
                    locale,
                    chunk.ordinal as i64,
                    chunk.range_start as i64,
                    chunk.range_end as i64,
                    chunk.body
                ])
                .map_err(db)?;
        }
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit replace_source_chunks: {e}")))?;
    Ok(ReplaceStats {
        removed,
        written: chunks.len(),
    })
}

/// Copies the previous build's staging rows forward into `new_build_id` for
/// every live source NOT in `exclude` (EPIC A PR2, the incremental-gap fix).
/// The incremental and replan pipelines only re-stage CHANGED sources, while
/// [`rebuild_source_fts`] indexes `build_id = new_build_id` rows only —
/// without the carry, every unchanged source would silently disappear from
/// the rebuilt index. `exclude` is added ∪ modified ∪ deleted: changed
/// sources are re-staged by [`replace_source_chunks`], and deleted sources are
/// never written, never carried (their rows simply do not re-enter the new
/// build). One `INSERT … SELECT` copies ALL columns verbatim (locale,
/// `canonical_url`, `product_version` included).
///
/// Idempotent: the carried set is delete-then-insert, so a repeated call
/// (defensive build retry) never duplicates rows. Takes `&Connection` (PR1
/// convention) — it joins the caller's transaction; call sites without an
/// open transaction wrap it so the delete and the insert commit atomically.
/// Returns the number of rows copied.
pub fn carry_source_chunks(
    conn: &Connection,
    prev_build_id: &BuildId,
    new_build_id: &BuildId,
    exclude: &[SourceId],
) -> Result<usize> {
    // The NOT-IN predicate repeats for the defensive delete and the copy.
    // The only interpolated SQL text is the `?` placeholder list — the ids
    // themselves travel as bound parameters (spec §42).
    let placeholders = vec!["?"; exclude.len()].join(", ");
    let not_in = if exclude.is_empty() {
        String::new()
    } else {
        format!(" AND source_id NOT IN ({placeholders})")
    };
    let exclude_ids: Vec<String> = exclude.iter().map(|id| id.as_str().to_owned()).collect();

    // Rows this build carried on a previous attempt are removed first, so a
    // retry lands on the same state instead of duplicating them.
    let mut delete_params: Vec<&str> = vec![new_build_id.as_str()];
    delete_params.extend(exclude_ids.iter().map(String::as_str));
    conn.prepare(&format!(
        "DELETE FROM source_chunk_text WHERE build_id = ?{not_in}"
    ))
    .map_err(|e| WikiError::Storage(format!("prepare carry delete: {e}")))?
    .execute(rusqlite::params_from_iter(delete_params))
    .map_err(db)?;

    let mut insert_params: Vec<&str> = vec![new_build_id.as_str(), prev_build_id.as_str()];
    insert_params.extend(exclude_ids.iter().map(String::as_str));
    let copied = conn
        .prepare(&format!(
            "INSERT INTO source_chunk_text
             (source_id, build_id, file_path, title, heading_path_json, locale,
              ordinal, range_start, range_end, body, canonical_url, product_version)
             SELECT source_id, ?, file_path, title, heading_path_json, locale,
                    ordinal, range_start, range_end, body, canonical_url, product_version
             FROM source_chunk_text
             WHERE build_id = ?{not_in}"
        ))
        .map_err(|e| WikiError::Storage(format!("prepare carry insert: {e}")))?
        .execute(rusqlite::params_from_iter(insert_params))
        .map_err(db)?;
    Ok(copied)
}

// ---------------------------------------------------------------------------
// FTS rebuild (runs inside the publish / recovery transaction)
// ---------------------------------------------------------------------------

/// Deletes ALL `source_fts` rows and re-inserts the tokenized rows of
/// `active_build_id` from `source_chunk_text`, pre-tokenized by the shared
/// [`SearchTokenizer`] — the SAME implementation the `wiki_fts` rebuild uses
/// (PRD §20: index and query MUST share one normalization). Takes
/// `&Connection` so it can run inside an open transaction (`Transaction`
/// derefs to `Connection`) — PR2's publish flow calls it within the §35
/// activate transaction. Returns the number of FTS rows written.
pub fn rebuild_source_fts(conn: &Connection, active_build_id: &BuildId) -> Result<usize> {
    if !crate::search_index::probe_fts5(conn) {
        return Err(WikiError::Index(
            crate::search_index::FTS5_UNAVAILABLE.to_owned(),
        ));
    }
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS source_fts USING fts5(
            title, headings, body, tokenize = 'unicode61'
        )",
    )
    .map_err(|e| WikiError::Storage(format!("create source_fts: {e}")))?;
    conn.execute("DELETE FROM source_fts", []).map_err(db)?;

    let tokenizer = crate::search_index::default_tokenizer();
    let mut insert_fts = conn
        .prepare("INSERT INTO source_fts (rowid, title, headings, body) VALUES (?1, ?2, ?3, ?4)")
        .map_err(|e| WikiError::Storage(format!("prepare fts insert: {e}")))?;
    let mut stmt = conn
        .prepare(
            "SELECT chunk_id, title, heading_path_json, body FROM source_chunk_text
             WHERE build_id = ?1 ORDER BY chunk_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare index chunks: {e}")))?;
    let rows = stmt
        .query_map(params![active_build_id.as_str()], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("index chunks: {e}")))?;

    let mut written = 0usize;
    for row in rows {
        let (chunk_id, title, heading_json, body) = row.map_err(db)?;
        let heading_path: Vec<String> = serde_json::from_str(&heading_json).unwrap_or_default();
        insert_fts
            .execute(params![
                chunk_id,
                tokenizer.analyze(&title).join(" "),
                tokenizer.analyze(&heading_path.join(" ")).join(" "),
                tokenizer.analyze(&body).join(" ")
            ])
            .map_err(db)?;
        written += 1;
    }
    Ok(written)
}

/// Recovery-side parity check (PRD §1.3 of EPIC A PR2, the counterpart of
/// `search_index::ensure_search_index_matches_active`): `source_fts` must
/// cover exactly the ACTIVE build's `source_chunk_text` rows. A missing table
/// or a row-count drift triggers [`rebuild_source_fts`] — the source index is
/// deliberately rebuilt FULLY (no wiki-side hash-diff; the carry-forward keeps
/// staging complete, so a full rebuild is correctness-equivalent). With
/// nothing active, leftover rows are cleared so search reports the truth.
/// Owns its transaction like the wiki-side check: the rebuild commits
/// atomically or not at all. An FTS5-less runtime with an ACTIVE build fails
/// loudly with `FTS5_UNAVAILABLE` (the probe in [`rebuild_source_fts`] is
/// unconditional) — mirroring the wiki-side contract, never silent
/// degradation; with nothing active the check is a no-op.
pub fn ensure_source_fts_matches_active(conn: &mut Connection) -> Result<()> {
    let Some(active) = crate::state::get_active_build_id(conn)? else {
        if source_fts_exists(conn)? {
            let rows: i64 = conn
                .query_row("SELECT COUNT(*) FROM source_fts", [], |row| row.get(0))
                .map_err(db)?;
            if rows > 0 {
                let tx = conn
                    .transaction()
                    .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
                tx.execute("DELETE FROM source_fts", []).map_err(db)?;
                tx.commit()
                    .map_err(|e| WikiError::Storage(format!("commit clear source_fts: {e}")))?;
                tracing::warn!("cleared source_fts: no build is active");
            }
        }
        return Ok(());
    };
    let staged: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM source_chunk_text WHERE build_id = ?1",
            params![active.as_str()],
            |row| row.get(0),
        )
        .map_err(db)?;
    let indexed: Option<i64> = if source_fts_exists(conn)? {
        Some(
            conn.query_row("SELECT COUNT(*) FROM source_fts", [], |row| row.get(0))
                .map_err(db)?,
        )
    } else {
        None
    };
    // Count equality alone can pass a WRONG-build index (same row count, other
    // build's content): consistency also requires every FTS row to map back to
    // THIS build's staging rows (rebuild contract: fts rowid == chunk_id).
    let foreign: i64 = if indexed.is_some() {
        conn.query_row(
            "SELECT COUNT(*) FROM source_fts WHERE rowid NOT IN
             (SELECT chunk_id FROM source_chunk_text WHERE build_id = ?1)",
            params![active.as_str()],
            |row| row.get(0),
        )
        .map_err(db)?
    } else {
        0
    };
    if indexed == Some(staged) && foreign == 0 {
        return Ok(());
    }
    tracing::warn!(
        build = %active,
        staged,
        indexed = indexed.unwrap_or_default(),
        foreign,
        "source_fts does not match the active build's staging rows; rebuilding"
    );
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let written = rebuild_source_fts(&tx, &active)?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit source_fts rebuild: {e}")))?;
    tracing::debug!(written, "source_fts rebuilt to the active build");
    Ok(())
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// One source-segment hit.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceFtsHit {
    pub chunk_id: i64,
    pub source_id: SourceId,
    /// The build whose rebuild produced this index row (the ACTIVE build by
    /// the [`rebuild_source_fts`] contract).
    pub build_id: String,
    pub file_path: String,
    pub heading_path: Vec<String>,
    pub ordinal: usize,
    /// Absolute offsets into the normalized source text.
    pub range_start: usize,
    pub range_end: usize,
    /// FTS5 `snippet()` over the body column with `[…]` highlight marks.
    pub snippet: String,
    /// FTS5 `bm25()` rank — numerically SMALLER is a better match.
    pub rank: f64,
}

/// True when the lazily-created FTS5 table exists (i.e. some rebuild ran).
fn source_fts_exists(conn: &Connection) -> Result<bool> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'source_fts'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| WikiError::Storage(format!("sqlite_master lookup: {e}")))?;
    Ok(count > 0)
}

/// Builds the FTS5 MATCH expression for raw query text: deduplicated tokens
/// as quoted string literals joined by `OR` — a verbatim mirror of
/// `llm-wiki-search::TextAnalyzer::fts_query`, kept local because storage
/// must not depend on the search crate (dependency direction: search →
/// storage). Quoting keeps every token a literal term and the expression
/// travels to SQLite as a bound parameter — user text is never interpolated.
fn fts_expression(query: &str) -> String {
    let tokenizer = crate::search_index::default_tokenizer();
    let mut seen = BTreeSet::new();
    let mut terms = Vec::new();
    for token in tokenizer.analyze(query) {
        if seen.insert(token.clone()) {
            terms.push(format!("\"{}\"", token.replace('"', "\"\"")));
        }
    }
    terms.join(" OR ")
}

/// Runs an FTS5 MATCH query over `source_fts` and returns the best `limit`
/// hits, ranked by bm25, with body snippets. The query is tokenized by the
/// SAME shared tokenizer the rebuild used. An index that was never built
/// (table absent — e.g. an FTS5-less runtime) yields an honest empty result;
/// punctuation-only input yields no expression and likewise no rows.
pub fn search_source_fts(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<SourceFtsHit>> {
    if !source_fts_exists(conn)? {
        return Ok(Vec::new());
    }
    let fts_query = fts_expression(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }

    let mut stmt = conn
        .prepare(
            "SELECT t.chunk_id, t.source_id, t.build_id, t.file_path, t.heading_path_json,
                    t.ordinal, t.range_start, t.range_end,
                    snippet(source_fts, 2, '[', ']', ' … ', 16), bm25(source_fts)
             FROM source_fts
             JOIN source_chunk_text t ON t.chunk_id = source_fts.rowid
             WHERE source_fts MATCH ?1
             ORDER BY bm25(source_fts)
             LIMIT ?2",
        )
        .map_err(|e| WikiError::Storage(format!("prepare source search: {e}")))?;
    let rows = stmt
        .query_map(params![fts_query, limit as i64], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, i64>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, f64>(9)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("source search: {e}")))?;

    let mut out = Vec::new();
    for row in rows {
        let (
            chunk_id,
            source_id,
            build_id,
            file_path,
            heading_json,
            ordinal,
            range_start,
            range_end,
            snippet,
            rank,
        ) = row.map_err(db)?;
        let heading_path: Vec<String> = serde_json::from_str(&heading_json).unwrap_or_default();
        out.push(SourceFtsHit {
            chunk_id,
            source_id: SourceId::from_validated(source_id),
            build_id,
            file_path,
            heading_path,
            ordinal: ordinal as usize,
            range_start: range_start as usize,
            range_end: range_end as usize,
            snippet,
            rank,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builds::{start_build, BuildDraft};
    use crate::connection::open_in_memory;
    use crate::search_index::{probe_fts5, FTS5_UNAVAILABLE};

    fn heading_path(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn chunk<'a>(title: &'a str, body: &'a str, ordinal: usize) -> ChunkInput<'a> {
        ChunkInput {
            title,
            heading_path: &[],
            ordinal,
            range_start: ordinal * 100,
            range_end: ordinal * 100 + body.len(),
            body,
        }
    }

    fn source(conn: &mut rusqlite::Connection, path: &str) -> SourceId {
        crate::sources::upsert_source(
            conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", path),
            path,
            "hash",
            1,
            None,
        )
        .unwrap()
        .0
    }

    /// Applies the first `upto` migration scripts (1-based count) to a RAW
    /// connection, stamping `user_version` after each one — the legacy-db
    /// simulation for the upgrade test (mirrors `connection::migrate_versions`).
    fn apply_migrations(conn: &rusqlite::Connection, upto: usize) {
        for (idx, script) in crate::migrations::MIGRATIONS.iter().enumerate().take(upto) {
            conn.execute_batch(script).unwrap();
            conn.execute_batch(&format!("PRAGMA user_version = {};", idx + 1))
                .unwrap();
        }
    }

    #[test]
    fn migration_chain_0001_to_0013_applies_and_preserves_legacy_rows() {
        // Legacy db at v12: migrations 0001-0012 applied, a Section-Matcher
        // row already present.
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        apply_migrations(&conn, 12);
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, 12);
        conn.execute_batch(
            "INSERT INTO sources (source_id, locator_key, rel_path, content_hash, size)
             VALUES ('src_legacy', 'ws|legacy.md', 'legacy.md', 'h', 1);
             INSERT INTO source_sections (section_id, source_id, heading_path_json,
                                          heading_path_key, content_fingerprint,
                                          range_start, range_end)
             VALUES ('sec_legacy', 'src_legacy', '[]', 'k', 'fp', 0, 1);",
        )
        .unwrap();

        // Upgrade to v13: the new staging table appears, legacy rows survive.
        conn.execute_batch(crate::migrations::MIGRATIONS[12])
            .unwrap();
        conn.execute_batch("PRAGMA user_version = 13;").unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, 13);
        conn.execute_batch(
            "INSERT INTO source_chunk_text (source_id, build_id, file_path, ordinal,
                                            range_start, range_end)
             VALUES ('src_legacy', 'bld_x', 'legacy.md', 0, 0, 1);",
        )
        .unwrap();
        let legacy: i64 = conn
            .query_row("SELECT COUNT(*) FROM source_sections", [], |r| r.get(0))
            .unwrap();
        assert_eq!(legacy, 1, "upgrade must not touch existing tables");
    }

    #[test]
    fn replace_source_chunks_is_idempotent_per_source_and_build() {
        let mut conn = open_in_memory().unwrap();
        let (source_id, _) = crate::sources::upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "a.md"),
            "a.md",
            "hash",
            1,
            None,
        )
        .unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();

        let chunks = vec![
            chunk("Intro", "intro 检查点 body", 0),
            chunk("Usage", "usage body", 1),
        ];
        let stats = replace_source_chunks(
            &mut conn,
            &source_id,
            &build,
            "docs/a.md",
            Some("zh"),
            &chunks,
        )
        .unwrap();
        assert_eq!(
            stats,
            ReplaceStats {
                removed: 0,
                written: 2
            }
        );

        // Same pair again: replaces, never duplicates.
        let stats = replace_source_chunks(
            &mut conn,
            &source_id,
            &build,
            "docs/a.md",
            Some("zh"),
            &chunks[..1],
        )
        .unwrap();
        assert_eq!(
            stats,
            ReplaceStats {
                removed: 2,
                written: 1
            }
        );
        let rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text WHERE source_id = ?1 AND build_id = ?2",
                params![source_id.as_str(), build.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rows, 1);

        // A different build for the same source coexists (staging is keyed by
        // the pair, not the source alone).
        let next_build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        replace_source_chunks(
            &mut conn,
            &source_id,
            &next_build,
            "docs/a.md",
            Some("zh"),
            &chunks,
        )
        .unwrap();
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text WHERE source_id = ?1",
                params![source_id.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(total, 3, "1 row for the old build + 2 for the new one");
    }

    #[test]
    fn rebuild_keeps_only_the_active_build_rows_and_flips_with_it() {
        let mut conn = open_in_memory().unwrap();
        let (source_id, _) = crate::sources::upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "a.md"),
            "a.md",
            "hash",
            1,
            None,
        )
        .unwrap();
        let build_a = start_build(&mut conn, &BuildDraft::default()).unwrap();
        let build_b = start_build(&mut conn, &BuildDraft::default()).unwrap();
        replace_source_chunks(
            &mut conn,
            &source_id,
            &build_a,
            "docs/a.md",
            None,
            &[chunk("Old", "alpha legacy content", 0)],
        )
        .unwrap();
        replace_source_chunks(
            &mut conn,
            &source_id,
            &build_b,
            "docs/a.md",
            None,
            &[chunk("New", "检查点默认持久化", 0)],
        )
        .unwrap();

        // Build B active: only its content is searchable.
        assert_eq!(rebuild_source_fts(&conn, &build_b).unwrap(), 1);
        let hits = search_source_fts(&conn, "检查点", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].build_id, build_b.as_str());
        assert_eq!(hits[0].source_id, source_id);
        assert_eq!(hits[0].file_path, "docs/a.md");
        assert!(search_source_fts(&conn, "legacy", 10).unwrap().is_empty());

        // Build A active again: the index flips to the other generation.
        assert_eq!(rebuild_source_fts(&conn, &build_a).unwrap(), 1);
        assert!(search_source_fts(&conn, "检查点", 10).unwrap().is_empty());
        let hits = search_source_fts(&conn, "legacy", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].build_id, build_a.as_str());

        // Staging rows are untouched by the rebuild — both builds remain.
        let staging: i64 = conn
            .query_row("SELECT COUNT(*) FROM source_chunk_text", [], |r| r.get(0))
            .unwrap();
        assert_eq!(staging, 2);
    }

    #[test]
    fn cjk_bigram_queries_hit_source_chunks_with_snippets() {
        let mut conn = open_in_memory().unwrap();
        let (source_id, _) = crate::sources::upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "流式.md"),
            "流式.md",
            "hash",
            1,
            None,
        )
        .unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        let hp = heading_path(&["流式处理", "检查点"]);
        let chunks = vec![ChunkInput {
            title: "检查点",
            heading_path: &hp,
            ordinal: 0,
            range_start: 0,
            range_end: 24,
            body: "检查点默认每 30 秒持久化一次。",
        }];
        replace_source_chunks(
            &mut conn,
            &source_id,
            &build,
            "docs/stream.md",
            Some("zh"),
            &chunks,
        )
        .unwrap();
        rebuild_source_fts(&conn, &build).unwrap();

        // Bigram query: title + body match, heading path round-trips.
        let hits = search_source_fts(&conn, "检查点", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].heading_path, hp);
        assert_eq!(hits[0].ordinal, 0);
        assert_eq!((hits[0].range_start, hits[0].range_end), (0, 24));
        assert!(hits[0].snippet.contains('['));
        assert!(hits[0].rank <= 0.0, "bm25 ranks are non-positive");

        // Single-character query rides the unigram path.
        assert_eq!(search_source_fts(&conn, "检", 10).unwrap().len(), 1);

        // English body terms through the same tokenizer.
        assert_eq!(
            search_source_fts(&conn, "persistence", 10).unwrap().len(),
            0
        );
        let hits = search_source_fts(&conn, "30", 10).unwrap();
        assert_eq!(hits.len(), 1);

        // Punctuation-only input builds no expression — no FTS5 syntax error.
        assert!(search_source_fts(&conn, " 。！ ", 10).unwrap().is_empty());

        // Limit is honored and ranks stay best-first.
        let hits = search_source_fts(&conn, "检查点 30", 1).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn fts5_unavailable_paths_degrade_loudly_never_silently() {
        let conn = open_in_memory().unwrap();
        let rebuilt = rebuild_source_fts(&conn, &BuildId::generate());
        if probe_fts5(&conn) {
            // Bundled SQLite ships FTS5: the rebuild succeeds (empty index —
            // no staging rows for a ghost build) and the unavailable branch is
            // exercised only on FTS5-less builds, where it must return the
            // shared FTS5_UNAVAILABLE error instead of panicking or degrading.
            assert!(rebuilt.is_ok());
        } else {
            let err = rebuilt.unwrap_err();
            assert!(matches!(err, WikiError::Index(ref m) if m == FTS5_UNAVAILABLE));
        }
        // Either way: an index that was never built searches as empty — no
        // panic, no fabricated hits.
        assert!(search_source_fts(&conn, "检查点", 10).unwrap().is_empty());
    }

    #[test]
    fn source_chunks_and_source_sections_coexist_without_interaction() {
        let mut conn = open_in_memory().unwrap();
        let (source_id, _) = crate::sources::upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "a.md"),
            "a.md",
            "hash",
            1,
            None,
        )
        .unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();

        // Section-Matcher registry side (the occupied `source_sections` name).
        let identity = llm_wiki_core::matcher::SectionIdentity::from_parts(
            &["Doc".to_string(), "Intro".to_string()],
            "intro",
        );
        let report = llm_wiki_core::matcher::match_sections(&[], &[identity]);
        let mut created = std::collections::BTreeMap::new();
        created.insert(0usize, llm_wiki_core::ids::SectionId::generate());
        let section_stats = crate::sections::apply_section_matches(
            &mut conn,
            &source_id,
            &report,
            &[llm_wiki_core::matcher::SectionIdentity::from_parts(
                &["Doc".to_string(), "Intro".to_string()],
                "intro",
            )],
            &[llm_wiki_core::model::SourceRange::new(0, 5)],
            &created,
            Some(build.as_str()),
        )
        .unwrap();
        assert_eq!(section_stats.created, 1);

        // Chunk side writes must not disturb the section registry.
        replace_source_chunks(
            &mut conn,
            &source_id,
            &build,
            "docs/a.md",
            None,
            &[chunk("Intro", "source 检查点 body", 0)],
        )
        .unwrap();
        rebuild_source_fts(&conn, &build).unwrap();

        let sections = crate::sections::load_active_sections(&conn, &source_id).unwrap();
        assert_eq!(sections.len(), 1, "section registry untouched by chunks");
        let hits = search_source_fts(&conn, "检查点", 10).unwrap();
        assert_eq!(hits.len(), 1, "chunk index is independently searchable");

        // And a second replace still leaves the section row alone.
        replace_source_chunks(
            &mut conn,
            &source_id,
            &build,
            "docs/a.md",
            None,
            &[chunk("Intro", "rewritten body", 0)],
        )
        .unwrap();
        assert_eq!(
            crate::sections::load_active_sections(&conn, &source_id)
                .unwrap()
                .len(),
            1
        );
    }

    /// PR2 incremental-gap fix: the carry copies the unchanged sources' rows
    /// forward verbatim (every column), skips exclude (changed/deleted), and
    /// is idempotent under a defensive retry.
    #[test]
    fn carry_source_chunks_copies_live_rows_verbatim_excludes_and_is_idempotent() {
        let mut conn = open_in_memory().unwrap();
        let s1 = source(&mut conn, "a.md");
        let s2 = source(&mut conn, "b.md");
        let s3 = source(&mut conn, "c.md");
        let prev = start_build(&mut conn, &BuildDraft::default()).unwrap();
        let next = start_build(&mut conn, &BuildDraft::default()).unwrap();

        replace_source_chunks(
            &mut conn,
            &s1,
            &prev,
            "docs/a.md",
            Some("zh"),
            &[chunk("Alpha", "alpha unchanged 检查点 body", 0)],
        )
        .unwrap();
        replace_source_chunks(
            &mut conn,
            &s2,
            &prev,
            "docs/b.md",
            None,
            &[chunk("Beta", "beta body", 0), chunk("Beta2", "second", 1)],
        )
        .unwrap();
        replace_source_chunks(
            &mut conn,
            &s3,
            &prev,
            "docs/c.md",
            Some("en"),
            &[chunk("Gamma", "gamma body", 0)],
        )
        .unwrap();
        // Forward-compat columns have no producer yet — set them by hand so
        // the copy is proven to preserve them too.
        conn.execute(
            "UPDATE source_chunk_text SET canonical_url = 'https://example.test/a',
             product_version = 'v1' WHERE source_id = ?1 AND build_id = ?2",
            params![s1.as_str(), prev.as_str()],
        )
        .unwrap();

        // The `next` build re-staged s2 (modified) and s3 left the workspace
        // (deleted): both are excluded, only s1's rows carry forward.
        replace_source_chunks(
            &mut conn,
            &s2,
            &next,
            "docs/b.md",
            None,
            &[chunk("Beta", "rewritten in next build", 0)],
        )
        .unwrap();
        let exclude = vec![s2.clone(), s3.clone()];
        let carried = carry_source_chunks(&conn, &prev, &next, &exclude).unwrap();
        assert_eq!(carried, 1, "only the untouched source copies forward");

        // Column fidelity: the carried row equals the prev row on EVERY column.
        let column_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text WHERE build_id = ?1",
                params![next.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(column_count, 2, "1 carried + 1 re-staged row, never more");
        let carried_pairs: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text prev
                 JOIN source_chunk_text next
                   ON next.source_id = prev.source_id AND next.ordinal = prev.ordinal
                  AND next.build_id = ?2
                 WHERE prev.build_id = ?1 AND prev.source_id = ?3",
                params![prev.as_str(), next.as_str(), s1.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(carried_pairs, 1);
        let mismatches: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text prev
                 JOIN source_chunk_text next
                   ON next.source_id = prev.source_id AND next.ordinal = prev.ordinal
                  AND next.build_id = ?1
                 WHERE prev.build_id = ?2 AND prev.source_id = ?3
                   AND (prev.file_path <> next.file_path
                    OR prev.title <> next.title
                    OR prev.heading_path_json <> next.heading_path_json
                    OR prev.locale IS NOT next.locale
                    OR prev.ordinal <> next.ordinal
                    OR prev.range_start <> next.range_start
                    OR prev.range_end <> next.range_end
                    OR prev.body <> next.body
                    OR prev.canonical_url IS NOT next.canonical_url
                    OR prev.product_version IS NOT next.product_version)",
                params![next.as_str(), prev.as_str(), s1.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(mismatches, 0, "carry must copy every column verbatim");

        // Deleted source: never carried into the new build.
        let deleted_rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text WHERE source_id = ?1 AND build_id = ?2",
                params![s3.as_str(), next.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(deleted_rows, 0);

        // Idempotent retry: same result, no duplicate rows; the re-staged
        // source's rows are untouched by the carry.
        let carried_again = carry_source_chunks(&conn, &prev, &next, &exclude).unwrap();
        assert_eq!(carried_again, 1);
        let column_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text WHERE build_id = ?1",
                params![next.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(column_count, 2, "a retry never duplicates rows");
        let rewritten: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text
                 WHERE source_id = ?1 AND build_id = ?2 AND body = 'rewritten in next build'",
                params![s2.as_str(), next.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rewritten, 1, "the re-staged source is not overwritten");
    }

    /// PR2 recovery parity: a drifted (or missing, or over-populated)
    /// `source_fts` is repaired to exactly the ACTIVE build's staging rows;
    /// a consistent index is a no-op; nothing active clears leftovers.
    #[test]
    fn ensure_source_fts_matches_active_repairs_drift_and_clears_without_active() {
        let mut conn = open_in_memory().unwrap();
        let s1 = source(&mut conn, "a.md");
        let build_a = start_build(&mut conn, &BuildDraft::default()).unwrap();
        let build_b = start_build(&mut conn, &BuildDraft::default()).unwrap();
        replace_source_chunks(
            &mut conn,
            &s1,
            &build_a,
            "docs/a.md",
            None,
            &[chunk("Old", "alpha stale content", 0)],
        )
        .unwrap();
        replace_source_chunks(
            &mut conn,
            &s1,
            &build_b,
            "docs/a.md",
            Some("zh"),
            &[chunk("New", "beta new 检查点 content", 0)],
        )
        .unwrap();
        crate::state::activate_build(&mut conn, &build_b).unwrap();

        // Stale index (built for the previous build): repaired on ensure.
        rebuild_source_fts(&conn, &build_a).unwrap();
        assert_eq!(search_source_fts(&conn, "alpha", 10).unwrap().len(), 1);
        ensure_source_fts_matches_active(&mut conn).unwrap();
        assert!(search_source_fts(&conn, "alpha", 10).unwrap().is_empty());
        assert_eq!(search_source_fts(&conn, "beta", 10).unwrap().len(), 1);

        // Missing table (e.g. wiped index): recreated and refilled.
        conn.execute("DROP TABLE source_fts", []).unwrap();
        ensure_source_fts_matches_active(&mut conn).unwrap();
        assert_eq!(search_source_fts(&conn, "检查点", 10).unwrap().len(), 1);

        // Consistent index: a no-op (idempotent).
        ensure_source_fts_matches_active(&mut conn).unwrap();
        assert_eq!(search_source_fts(&conn, "beta", 10).unwrap().len(), 1);

        // Nothing active: leftover rows are cleared, not served.
        crate::state::set_active_build(&mut conn, None).unwrap();
        ensure_source_fts_matches_active(&mut conn).unwrap();
        assert!(search_source_fts(&conn, "beta", 10).unwrap().is_empty());
    }
}
