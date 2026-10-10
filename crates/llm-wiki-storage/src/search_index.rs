//! Full-text search index over the ACTIVE generation (PRD §20, §5.5).
//!
//! Storage model: `wiki_page_text` (migration 0008) holds one row per PAGE
//! SECTION — `(page_id, build_id, slug, title, aliases_json,
//! heading_path_json, body)` — and the `wiki_fts` FTS5 virtual table indexes
//! `(title, headings, body, aliases)` with the `unicode61` tokenizer.
//!
//! CJK strategy (PRD §20 hard rule): the default Latin tokenizer is NEVER
//! relied on for Chinese. The [`SearchTokenizer`] pre-tokenizes every field
//! (shared normalization: NFKC, lowercase, full→half width, common
//! punctuation stripped; Latin/digit runs become word tokens, Han runs become
//! bigrams PLUS unigrams); tokens are space-joined on insert so unicode61
//! preserves each token as one term. Queries must pass through the same
//! tokenizer (`llm-wiki-search::TextAnalyzer::fts_query`).
//!
//! Publish integration (PRD §35 step 6): [`activate_build_with_search_index`]
//! performs the §35 commit point — `active_build_id` + build COMPLETED — and
//! the full index rebuild in ONE transaction, so FTS and the active pointer
//! flip atomically. [`ensure_search_index_matches_active`] is the recovery
//! side check (index page set == active generation pages, rebuild on drift).

use std::collections::BTreeSet;

use rusqlite::{params, Connection};
use serde_json::json;

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, WikiPageId};
use llm_wiki_markdown::parse_document;

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// Error text used wherever FTS5 is required but the runtime SQLite lacks it.
/// Never silently degraded: search refuses to run and the publish transaction
/// aborts with this message (PRD §20).
pub const FTS5_UNAVAILABLE: &str = "SQLite build lacks FTS5; full-text search will refuse to run \
     (SQLite must be compiled with SQLITE_ENABLE_FTS5; config search.full_text = true)";

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// Pre-tokenizer shared by index time and query time (PRD §20: query and
/// index MUST use the same normalization). Implementations produce the
/// space-joined token streams stored in `wiki_fts`.
pub trait SearchTokenizer: Send + Sync {
    /// Normalizes (`NFKC` + lowercase, which folds full-width forms to their
    /// half-width ASCII equivalents) and tokenizes: runs of Latin letters and
    /// digits become single word tokens; runs of Han characters become every
    /// unigram AND every adjacent bigram, so 2–4 character words and
    /// single-character queries both match.
    fn analyze(&self, text: &str) -> Vec<String>;
}

/// The V0.2 default tokenizer implementing the §20 shared normalization.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultSearchTokenizer;

impl SearchTokenizer for DefaultSearchTokenizer {
    fn analyze(&self, text: &str) -> Vec<String> {
        tokenize(text)
    }
}

/// Static handle for call sites that only need the default strategy
/// (the publish flow passes this into the storage helpers).
pub fn default_tokenizer() -> &'static dyn SearchTokenizer {
    &DefaultSearchTokenizer
}

/// True for CJK Unified Ideographs (incl. Extension A and compatibility
/// forms) — the script that must never go through the Latin tokenizer.
///
/// Kana/Hangul are deliberately OUT of V0.2 scope: PRD §20's CJK strategy
/// targets Chinese (the zh-CN fixtures). Extending to Kana/Hangul requires
/// re-running every Top-K gate with versioned thresholds — a V0.3 candidate,
/// not a drive-by change.
fn is_han_char(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF)
}

fn normalize(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkc().flat_map(char::to_lowercase).collect()
}

fn tokenize(text: &str) -> Vec<String> {
    let normalized: String = normalize(text);
    let mut tokens = Vec::new();
    let mut run = String::new();
    for c in normalized.chars().chain(std::iter::once(' ')) {
        if c.is_alphanumeric() {
            run.push(c);
            continue;
        }
        if !run.is_empty() {
            split_run(&run, &mut tokens);
            run.clear();
        }
    }
    tokens
}

/// Splits one alphanumeric run by script: non-Han sub-runs are single word
/// tokens; Han sub-runs expand to unigrams + adjacent bigrams.
fn split_run(run: &str, tokens: &mut Vec<String>) {
    let mut sub: Vec<char> = Vec::new();
    let mut sub_is_han = is_han_char(run.chars().next().unwrap_or('a'));
    let flush = |sub: &mut Vec<char>, is_han: bool, tokens: &mut Vec<String>| {
        if sub.is_empty() {
            return;
        }
        if is_han {
            for (idx, c) in sub.iter().enumerate() {
                tokens.push(c.to_string());
                if idx + 1 < sub.len() {
                    tokens.push(format!("{}{}", c, sub[idx + 1]));
                }
            }
        } else {
            tokens.push(sub.iter().collect());
        }
        sub.clear();
    };
    for c in run.chars() {
        let han = is_han_char(c);
        if han != sub_is_han {
            flush(&mut sub, sub_is_han, tokens);
            sub_is_han = han;
        }
        sub.push(c);
    }
    flush(&mut sub, sub_is_han, tokens);
}

// ---------------------------------------------------------------------------
// FTS5 availability
// ---------------------------------------------------------------------------

/// Whether this SQLite build can host FTS5 virtual tables: probed with a
/// `CREATE VIRTUAL TABLE … USING fts5` attempt in the connection-local temp
/// database (never guesses from version strings).
pub fn probe_fts5(conn: &Connection) -> bool {
    let created = conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.llm_wiki_fts5_probe USING fts5(text)",
    );
    let available = created.is_ok();
    if available {
        let _ = conn.execute_batch("DROP TABLE IF EXISTS temp.llm_wiki_fts5_probe");
    }
    available
}

/// True when the lazily-created `wiki_fts` virtual table exists (i.e. some
/// rebuild ran). Public since PR3: the fusion layer probes it to report an
/// HONEST `IndexNotBuilt` degradation instead of an ambiguous no-match.
pub fn fts_table_exists(conn: &Connection) -> Result<bool> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'wiki_fts'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| WikiError::Storage(format!("sqlite_master lookup: {e}")))?;
    Ok(count > 0)
}

// ---------------------------------------------------------------------------
// Rebuild (runs inside the publish / recovery transaction)
// ---------------------------------------------------------------------------

/// What one rebuild wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchIndexStats {
    pub pages: usize,
    pub sections: usize,
}

/// Deletes ALL index rows and re-inserts the sections of `build_id` from
/// `wiki_pages`, tokenized by the shared analyzer. Takes `&Connection` so it
/// can run inside an open transaction (`Transaction` derefs to `Connection`) —
/// the publish flow calls it within the §35 activate transaction, where a
/// failure aborts the whole commit point.
pub fn rebuild_search_index(
    conn: &Connection,
    build_id: &BuildId,
    tokenizer: &dyn SearchTokenizer,
) -> Result<SearchIndexStats> {
    if !probe_fts5(conn) {
        return Err(WikiError::Index(FTS5_UNAVAILABLE.to_owned()));
    }
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
            title, headings, body, aliases, tokenize = 'unicode61'
        )",
    )
    .map_err(|e| WikiError::Storage(format!("create wiki_fts: {e}")))?;
    conn.execute("DELETE FROM wiki_fts", []).map_err(db)?;
    conn.execute("DELETE FROM wiki_page_text", []).map_err(db)?;

    // INSERT prepares are hoisted out of the per-page loop (rusqlite
    // `Statement::insert` also returns the rowid, so the FTS row can link to
    // its content row without a separate `last_insert_rowid` round-trip).
    let mut insert_text = conn
        .prepare(
            "INSERT INTO wiki_page_text (page_id, build_id, slug, title, aliases_json, heading_path_json, body)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )
        .map_err(|e| WikiError::Storage(format!("prepare text insert: {e}")))?;
    let mut insert_fts = conn
        .prepare(
            "INSERT INTO wiki_fts (rowid, title, headings, body, aliases) VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .map_err(|e| WikiError::Storage(format!("prepare fts insert: {e}")))?;

    let mut stmt = conn
        .prepare(
            "SELECT page_id, slug, title, content FROM wiki_pages
             WHERE build_id = ?1 ORDER BY slug",
        )
        .map_err(|e| WikiError::Storage(format!("prepare index pages: {e}")))?;
    let rows = stmt
        .query_map(params![build_id.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("index pages: {e}")))?;

    let mut pages = 0usize;
    let mut sections = 0usize;
    for row in rows {
        let (page_id, slug, title, content) = row.map_err(db)?;
        pages += 1;
        sections += index_page(
            &mut insert_text,
            &mut insert_fts,
            tokenizer,
            &page_id,
            build_id,
            &slug,
            &title,
            &content,
        )?;
    }
    Ok(SearchIndexStats { pages, sections })
}

/// Indexes ONE page: one structural parse, then a text row + FTS row per
/// section. Shared by the full rebuild and the incremental update. Returns
/// the number of section rows written.
#[allow(clippy::too_many_arguments)]
fn index_page(
    insert_text: &mut rusqlite::Statement<'_>,
    insert_fts: &mut rusqlite::Statement<'_>,
    tokenizer: &dyn SearchTokenizer,
    page_id: &str,
    build_id: &BuildId,
    slug: &str,
    title: &str,
    content: &str,
) -> Result<usize> {
    // One structural parse per page: sections (heading path + body) plus
    // the frontmatter `aliases` key — the fourth §20 index object (always
    // empty in the V0.2 fixtures).
    let parsed = parse_document(content, &format!("{slug}.md"));
    let aliases = page_aliases(parsed.frontmatter.get("aliases"));
    let section_rows: Vec<(&[String], &str)> = if parsed.sections.is_empty() {
        // A page that yields no sections still gets one row so its title
        // stays searchable.
        vec![(&[], "")]
    } else {
        parsed
            .sections
            .iter()
            .map(|section| (section.heading_path.as_slice(), section.content.as_str()))
            .collect()
    };
    let mut sections = 0usize;
    for (heading_path, body) in section_rows {
        insert_section_row(
            insert_text,
            insert_fts,
            tokenizer,
            page_id,
            build_id,
            slug,
            title,
            &aliases,
            heading_path,
            body,
        )?;
        sections += 1;
    }
    Ok(sections)
}

/// `page_id → body_hash` of one build's pages — the incremental diff key.
fn page_body_hashes(
    conn: &Connection,
    build_id: &BuildId,
) -> Result<std::collections::BTreeMap<String, String>> {
    let mut stmt = conn
        .prepare("SELECT page_id, body_hash FROM wiki_pages WHERE build_id = ?1")
        .map_err(|e| WikiError::Storage(format!("prepare page hashes: {e}")))?;
    let rows = stmt
        .query_map(params![build_id.as_str()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| WikiError::Storage(format!("page hashes: {e}")))?;
    let mut map = std::collections::BTreeMap::new();
    for row in rows {
        let (page_id, body_hash) = row.map_err(db)?;
        map.insert(page_id, body_hash);
    }
    Ok(map)
}

/// Removes one page's text rows and their FTS entries.
fn delete_page_index_rows(conn: &Connection, page_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM wiki_fts WHERE rowid IN (SELECT text_id FROM wiki_page_text WHERE page_id = ?1)",
        params![page_id],
    )
    .map_err(db)?;
    conn.execute(
        "DELETE FROM wiki_page_text WHERE page_id = ?1",
        params![page_id],
    )
    .map_err(db)?;
    Ok(())
}

/// Incremental index update (audit FIX-010): diff `build_id` against the
/// previous ACTIVE generation and touch only pages whose (page_id,
/// body_hash) pair changed — unchanged pages keep their tokenized rows (the
/// build stamp moves) and their FTS entries; removed pages lose theirs.
/// `prev_build = None` (first publish) falls back to the full rebuild. Same
/// transaction contract as the rebuild: runs inside the §35 activate
/// transaction, where a failure aborts the whole commit point.
pub fn update_search_index(
    conn: &Connection,
    prev_build: Option<&BuildId>,
    build_id: &BuildId,
    tokenizer: &dyn SearchTokenizer,
) -> Result<SearchIndexStats> {
    if !probe_fts5(conn) {
        return Err(WikiError::Index(FTS5_UNAVAILABLE.to_owned()));
    }
    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
            title, headings, body, aliases, tokenize = 'unicode61'
        )",
    )
    .map_err(|e| WikiError::Storage(format!("create wiki_fts: {e}")))?;
    let Some(prev_build) = prev_build else {
        return rebuild_search_index(conn, build_id, tokenizer);
    };

    let prev_hashes = page_body_hashes(conn, prev_build)?;
    let new_hashes = page_body_hashes(conn, build_id)?;

    let mut insert_text = conn
        .prepare(
            "INSERT INTO wiki_page_text (page_id, build_id, slug, title, aliases_json, heading_path_json, body)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )
        .map_err(|e| WikiError::Storage(format!("prepare text insert: {e}")))?;
    let mut insert_fts = conn
        .prepare(
            "INSERT INTO wiki_fts (rowid, title, headings, body, aliases) VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .map_err(|e| WikiError::Storage(format!("prepare fts insert: {e}")))?;
    let mut select_page = conn
        .prepare("SELECT slug, title, content FROM wiki_pages WHERE build_id = ?1 AND page_id = ?2")
        .map_err(|e| WikiError::Storage(format!("prepare index page: {e}")))?;

    let mut reindexed = 0usize;
    for (page_id, body_hash) in &new_hashes {
        match prev_hashes.get(page_id) {
            // Byte-identical carried page: keep the tokenized rows, move the
            // build stamp so the active-build filter keeps finding it.
            Some(prev_hash) if prev_hash == body_hash => {
                conn.execute(
                    "UPDATE wiki_page_text SET build_id = ?1 WHERE page_id = ?2",
                    params![build_id.as_str(), page_id],
                )
                .map_err(db)?;
            }
            // Changed or brand-new page: re-parse and re-tokenize.
            _ => {
                delete_page_index_rows(conn, page_id)?;
                let (slug, title, content) = select_page
                    .query_row(params![build_id.as_str(), page_id], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    })
                    .map_err(db)?;
                index_page(
                    &mut insert_text,
                    &mut insert_fts,
                    tokenizer,
                    page_id,
                    build_id,
                    &slug,
                    &title,
                    &content,
                )?;
                reindexed += 1;
            }
        }
    }
    for page_id in prev_hashes.keys() {
        if !new_hashes.contains_key(page_id) {
            delete_page_index_rows(conn, page_id)?;
        }
    }
    let sections: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM wiki_page_text WHERE build_id = ?1",
            params![build_id.as_str()],
            |row| row.get(0),
        )
        .map_err(db)?;
    tracing::debug!(
        pages = new_hashes.len(),
        reindexed,
        "incremental FTS update"
    );
    Ok(SearchIndexStats {
        pages: new_hashes.len(),
        sections: sections as usize,
    })
}

/// Frontmatter `aliases` (comma-separated, V0.2 fixtures leave it empty) as
/// the fourth §20 index object.
fn page_aliases(parsed_aliases: Option<&str>) -> Vec<String> {
    parsed_aliases
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|alias| !alias.is_empty())
        .map(str::to_owned)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn insert_section_row(
    text_stmt: &mut rusqlite::Statement<'_>,
    fts_stmt: &mut rusqlite::Statement<'_>,
    tokenizer: &dyn SearchTokenizer,
    page_id: &str,
    build_id: &BuildId,
    slug: &str,
    title: &str,
    aliases: &[String],
    heading_path: &[String],
    body: &str,
) -> Result<()> {
    // `Statement::insert` executes AND returns the rowid — the FTS row links
    // to its content row without a second round-trip.
    let rowid = text_stmt
        .insert(params![
            page_id,
            build_id.as_str(),
            slug,
            title,
            json!(aliases).to_string(),
            json!(heading_path).to_string(),
            body
        ])
        .map_err(db)?;
    fts_stmt
        .insert(params![
            rowid,
            tokenizer.analyze(title).join(" "),
            tokenizer.analyze(&heading_path.join(" ")).join(" "),
            tokenizer.analyze(body).join(" "),
            tokenizer.analyze(&aliases.join(" ")).join(" ")
        ])
        .map_err(db)?;
    Ok(())
}

/// The §35 publish commit point (step 6) with the derived indexes joined in:
/// ONE transaction switches `active_build_id` + marks the build COMPLETED,
/// rebuilds the FTS index for the activated generation, rebuilds the §17 Wiki
/// Graph AND rebuilds the raw-source index (`chunks::rebuild_source_fts` —
/// EPIC A PR2, FULLY for the activated build: staging is complete by then via
/// the build-time writes + carry-forward). Any failure — including an
/// FTS5-less SQLite — aborts the transaction, leaving the previous
/// generation fully visible (PRD §35). The graph rebuild is deliberately
/// NOT config-gated: `search.graph` gates query-side consumption only, the
/// stored graph always matches the active generation.
pub fn activate_build_with_search_index(
    conn: &mut Connection,
    build_id: &BuildId,
    tokenizer: &dyn SearchTokenizer,
) -> Result<SearchIndexStats> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    // The diff base is the CURRENT active generation, read before the swap:
    // the derived indexes move to the new state first (incrementally, audit
    // FIX-010/011 — O(changed), not O(total)), THEN the pointer flips, so the
    // activation itself stays short (audit FIX-012). The source index is the
    // deliberate exception: it rebuilds in full every publish (EPIC A PR2
    // decision — the carry keeps staging correct; profiling waits for EPIC H).
    let prev_build = crate::state::get_active_build_id(&tx)?;
    let stats = update_search_index(&tx, prev_build.as_ref(), build_id, tokenizer)?;
    let graph_stats = crate::graph::update_graph(&tx, prev_build.as_ref(), build_id)?;
    let source_fts_rows = crate::chunks::rebuild_source_fts(&tx, build_id)?;
    crate::state::activate_in_tx(&tx, build_id)?;
    tracing::debug!(
        nodes = graph_stats.nodes,
        edges = graph_stats.edges,
        skipped = graph_stats.skipped_edges,
        source_chunks = source_fts_rows,
        "wiki graph and source index updated with the activated generation"
    );
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit activate_build: {e}")))?;
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Recovery-side verification
// ---------------------------------------------------------------------------

/// Verifies that the index covers exactly the ACTIVE generation's pages and
/// rebuilds on drift (idempotent, cheap: two page-id set reads). Called at the
/// end of publish recovery; `Ok(None)` means the index already matched. With
/// nothing active, a non-empty index is cleared so search reports the truth.
pub fn ensure_search_index_matches_active(
    conn: &mut Connection,
    tokenizer: &dyn SearchTokenizer,
) -> Result<Option<SearchIndexStats>> {
    let active = crate::state::get_active_build_id(conn)?;
    let Some(active) = active else {
        if index_has_rows(conn)? {
            // One transaction: both index tables clear together (spec §42 —
            // mutations run in a transaction; idempotent if a crash re-runs it).
            let tx = conn
                .transaction()
                .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
            clear_search_index(&tx)?;
            tx.commit()
                .map_err(|e| WikiError::Storage(format!("commit clear index: {e}")))?;
            tracing::warn!("cleared the search index: no generation is active");
        }
        return Ok(None);
    };
    if index_page_ids(conn)? == active_page_ids(conn, &active)? {
        return Ok(None);
    }
    tracing::warn!(
        build = %active,
        "search index does not match the active generation; rebuilding"
    );
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let stats = rebuild_search_index(&tx, &active, tokenizer)?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit index rebuild: {e}")))?;
    Ok(Some(stats))
}

fn index_has_rows(conn: &Connection) -> Result<bool> {
    Ok(conn
        .query_row("SELECT COUNT(*) FROM wiki_page_text", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(db)?
        > 0)
}

fn index_page_ids(conn: &Connection) -> Result<BTreeSet<String>> {
    let mut stmt = conn
        .prepare("SELECT DISTINCT page_id FROM wiki_page_text")
        .map_err(|e| WikiError::Storage(format!("prepare index page ids: {e}")))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| WikiError::Storage(format!("index page ids: {e}")))?;
    let mut set = BTreeSet::new();
    for row in rows {
        set.insert(row.map_err(db)?);
    }
    Ok(set)
}

fn active_page_ids(conn: &Connection, build_id: &BuildId) -> Result<BTreeSet<String>> {
    let mut stmt = conn
        .prepare("SELECT DISTINCT page_id FROM wiki_pages WHERE build_id = ?1")
        .map_err(|e| WikiError::Storage(format!("prepare active page ids: {e}")))?;
    let rows = stmt
        .query_map(params![build_id.as_str()], |row| row.get::<_, String>(0))
        .map_err(|e| WikiError::Storage(format!("active page ids: {e}")))?;
    let mut set = BTreeSet::new();
    for row in rows {
        set.insert(row.map_err(db)?);
    }
    Ok(set)
}

/// Empties the index (recovery path: nothing is published).
pub fn clear_search_index(conn: &Connection) -> Result<()> {
    if fts_table_exists(conn)? {
        conn.execute("DELETE FROM wiki_fts", []).map_err(db)?;
    }
    conn.execute("DELETE FROM wiki_page_text", []).map_err(db)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// One section-level hit (PRD §5.5: search returns page/section, not answers).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchIndexRow {
    pub page_id: WikiPageId,
    pub slug: String,
    pub title: String,
    pub heading_path: Vec<String>,
    /// FTS5 `snippet()` over the body column with `[…]` highlight marks.
    pub snippet: String,
    /// FTS5 `bm25()` rank — numerically SMALLER is a better match.
    pub rank: f64,
}

/// Runs an FTS5 MATCH query (a pre-built, quoted expression from the shared
/// analyzer — never raw user input) and returns the best `limit` hits of the
/// ACTIVE generation, ranked by bm25, with body snippets. A never-built index
/// yields no rows; no active build is a hard `Index` error.
pub fn search_index(
    conn: &Connection,
    fts_query: &str,
    limit: usize,
) -> Result<Vec<SearchIndexRow>> {
    // No active build is a hard error (never a silent empty result); an
    // index that was never built is an honest empty result.
    let active = crate::state::get_active_build_id(conn)?
        .ok_or_else(|| WikiError::Index("nothing published; run build first".into()))?;
    if !fts_table_exists(conn)? {
        return Ok(Vec::new());
    }

    let mut stmt = conn
        .prepare(
            "SELECT t.page_id, t.slug, t.title, t.heading_path_json,
                    snippet(wiki_fts, 2, '[', ']', ' … ', 16), bm25(wiki_fts)
             FROM wiki_fts
             JOIN wiki_page_text t ON t.text_id = wiki_fts.rowid
             WHERE wiki_fts MATCH ?1 AND t.build_id = ?2
             ORDER BY bm25(wiki_fts)
             LIMIT ?3",
        )
        .map_err(|e| WikiError::Storage(format!("prepare search: {e}")))?;
    let rows = stmt
        .query_map(params![fts_query, active.as_str(), limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("search: {e}")))?;

    let mut out = Vec::new();
    for row in rows {
        let (page_id, slug, title, heading_json, snippet, rank) = row.map_err(db)?;
        let heading_path: Vec<String> = serde_json::from_str(&heading_json).unwrap_or_default();
        out.push(SearchIndexRow {
            page_id: WikiPageId::from_validated(page_id),
            slug,
            title,
            heading_path,
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
    use crate::state::get_active_build_id;

    use super::activate_build_with_search_index as _activate;

    fn page(slug: &str, title: &str, content: &str) -> crate::wiki::WikiPageRecord {
        crate::wiki::WikiPageRecord {
            page_id: WikiPageId::generate(),
            slug: slug.to_owned(),
            title: title.to_owned(),
            category: "concepts".into(),
            language: "en".into(),
            body_hash: llm_wiki_core::hash::sha256_hex(content.as_bytes()),
            content: content.to_owned(),
            knowledge_refs: Vec::new(),
            citations: Vec::new(),
            links: Vec::new(),
        }
    }

    fn persist(conn: &mut Connection, build_id: &BuildId, pages: &[crate::wiki::WikiPageRecord]) {
        crate::wiki::persist_generation(conn, build_id, pages).unwrap();
    }

    fn search(conn: &Connection, query: &str, limit: usize) -> Vec<SearchIndexRow> {
        let fts_query = DefaultSearchTokenizer.fts_expression(query);
        search_index(conn, &fts_query, limit).unwrap()
    }

    trait FtsExpr {
        fn fts_expression(&self, text: &str) -> String;
    }
    impl FtsExpr for DefaultSearchTokenizer {
        // Minimal OR-of-quoted-terms builder mirroring the search crate's
        // `fts_query`; keeps these tests independent of that crate.
        fn fts_expression(&self, text: &str) -> String {
            let mut seen = BTreeSet::new();
            let mut terms = Vec::new();
            for token in self.analyze(text) {
                if seen.insert(token.clone()) {
                    terms.push(format!("\"{}\"", token.replace('"', "\"\"")));
                }
            }
            terms.join(" OR ")
        }
    }

    #[test]
    fn probe_reports_fts5_on_bundled_sqlite() {
        let conn = open_in_memory().unwrap();
        if probe_fts5(&conn) {
            // Bundled SQLite compiles with SQLITE_ENABLE_FTS5; a second probe
            // is stable and leaves no temp table behind.
            assert!(probe_fts5(&conn));
            let leftovers: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_temp_master WHERE name LIKE 'llm_wiki_fts5_probe%'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(leftovers, 0);
        }
        // A build without FTS5 simply reports false — callers degrade loudly
        // (search refuses, doctor FAILs); nothing to assert here.
    }

    #[test]
    fn tokenizer_normalizes_nfkc_case_and_width() {
        let t = DefaultSearchTokenizer;
        assert_eq!(t.analyze("Access Tokens"), vec!["access", "tokens"]);
        // NFKC folds full-width forms onto ASCII and lowercases.
        assert_eq!(t.analyze("ＡＣＣＥＳＳ １２３"), vec!["access", "123"]);
        assert_eq!(t.analyze("ＳＳＯ"), vec!["sso"]);
        // Common punctuation (ASCII and CJK) separates tokens.
        assert_eq!(
            t.analyze("single sign-on (SSO)"),
            vec!["single", "sign", "on", "sso"]
        );
        assert_eq!(
            t.analyze("处理失败后，流任务会恢复。"),
            t.analyze("处理失败后 流任务会恢复"),
        );
    }

    #[test]
    fn tokenizer_han_runs_produce_unigrams_and_bigrams() {
        let t = DefaultSearchTokenizer;
        // Interleaved per position: unigram, then the bigram it starts.
        assert_eq!(
            t.analyze("检查点"),
            vec!["检", "检查", "查", "查点", "点"],
            "2-char words and single-char queries must both match"
        );
        assert_eq!(t.analyze("登录"), vec!["登", "登录", "录"]);
        assert_eq!(t.analyze("流"), vec!["流"], "single char stays usable");
    }

    #[test]
    fn tokenizer_mixed_scripts_keep_both_strategies() {
        let t = DefaultSearchTokenizer;
        assert_eq!(
            t.analyze("SSO登录"),
            vec!["sso", "登", "登录", "录"],
            "Latin word token + Han bigram/unigram in one run"
        );
        assert_eq!(
            t.analyze("ＳＳＯ单点登录"),
            vec!["sso", "单", "单点", "点", "点登", "登", "登录", "录"],
            "NFKC + script split on full-width input"
        );
        // Latin+digit runs are one word token; CJK neighbors split the run.
        assert_eq!(
            t.analyze("HMAC-SHA256签名"),
            vec!["hmac", "sha256", "签", "签名", "名"]
        );
    }

    #[test]
    fn rebuild_and_search_roundtrip_sections_of_the_active_build() {
        let mut conn = open_in_memory().unwrap();
        assert!(probe_fts5(&conn), "bundled SQLite ships FTS5");
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        let older = start_build(&mut conn, &BuildDraft::default()).unwrap();
        persist(
            &mut conn,
            &older,
            &[page(
                "stale",
                "Stale Page",
                "# Stale Page\n\nstale 检查点 body",
            )],
        );
        persist(
            &mut conn,
            &build,
            &[
                page(
                    "streaming",
                    "Streaming Processing",
                    "# Streaming Processing\n\n## Checkpoints\n\n检查点默认每 30 秒持久化一次。\n\n## Windows\n\ntumbling windows close on fixed boundaries\n",
                ),
                page(
                    "sso",
                    "Identity & Access",
                    "# Identity & Access\n\n企业可以通过 SAML 或 OIDC 配置单点登录（SSO）。\n",
                ),
            ],
        );

        // Nothing active yet → search is a hard error (never silent empty).
        assert!(search_index(&conn, "\"检查\"", 10).is_err());

        let stats = _activate(&mut conn, &build, default_tokenizer()).unwrap();
        assert_eq!(stats.pages, 2);
        assert_eq!(stats.sections, 4, "2+1+1 section rows");
        assert_eq!(get_active_build_id(&conn).unwrap(), Some(build.clone()));

        // CJK bigram query hits the right section with its heading path.
        let hits = search(&conn, "检查点", 10);
        assert_eq!(hits[0].slug, "streaming");
        assert_eq!(
            hits[0].heading_path,
            vec!["Streaming Processing", "Checkpoints"]
        );
        assert!(hits[0].snippet.contains('[') || !hits[0].snippet.is_empty());
        assert!(hits[0].rank <= 0.0, "bm25 ranks are non-positive");

        // Mixed Latin/CJK: the SSO page matches both terms of its body.
        let hits = search(&conn, "SSO 登录", 10);
        assert_eq!(hits[0].slug, "sso");

        // EN body term through the same analyzer.
        let hits = search(&conn, "tumbling windows", 10);
        assert!(hits.iter().any(|hit| hit.slug == "streaming"));

        // Older, non-active generations are invisible even when re-searched.
        let hits = search(&conn, "stale 检查点", 10);
        assert!(
            hits.iter().all(|hit| hit.slug != "stale"),
            "index covers the active generation only"
        );

        // Limit is honored and ranks are ordered best-first.
        let hits = search(&conn, "检查点 单点登录 streaming tumbling", 2);
        assert_eq!(hits.len(), 2);
        assert!(hits.windows(2).all(|w| w[0].rank <= w[1].rank));
    }

    #[test]
    fn rebuild_is_idempotent_and_replaces_previous_content() {
        let mut conn = open_in_memory().unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        persist(
            &mut conn,
            &build,
            &[page("docs", "Docs", "# Docs\n\nalpha content")],
        );
        _activate(&mut conn, &build, default_tokenizer()).unwrap();
        let stats = _activate(&mut conn, &build, default_tokenizer()).unwrap();
        assert_eq!(stats.pages, 1);
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM wiki_page_text", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 1, "delete-all + reinsert never duplicates");
    }

    #[test]
    fn activation_failure_rolls_back_pointer_and_index() {
        let mut conn = open_in_memory().unwrap();
        // No builds row → activate_in_tx fails and the whole transaction
        // (pointer flip + index rebuild) rolls back.
        let ghost = BuildId::generate();
        persist(
            &mut conn,
            &ghost,
            &[page("orphan", "Orphan", "# Orphan\n\norphan body")],
        );
        assert!(_activate(&mut conn, &ghost, default_tokenizer()).is_err());
        assert!(get_active_build_id(&conn).unwrap().is_none());
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM wiki_page_text", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0, "the index rebuild joined the aborted transaction");
    }

    /// EPIC A PR2: the raw-source index rebuild joins the SAME commit point —
    /// an activation failure must leave `source_fts` serving the previous
    /// state exactly like `wiki_fts`, never a half-flipped source index.
    #[test]
    fn source_fts_rebuild_joins_the_activation_transaction() {
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

        // A previous build's source index exists and is searchable. (Staging
        // needs no builds row; the ghost build below deliberately has none so
        // activate_in_tx fails.)
        let live = BuildId::generate();
        crate::chunks::replace_source_chunks(
            &mut conn,
            &source_id,
            &live,
            "docs/a.md",
            None,
            &[crate::chunks::ChunkInput {
                title: "Live",
                heading_path: &[],
                ordinal: 0,
                range_start: 0,
                range_end: 17,
                body: "alpha legacy body",
            }],
        )
        .unwrap();
        crate::chunks::rebuild_source_fts(&conn, &live).unwrap();
        assert_eq!(
            crate::chunks::search_source_fts(&conn, "alpha", 10)
                .unwrap()
                .len(),
            1
        );

        // The ghost build would flip the source index to its own staging
        // rows — but its activation fails, so the flip must roll back.
        let ghost = BuildId::generate();
        crate::chunks::replace_source_chunks(
            &mut conn,
            &source_id,
            &ghost,
            "docs/a.md",
            None,
            &[crate::chunks::ChunkInput {
                title: "Ghost",
                heading_path: &[],
                ordinal: 0,
                range_start: 0,
                range_end: 15,
                body: "ghost staging body",
            }],
        )
        .unwrap();
        persist(
            &mut conn,
            &ghost,
            &[page("orphan", "Orphan", "# Orphan\n\norphan body")],
        );

        assert!(_activate(&mut conn, &ghost, default_tokenizer()).is_err());
        assert!(get_active_build_id(&conn).unwrap().is_none());
        // Sanity: the ghost staging rows existed, so the in-tx rebuild really
        // had a full set to flip to before the rollback restored the old one.
        let ghost_staged: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM source_chunk_text WHERE build_id = ?1",
                params![ghost.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ghost_staged, 1);
        assert!(
            crate::chunks::search_source_fts(&conn, "ghost", 10)
                .unwrap()
                .is_empty(),
            "the ghost index flip must not survive the failed activation"
        );
        let hits = crate::chunks::search_source_fts(&conn, "alpha", 10).unwrap();
        assert_eq!(hits.len(), 1, "the aborted source rebuild rolled back");
        assert_eq!(hits[0].build_id, live.as_str());
    }

    /// Audit FIX-010 acceptance: the incremental update path must land in
    /// EXACTLY the same state as a full rebuild of the same generation —
    /// carried pages keep their rows (re-stamped), changed pages re-tokenize,
    /// removed pages leave the index.
    #[test]
    fn incremental_update_equals_full_rebuild() {
        let tokenizer = default_tokenizer();
        let ids: Vec<WikiPageId> = [
            "wp_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "wp_01BX5ZZKBKACTAV9WEVGEMMVRZ",
            "wp_01CZZZZZZZZZZZZZZZZZZZZZZZ",
            "wp_01DZZZZZZZZZZZZZZZZZZZZZZZ",
        ]
        .map(WikiPageId::parse)
        .map(Result::unwrap)
        .to_vec();
        let with_id =
            |id: &WikiPageId, slug: &str, title: &str, content: &str| crate::wiki::WikiPageRecord {
                page_id: id.clone(),
                slug: slug.to_owned(),
                title: title.to_owned(),
                category: "concepts".into(),
                language: "en".into(),
                body_hash: llm_wiki_core::hash::sha256_hex(content.as_bytes()),
                content: content.to_owned(),
                knowledge_refs: Vec::new(),
                citations: Vec::new(),
                links: Vec::new(),
            };
        // Generation A: three pages.
        let a_pages = vec![
            with_id(
                &ids[0],
                "carried",
                "Carried Page",
                "# Carried\n\ncarried 检查点 body",
            ),
            with_id(
                &ids[1],
                "changed",
                "Changed Page",
                "# Changed\n\nold 单点登录 body",
            ),
            with_id(
                &ids[2],
                "dropped",
                "Dropped Page",
                "# Dropped\n\ndropped content",
            ),
        ];
        // Generation B: page 0 carried verbatim, page 1 recompiled (same id,
        // new content), page 3 added, page 2 removed.
        let b_pages = vec![
            with_id(
                &ids[0],
                "carried",
                "Carried Page",
                "# Carried\n\ncarried 检查点 body",
            ),
            with_id(
                &ids[1],
                "changed",
                "Changed Page",
                "# Changed\n\nnew 流任务 body",
            ),
            with_id(
                &ids[3],
                "fresh",
                "Fresh Page",
                "# Fresh\n\nfresh SSO 登录 body",
            ),
        ];

        // Connection 1: A activated fully, then B via the INCREMENTAL path.
        let mut incremental = open_in_memory().unwrap();
        let build_a = start_build(&mut incremental, &BuildDraft::default()).unwrap();
        persist(&mut incremental, &build_a, &a_pages);
        _activate(&mut incremental, &build_a, tokenizer).unwrap();
        let build_b = start_build(&mut incremental, &BuildDraft::default()).unwrap();
        persist(&mut incremental, &build_b, &b_pages);
        _activate(&mut incremental, &build_b, tokenizer).unwrap();

        // Connection 2: B via the FULL rebuild path only.
        let mut full = open_in_memory().unwrap();
        let build_b2 = start_build(&mut full, &BuildDraft::default()).unwrap();
        persist(&mut full, &build_b2, &b_pages);
        _activate(&mut full, &build_b2, tokenizer).unwrap();

        let snapshot = |conn: &Connection| -> Vec<(String, String, String, String, String)> {
            let mut stmt = conn
                .prepare(
                    "SELECT t.page_id, t.slug, t.title, t.heading_path_json, f.body
                     FROM wiki_page_text t JOIN wiki_fts f ON f.rowid = t.text_id
                     ORDER BY t.page_id, t.heading_path_json",
                )
                .unwrap();
            stmt.query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
        };
        assert_eq!(
            snapshot(&incremental),
            snapshot(&full),
            "incremental update must equal a full rebuild of the same generation"
        );

        // The active-build filter still finds carried content after the
        // re-stamp, and removed content is gone.
        let hits = search(&incremental, "检查点", 10);
        assert!(hits.iter().any(|hit| hit.slug == "carried"));
        let hits = search(&incremental, "dropped", 10);
        assert!(hits.is_empty(), "removed page leaves the index");
    }

    #[test]
    fn recovery_check_rebuilds_when_the_index_drifts() {
        let mut conn = open_in_memory().unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        persist(
            &mut conn,
            &build,
            &[page("alpha", "Alpha", "# Alpha\n\nalpha body")],
        );
        _activate(&mut conn, &build, default_tokenizer()).unwrap();
        assert_eq!(
            ensure_search_index_matches_active(&mut conn, default_tokenizer()).unwrap(),
            None
        );

        // A page appears without the index noticing (simulated drift).
        let extra = page("beta", "Beta", "# Beta\n\nbeta 单点登录 body");
        crate::wiki::persist_generation(&mut conn, &build, &[extra]).unwrap();
        let stats = ensure_search_index_matches_active(&mut conn, default_tokenizer())
            .unwrap()
            .expect("drift triggers a rebuild");
        assert_eq!(stats.pages, 2);
        assert_eq!(
            ensure_search_index_matches_active(&mut conn, default_tokenizer()).unwrap(),
            None
        );

        // Clearing the active build empties the index.
        crate::state::set_active_build(&mut conn, None).unwrap();
        ensure_search_index_matches_active(&mut conn, default_tokenizer()).unwrap();
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM wiki_page_text", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0);
    }
}
