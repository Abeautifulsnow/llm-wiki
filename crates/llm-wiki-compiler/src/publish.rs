//! Atomic publish (PRD §35): every build writes an immutable generation under
//! `{wiki_dir}/generations/{build_id}/` and the visible wiki is switched by
//! replacing the small `current.json` pointer file — never by overwriting a
//! live wiki.
//!
//! Publish flow, in order:
//! 1. write all page files into `generations/{build_id}/`;
//! 2. validate the generation (non-empty page set, every file present, content
//!    hash matches `WikiPageRecord.body_hash`);
//! 3. mark the build READY (generation rows are persisted by the caller);
//! 4. record the publish intent `{old_build_id, new_build_id}` in
//!    `.publish-journal.json` (temp file + rename);
//! 5. atomically replace `current.json` (temp file in the same directory,
//!    then rename over the target);
//! 6. ONE database transaction: switch `active_build_id` AND mark the build
//!    COMPLETED, with the FTS search-index rebuild joined into the same
//!    transaction (PRD §20 — index and pointer flip atomically);
//! 7. delete the journal file.
//!
//! A crash after any step is resolved by [`recover_if_needed`]: the journal
//! plus the `current.json` vs DB comparison selects exactly one outcome —
//! complete the verified new version, or roll back to the explicit old
//! version. A pointer/DB mismatch WITHOUT a journal is
//! `WikiError::PublishRecovery`; recovery never guesses.
//!
//! Windows constraints: only small pointer files are atomically replaced;
//! rename uses a bounded retry for transient sharing violations; no
//! directory-swap atomicity is assumed and the previous good generation is
//! always retained.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rusqlite::params;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::BuildId;
use llm_wiki_storage::{
    activate_build_with_search_index, default_tokenizer, ensure_graph_matches_active,
    ensure_search_index_matches_active, get_active_build_id, load_generation_pages,
    set_active_build, update_build_status, WikiPageRecord,
};

/// Bounded rename retry (PRD §35): a reader holding the pointer file open on
/// Windows produces a transient sharing violation; give up after this many
/// attempts with a clear error.
pub const RENAME_ATTEMPTS: u32 = 10;
/// Pause between rename retries.
pub const RENAME_RETRY_DELAY: Duration = Duration::from_millis(25);

const POINTER_FILE: &str = "current.json";
const JOURNAL_FILE: &str = ".publish-journal.json";
const GENERATIONS_DIR: &str = "generations";

// ---------------------------------------------------------------------------
// Paths and on-disk documents
// ---------------------------------------------------------------------------

/// Layout of one `wiki_dir` (PRD §35).
#[derive(Debug, Clone)]
pub struct PublishPaths {
    wiki_dir: PathBuf,
}

impl PublishPaths {
    pub fn new(wiki_dir: &Path) -> Self {
        Self {
            wiki_dir: wiki_dir.to_path_buf(),
        }
    }

    pub fn wiki_dir(&self) -> &Path {
        &self.wiki_dir
    }

    pub fn generations_dir(&self) -> PathBuf {
        self.wiki_dir.join(GENERATIONS_DIR)
    }

    pub fn generation_dir(&self, build_id: &BuildId) -> PathBuf {
        self.generations_dir().join(build_id.as_str())
    }

    pub fn pointer_path(&self) -> PathBuf {
        self.wiki_dir.join(POINTER_FILE)
    }

    pub fn journal_path(&self) -> PathBuf {
        self.wiki_dir.join(JOURNAL_FILE)
    }
}

/// `current.json`: ONLY the visible build id plus minimal metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentPointer {
    pub build_id: String,
    pub published_at: String,
}

/// `.publish-journal.json`: the very short-lived publish intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishJournal {
    pub old_build_id: Option<String>,
    pub new_build_id: String,
}

/// Outcome of a successful publish.
#[derive(Debug, Clone)]
pub struct PublishReport {
    pub build_id: BuildId,
    /// Generation the pointer pointed at before this publish (None on the
    /// first publish).
    pub previous_build: Option<BuildId>,
    pub published_path: PathBuf,
    /// Warnings emitted by post-publish cleanup (cleanup failures are never
    /// publish failures, PRD §35).
    pub cleanup_warnings: Vec<String>,
}

/// What recovery did, reported to logs and callers (PRD §35).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryReport {
    /// `completed` (new version finished) or `rolled-back` (explicit old
    /// version restored).
    pub action: RecoveryAction,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// The pointer had already moved and the verified new version was finished
    /// in the database (or the journal was a leftover after a full commit).
    CompletedNew,
    /// The publish rolled back to the explicitly recorded old version.
    RolledBack,
}

// ---------------------------------------------------------------------------
// Atomic small-file replacement
// ---------------------------------------------------------------------------

/// Windows sharing violations / access denied during rename are transient —
/// another process (or a virus scanner) holds the target open. Everything
/// else fails immediately.
fn is_transient_rename_error(err: &std::io::Error) -> bool {
    matches!(err.raw_os_error(), Some(5 | 32))
}

/// Runs one rename-like operation with a bounded retry on transient errors.
/// Unit-testable seam: the operation is injected.
fn rename_with_retry<F>(mut op: F, attempts: u32, delay: Duration) -> Result<()>
where
    F: FnMut() -> std::io::Result<()>,
{
    let attempts = attempts.max(1);
    for attempt in 1..=attempts {
        match op() {
            Ok(()) => return Ok(()),
            Err(err) if is_transient_rename_error(&err) && attempt < attempts => {
                tracing::debug!(attempt, error = %err, "transient rename failure, retrying");
                std::thread::sleep(delay);
            }
            Err(err) => {
                return Err(WikiError::Storage(format!(
                    "rename failed after {attempt} attempt(s): {err}"
                )));
            }
        }
    }
    Ok(())
}

/// Atomically replaces `target` with `contents`: the temp file is written in
/// the SAME directory (so the rename stays on one filesystem) and then renamed
/// over the target with the bounded retry.
pub fn atomic_write(target: &Path, contents: &str) -> Result<()> {
    let tmp = target.with_extension(format!("tmp.{}", std::process::id()));
    if let Err(e) = std::fs::write(&tmp, contents) {
        let _ = std::fs::remove_file(&tmp);
        return Err(WikiError::Storage(format!(
            "cannot write {}: {e}",
            tmp.display()
        )));
    }
    let result = rename_with_retry(
        || std::fs::rename(&tmp, target),
        RENAME_ATTEMPTS,
        RENAME_RETRY_DELAY,
    );
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

// ---------------------------------------------------------------------------
// Pointer + journal IO
// ---------------------------------------------------------------------------

/// Reads `current.json`. A missing file is `None`; a corrupt file is a
/// `PublishRecovery` error — the pointer must never be guessed around.
pub fn read_current_pointer(paths: &PublishPaths) -> Result<Option<CurrentPointer>> {
    let path = paths.pointer_path();
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| WikiError::PublishRecovery(format!("cannot read {}: {e}", path.display())))?;
    let pointer: CurrentPointer = serde_json::from_str(&text)
        .map_err(|e| WikiError::PublishRecovery(format!("{} is corrupt: {e}", path.display())))?;
    Ok(Some(pointer))
}

/// Step 5: atomically replaces `current.json`.
pub fn write_current_pointer(paths: &PublishPaths, build_id: &BuildId) -> Result<()> {
    let pointer = CurrentPointer {
        build_id: build_id.as_str().to_owned(),
        published_at: chrono::Utc::now().to_rfc3339(),
    };
    let json = serde_json::to_string_pretty(&pointer)
        .map_err(|e| WikiError::Storage(format!("serialize pointer: {e}")))?;
    atomic_write(&paths.pointer_path(), &json)
}

/// Whether an un-recovered publish intent exists (doctor check, PRD §29).
pub fn journal_exists(paths: &PublishPaths) -> bool {
    paths.journal_path().exists()
}

/// Reads the publish journal; a corrupt journal is a `PublishRecovery` error
/// (recovery never guesses, PRD §35).
pub fn read_journal(paths: &PublishPaths) -> Result<Option<PublishJournal>> {
    let path = paths.journal_path();
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| WikiError::PublishRecovery(format!("cannot read publish journal: {e}")))?;
    let journal: PublishJournal = serde_json::from_str(&text)
        .map_err(|e| WikiError::PublishRecovery(format!("publish journal is corrupt: {e}")))?;
    Ok(Some(journal))
}

/// Step 4: records the publish intent with the same atomicity as the pointer.
pub fn write_journal(
    paths: &PublishPaths,
    old_build_id: Option<&BuildId>,
    new_build_id: &BuildId,
) -> Result<()> {
    let journal = PublishJournal {
        old_build_id: old_build_id.map(|b| b.as_str().to_owned()),
        new_build_id: new_build_id.as_str().to_owned(),
    };
    let json = serde_json::to_string_pretty(&journal)
        .map_err(|e| WikiError::Storage(format!("serialize journal: {e}")))?;
    atomic_write(&paths.journal_path(), &json)
}

/// Step 7: clears the publish intent (missing file is fine).
pub fn clear_journal(paths: &PublishPaths) -> Result<()> {
    let path = paths.journal_path();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(WikiError::Storage(format!(
            "cannot clear publish journal {}: {e}",
            path.display()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Generation writing / validation
// ---------------------------------------------------------------------------

/// Filesystem-safe file name for one page. Planner slugs are lowercase
/// alphanumeric + dashes, but pages compiled from other sources must never be
/// able to escape the generation directory. Public so lint (§36) maps page
/// rows back onto their generation files with the exact same rule.
pub fn page_file_name(slug: &str, page_id: &str) -> String {
    let safe: String = slug
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = safe.trim_matches('-');
    if trimmed.is_empty() {
        format!("{page_id}.md")
    } else {
        format!("{trimmed}.md")
    }
}

/// Computes the reuse set for one publish: pages whose (page_id, body_hash)
/// pair exists in the previous generation AND whose previous-generation file
/// is still on disk. Body hash equality means the compiled content is
/// byte-identical, so the old file can back the new one via hardlink.
fn reusable_pages(
    conn: &rusqlite::Connection,
    paths: &PublishPaths,
    previous: Option<&BuildId>,
    pages: &[WikiPageRecord],
) -> Result<ReusablePages> {
    let Some(previous) = previous else {
        return Ok(BTreeMap::new());
    };
    let mut stmt = conn
        .prepare("SELECT page_id, body_hash, slug FROM wiki_pages WHERE build_id = ?1")
        .map_err(|e| WikiError::Storage(format!("prepare prev pages: {e}")))?;
    let prev_rows = stmt
        .query_map(params![previous.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("prev pages: {e}")))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| WikiError::Storage(e.to_string()))?;
    drop(stmt);
    let prev_dir = paths.generation_dir(previous);
    let mut reuse = BTreeMap::new();
    for page in pages {
        let Some((_, _, prev_slug)) = prev_rows.iter().find(|(prev_id, prev_hash, _)| {
            prev_id == page.page_id.as_str() && prev_hash == &page.body_hash
        }) else {
            continue;
        };
        let prev_path = prev_dir.join(page_file_name(prev_slug, page.page_id.as_str()));
        if prev_path.is_file() {
            reuse.insert(page.page_id.as_str().to_owned(), prev_path);
        }
    }
    Ok(reuse)
}

/// Pages whose (page_id, body_hash) pair is byte-identical to the previous
/// generation: page_id → that generation's on-disk file. Hardlinking a
/// carried page reuses the object instead of rewriting the bytes (audit
/// FIX-013) — no data copy on publish, no re-hash on validation.
pub type ReusablePages = BTreeMap<String, PathBuf>;

/// Step 1: writes the immutable generation (one .md per page). Pages must be
/// fully written before any pointer move. Distinct slugs that sanitize onto
/// the same file name are rejected BEFORE anything is written — a silent
/// overwrite would drop a page from the published wiki. Carried pages in
/// `reuse` are hardlinked from their previous-generation file (falling back
/// to a plain write when the filesystem refuses) instead of rewritten.
pub fn write_generation(
    paths: &PublishPaths,
    build_id: &BuildId,
    pages: &[WikiPageRecord],
    reuse: &ReusablePages,
) -> Result<()> {
    let dir = paths.generation_dir(build_id);
    // Pre-compute every file name: no partial generation on a slug collision.
    let mut assigned: HashMap<String, String> = HashMap::with_capacity(pages.len());
    for page in pages {
        let name = page_file_name(&page.slug, page.page_id.as_str());
        if let Some(first_slug) = assigned.get(&name) {
            return Err(WikiError::Compilation(format!(
                "generation {build_id}: slugs {first_slug:?} and {:?} both map to file name {name:?}; refusing to write a lossy generation",
                page.slug
            )));
        }
        assigned.insert(name, page.slug.clone());
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| WikiError::Storage(format!("cannot create {}: {e}", dir.display())))?;
    for page in pages {
        let path = dir.join(page_file_name(&page.slug, page.page_id.as_str()));
        if let Some(source) = reuse.get(page.page_id.as_str()) {
            // A hardlink shares the inode: no content bytes are copied. The
            // shared object was hash-validated at its own publish; hand
            // edits to it are a broken-immutability condition that lint
            // (§36 hand-edited-file) flags on the active generation.
            if std::fs::hard_link(source, &path).is_ok() {
                continue;
            }
            tracing::debug!(
                source = %source.display(),
                "hardlink refused by the filesystem; writing the carried page instead"
            );
        }
        std::fs::write(&path, &page.content)
            .map_err(|e| WikiError::Storage(format!("cannot write {}: {e}", path.display())))?;
    }
    Ok(())
}

/// Step 2: validates the generation — page set non-empty, every file present,
/// content hash matches `body_hash` (PRD §35). Pages in `reused` were
/// hardlinked from the previous generation's validated file: the shared
/// inode is checked for presence and size only, skipping the O(unchanged)
/// read-back + re-hash (audit FIX-013).
pub fn validate_generation(
    paths: &PublishPaths,
    build_id: &BuildId,
    pages: &[WikiPageRecord],
    reused: &BTreeSet<String>,
) -> Result<()> {
    if pages.is_empty() {
        return Err(WikiError::Compilation(format!(
            "generation {build_id} is empty; refusing to publish (PRD §34: no partial wikis)"
        )));
    }
    let dir = paths.generation_dir(build_id);
    for page in pages {
        let path = dir.join(page_file_name(&page.slug, page.page_id.as_str()));
        if reused.contains(page.page_id.as_str()) {
            let len = std::fs::metadata(&path)
                .map(|meta| meta.len() as usize)
                .map_err(|e| {
                    WikiError::Compilation(format!(
                        "generation {build_id} invalid: {} is missing or unreadable: {e}",
                        path.display()
                    ))
                })?;
            if len != page.content.len() {
                return Err(WikiError::Compilation(format!(
                    "generation {build_id} invalid: {} has {} bytes, expected {}",
                    path.display(),
                    len,
                    page.content.len()
                )));
            }
            continue;
        }
        let content = std::fs::read(&path).map_err(|e| {
            WikiError::Compilation(format!(
                "generation {build_id} invalid: {} is missing or unreadable: {e}",
                path.display()
            ))
        })?;
        if sha256_hex(&content) != page.body_hash {
            return Err(WikiError::Compilation(format!(
                "generation {build_id} invalid: {} does not match body_hash {}",
                path.display(),
                page.body_hash
            )));
        }
    }
    Ok(())
}

/// Validates an on-disk generation against the persisted `wiki_pages` rows
/// (recovery path: the pages are not in memory).
fn validate_generation_from_db(
    conn: &rusqlite::Connection,
    paths: &PublishPaths,
    build_id: &BuildId,
) -> Result<()> {
    let pages = load_generation_pages(conn, build_id)?;
    // Recovery validates EVERYTHING from disk: the reused-object assumption
    // (validated at the source publish) must not be trusted after a crash.
    validate_generation(paths, build_id, &pages, &BTreeSet::new())
}

// ---------------------------------------------------------------------------
// Publish
// ---------------------------------------------------------------------------

/// Runs the full §35 publish flow for one compiled generation. The generation
/// rows (`persist_generation`) are persisted by the caller beforehand; this
/// function owns everything from READY through journal clearing and cleanup.
pub fn publish(
    conn: &mut rusqlite::Connection,
    wiki_dir: &Path,
    build_id: &BuildId,
    pages: &[WikiPageRecord],
    keep_generations: u32,
) -> Result<PublishReport> {
    let paths = PublishPaths::new(wiki_dir);
    if keep_generations < 1 {
        return Err(WikiError::Config(
            "build.keep_generations must be >= 1".into(),
        ));
    }
    // An un-resolved publish intent must be recovered, never overwritten:
    // step 4 would otherwise replace the old version's rollback target (PRD
    // §35: recovery resolves the journal, publishing never guesses over it).
    if journal_exists(&paths) {
        return Err(WikiError::PublishRecovery(format!(
            "an un-recovered publish journal exists at {}; publish recovery must resolve it before a new publish starts",
            paths.journal_path().display()
        )));
    }

    // The generation the pointer names is the explicit rollback target; it is
    // also the reuse base for byte-identical carried pages (audit FIX-013).
    let previous = match read_current_pointer(&paths)? {
        Some(pointer) => Some(BuildId::parse(pointer.build_id.clone()).map_err(|e| {
            WikiError::PublishRecovery(format!("current.json holds an invalid build id: {e}"))
        })?),
        None => get_active_build_id(conn)?,
    };
    let reuse = reusable_pages(conn, &paths, previous.as_ref(), pages)?;
    let reused: BTreeSet<String> = reuse.keys().cloned().collect();

    // Step 1+2: write and validate the immutable generation FIRST — until the
    // files verify, nothing else happens.
    write_generation(&paths, build_id, pages, &reuse)?;
    validate_generation(&paths, build_id, pages, &reused)?;

    // Step 3: generation rows are already persisted; the build goes READY.
    update_build_status(conn, build_id, "READY")?;

    // Step 4: record the publish intent BEFORE the pointer moves.
    write_journal(&paths, previous.as_ref(), build_id)?;

    // Step 5: atomically replace the pointer.
    write_current_pointer(&paths, build_id)?;

    // Step 6: ONE transaction — active_build_id AND COMPLETED, with the FTS
    // index rebuild for the activated generation joined into the same
    // transaction (PRD §20: FTS and active_build_id flip atomically; a
    // failure — e.g. an FTS5-less SQLite — aborts the publish and the
    // previous generation stays fully visible).
    activate_build_with_search_index(conn, build_id, default_tokenizer())?;

    // Step 7: clear the intent.
    clear_journal(&paths)?;

    // Post-publish cleanup is best-effort (PRD §35: warnings, not errors).
    let cleanup_warnings = cleanup_generations(&paths, build_id, keep_generations);
    for warning in &cleanup_warnings {
        tracing::warn!(warning, "generation cleanup skipped");
    }

    tracing::info!(build = %build_id, previous = ?previous, "published generation");
    Ok(PublishReport {
        build_id: build_id.clone(),
        previous_build: previous,
        published_path: paths.generation_dir(build_id),
        cleanup_warnings,
    })
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

/// Length note: ~118 lines — the §35 crash-recovery state machine; each branch maps 1:1 to a test in the crash matrix, so splitting would break the auditability of the decision table.
/// Resolves a possibly-interrupted publish under a single consistent view
/// (PRD §35). Called before every build and exposed for `doctor`:
///
/// - journal present → complete the verified new version, or roll back to the
///   explicit old version recorded in the journal;
/// - `current.json` vs DB `active_build_id` mismatch WITHOUT a journal →
///   `WikiError::PublishRecovery` (never guess);
/// - consistent state → `Ok(None)`.
pub fn recover_if_needed(
    conn: &mut rusqlite::Connection,
    wiki_dir: &Path,
) -> Result<Option<RecoveryReport>> {
    let paths = PublishPaths::new(wiki_dir);
    // Corrupt journal/pointer files surface as PublishRecovery from the
    // readers; recovery refuses to interpret damaged state by guessing.
    let journal = read_journal(&paths)?;
    let pointer = read_current_pointer(&paths)?;
    let db_active = get_active_build_id(conn)?;

    let Some(journal) = journal else {
        return match (&pointer, &db_active) {
            (None, None) => verify_recovery_index(conn).map(|()| None),
            (Some(pointer), Some(active)) if pointer.build_id == active.as_str() => {
                verify_recovery_index(conn).map(|()| None)
            }
            (Some(pointer), Some(active)) => Err(WikiError::PublishRecovery(format!(
                "current.json points at {} but the database active_build_id is {} and there is no publish journal; refusing to guess",
                pointer.build_id,
                active
            ))),
            (Some(pointer), None) => Err(WikiError::PublishRecovery(format!(
                "current.json points at {} but the database has no active_build_id and there is no publish journal; refusing to guess",
                pointer.build_id
            ))),
            (None, Some(active)) => Err(WikiError::PublishRecovery(format!(
                "the database active_build_id is {} but current.json is missing and there is no publish journal; refusing to guess",
                active
            ))),
        };
    };

    let new_id = BuildId::parse(journal.new_build_id.clone()).map_err(|e| {
        WikiError::PublishRecovery(format!("publish journal holds an invalid build id: {e}"))
    })?;
    let old_id = journal
        .old_build_id
        .as_ref()
        .map(|value| {
            BuildId::parse(value.clone()).map_err(|e| {
                WikiError::PublishRecovery(format!(
                    "publish journal holds an invalid old build id: {e}"
                ))
            })
        })
        .transpose()?;
    let pointer_is_new = pointer
        .as_ref()
        .is_some_and(|pointer| pointer.build_id == journal.new_build_id);
    let db_is_new = db_active
        .as_ref()
        .is_some_and(|active| active.as_str() == journal.new_build_id);

    let report = if pointer_is_new && db_is_new {
        // Crash after step 6, before step 7: the publish committed fully —
        // only the journal file is left over.
        clear_journal(&paths)?;
        RecoveryReport {
            action: RecoveryAction::CompletedNew,
            detail: format!(
                "publish of {new_id} had already committed; cleared the leftover publish journal"
            ),
        }
    } else if pointer_is_new {
        // Crash between steps 5 and 6: pointer moved, database not committed.
        // Complete the verified new version — INCLUDING the search index
        // rebuild, joined to the same commit-point transaction — when it
        // still validates.
        if validate_generation_from_db(conn, &paths, &new_id).is_ok() {
            activate_build_with_search_index(conn, &new_id, default_tokenizer())?;
            clear_journal(&paths)?;
            RecoveryReport {
                action: RecoveryAction::CompletedNew,
                detail: format!(
                    "publish of {new_id} was interrupted after the pointer move; the generation re-validated and the database side was completed"
                ),
            }
        } else {
            rollback(
                conn,
                &paths,
                old_id.as_ref(),
                &new_id,
                "the new generation no longer validates",
            )?
        }
    } else if db_is_new {
        // Database committed but the pointer never moved (crash between 5 and
        // 6 with the rename lost): the visible wiki is the old generation, so
        // the database is reverted to match it.
        rollback(
            conn,
            &paths,
            old_id.as_ref(),
            &new_id,
            "the pointer never moved",
        )?
    } else {
        // Crash between steps 4 and 5: intent recorded, pointer untouched —
        // the new version was never published.
        rollback(
            conn,
            &paths,
            old_id.as_ref(),
            &new_id,
            "the pointer was never moved to the new generation",
        )?
    };

    // Post-recovery invariant (PRD §20/§35): the FTS index must cover exactly
    // the ACTIVE generation's pages — recovery rebuilds on drift (and clears
    // it when nothing is active) so search never serves a stale generation.
    verify_recovery_index(conn)?;

    tracing::warn!(action = ?report.action, detail = %report.detail, "publish recovery ran");
    Ok(Some(report))
}

/// After ANY recovery outcome the derived indexes are re-verified against the
/// active generation (rebuild on drift, clear when nothing is published):
/// the FTS index (PRD §20) and the §17 Wiki Graph.
fn verify_recovery_index(conn: &mut rusqlite::Connection) -> Result<()> {
    ensure_search_index_matches_active(conn, default_tokenizer())?;
    ensure_graph_matches_active(conn)?;
    Ok(())
}

/// Rolls back to the explicit old build recorded in the journal (PRD §35:
/// never guess, never mix generations).
fn rollback(
    conn: &mut rusqlite::Connection,
    paths: &PublishPaths,
    old: Option<&BuildId>,
    new: &BuildId,
    because: &str,
) -> Result<RecoveryReport> {
    let detail = match old {
        Some(old) => {
            let old_dir = paths.generation_dir(old);
            if !old_dir.is_dir() {
                return Err(WikiError::PublishRecovery(format!(
                    "cannot roll back: previous generation {} is missing on disk; refusing to guess",
                    old
                )));
            }
            write_current_pointer(paths, old)?;
            set_active_build(conn, Some(old))?;
            format!("rolled back to the previous generation {old} because {because}")
        }
        None => {
            // First publish: there is no old version. Make sure neither the
            // pointer nor the database references the unverified new one.
            set_active_build(conn, None)?;
            let pointer_path = paths.pointer_path();
            if pointer_path.exists() {
                std::fs::remove_file(&pointer_path).map_err(|e| {
                    WikiError::PublishRecovery(format!(
                        "cannot remove pointer {}: {e}",
                        pointer_path.display()
                    ))
                })?;
            }
            format!("rolled back the first publish of {new} to an empty wiki because {because}")
        }
    };
    clear_journal(paths)?;
    // The interrupted build never published; mark it so §31 states stay sane.
    // Best-effort: the row always exists in real flows.
    let _ = update_build_status(conn, new, "INTERRUPTED");
    Ok(RecoveryReport {
        action: RecoveryAction::RolledBack,
        detail,
    })
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

/// Deletes generations that are neither the current one nor inside the
/// `build.keep_generations` retention window (PRD §35). Ordering is by build
/// id: ids are `<prefix>_<ULID>` so lexicographic order is chronological.
/// Never deletes the current generation; every failure becomes a warning.
pub fn cleanup_generations(
    paths: &PublishPaths,
    current: &BuildId,
    keep_generations: u32,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let dir = paths.generations_dir();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) => {
            warnings.push(format!("cannot list {}: {e}", dir.display()));
            return warnings;
        }
    };

    let mut others: Vec<BuildId> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == current.as_str() {
            continue; // never delete the current generation
        }
        match BuildId::parse(name) {
            Ok(build_id) => others.push(build_id),
            Err(_) => warnings.push(format!(
                "skipping {} (not a build id); manual cleanup may be needed",
                entry.path().display()
            )),
        }
    }

    // Newest first; keep `keep_generations - 1` besides the current one.
    others.sort_by(|a, b| b.cmp(a));
    let keep = keep_generations.saturating_sub(1) as usize;
    for stale in others.into_iter().skip(keep) {
        let path = paths.generation_dir(&stale);
        if let Err(e) = std::fs::remove_dir_all(&path) {
            warnings.push(format!("cannot delete {}: {e}", path.display()));
        } else {
            tracing::info!(generation = %stale, "deleted stale generation");
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_storage::{open_in_memory, start_build, BuildDraft};

    fn temp_wiki_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llm-wiki-publish-{tag}-{}-{}",
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

    fn seeded_build(conn: &mut rusqlite::Connection) -> BuildId {
        start_build(conn, &BuildDraft::default()).unwrap()
    }

    #[test]
    fn rename_retry_gives_up_after_the_bound() {
        let mut calls = 0u32;
        let err = rename_with_retry(
            || {
                calls += 1;
                Err(std::io::Error::from_raw_os_error(32))
            },
            4,
            Duration::from_millis(1),
        )
        .unwrap_err();
        assert_eq!(calls, 4, "retries only within the bound");
        assert!(err.to_string().contains("4 attempt(s)"), "{err}");
    }

    #[test]
    fn rename_retry_succeeds_after_transient_failures() {
        let mut calls = 0u32;
        rename_with_retry(
            || {
                calls += 1;
                if calls < 3 {
                    Err(std::io::Error::from_raw_os_error(5))
                } else {
                    Ok(())
                }
            },
            RENAME_ATTEMPTS,
            Duration::from_millis(1),
        )
        .unwrap();
        assert_eq!(calls, 3);
    }

    #[test]
    fn rename_retry_fails_fast_on_permanent_errors() {
        let mut calls = 0u32;
        let err = rename_with_retry(
            || {
                calls += 1;
                Err(std::io::Error::from_raw_os_error(2)) // not transient
            },
            RENAME_ATTEMPTS,
            Duration::from_millis(1),
        )
        .unwrap_err();
        assert_eq!(calls, 1, "permanent errors must not be retried");
        assert!(err.to_string().contains("1 attempt(s)"), "{err}");
    }

    #[test]
    fn atomic_write_replaces_target_in_place() {
        let dir = temp_wiki_dir("atomic-write");
        let target = dir.join("current.json");
        atomic_write(&target, "{\"v\":1}").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "{\"v\":1}");
        atomic_write(&target, "{\"v\":2}").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "{\"v\":2}");
        // No temp leftovers.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn pointer_and_journal_roundtrip() {
        let dir = temp_wiki_dir("roundtrip");
        let paths = PublishPaths::new(&dir);
        assert!(read_current_pointer(&paths).unwrap().is_none());
        assert!(!journal_exists(&paths));

        let build = BuildId::generate();
        write_current_pointer(&paths, &build).unwrap();
        let pointer = read_current_pointer(&paths).unwrap().unwrap();
        assert_eq!(pointer.build_id, build.as_str());
        assert!(!pointer.published_at.is_empty());

        write_journal(&paths, None, &build).unwrap();
        assert!(journal_exists(&paths));
        let journal = read_journal(&paths).unwrap().unwrap();
        assert_eq!(journal.new_build_id, build.as_str());
        assert_eq!(journal.old_build_id, None);
        clear_journal(&paths).unwrap();
        assert!(!journal_exists(&paths));
    }

    #[test]
    fn corrupt_pointer_or_journal_is_publish_recovery_not_guessing() {
        let dir = temp_wiki_dir("corrupt");
        let paths = PublishPaths::new(&dir);
        std::fs::write(paths.pointer_path(), "{ not json").unwrap();
        assert!(read_current_pointer(&paths).is_err());
        std::fs::write(paths.journal_path(), "garbage").unwrap();
        assert!(read_journal(&paths).is_err());
    }

    #[test]
    fn validation_rejects_missing_files_and_hash_mismatches() {
        let dir = temp_wiki_dir("validate");
        let paths = PublishPaths::new(&dir);
        let build = BuildId::generate();
        let pages = vec![page("runtime", "# Runtime\n\nbody")];

        assert!(
            validate_generation(&paths, &build, &pages, &BTreeSet::new()).is_err(),
            "empty dir"
        );

        write_generation(&paths, &build, &pages, &BTreeMap::new()).unwrap();
        validate_generation(&paths, &build, &pages, &BTreeSet::new()).unwrap();

        // Tamper with the file: hash must stop matching.
        let file = paths.generation_dir(&build).join("runtime.md");
        std::fs::write(&file, "# Tampered").unwrap();
        assert!(validate_generation(&paths, &build, &pages, &BTreeSet::new()).is_err());

        // Empty page set is rejected outright.
        assert!(validate_generation(&paths, &build, &[], &BTreeSet::new()).is_err());
    }

    #[test]
    fn page_file_names_are_sanitized() {
        assert_eq!(
            page_file_name("plugin-runtime", "wp_1"),
            "plugin-runtime.md"
        );
        // Path separators and dots collapse to dashes and are trimmed, so the
        // name can never escape the generation directory (".." is impossible).
        assert_eq!(page_file_name("../evil/slug", "wp_2"), "evil-slug.md");
        assert_eq!(page_file_name("", "wp_3"), "wp_3.md");
        assert_eq!(page_file_name("标题", "wp_4"), "wp_4.md");
    }

    #[test]
    fn publish_happy_path_advances_pointer_db_and_clears_journal() {
        let dir = temp_wiki_dir("happy");
        let mut conn = open_in_memory().unwrap();
        let first = seeded_build(&mut conn);
        let second = seeded_build(&mut conn);
        let pages = vec![page("runtime", "# Runtime")];

        // First publish: no previous generation.
        let report = publish(&mut conn, &dir, &first, &pages, 3).unwrap();
        assert_eq!(report.previous_build, None);
        assert_eq!(report.cleanup_warnings, Vec::<String>::new());
        let generation_file = dir
            .join("generations")
            .join(first.as_str())
            .join("runtime.md");
        assert_eq!(
            std::fs::read_to_string(&generation_file).unwrap(),
            "# Runtime"
        );
        let pointer = read_current_pointer(&PublishPaths::new(&dir))
            .unwrap()
            .unwrap();
        assert_eq!(pointer.build_id, first.as_str());
        assert_eq!(get_active_build_id(&conn).unwrap(), Some(first.clone()));
        assert!(!journal_exists(&PublishPaths::new(&dir)));
        let status: String = conn
            .query_row(
                "SELECT status FROM builds WHERE build_id = ?1",
                [first.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "COMPLETED");

        // Second publish advances everything and keeps the old generation.
        let report = publish(&mut conn, &dir, &second, &pages, 3).unwrap();
        assert_eq!(report.previous_build, Some(first.clone()));
        let pointer = read_current_pointer(&PublishPaths::new(&dir))
            .unwrap()
            .unwrap();
        assert_eq!(pointer.build_id, second.as_str());
        assert_eq!(get_active_build_id(&conn).unwrap(), Some(second.clone()));
        assert!(dir.join("generations").join(first.as_str()).is_dir());
        assert!(dir.join("generations").join(second.as_str()).is_dir());
    }

    #[test]
    fn cleanup_respects_retention_and_never_deletes_current() {
        let dir = temp_wiki_dir("cleanup");
        let mut conn = open_in_memory().unwrap();
        let pages = vec![page("runtime", "# Runtime")];

        // Deterministic, strictly ascending build ids: retention orders by id
        // (ids are `bld_<ULID>`, so lexicographic is chronological), and four
        // ids minted in the same millisecond would order randomly.
        let ulids = [
            "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "01BX5ZZKBKACTAV9WEVGEMMVRZ",
            "01CZZZZZZZZZZZZZZZZZZZZZZZ",
            "01DARZ3NDEKTSV4RRFFQ69G5FAV",
        ];
        let builds: Vec<BuildId> = ulids
            .iter()
            .map(|ulid| {
                let id = BuildId::parse(format!("bld_{ulid}")).unwrap();
                conn.execute(
                    "INSERT INTO builds (build_id, started_at, status) VALUES (?1, '2026-01-01T00:00:00Z', 'RUNNING')",
                    [id.as_str()],
                )
                .unwrap();
                id
            })
            .collect();

        for build in &builds {
            publish(&mut conn, &dir, build, &pages, 2).unwrap();
        }

        let paths = PublishPaths::new(&dir);
        // keep_generations = 2 → current + the one previous build survive;
        // the two oldest generations are deleted.
        let current = builds[3].clone();
        let mut survivors: Vec<String> = std::fs::read_dir(paths.generations_dir())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        survivors.sort();
        let mut expected = vec![builds[2].as_str().to_owned(), current.as_str().to_owned()];
        expected.sort();
        assert_eq!(survivors, expected);
        assert!(dir.join("generations").join(current.as_str()).is_dir());
    }

    #[test]
    fn cleanup_leaves_unparseable_directories_alone() {
        let dir = temp_wiki_dir("cleanup-junk");
        let mut conn = open_in_memory().unwrap();
        let build = seeded_build(&mut conn);
        let pages = vec![page("runtime", "# Runtime")];
        publish(&mut conn, &dir, &build, &pages, 1).unwrap();

        let junk = dir.join("generations").join("not-a-build-id");
        std::fs::create_dir_all(&junk).unwrap();
        let warnings = cleanup_generations(&PublishPaths::new(&dir), &build, 1);
        assert!(warnings.iter().any(|w| w.contains("not-a-build-id")));
        assert!(junk.is_dir(), "unparseable directories are never deleted");
    }

    #[test]
    fn write_generation_rejects_slug_filename_collisions() {
        let dir = temp_wiki_dir("collision");
        let paths = PublishPaths::new(&dir);
        let build = BuildId::generate();
        // "a b" and "a-b" sanitize onto the same file name.
        let pages = vec![page("a b", "# A"), page("a-b", "# B")];
        let err = write_generation(&paths, &build, &pages, &BTreeMap::new()).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("\"a b\"") && message.contains("\"a-b\""),
            "{message}"
        );
        assert!(
            !paths.generation_dir(&build).exists(),
            "nothing is written when a collision is detected"
        );
    }

    #[test]
    fn publish_refuses_to_overwrite_a_stale_journal() {
        let dir = temp_wiki_dir("stale-journal");
        let mut conn = open_in_memory().unwrap();
        let build = seeded_build(&mut conn);
        let pages = vec![page("runtime", "# Runtime")];
        let paths = PublishPaths::new(&dir);
        write_journal(&paths, None, &build).unwrap();

        let err = publish(&mut conn, &dir, &build, &pages, 3).unwrap_err();
        assert!(matches!(err, WikiError::PublishRecovery(_)), "{err}");
        assert!(journal_exists(&paths), "the journal is left for recovery");
        assert!(
            read_current_pointer(&paths).unwrap().is_none(),
            "nothing was published over the unresolved intent"
        );
    }
}
