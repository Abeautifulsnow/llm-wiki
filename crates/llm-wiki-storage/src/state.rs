//! Publish/state support (PRD §31, §35): the `wiki_state` key-value store,
//! build stage transitions and stale-build recovery.
//!
//! The `active_build_id` key names the generation the database considers
//! current. [`activate_build`] switches it in the SAME transaction that marks
//! the build COMPLETED, so the filesystem pointer swap (§35) and the database
//! can only disagree within a publish critical section, which the publish
//! journal recovery resolves.

use rusqlite::{params, Connection, Transaction};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::BuildId;

/// Key under which the currently published build id is stored.
pub const ACTIVE_BUILD_KEY: &str = "active_build_id";

/// §31 states a build row may hold. Stage updates come from the build
/// pipeline; terminal states additionally record `finished_at` via
/// `finish_build`/`activate_build`.
pub const BUILD_STATUSES: &[&str] = &[
    "SCANNING",
    "PARSING",
    "ANALYZING",
    "PLANNING",
    "COMPILING",
    "INDEXING",
    "READY",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
    "REPLAN_REQUIRED",
];

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// Reads one `wiki_state` value.
pub fn get_state(conn: &Connection, key: &str) -> Result<Option<String>> {
    let mut stmt = conn
        .prepare("SELECT value FROM wiki_state WHERE key = ?1")
        .map_err(|e| WikiError::Storage(format!("prepare get_state: {e}")))?;
    let mut rows = stmt
        .query(params![key])
        .map_err(|e| WikiError::Storage(format!("get_state: {e}")))?;
    match rows.next() {
        Ok(Some(row)) => Ok(Some(row.get(0).map_err(db)?)),
        Ok(None) => Ok(None),
        Err(e) => Err(WikiError::Storage(format!("get_state: {e}"))),
    }
}

/// The build id the database considers currently published, if any.
pub fn get_active_build_id(conn: &Connection) -> Result<Option<BuildId>> {
    Ok(get_state(conn, ACTIVE_BUILD_KEY)?.map(BuildId::from_validated))
}

/// Sets or clears one `wiki_state` value inside a transaction.
pub fn set_state(conn: &mut Connection, key: &str, value: Option<&str>) -> Result<()> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    match value {
        Some(value) => {
            tx.execute(
                "INSERT INTO wiki_state (key, value) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )
            .map_err(db)?;
        }
        None => {
            tx.execute("DELETE FROM wiki_state WHERE key = ?1", params![key])
                .map_err(db)?;
        }
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit set_state: {e}")))?;
    Ok(())
}

/// Switches the persisted active build without touching build statuses
/// (publish-journal rollback path, PRD §35 recovery).
pub fn set_active_build(conn: &mut Connection, build_id: Option<&BuildId>) -> Result<()> {
    set_state(conn, ACTIVE_BUILD_KEY, build_id.map(|b| b.as_str()))
}

/// The §35 step-6 statements scoped to an OPEN transaction: switch
/// `active_build_id` and mark the build COMPLETED with its finish timestamp.
/// Shared by [`activate_build`] and the search-index composite
/// (`search_index::activate_build_with_search_index`) so the FTS rebuild can
/// join the SAME transaction and flip atomically with the pointer.
pub(crate) fn activate_in_tx(tx: &Transaction, build_id: &BuildId) -> Result<()> {
    tx.execute(
        "INSERT INTO wiki_state (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![ACTIVE_BUILD_KEY, build_id.as_str()],
    )
    .map_err(db)?;
    let changed = tx
        .execute(
            "UPDATE builds SET status = 'COMPLETED', finished_at = ?1 WHERE build_id = ?2",
            params![chrono::Utc::now().to_rfc3339(), build_id.as_str()],
        )
        .map_err(db)?;
    if changed == 0 {
        return Err(WikiError::Storage(format!("unknown build {build_id}")));
    }
    Ok(())
}

/// The publish commit point (PRD §35 step 6): ONE transaction that switches
/// `active_build_id` and marks the build COMPLETED with its finish timestamp.
pub fn activate_build(conn: &mut Connection, build_id: &BuildId) -> Result<()> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    activate_in_tx(&tx, build_id)?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit activate_build: {e}")))?;
    Ok(())
}

/// Persists one §31 stage transition (SCANNING/PARSING/…/READY and terminal
/// states). Terminal status timestamps belong to `finish_build`/`activate_build`.
pub fn update_build_status(conn: &mut Connection, build_id: &BuildId, status: &str) -> Result<()> {
    if !BUILD_STATUSES.contains(&status) {
        return Err(WikiError::Storage(format!(
            "invalid build status '{status}'"
        )));
    }
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let changed = tx
        .execute(
            "UPDATE builds SET status = ?1 WHERE build_id = ?2",
            params![status, build_id.as_str()],
        )
        .map_err(db)?;
    if changed == 0 {
        return Err(WikiError::Storage(format!("unknown build {build_id}")));
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit update_build_status: {e}")))?;
    Ok(())
}

/// Startup recovery (PRD §31): a server restart must not leave RUNNING/READY
/// builds in a fake-live state. Returns how many rows were marked INTERRUPTED.
pub fn mark_stale_builds_interrupted(conn: &mut Connection) -> Result<u64> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let changed = tx
        .execute(
            "UPDATE builds SET status = 'INTERRUPTED', finished_at = ?1
             WHERE status IN ('RUNNING', 'READY')",
            params![chrono::Utc::now().to_rfc3339()],
        )
        .map_err(db)?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit mark_stale_builds: {e}")))?;
    Ok(changed as u64)
}

/// Distinct build ids that have persisted generation rows — the cleanup
/// bookkeeping view of which generations exist machine-side (PRD §35).
pub fn list_generation_build_ids(conn: &Connection) -> Result<Vec<BuildId>> {
    let mut stmt = conn
        .prepare("SELECT DISTINCT build_id FROM wiki_pages ORDER BY build_id")
        .map_err(|e| WikiError::Storage(format!("prepare list_generations: {e}")))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| WikiError::Storage(format!("list_generations: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(BuildId::from_validated(row.map_err(db)?));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builds::{finish_build, start_build, BuildDraft};
    use crate::connection::open_in_memory;

    fn new_build(conn: &mut Connection) -> BuildId {
        start_build(conn, &BuildDraft::default()).unwrap()
    }

    #[test]
    fn state_roundtrip_and_delete() {
        let mut conn = open_in_memory().unwrap();
        assert_eq!(get_state(&conn, "anything").unwrap(), None);
        set_state(&mut conn, "anything", Some("v1")).unwrap();
        assert_eq!(get_state(&conn, "anything").unwrap().as_deref(), Some("v1"));
        set_state(&mut conn, "anything", Some("v2")).unwrap();
        assert_eq!(get_state(&conn, "anything").unwrap().as_deref(), Some("v2"));
        set_state(&mut conn, "anything", None).unwrap();
        assert_eq!(get_state(&conn, "anything").unwrap(), None);
    }

    #[test]
    fn activate_build_switches_state_and_completes_in_one_tx() {
        let mut conn = open_in_memory().unwrap();
        let first = new_build(&mut conn);
        let second = new_build(&mut conn);
        assert!(get_active_build_id(&conn).unwrap().is_none());

        activate_build(&mut conn, &first).unwrap();
        assert_eq!(get_active_build_id(&conn).unwrap(), Some(first.clone()));
        let status: String = conn
            .query_row(
                "SELECT status FROM builds WHERE build_id = ?1",
                params![first.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "COMPLETED");

        // Second publish advances both atomically.
        activate_build(&mut conn, &second).unwrap();
        assert_eq!(get_active_build_id(&conn).unwrap(), Some(second.clone()));
    }

    #[test]
    fn activate_unknown_build_is_an_error() {
        let mut conn = open_in_memory().unwrap();
        let ghost = BuildId::generate();
        assert!(activate_build(&mut conn, &ghost).is_err());
    }

    #[test]
    fn set_active_build_can_clear_the_key() {
        let mut conn = open_in_memory().unwrap();
        let build = new_build(&mut conn);
        activate_build(&mut conn, &build).unwrap();
        set_active_build(&mut conn, None).unwrap();
        assert!(get_active_build_id(&conn).unwrap().is_none());
    }

    #[test]
    fn status_transitions_validate_the_state_machine() {
        let mut conn = open_in_memory().unwrap();
        let build = new_build(&mut conn);
        for stage in [
            "SCANNING",
            "PARSING",
            "ANALYZING",
            "PLANNING",
            "COMPILING",
            "INDEXING",
        ] {
            update_build_status(&mut conn, &build, stage).unwrap();
        }
        let status: String = conn
            .query_row(
                "SELECT status FROM builds WHERE build_id = ?1",
                params![build.as_str()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "INDEXING");
        update_build_status(&mut conn, &build, "NOT_A_STATE").unwrap_err();
        update_build_status(&mut conn, &BuildId::generate(), "READY").unwrap_err();
    }

    #[test]
    fn stale_running_and_ready_builds_become_interrupted() {
        let mut conn = open_in_memory().unwrap();
        let running = new_build(&mut conn);
        let ready = new_build(&mut conn);
        let failed = new_build(&mut conn);
        update_build_status(&mut conn, &ready, "READY").unwrap();
        finish_build(&mut conn, &failed, "FAILED", None, None).unwrap();

        let marked = mark_stale_builds_interrupted(&mut conn).unwrap();
        assert_eq!(marked, 2, "RUNNING + READY");

        let status = |id: &BuildId| -> String {
            conn.query_row(
                "SELECT status FROM builds WHERE build_id = ?1",
                params![id.as_str()],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(status(&running), "INTERRUPTED");
        assert_eq!(status(&ready), "INTERRUPTED");
        assert_eq!(status(&failed), "FAILED", "terminal states are untouched");
    }

    #[test]
    fn generation_listing_is_distinct_and_sorted() {
        let conn = open_in_memory().unwrap();
        assert!(list_generation_build_ids(&conn).unwrap().is_empty());
        let a = BuildId::generate();
        let b = BuildId::generate();
        for (build, slug) in [(&a, "page-a"), (&a, "page-a2"), (&b, "page-b")] {
            conn.execute(
                "INSERT INTO wiki_pages (page_id, build_id, slug, title, category, language, body_hash, content, created_at)
                 VALUES (?1, ?2, ?3, 't', 'c', 'und', 'h', 'body', 'now')",
                params![llm_wiki_core::ids::WikiPageId::generate().as_str(), build.as_str(), slug],
            )
            .unwrap();
        }
        let ids = list_generation_build_ids(&conn).unwrap();
        let mut expected = vec![a.clone(), b.clone()];
        expected.sort();
        assert_eq!(ids, expected, "distinct ids in build_id order");
    }
}
