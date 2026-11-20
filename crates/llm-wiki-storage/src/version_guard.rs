//! Workspace version guard (operational safety).
//!
//! A workspace is a directory pair — `.llm-wiki/state.db` plus `wiki/`. When one
//! deployment serves many workspaces (one directory each) an upgrade that only
//! touches some of them leaves the rest readable but *stale*: SQLite happily
//! opens an older schema, and the published generation keeps serving even though
//! it was produced by a different compiler. That mismatch is silent — pages look
//! fine, they are just missing whatever the newer pipeline started producing.
//!
//! This module reads both facts once, so `serve` can decide instead of finding
//! out months later.
//!
//! Two distinct questions, deliberately not conflated:
//!
//! - **Can this binary even read the db?** `PRAGMA user_version` above the
//!   migration count this binary was built with means the workspace was written
//!   by a NEWER release. Nothing downstream can be trusted, so this is fatal.
//! - **Was the published generation produced by this binary?** A different
//!   `compiler_version` on the ACTIVE build means "readable but stale": the wiki
//!   predates this binary's pipeline. Serving it is legitimate (the pages are
//!   still grounded and correct), it just needs a rebuild — so this is a warning
//!   unless the deployment opts into strict mode.

use rusqlite::Connection;

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::BuildId;

use crate::state::get_active_build_id;

/// What one workspace's state db says about the binary that wrote it.
#[derive(Debug, Clone)]
pub struct WorkspaceVersions {
    /// Schema version recorded in the db (`PRAGMA user_version`).
    pub db_schema_version: i64,
    /// Schema version the running binary knows — `MIGRATIONS.len()`.
    pub supported_schema_version: i64,
    /// The currently published build, if any.
    pub active_build_id: Option<BuildId>,
    /// `compiler_version` recorded on the active build (None when unknown or
    /// when the build predates the column).
    pub generation_compiler_version: Option<String>,
    /// `schema_version` recorded on the active build. Diagnostic only: a change
    /// here is handled by the build-time fingerprint guard (REPLAN_REQUIRED),
    /// not by the server.
    pub generation_schema_version: Option<String>,
}

impl WorkspaceVersions {
    /// The db was written by a newer release: this binary cannot read it.
    ///
    /// Migrations only move forward, so a higher `user_version` means tables
    /// (or columns) this build has never heard of — reading them would fail at
    /// the first query rather than here.
    pub fn db_is_from_the_future(&self) -> bool {
        self.db_schema_version > self.supported_schema_version
    }

    /// The db is older than this binary: `open()` migrates it forward, so this
    /// is informational (it tells you the upgrade actually happened).
    pub fn db_is_from_the_past(&self) -> bool {
        self.db_schema_version < self.supported_schema_version
    }

    /// True when a generation is published but was compiled by another binary
    /// version. `None` (unknown) is treated as stale: it cannot be proven
    /// current, and claiming otherwise would hide exactly what this guard
    /// exists to surface.
    pub fn generation_is_stale(&self, binary_version: &str) -> bool {
        match (self.active_build_id.as_ref(), self.generation_compiler_version.as_deref()) {
            (None, _) => false,
            (Some(_), Some(version)) => version != binary_version,
            (Some(_), None) => true,
        }
    }
}

/// Reads the two version facts of one workspace. Never mutates the db.
pub fn check_workspace_versions(conn: &Connection) -> Result<WorkspaceVersions> {
    let db_schema_version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| WikiError::Storage(format!("cannot read user_version: {e}")))?;
    let active_build_id = get_active_build_id(conn)?;

    let (mut compiler_version, mut schema_version) = (None, None);
    if let Some(build_id) = active_build_id.as_ref() {
        let row = conn.query_row(
            "SELECT compiler_version, schema_version FROM builds WHERE build_id = ?1",
            [build_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<String>>(1)?,
                ))
            },
        );
        match row {
            Ok((compiler, schema)) => {
                compiler_version = compiler;
                schema_version = schema;
            }
            // The pointer names a build whose row is gone (hand-edited db, or a
            // pre-builds-table workspace). Treated as "unknown", never as an
            // error: the guard must not be the thing that takes a server down.
            Err(rusqlite::Error::QueryReturnedNoRows) => {}
            Err(e) => {
                return Err(WikiError::Storage(format!(
                    "cannot read active build versions: {e}"
                )))
            }
        }
    }

    Ok(WorkspaceVersions {
        db_schema_version,
        supported_schema_version: crate::migrations::MIGRATIONS.len() as i64,
        active_build_id,
        generation_compiler_version: compiler_version,
        generation_schema_version: schema_version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(db: i64, supported: i64, gen: Option<&str>) -> WorkspaceVersions {
        WorkspaceVersions {
            db_schema_version: db,
            supported_schema_version: supported,
            active_build_id: gen.map(|_| BuildId::generate()),
            generation_compiler_version: gen.map(str::to_owned),
            generation_schema_version: None,
        }
    }

    #[test]
    fn future_schema_is_fatal_but_past_schema_is_not() {
        assert!(report(9, 8, None).db_is_from_the_future());
        assert!(!report(7, 8, None).db_is_from_the_future());
        assert!(report(7, 8, None).db_is_from_the_past());
        assert!(!report(8, 8, None).db_is_from_the_past());
    }

    #[test]
    fn stale_generation_is_detected_and_unknown_counts_as_stale() {
        assert!(report(8, 8, Some("0.1.0")).generation_is_stale("0.2.0"));
        assert!(!report(8, 8, Some("0.2.0")).generation_is_stale("0.2.0"));
        // No generation: nothing to compare, not stale.
        assert!(!report(8, 8, None).generation_is_stale("0.2.0"));
        // A published generation with no recorded version cannot be proven
        // current, so it is reported rather than silently accepted.
        let mut unknown = report(8, 8, Some("0.2.0"));
        unknown.generation_compiler_version = None;
        assert!(unknown.generation_is_stale("0.2.0"));
    }

    #[test]
    fn check_reads_user_version_and_active_build() {
        let conn = crate::connection::open_in_memory().unwrap();
        let report = check_workspace_versions(&conn).unwrap();
        assert_eq!(report.db_schema_version, report.supported_schema_version);
        assert!(report.active_build_id.is_none());
        assert!(!report.db_is_from_the_future());
        assert!(!report.generation_is_stale("0.0.0"));
    }
}
