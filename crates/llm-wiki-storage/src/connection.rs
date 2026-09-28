//! Connection setup and migrations (PRD §42).
//!
//! Every connection gets WAL mode, `foreign_keys=ON`, a busy timeout and
//! NORMAL synchronous mode. Schema is versioned via `PRAGMA user_version`;
//! migrations run inside a transaction and are idempotent.

use std::path::Path;
use std::time::Duration;

pub use rusqlite::Connection;

use llm_wiki_core::error::{Result, WikiError};

use crate::migrations::MIGRATIONS;

pub fn open(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)
        .map_err(|e| WikiError::Storage(format!("cannot open {}: {e}", path.display())))?;
    configure(&conn)?;
    migrate(&conn)?;
    Ok(conn)
}

pub fn open_in_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory()
        .map_err(|e| WikiError::Storage(format!("cannot open in-memory db: {e}")))?;
    configure(&conn)?;
    migrate(&conn)?;
    Ok(conn)
}

fn configure(conn: &Connection) -> Result<()> {
    // WAL is the PRD §42 recommendation; it degrades to `memory` journal on
    // in-memory databases, which is fine.
    let _ = conn.query_row("PRAGMA journal_mode=WAL", [], |row| {
        let mode: String = row.get(0)?;
        Ok(mode)
    });
    conn.pragma_update(None, "foreign_keys", true)
        .map_err(|e| WikiError::Storage(format!("cannot enable foreign_keys: {e}")))?;
    conn.pragma_update(None, "synchronous", "NORMAL")
        .map_err(|e| WikiError::Storage(format!("cannot set synchronous mode: {e}")))?;
    conn.busy_timeout(Duration::from_millis(5_000))
        .map_err(|e| WikiError::Storage(format!("cannot set busy timeout: {e}")))?;
    Ok(())
}

fn migrate(conn: &Connection) -> Result<()> {
    let current: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| WikiError::Storage(format!("cannot read user_version: {e}")))?;
    if current as usize >= MIGRATIONS.len() {
        return Ok(());
    }
    // Schema migrations follow the documented SQLite table-rebuild procedure:
    // foreign-key enforcement is disabled around the whole run (PRAGMA
    // foreign_keys is a no-op inside a transaction, so it cannot be toggled
    // from within a script) and re-enabled with a foreign_key_check after.
    // The old generation tables are therefore not implicitly deleted when
    // dropped and FK targets need not exist mid-rebuild.
    conn.execute_batch("PRAGMA foreign_keys=OFF;")
        .map_err(|e| WikiError::Storage(format!("cannot disable foreign_keys: {e}")))?;
    let migrate_result = migrate_versions(conn, current);
    let re_enabled = conn.execute_batch("PRAGMA foreign_keys=ON;");
    let violations = foreign_key_violations(conn)?;
    migrate_result?;
    re_enabled.map_err(|e| WikiError::Storage(format!("cannot re-enable foreign_keys: {e}")))?;
    if violations > 0 {
        return Err(WikiError::Storage(format!(
            "migration left {violations} foreign-key violation(s) in the state db"
        )));
    }
    Ok(())
}

fn migrate_versions(conn: &Connection, current: i64) -> Result<()> {
    for (idx, script) in MIGRATIONS.iter().enumerate().skip(current as usize) {
        let version = (idx + 1) as i64;
        conn.execute_batch("BEGIN;")
            .map_err(|e| WikiError::Storage(format!("migration {version} begin: {e}")))?;
        if let Err(e) = conn
            .execute_batch(script)
            .and_then(|_| conn.execute_batch(&format!("PRAGMA user_version = {version};")))
        {
            let _ = conn.execute_batch("ROLLBACK;");
            return Err(WikiError::Storage(format!(
                "migration {version} failed: {e}"
            )));
        }
        conn.execute_batch("COMMIT;")
            .map_err(|e| WikiError::Storage(format!("migration {version} commit: {e}")))?;
        tracing::info!(version, "applied migration");
    }
    Ok(())
}

/// Counts rows that violate an enabled foreign key (`PRAGMA foreign_key_check`).
fn foreign_key_violations(conn: &Connection) -> Result<u64> {
    let mut stmt = conn
        .prepare("PRAGMA foreign_key_check")
        .map_err(|e| WikiError::Storage(format!("prepare foreign_key_check: {e}")))?;
    let mut count = 0u64;
    let mut rows = stmt
        .query([])
        .map_err(|e| WikiError::Storage(format!("foreign_key_check: {e}")))?;
    while rows
        .next()
        .map_err(|e| WikiError::Storage(format!("foreign_key_check: {e}")))?
        .is_some()
    {
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_db_gets_latest_schema_and_pragmas() {
        let conn = open_in_memory().unwrap();
        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version as usize, MIGRATIONS.len());

        let fk: i64 = conn
            .query_row("PRAGMA foreign_keys", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fk, 1);

        // meta table seeded with the revision counter.
        let rev: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'registry_revision'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rev, "0");
    }

    #[test]
    fn reopening_is_idempotent() {
        let dir = std::env::temp_dir().join(format!("llm-wiki-db-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.db");
        let _ = std::fs::remove_file(&path);
        drop(open(&path).unwrap());
        drop(open(&path).unwrap());
        let _ = std::fs::remove_file(&path);
    }
}
