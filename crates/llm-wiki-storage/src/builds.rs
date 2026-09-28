//! Build records (PRD §18.1): enough metadata to reproduce "why did the wiki
//! change" — model, prompt/parser/compiler versions, schema version, effective
//! config hash, source snapshot hash and the BuildFingerprint.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::BuildId;

#[derive(Debug, Clone, Default)]
pub struct BuildDraft {
    pub source_snapshot_hash: Option<String>,
    pub model: Option<String>,
    pub prompt_version: Option<String>,
    pub compiler_version: Option<String>,
    pub parser_version: Option<String>,
    pub schema_version: Option<String>,
    pub config_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BuildRecord {
    pub build_id: BuildId,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub status: String,
    pub source_snapshot_hash: Option<String>,
    pub build_fingerprint: Option<String>,
    pub model: Option<String>,
    pub prompt_version: Option<String>,
    pub compiler_version: Option<String>,
    pub parser_version: Option<String>,
    pub schema_version: Option<String>,
    pub config_hash: Option<String>,
    pub registry_revision: Option<i64>,
}

/// Creates a RUNNING build row and returns its id.
pub fn start_build(conn: &mut Connection, draft: &BuildDraft) -> Result<BuildId> {
    let build_id = BuildId::generate();
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    tx.execute(
        "INSERT INTO builds (build_id, started_at, status, source_snapshot_hash, model, prompt_version, compiler_version, parser_version, schema_version, config_hash)
         VALUES (?1, ?2, 'RUNNING', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            build_id.as_str(),
            Utc::now().to_rfc3339(),
            draft.source_snapshot_hash,
            draft.model,
            draft.prompt_version,
            draft.compiler_version,
            draft.parser_version,
            draft.schema_version,
            draft.config_hash
        ],
    )
    .map_err(|e| WikiError::Storage(format!("insert build: {e}")))?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit build: {e}")))?;
    Ok(build_id)
}

/// Transitions a build to a terminal state (COMPLETED / FAILED / INTERRUPTED /
/// REPLAN_REQUIRED, PRD §31) with its fingerprint and the registry revision it
/// committed at.
pub fn finish_build(
    conn: &mut Connection,
    build_id: &BuildId,
    status: &str,
    build_fingerprint: Option<&str>,
    registry_revision: Option<u64>,
) -> Result<()> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let changed = tx
        .execute(
            "UPDATE builds SET status = ?1, finished_at = ?2, build_fingerprint = ?3, registry_revision = ?4 WHERE build_id = ?5",
            params![
                status,
                Utc::now().to_rfc3339(),
                build_fingerprint,
                registry_revision.map(|r| r as i64),
                build_id.as_str()
            ],
        )
        .map_err(|e| WikiError::Storage(format!("finish build: {e}")))?;
    if changed == 0 {
        return Err(WikiError::Storage(format!("unknown build {build_id}")));
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit finish build: {e}")))?;
    Ok(())
}

/// Records the scanned source snapshot hash once the scan stage produced it
/// (the build row is created before scanning so §31 stage transitions can be
/// persisted from the very start).
pub fn set_build_snapshot_hash(
    conn: &mut Connection,
    build_id: &BuildId,
    snapshot_hash: &str,
) -> Result<()> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let changed = tx
        .execute(
            "UPDATE builds SET source_snapshot_hash = ?1 WHERE build_id = ?2",
            params![snapshot_hash, build_id.as_str()],
        )
        .map_err(|e| WikiError::Storage(format!("set snapshot hash: {e}")))?;
    if changed == 0 {
        return Err(WikiError::Storage(format!("unknown build {build_id}")));
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit snapshot hash: {e}")))?;
    Ok(())
}

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

fn row_to_record(row: &rusqlite::Row) -> Result<BuildRecord> {
    let started: String = row.get("started_at").map_err(db)?;
    let finished: Option<String> = row.get("finished_at").map_err(db)?;
    let started_dt = DateTime::parse_from_rfc3339(&started)
        .map_err(|e| WikiError::Storage(format!("build started_at is not rfc3339: {e}")))?;
    Ok(BuildRecord {
        build_id: BuildId::from_validated(row.get::<_, String>("build_id").map_err(db)?),
        started_at: started_dt.with_timezone(&Utc),
        finished_at: finished
            .and_then(|f| DateTime::parse_from_rfc3339(&f).ok())
            .map(|f| f.with_timezone(&Utc)),
        status: row.get("status").map_err(db)?,
        source_snapshot_hash: row.get("source_snapshot_hash").map_err(db)?,
        build_fingerprint: row.get("build_fingerprint").map_err(db)?,
        model: row.get("model").map_err(db)?,
        prompt_version: row.get("prompt_version").map_err(db)?,
        compiler_version: row.get("compiler_version").map_err(db)?,
        parser_version: row.get("parser_version").map_err(db)?,
        schema_version: row.get("schema_version").map_err(db)?,
        config_hash: row.get("config_hash").map_err(db)?,
        registry_revision: row.get("registry_revision").map_err(db)?,
    })
}

const SELECT_COLS: &str = "build_id, started_at, finished_at, status, source_snapshot_hash, build_fingerprint, model, prompt_version, compiler_version, parser_version, schema_version, config_hash, registry_revision";

pub fn latest_build(conn: &Connection) -> Result<Option<BuildRecord>> {
    let sql = format!("SELECT {SELECT_COLS} FROM builds ORDER BY started_at DESC LIMIT 1");
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| WikiError::Storage(format!("prepare latest_build: {e}")))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| WikiError::Storage(format!("latest_build: {e}")))?;
    match rows.next() {
        Ok(Some(row)) => Ok(Some(row_to_record(row)?)),
        Ok(None) => Ok(None),
        Err(e) => Err(WikiError::Storage(format!("latest_build: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;

    #[test]
    fn build_lifecycle_roundtrip() {
        let mut conn = open_in_memory().unwrap();
        let draft = BuildDraft {
            source_snapshot_hash: Some("snap".into()),
            model: Some("m".into()),
            prompt_version: Some("p1".into()),
            compiler_version: Some("0.1.0".into()),
            parser_version: Some("parser0".into()),
            schema_version: Some("1".into()),
            config_hash: Some("cfg".into()),
        };
        let id = start_build(&mut conn, &draft).unwrap();
        assert!(latest_build(&conn).unwrap().unwrap().status == "RUNNING");

        finish_build(&mut conn, &id, "COMPLETED", Some("fp"), Some(7)).unwrap();
        let record = latest_build(&conn).unwrap().unwrap();
        assert_eq!(record.status, "COMPLETED");
        assert_eq!(record.build_fingerprint.as_deref(), Some("fp"));
        assert_eq!(record.registry_revision, Some(7));
        assert!(record.finished_at.is_some());
    }

    #[test]
    fn snapshot_hash_is_recorded_after_build_start() {
        let mut conn = open_in_memory().unwrap();
        let id = start_build(&mut conn, &BuildDraft::default()).unwrap();
        assert_eq!(
            latest_build(&conn).unwrap().unwrap().source_snapshot_hash,
            None
        );
        set_build_snapshot_hash(&mut conn, &id, "snap-1").unwrap();
        assert_eq!(
            latest_build(&conn).unwrap().unwrap().source_snapshot_hash,
            Some("snap-1".to_owned())
        );
        assert!(set_build_snapshot_hash(&mut conn, &BuildId::generate(), "x").is_err());
    }
}
