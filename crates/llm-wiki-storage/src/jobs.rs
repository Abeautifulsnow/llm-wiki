//! Server job persistence (PRD §30/§31): queued and running build jobs
//! survive server restarts.
//!
//! §31: no job may stay RUNNING in a fake state across a restart — startup
//! recovery marks every QUEUED/RUNNING row INTERRUPTED. Write operations
//! accept an idempotency key; the same key resolves to the SAME job row
//! within its retention window instead of starting a second LLM run.

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, JobId};

/// §31 job statuses (the pipeline stage phases live in `phase`).
pub const JOB_STATUSES: &[&str] = &[
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
    "REPLAN_REQUIRED",
];

/// Failure codes surfaced on terminal job rows (stable API contract).
pub const FAILURE_CANCELLED: &str = "cancelled";
pub const FAILURE_INTERRUPTED: &str = "server_restart";
pub const FAILURE_LLM: &str = "llm_error";
pub const FAILURE_PLANNING: &str = "planning_error";
pub const FAILURE_REPLAN_REQUIRED: &str = "replan_required";
pub const FAILURE_PUBLISH: &str = "publish_recovery";
pub const FAILURE_INTERNAL: &str = "internal_error";

/// One persisted server job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerJobRecord {
    pub job_id: JobId,
    /// `build` (the only v1 job kind).
    pub kind: String,
    pub status: String,
    /// Live §31 stage phase (SCANNING..INDEXING) while RUNNING.
    pub phase: Option<String>,
    pub build_id: Option<BuildId>,
    pub failure_code: Option<String>,
    pub retryable: bool,
    pub error: Option<String>,
    pub request_id: Option<String>,
    pub idempotency_key: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
}

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(format!("server_jobs: {e}"))
}

fn row_to_record(row: &rusqlite::Row) -> Result<ServerJobRecord> {
    Ok(ServerJobRecord {
        job_id: JobId::from_validated(row.get::<_, String>("job_id").map_err(db)?),
        kind: row.get("kind").map_err(db)?,
        status: row.get("status").map_err(db)?,
        phase: row.get("phase").map_err(db)?,
        build_id: row
            .get::<_, Option<String>>("build_id")
            .map_err(db)?
            .map(BuildId::from_validated),
        failure_code: row.get("failure_code").map_err(db)?,
        retryable: row.get::<_, i64>("retryable").map_err(db)? != 0,
        error: row.get("error").map_err(db)?,
        request_id: row.get("request_id").map_err(db)?,
        idempotency_key: row.get("idempotency_key").map_err(db)?,
        created_at: row.get("created_at").map_err(db)?,
        started_at: row.get("started_at").map_err(db)?,
        finished_at: row.get("finished_at").map_err(db)?,
    })
}

const RECORD_COLUMNS: &str = "job_id, kind, status, phase, build_id, failure_code, retryable, \
     error, request_id, idempotency_key, created_at, started_at, finished_at";

/// Inserts a QUEUED job row.
pub fn insert_job(conn: &Connection, record: &ServerJobRecord) -> Result<()> {
    conn.execute(
        format!(
            "INSERT INTO server_jobs ({RECORD_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"
        )
        .as_str(),
        params![
            record.job_id.as_str(),
            record.kind,
            record.status,
            record.phase,
            record.build_id.as_ref().map(|b| b.as_str()),
            record.failure_code,
            i64::from(record.retryable),
            record.error,
            record.request_id,
            record.idempotency_key,
            record.created_at,
            record.started_at,
            record.finished_at,
        ],
    )
    .map_err(db)?;
    Ok(())
}

/// One job by id.
pub fn get_job(conn: &Connection, job_id: &JobId) -> Result<Option<ServerJobRecord>> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {RECORD_COLUMNS} FROM server_jobs WHERE job_id = ?1"
        ))
        .map_err(db)?;
    let mut rows = stmt.query(params![job_id.as_str()]).map_err(db)?;
    match rows.next().map_err(db)? {
        Some(row) => Ok(Some(row_to_record(row)?)),
        None => Ok(None),
    }
}

/// The job an idempotency key maps to (§30: same key → same job).
pub fn get_job_by_idempotency_key(conn: &Connection, key: &str) -> Result<Option<ServerJobRecord>> {
    let mut stmt = conn
        .prepare(&format!(
            "SELECT {RECORD_COLUMNS} FROM server_jobs WHERE idempotency_key = ?1"
        ))
        .map_err(db)?;
    let mut rows = stmt.query(params![key]).map_err(db)?;
    match rows.next().map_err(db)? {
        Some(row) => Ok(Some(row_to_record(row)?)),
        None => Ok(None),
    }
}

/// Lists jobs newest-first with cursor pagination (§30: list endpoints are
/// cursor-paginated). `cursor` is the previous page's last job id; the page
/// starts strictly after it.
pub fn list_jobs(
    conn: &Connection,
    status: Option<&str>,
    limit: usize,
    cursor: Option<&JobId>,
) -> Result<Vec<ServerJobRecord>> {
    let cursor_created: Option<String> = match cursor {
        Some(job_id) => {
            let Some(record) = get_job(conn, job_id)? else {
                return Err(WikiError::Storage(format!("cursor job {job_id} not found")));
            };
            Some(record.created_at)
        }
        None => None,
    };
    let sql = format!(
        "SELECT {RECORD_COLUMNS} FROM server_jobs \
         WHERE (?1 IS NULL OR status = ?1) \
           AND (?2 IS NULL OR created_at < ?2 OR (created_at = ?2 AND job_id < ?3)) \
         ORDER BY created_at DESC, job_id DESC LIMIT ?4"
    );
    let mut stmt = conn.prepare(&sql).map_err(db)?;
    let mut rows = stmt
        .query(params![
            status,
            cursor_created,
            cursor.map(|c| c.as_str()),
            limit as i64,
        ])
        .map_err(db)?;
    let mut records = Vec::new();
    while let Some(row) = rows.next().map_err(db)? {
        records.push(row_to_record(row)?);
    }
    Ok(records)
}

/// Counts jobs per status bucket (queue-depth reporting on /v1/status).
pub fn count_jobs_by_status(
    conn: &Connection,
    statuses: &[&str],
) -> Result<std::collections::BTreeMap<String, u32>> {
    let mut counts = std::collections::BTreeMap::new();
    for status in statuses {
        let n: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM server_jobs WHERE status = ?1",
                params![status],
                |row| row.get(0),
            )
            .map_err(db)?;
        counts.insert((*status).to_owned(), n);
    }
    Ok(counts)
}

/// Marks a job RUNNING (idempotent; also stamps `started_at` once).
pub fn set_job_running(conn: &Connection, job_id: &JobId) -> Result<()> {
    conn.execute(
        "UPDATE server_jobs SET status = 'RUNNING', \
         started_at = COALESCE(started_at, ?2) WHERE job_id = ?1 AND status = 'QUEUED'",
        params![job_id.as_str(), chrono::Utc::now().to_rfc3339()],
    )
    .map_err(db)?;
    Ok(())
}

/// Mirrors a pipeline stage transition onto the job row.
pub fn set_job_phase(conn: &Connection, job_id: &JobId, phase: &str) -> Result<()> {
    conn.execute(
        "UPDATE server_jobs SET phase = ?2 WHERE job_id = ?1",
        params![job_id.as_str(), phase],
    )
    .map_err(db)?;
    Ok(())
}

/// Links the job to the build row the pipeline created.
pub fn attach_job_build(conn: &Connection, job_id: &JobId, build_id: &BuildId) -> Result<()> {
    conn.execute(
        "UPDATE server_jobs SET build_id = ?2 WHERE job_id = ?1",
        params![job_id.as_str(), build_id.as_str()],
    )
    .map_err(db)?;
    Ok(())
}

/// Terminalizes a job with its §31 outcome.
pub fn finish_job(
    conn: &Connection,
    job_id: &JobId,
    status: &str,
    failure_code: Option<&str>,
    retryable: bool,
    error: Option<&str>,
) -> Result<()> {
    if !JOB_STATUSES.contains(&status) {
        return Err(WikiError::Storage(format!("invalid job status {status}")));
    }
    conn.execute(
        "UPDATE server_jobs SET status = ?2, failure_code = ?3, retryable = ?4, error = ?5, \
         finished_at = ?6 WHERE job_id = ?1",
        params![
            job_id.as_str(),
            status,
            failure_code,
            i64::from(retryable),
            error,
            chrono::Utc::now().to_rfc3339()
        ],
    )
    .map_err(db)?;
    Ok(())
}

/// Startup recovery (§31): every QUEUED/RUNNING job becomes INTERRUPTED with
/// `failure_code = server_restart`. Returns how many rows were recovered.
pub fn mark_stale_jobs_interrupted(conn: &Connection) -> Result<u64> {
    let changed = conn
        .execute(
            "UPDATE server_jobs SET status = 'INTERRUPTED', failure_code = ?1, retryable = 1, \
             finished_at = ?2 WHERE status IN ('QUEUED', 'RUNNING')",
            params![FAILURE_INTERRUPTED, chrono::Utc::now().to_rfc3339()],
        )
        .map_err(db)?;
    Ok(changed as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_in_memory;

    fn queued(id: &str, key: Option<String>) -> ServerJobRecord {
        ServerJobRecord {
            job_id: JobId::from_validated(id.to_owned()),
            kind: "build".into(),
            status: "QUEUED".into(),
            phase: None,
            build_id: None,
            failure_code: None,
            retryable: false,
            error: None,
            request_id: Some("req-1".into()),
            idempotency_key: key,
            created_at: "2026-01-01T00:00:00+00:00".into(),
            started_at: None,
            finished_at: None,
        }
    }

    #[test]
    fn insert_get_and_idempotency_lookup() {
        let conn = open_in_memory().unwrap();
        insert_job(&conn, &queued("job_A", Some("idem-1".into()))).unwrap();

        let got = get_job(&conn, &JobId::from_validated("job_A"))
            .unwrap()
            .expect("row exists");
        assert_eq!(got.status, "QUEUED");
        assert_eq!(got.idempotency_key.as_deref(), Some("idem-1"));

        let by_key = get_job_by_idempotency_key(&conn, "idem-1")
            .unwrap()
            .expect("key resolves");
        assert_eq!(by_key.job_id, got.job_id);

        assert!(get_job(&conn, &JobId::from_validated("job_MISSING"))
            .unwrap()
            .is_none());
    }

    #[test]
    fn lifecycle_running_phase_terminal_and_recovery() {
        let conn = open_in_memory().unwrap();
        insert_job(&conn, &queued("job_A", None)).unwrap();
        insert_job(&conn, &queued("job_B", None)).unwrap();

        set_job_running(&conn, &JobId::from_validated("job_A")).unwrap();
        set_job_phase(&conn, &JobId::from_validated("job_A"), "ANALYZING").unwrap();
        let running = get_job(&conn, &JobId::from_validated("job_A"))
            .unwrap()
            .unwrap();
        assert_eq!(running.status, "RUNNING");
        assert_eq!(running.phase.as_deref(), Some("ANALYZING"));
        assert!(running.started_at.is_some());

        finish_job(
            &conn,
            &JobId::from_validated("job_A"),
            "FAILED",
            Some(FAILURE_LLM),
            true,
            Some("boom"),
        )
        .unwrap();
        let failed = get_job(&conn, &JobId::from_validated("job_A"))
            .unwrap()
            .unwrap();
        assert_eq!(failed.status, "FAILED");
        assert_eq!(failed.failure_code.as_deref(), Some(FAILURE_LLM));
        assert!(failed.retryable);
        assert!(failed.finished_at.is_some());

        // job_B stays QUEUED → startup recovery interrupts it.
        let recovered = mark_stale_jobs_interrupted(&conn).unwrap();
        assert_eq!(recovered, 1);
        let interrupted = get_job(&conn, &JobId::from_validated("job_B"))
            .unwrap()
            .unwrap();
        assert_eq!(interrupted.status, "INTERRUPTED");
        assert_eq!(
            interrupted.failure_code.as_deref(),
            Some(FAILURE_INTERRUPTED)
        );
        // Recovery never touches terminal rows.
        assert_eq!(mark_stale_jobs_interrupted(&conn).unwrap(), 0);
    }

    #[test]
    fn list_orders_newest_first_and_paginates_by_cursor() {
        let conn = open_in_memory().unwrap();
        for (id, created) in [
            ("job_A", "2026-01-01T00:00:01+00:00"),
            ("job_B", "2026-01-01T00:00:02+00:00"),
            ("job_C", "2026-01-01T00:00:03+00:00"),
        ] {
            let mut record = queued(id, None);
            record.created_at = created.to_owned();
            insert_job(&conn, &record).unwrap();
        }
        let page_one = list_jobs(&conn, None, 2, None).unwrap();
        let ids: Vec<String> = page_one
            .iter()
            .map(|r| r.job_id.as_str().to_owned())
            .collect();
        assert_eq!(ids, ["job_C", "job_B"]);

        let page_two = list_jobs(&conn, None, 2, Some(&page_one[1].job_id)).unwrap();
        let ids: Vec<String> = page_two
            .iter()
            .map(|r| r.job_id.as_str().to_owned())
            .collect();
        assert_eq!(ids, ["job_A"]);

        // Status filter composes with the cursor.
        finish_job(
            &conn,
            &JobId::from_validated("job_C"),
            "COMPLETED",
            None,
            false,
            None,
        )
        .unwrap();
        let completed = list_jobs(&conn, Some("COMPLETED"), 10, None).unwrap();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].job_id.as_str(), "job_C");

        let counts = count_jobs_by_status(&conn, JOB_STATUSES).unwrap();
        assert_eq!(counts.get("COMPLETED"), Some(&1));
        assert_eq!(counts.get("QUEUED"), Some(&2));
    }

    #[test]
    fn unknown_cursor_is_an_error_not_an_empty_page() {
        let conn = open_in_memory().unwrap();
        let err =
            list_jobs(&conn, None, 10, Some(&JobId::from_validated("job_MISSING"))).unwrap_err();
        assert!(err.to_string().contains("cursor job"));
    }
}
