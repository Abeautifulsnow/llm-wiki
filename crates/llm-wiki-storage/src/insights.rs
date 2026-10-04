//! Verified insight write-back storage (audit FIX-020): the curated layer
//! for query-derived syntheses that passed citation verification. Read-side
//! helpers for future consumers (semantic lint, plan hints) live next to the
//! writer.

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, InsightId};

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// One persisted insight: the verified answer plus its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct InsightRecord {
    pub insight_id: InsightId,
    pub build_id: BuildId,
    pub query: String,
    pub answer: String,
    /// Expanded citation records at synthesis time (claim, source path,
    /// heading path, range, evidence digest).
    pub citations: Vec<InsightCitation>,
    pub created_at: String,
}

/// One cited claim, flattened for storage and display.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InsightCitation {
    pub claim_node_id: String,
    pub source: String,
    pub heading_path: Vec<String>,
    pub range: (usize, usize),
    pub evidence_digest: String,
}

/// Persists one verified insight.
pub fn insert_insight(conn: &mut Connection, record: &InsightRecord) -> Result<()> {
    let citations_json = serde_json::to_string(&record.citations)
        .map_err(|e| WikiError::Storage(format!("serialize insight citations: {e}")))?;
    conn.execute(
        "INSERT INTO wiki_insights (insight_id, build_id, query, answer, citations_json, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            record.insight_id.as_str(),
            record.build_id.as_str(),
            record.query,
            record.answer,
            citations_json,
            record.created_at
        ],
    )
    .map_err(db)?;
    Ok(())
}

/// Whether the cursor insight id exists (pagination input validation).
pub fn insight_exists(conn: &Connection, insight_id: &InsightId) -> Result<bool> {
    let found: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM wiki_insights WHERE insight_id = ?1",
            params![insight_id.as_str()],
            |row| row.get(0),
        )
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            other => Err(WikiError::Storage(format!("insight_exists: {other}"))),
        })?;
    Ok(found.is_some())
}

/// All insights (newest first), for `doctor`/future consumers.
pub fn list_insights(conn: &Connection) -> Result<Vec<InsightRecord>> {
    let mut stmt = conn
        .prepare(
            "SELECT insight_id, build_id, query, answer, citations_json, created_at
             FROM wiki_insights ORDER BY created_at DESC, insight_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare list insights: {e}")))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("list insights: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        let (insight_id, build_id, query, answer, citations_json, created_at) = row.map_err(db)?;
        let citations: Vec<InsightCitation> = serde_json::from_str(&citations_json)
            .map_err(|e| WikiError::Storage(format!("parse insight citations: {e}")))?;
        out.push(InsightRecord {
            insight_id: InsightId::from_validated(insight_id),
            build_id: BuildId::from_validated(build_id),
            query,
            answer,
            citations,
            created_at,
        });
    }
    Ok(out)
}

/// One page of insights, newest first, cursor-paginated (PRD §30: list
/// endpoints paginate). `cursor` is the previous page's last insight id; the
/// page starts strictly after it.
pub fn list_insights_paged(
    conn: &Connection,
    limit: usize,
    cursor: Option<&InsightId>,
) -> Result<Vec<InsightRecord>> {
    let cursor_created: Option<String> = match cursor {
        Some(insight_id) => conn
            .query_row(
                "SELECT created_at FROM wiki_insights WHERE insight_id = ?1",
                params![insight_id.as_str()],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(WikiError::Storage(format!("insight cursor: {other}"))),
            })?
            .ok_or_else(|| WikiError::Storage(format!("cursor insight {insight_id} not found")))?,
        None => None,
    };
    let mut stmt = conn
        .prepare(
            "SELECT insight_id, build_id, query, answer, citations_json, created_at
             FROM wiki_insights
             WHERE (?1 IS NULL OR created_at < ?1 OR (created_at = ?1 AND insight_id < ?2))
             ORDER BY created_at DESC, insight_id DESC LIMIT ?3",
        )
        .map_err(|e| WikiError::Storage(format!("prepare list insights paged: {e}")))?;
    let mut rows = stmt
        .query(params![
            cursor_created,
            cursor.map(|c| c.as_str()),
            limit as i64
        ])
        .map_err(|e| WikiError::Storage(format!("list insights paged: {e}")))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(db)? {
        let insight_id: String = row.get(0).map_err(db)?;
        let build_id: String = row.get(1).map_err(db)?;
        let query: String = row.get(2).map_err(db)?;
        let answer: String = row.get(3).map_err(db)?;
        let citations_json: String = row.get(4).map_err(db)?;
        let created_at: String = row.get(5).map_err(db)?;
        let citations: Vec<InsightCitation> = serde_json::from_str(&citations_json)
            .map_err(|e| WikiError::Storage(format!("parse insight citations: {e}")))?;
        out.push(InsightRecord {
            insight_id: InsightId::from_validated(insight_id),
            build_id: BuildId::from_validated(build_id),
            query,
            answer,
            citations,
            created_at,
        });
    }
    Ok(out)
}
