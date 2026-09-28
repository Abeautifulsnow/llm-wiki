//! LLM response cache persistence (PRD §28).
//!
//! Rows are keyed by the §28 hash: task type, canonical request payload hash,
//! model, prompt/schema/parser versions and effective config hash. Each row
//! also records the source snapshot hash so cross-version reuse is impossible.
//!
//! First write wins (`INSERT OR IGNORE`): responses are deterministic for a
//! given key, so overwriting would never change the value. Never regressing
//! protects against a racing rebuild.

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// One cache entry to persist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRow {
    pub cache_key: String,
    pub task_type: String,
    pub model: String,
    pub prompt_version: String,
    pub schema_version: String,
    pub parser_version: String,
    pub config_hash: String,
    pub source_snapshot_hash: String,
    pub response: String,
}

/// Returns the cached response text for `cache_key`, or `None` on a miss.
pub fn get_cached_response(conn: &Connection, cache_key: &str) -> Result<Option<String>> {
    let mut stmt = conn
        .prepare("SELECT response FROM llm_cache WHERE cache_key = ?1")
        .map_err(|e| WikiError::Storage(format!("prepare get_cached_response: {e}")))?;
    let mut rows = stmt
        .query(params![cache_key])
        .map_err(|e| WikiError::Storage(format!("get_cached_response: {e}")))?;
    match rows.next() {
        Ok(Some(row)) => Ok(Some(row.get(0).map_err(db)?)),
        Ok(None) => Ok(None),
        Err(e) => Err(WikiError::Storage(format!("get_cached_response: {e}"))),
    }
}

/// Stores one cache entry inside a transaction. Existing keys are left
/// untouched (first write wins).
pub fn put_cached_response(conn: &mut Connection, row: &CacheRow) -> Result<()> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    tx.execute(
        "INSERT OR IGNORE INTO llm_cache
         (cache_key, task_type, model, prompt_version, schema_version, parser_version, config_hash, source_snapshot_hash, response, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            row.cache_key,
            row.task_type,
            row.model,
            row.prompt_version,
            row.schema_version,
            row.parser_version,
            row.config_hash,
            row.source_snapshot_hash,
            row.response,
            chrono::Utc::now().to_rfc3339()
        ],
    )
    .map_err(db)?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit put_cached_response: {e}")))?;
    Ok(())
}

/// Number of cache rows (test/diagnostic convenience).
pub fn count_cache_entries(conn: &Connection) -> Result<u64> {
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM llm_cache", [], |r| r.get(0))
        .map_err(|e| WikiError::Storage(format!("count_cache_entries: {e}")))?;
    Ok(count as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;

    fn row(key: &str, response: &str) -> CacheRow {
        CacheRow {
            cache_key: key.to_owned(),
            task_type: "document-analysis".into(),
            model: "fake".into(),
            prompt_version: "document-analysis@1".into(),
            schema_version: "1".into(),
            parser_version: "0.1.0".into(),
            config_hash: "cfg".into(),
            source_snapshot_hash: "snap".into(),
            response: response.to_owned(),
        }
    }

    #[test]
    fn cache_roundtrip_and_first_write_wins() {
        let mut conn = open_in_memory().unwrap();
        assert_eq!(get_cached_response(&conn, "k1").unwrap(), None);
        put_cached_response(&mut conn, &row("k1", "first")).unwrap();
        assert_eq!(
            get_cached_response(&conn, "k1").unwrap().as_deref(),
            Some("first")
        );
        // A second write under the same key never regresses the value.
        put_cached_response(&mut conn, &row("k1", "second")).unwrap();
        assert_eq!(
            get_cached_response(&conn, "k1").unwrap().as_deref(),
            Some("first")
        );
        put_cached_response(&mut conn, &row("k2", "other")).unwrap();
        assert_eq!(count_cache_entries(&conn).unwrap(), 2);
    }
}
