//! Source Registry (PRD §8.2): maps recomputable locator keys to persistent
//! opaque `SourceId`s. Path changes are delete/add in V0.1 (PRD §43); rename
//! matching arrives with V0.2.

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{SourceId, SourceLocatorKey};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRecord {
    pub source_id: SourceId,
    pub locator_key: SourceLocatorKey,
    pub rel_path: String,
    pub content_hash: String,
    pub size: i64,
    pub status: String,
}

const SELECT_COLS: &str = "source_id, locator_key, rel_path, content_hash, size, status";

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

fn row_to_record(row: &rusqlite::Row) -> Result<SourceRecord> {
    Ok(SourceRecord {
        source_id: SourceId::from_validated(row.get::<_, String>("source_id").map_err(db)?),
        locator_key: SourceLocatorKey::from_validated(
            row.get::<_, String>("locator_key").map_err(db)?,
        ),
        rel_path: row.get("rel_path").map_err(db)?,
        content_hash: row.get("content_hash").map_err(db)?,
        size: row.get("size").map_err(db)?,
        status: row.get("status").map_err(db)?,
    })
}

/// One source to upsert in a [`upsert_sources_batch`] call.
#[derive(Debug, Clone)]
pub struct SourceUpsert<'a> {
    pub locator_key: &'a SourceLocatorKey,
    pub rel_path: &'a str,
    pub content_hash: &'a str,
    pub size: i64,
}

/// Inserts the source or refreshes its hash/size/path. Returns the persistent
/// id and whether it was newly created.
pub fn upsert_source(
    conn: &mut Connection,
    locator_key: &SourceLocatorKey,
    rel_path: &str,
    content_hash: &str,
    size: i64,
    build_id: Option<&str>,
) -> Result<(SourceId, bool)> {
    let source = SourceUpsert {
        locator_key,
        rel_path,
        content_hash,
        size,
    };
    let mut result = upsert_sources_batch(conn, std::slice::from_ref(&source), build_id)?;
    Ok(result.remove(0))
}

/// Upserts many sources inside ONE transaction — no per-file commits, so a
/// failure rolls back the whole batch and the scan loop never issues N
/// separate fsyncs. Statements are prepared once and reused.
///
/// Returns each source's `(id, created)` in input order.
pub fn upsert_sources_batch(
    conn: &mut Connection,
    sources: &[SourceUpsert<'_>],
    build_id: Option<&str>,
) -> Result<Vec<(SourceId, bool)>> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;

    let out = {
        let mut select_stmt = tx
            .prepare("SELECT source_id FROM sources WHERE locator_key = ?1")
            .map_err(db)?;
        let mut update_stmt = tx
            .prepare(
                "UPDATE sources SET rel_path = ?1, content_hash = ?2, size = ?3, status = 'active', last_seen_build_id = ?4 WHERE locator_key = ?5",
            )
            .map_err(db)?;
        let mut insert_stmt = tx
            .prepare(
                "INSERT INTO sources (source_id, locator_key, rel_path, content_hash, size, first_seen_build_id, last_seen_build_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            )
            .map_err(db)?;

        let mut out = Vec::with_capacity(sources.len());
        for source in sources {
            let existing: Option<String> = select_stmt
                .query_row(params![source.locator_key.as_str()], |row| row.get(0))
                .ok();
            let created = existing.is_none();

            let source_id = match existing {
                Some(id) => {
                    update_stmt
                        .execute(params![
                            source.rel_path,
                            source.content_hash,
                            source.size,
                            build_id,
                            source.locator_key.as_str()
                        ])
                        .map_err(db)?;
                    SourceId::from_validated(id)
                }
                None => {
                    let id = SourceId::generate();
                    insert_stmt
                        .execute(params![
                            id.as_str(),
                            source.locator_key.as_str(),
                            source.rel_path,
                            source.content_hash,
                            source.size,
                            build_id
                        ])
                        .map_err(db)?;
                    id
                }
            };
            out.push((source_id, created));
        }
        drop(select_stmt);
        drop(update_stmt);
        drop(insert_stmt);
        out
    };

    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit batch upsert: {e}")))?;
    Ok(out)
}

pub fn get_by_locator(
    conn: &Connection,
    locator_key: &SourceLocatorKey,
) -> Result<Option<SourceRecord>> {
    let sql = format!("SELECT {SELECT_COLS} FROM sources WHERE locator_key = ?1");
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| WikiError::Storage(format!("prepare get_by_locator: {e}")))?;
    let mut rows = stmt
        .query(params![locator_key.as_str()])
        .map_err(|e| WikiError::Storage(format!("get_by_locator: {e}")))?;
    match rows.next() {
        Ok(Some(row)) => Ok(Some(row_to_record(row)?)),
        Ok(None) => Ok(None),
        Err(e) => Err(WikiError::Storage(format!("get_by_locator: {e}"))),
    }
}

pub fn list_sources(conn: &Connection) -> Result<Vec<SourceRecord>> {
    let sql =
        format!("SELECT {SELECT_COLS} FROM sources WHERE status = 'active' ORDER BY rel_path");
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| WikiError::Storage(format!("prepare list_sources: {e}")))?;
    let mut rows = stmt
        .query([])
        .map_err(|e| WikiError::Storage(format!("list_sources: {e}")))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().map_err(db)? {
        out.push(row_to_record(row)?);
    }
    Ok(out)
}

pub fn count_sources(conn: &Connection) -> Result<u64> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sources WHERE status = 'active'",
            [],
            |r| r.get(0),
        )
        .map_err(|e| WikiError::Storage(format!("count_sources: {e}")))?;
    Ok(count as u64)
}

/// Marks a source removed (PRD §19.3 starts from this state; V0.2 consumes it).
pub fn mark_removed(conn: &mut Connection, locator_key: &SourceLocatorKey) -> Result<bool> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let changed = tx
        .execute(
            "UPDATE sources SET status = 'removed' WHERE locator_key = ?1 AND status = 'active'",
            params![locator_key.as_str()],
        )
        .map_err(|e| WikiError::Storage(format!("mark_removed: {e}")))?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit mark_removed: {e}")))?;
    Ok(changed > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;

    fn locator(rel: &str) -> SourceLocatorKey {
        SourceLocatorKey::compute("ws", rel)
    }

    #[test]
    fn upsert_creates_once_then_updates() {
        let mut conn = open_in_memory().unwrap();
        let loc = locator("docs/a.md");

        let (id1, created1) =
            upsert_source(&mut conn, &loc, "docs/a.md", "hash1", 10, None).unwrap();
        assert!(created1);

        let (id2, created2) =
            upsert_source(&mut conn, &loc, "docs/a.md", "hash2", 12, None).unwrap();
        assert!(!created2);
        assert_eq!(id1, id2, "locator reuse must keep the opaque SourceId");

        let record = get_by_locator(&conn, &loc).unwrap().unwrap();
        assert_eq!(record.content_hash, "hash2");
        assert_eq!(record.size, 12);
        assert_eq!(count_sources(&conn).unwrap(), 1);
    }

    #[test]
    fn batch_upsert_matches_single_upsert_results() {
        let loc_a = locator("docs/a.md");
        let loc_b = locator("docs/b.md");

        let mut single_conn = open_in_memory().unwrap();
        let (_, created_a) =
            upsert_source(&mut single_conn, &loc_a, "docs/a.md", "h1", 1, None).unwrap();
        let (_, created_b) =
            upsert_source(&mut single_conn, &loc_b, "docs/b.md", "h2", 2, None).unwrap();

        let mut conn = open_in_memory().unwrap();
        let batch = vec![
            SourceUpsert {
                locator_key: &loc_a,
                rel_path: "docs/a.md",
                content_hash: "h1",
                size: 1,
            },
            SourceUpsert {
                locator_key: &loc_b,
                rel_path: "docs/b.md",
                content_hash: "h2",
                size: 2,
            },
        ];
        // IDs are opaque and per-database, so cross-database equality is
        // asserted on the `created` flags and on the row payloads, not on the
        // generated id strings.
        let results = upsert_sources_batch(&mut conn, &batch, None).unwrap();
        assert_eq!(
            results
                .iter()
                .map(|(_, created)| *created)
                .collect::<Vec<_>>(),
            vec![created_a, created_b]
        );
        let content = |conn: &Connection| {
            list_sources(conn)
                .unwrap()
                .into_iter()
                .map(|record| {
                    (
                        record.rel_path,
                        record.content_hash,
                        record.size,
                        record.status,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(content(&conn), content(&single_conn));

        // Within one database the identity is stable: a second batch updates
        // in a single transaction and must reuse the batch's own SourceId.
        let id_a_batch = results[0].0.clone();
        let update_batch = vec![SourceUpsert {
            locator_key: &loc_a,
            rel_path: "docs/a.md",
            content_hash: "h9",
            size: 9,
        }];
        let results2 = upsert_sources_batch(&mut conn, &update_batch, None).unwrap();
        assert_eq!(results2, vec![(id_a_batch, false)]);
        assert_eq!(
            get_by_locator(&conn, &loc_a).unwrap().unwrap().content_hash,
            "h9"
        );
    }

    #[test]
    fn mark_removed_then_reupsert_reactivates() {
        let mut conn = open_in_memory().unwrap();
        let loc = locator("docs/gone.md");
        let (id, _) = upsert_source(&mut conn, &loc, "docs/gone.md", "h", 1, None).unwrap();
        assert!(mark_removed(&mut conn, &loc).unwrap());
        assert_eq!(count_sources(&conn).unwrap(), 0);

        let (id2, created) = upsert_source(&mut conn, &loc, "docs/gone.md", "h2", 2, None).unwrap();
        assert!(!created);
        assert_eq!(id, id2);
        assert_eq!(count_sources(&conn).unwrap(), 1);
    }
}
