//! Stable page-identity map (PRD §45, migration 0007): merge/split/retire/
//! keep relations between predecessor and successor `WikiPageId`s recorded by
//! the explicit global re-plan (`llm-wiki replan`, PRD §19.2). The map is the
//! substrate for link, cache and old-URL migration consumers.
//!
//! Page rows persist across builds (`wiki_pages.page_id` is the primary key),
//! so a predecessor of a superseded generation remains a valid foreign-key
//! target. Rows are written AFTER `persist_generation` of the new build so
//! successor references resolve.

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, PageMapId, WikiPageId};

/// Relation kinds (checked by migration 0007).
pub const MAP_KIND_MERGE: &str = "merge";
pub const MAP_KIND_SPLIT: &str = "split";
pub const MAP_KIND_RETIRE: &str = "retire";
pub const MAP_KIND_KEEP: &str = "keep";

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// One page-identity relation to persist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageIdMapRow {
    /// The old page the identity continues from (`None` is never written by
    /// the replan flow; the column stays nullable per PRD §45).
    pub predecessor_page_id: Option<WikiPageId>,
    /// The new page taking over (`None` for retired pages).
    pub successor_page_id: Option<WikiPageId>,
    /// `merge` | `split` | `retire` | `keep`.
    pub kind: String,
    /// The replan build that wrote the relation.
    pub build_id: BuildId,
}

/// A persisted page-identity relation (audit view).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageIdMapEntry {
    pub mapping_id: PageMapId,
    pub predecessor_page_id: Option<WikiPageId>,
    pub successor_page_id: Option<WikiPageId>,
    pub kind: String,
    pub build_id: BuildId,
    pub created_at: String,
}

/// Persists page-identity relations in ONE transaction and returns the
/// mapping ids in input order.
pub fn insert_page_id_maps(conn: &mut Connection, rows: &[PageIdMapRow]) -> Result<Vec<PageMapId>> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let mut ids = Vec::with_capacity(rows.len());
    for row in rows {
        let mapping_id = PageMapId::generate();
        tx.execute(
            "INSERT INTO page_id_map
             (mapping_id, predecessor_page_id, successor_page_id, kind, build_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                mapping_id.as_str(),
                row.predecessor_page_id.as_ref().map(|id| id.as_str()),
                row.successor_page_id.as_ref().map(|id| id.as_str()),
                row.kind,
                row.build_id.as_str(),
                chrono::Utc::now().to_rfc3339()
            ],
        )
        .map_err(db)?;
        ids.push(mapping_id);
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit page_id_map: {e}")))?;
    Ok(ids)
}

// Table/column names below are compile-time constants — the `format!`
// concatenation never interpolates user input (all values are bound).
const SELECT_ENTRY: &str = "SELECT mapping_id, predecessor_page_id, successor_page_id, kind, build_id, created_at FROM page_id_map";

fn row_to_entry(row: &rusqlite::Row) -> rusqlite::Result<PageIdMapEntry> {
    let predecessor: Option<String> = row.get("predecessor_page_id")?;
    let successor: Option<String> = row.get("successor_page_id")?;
    Ok(PageIdMapEntry {
        mapping_id: PageMapId::from_validated(row.get::<_, String>("mapping_id")?),
        predecessor_page_id: predecessor.map(WikiPageId::from_validated),
        successor_page_id: successor.map(WikiPageId::from_validated),
        kind: row.get("kind")?,
        build_id: BuildId::from_validated(row.get::<_, String>("build_id")?),
        created_at: row.get("created_at")?,
    })
}

/// Every relation whose PREDECESSOR is `page_id` — "what happened to this old
/// page" (deterministic order: mapping id, which is chronological).
pub fn list_page_id_maps_by_predecessor(
    conn: &Connection,
    page_id: &WikiPageId,
) -> Result<Vec<PageIdMapEntry>> {
    let mut stmt = conn
        .prepare(&format!(
            "{SELECT_ENTRY} WHERE predecessor_page_id = ?1 ORDER BY mapping_id"
        ))
        .map_err(|e| WikiError::Storage(format!("prepare page_id_map by predecessor: {e}")))?;
    let rows = stmt
        .query_map(params![page_id.as_str()], row_to_entry)
        .map_err(|e| WikiError::Storage(format!("page_id_map by predecessor: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(db)?);
    }
    Ok(out)
}

/// Every relation whose SUCCESSOR is `page_id` — "where did this page come
/// from" (deterministic order: mapping id).
pub fn list_page_id_maps_by_successor(
    conn: &Connection,
    page_id: &WikiPageId,
) -> Result<Vec<PageIdMapEntry>> {
    let mut stmt = conn
        .prepare(&format!(
            "{SELECT_ENTRY} WHERE successor_page_id = ?1 ORDER BY mapping_id"
        ))
        .map_err(|e| WikiError::Storage(format!("prepare page_id_map by successor: {e}")))?;
    let rows = stmt
        .query_map(params![page_id.as_str()], row_to_entry)
        .map_err(|e| WikiError::Storage(format!("page_id_map by successor: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(db)?);
    }
    Ok(out)
}

/// Every relation written by one build (deterministic order: mapping id).
pub fn list_page_id_maps_for_build(
    conn: &Connection,
    build_id: &BuildId,
) -> Result<Vec<PageIdMapEntry>> {
    let mut stmt = conn
        .prepare(&format!(
            "{SELECT_ENTRY} WHERE build_id = ?1 ORDER BY mapping_id"
        ))
        .map_err(|e| WikiError::Storage(format!("prepare page_id_map by build: {e}")))?;
    let rows = stmt
        .query_map(params![build_id.as_str()], row_to_entry)
        .map_err(|e| WikiError::Storage(format!("page_id_map by build: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(db)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builds::start_build;
    use crate::connection::open_in_memory;
    use crate::wiki::persist_generation;
    use llm_wiki_core::hash::sha256_hex;

    /// Persists a minimal generation containing `slugs` so page rows exist as
    /// foreign-key targets, returning the page ids in slug order.
    fn seed_generation(
        conn: &mut Connection,
        build_id: &BuildId,
        slugs: &[&str],
    ) -> Vec<WikiPageId> {
        let pages: Vec<crate::wiki::WikiPageRecord> = slugs
            .iter()
            .map(|slug| crate::wiki::WikiPageRecord {
                page_id: WikiPageId::generate(),
                slug: (*slug).to_owned(),
                title: (*slug).to_owned(),
                category: "concepts".into(),
                language: "en".into(),
                body_hash: sha256_hex(slug.as_bytes()),
                content: format!("# {slug}"),
                knowledge_refs: Vec::new(),
                citations: Vec::new(),
                links: Vec::new(),
            })
            .collect();
        let ids = pages.iter().map(|page| page.page_id.clone()).collect();
        persist_generation(conn, build_id, &pages).unwrap();
        ids
    }

    #[test]
    fn page_id_map_rows_roundtrip_by_predecessor_successor_and_build() {
        let mut conn = open_in_memory().unwrap();
        let old_build = start_build(&mut conn, &crate::builds::BuildDraft::default()).unwrap();
        let new_build = start_build(&mut conn, &crate::builds::BuildDraft::default()).unwrap();
        let old_ids = seed_generation(&mut conn, &old_build, &["alpha", "beta", "gamma"]);
        let new_ids = seed_generation(&mut conn, &new_build, &["alpha-beta", "delta"]);

        let rows = vec![
            // alpha + beta merged into alpha-beta (dominant: alpha).
            PageIdMapRow {
                predecessor_page_id: Some(old_ids[0].clone()),
                successor_page_id: Some(new_ids[0].clone()),
                kind: MAP_KIND_MERGE.to_owned(),
                build_id: new_build.clone(),
            },
            PageIdMapRow {
                predecessor_page_id: Some(old_ids[1].clone()),
                successor_page_id: Some(new_ids[0].clone()),
                kind: MAP_KIND_MERGE.to_owned(),
                build_id: new_build.clone(),
            },
            // gamma split into delta (plus the absorbed half already covered
            // above); here: gamma retired, delta freshly created.
            PageIdMapRow {
                predecessor_page_id: Some(old_ids[2].clone()),
                successor_page_id: None,
                kind: MAP_KIND_RETIRE.to_owned(),
                build_id: new_build.clone(),
            },
            PageIdMapRow {
                predecessor_page_id: None,
                successor_page_id: Some(new_ids[1].clone()),
                kind: MAP_KIND_SPLIT.to_owned(),
                build_id: new_build.clone(),
            },
        ];
        let ids = insert_page_id_maps(&mut conn, &rows).unwrap();
        assert_eq!(ids.len(), 4);
        assert!(ids.iter().all(|id| id.as_str().starts_with("pgm_")));

        // By predecessor: alpha-beta's two merge predecessors resolve.
        let by_successor = list_page_id_maps_by_successor(&conn, &new_ids[0]).unwrap();
        assert_eq!(by_successor.len(), 2);
        assert!(by_successor
            .iter()
            .all(|entry| entry.kind == MAP_KIND_MERGE));
        assert_eq!(
            by_successor[0].predecessor_page_id,
            Some(old_ids[0].clone())
        );
        assert_eq!(
            by_successor[1].predecessor_page_id,
            Some(old_ids[1].clone())
        );

        let by_predecessor = list_page_id_maps_by_predecessor(&conn, &old_ids[2]).unwrap();
        assert_eq!(by_predecessor.len(), 1);
        assert_eq!(by_predecessor[0].kind, MAP_KIND_RETIRE);
        assert_eq!(by_predecessor[0].successor_page_id, None);
        assert_eq!(by_predecessor[0].build_id, new_build);
        assert!(!by_predecessor[0].created_at.is_empty());

        // By build: everything written by this replan, in mapping order.
        let all = list_page_id_maps_for_build(&conn, &new_build).unwrap();
        assert_eq!(all.len(), 4);
        let kinds: Vec<&str> = all.iter().map(|entry| entry.kind.as_str()).collect();
        assert_eq!(kinds, ["merge", "merge", "retire", "split"]);

        // Other builds see nothing.
        assert!(list_page_id_maps_for_build(&conn, &old_build)
            .unwrap()
            .is_empty());
        assert!(list_page_id_maps_by_predecessor(&conn, &new_ids[0])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn page_id_map_rejects_invalid_kind() {
        // The kind CHECK constraint rejects unknown relations; the page
        // columns carry no FK by design (page_id is not UNIQUE since the
        // (build_id, page_id) key of migration 0005 — identity continuity is
        // the replan flow's responsibility, not the schema's).
        let mut conn = open_in_memory().unwrap();
        let build = start_build(&mut conn, &crate::builds::BuildDraft::default()).unwrap();
        let bad_kind = PageIdMapRow {
            predecessor_page_id: None,
            successor_page_id: None,
            kind: "nonsense".to_owned(),
            build_id: build,
        };
        assert!(insert_page_id_maps(&mut conn, &[bad_kind]).is_err());
    }

    #[test]
    fn page_id_map_empty_insert_is_a_noop_transaction() {
        let mut conn = open_in_memory().unwrap();
        let ids = insert_page_id_maps(&mut conn, &[]).unwrap();
        assert!(ids.is_empty());
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM page_id_map", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
}
