//! Source-local Section Registry (PRD §45, §51 step 06/07).
//!
//! Persists per-source sections with opaque `SectionId`s and applies the
//! deterministic [`MatchReport`] of the Section Matcher: carried identities
//! get their range/fingerprint refreshed, created ones receive the
//! caller-assigned new IDs, unmatched ones retire. Ambiguous sources are
//! handled by the caller (re-analysis required) — this layer never guesses.

use std::collections::BTreeMap;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{SectionId, SourceId};
use llm_wiki_core::matcher::{MatchReport, SectionIdentity, SectionOutcome};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSection {
    pub section_id: SectionId,
    pub heading_path: Vec<String>,
    pub fingerprint: String,
    pub range_start: usize,
    pub range_end: usize,
}

fn row_to_section(row: &rusqlite::Row) -> rusqlite::Result<StoredSection> {
    let path_json: String = row.get("heading_path_json")?;
    let heading_path: Vec<String> = serde_json::from_str(&path_json).unwrap_or_default();
    Ok(StoredSection {
        section_id: SectionId::from_validated(row.get::<_, String>("section_id")?),
        heading_path,
        fingerprint: row.get("content_fingerprint")?,
        range_start: row.get::<_, i64>("range_start")? as usize,
        range_end: row.get::<_, i64>("range_end")? as usize,
    })
}

pub fn load_active_sections(conn: &Connection, source_id: &SourceId) -> Result<Vec<StoredSection>> {
    let mut stmt = conn
        .prepare(
            "SELECT section_id, heading_path_json, content_fingerprint, range_start, range_end
             FROM source_sections
             WHERE source_id = ?1 AND status = 'active'
             ORDER BY range_start",
        )
        .map_err(|e| WikiError::Storage(format!("prepare load_sections: {e}")))?;
    let rows = stmt
        .query_map(params![source_id.as_str()], row_to_section)
        .map_err(|e| WikiError::Storage(format!("load_sections: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| WikiError::Storage(format!("load_sections row: {e}")))?);
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionApplyStats {
    pub carried: usize,
    pub created: usize,
    pub retired: usize,
}

/// Applies a match report inside one transaction.
///
/// `created_ids` maps current-section indices with outcome [`SectionOutcome::Created`]
/// to freshly generated [`SectionId`]s.
pub fn apply_section_matches(
    conn: &mut Connection,
    source_id: &SourceId,
    report: &MatchReport,
    current: &[SectionIdentity],
    ranges: &[llm_wiki_core::model::SourceRange],
    created_ids: &BTreeMap<usize, SectionId>,
    build_id: Option<&str>,
) -> Result<SectionApplyStats> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;

    let mut stats = SectionApplyStats {
        carried: 0,
        created: 0,
        retired: 0,
    };

    for assignment in &report.assignments {
        let identity = &current[assignment.cur_index];
        let range = ranges
            .get(assignment.cur_index)
            .copied()
            .unwrap_or(llm_wiki_core::model::SourceRange::new(0, 0));
        let path_json = serde_json::to_string(&identity.heading_path)
            .map_err(|e| WikiError::Storage(format!("serialize heading path: {e}")))?;
        let path_key = identity.path_key();

        match &assignment.outcome {
            SectionOutcome::Carried { prev } => {
                tx.execute(
                    "UPDATE source_sections
                     SET heading_path_json = ?1, heading_path_key = ?2, content_fingerprint = ?3,
                         range_start = ?4, range_end = ?5, status = 'active'
                     WHERE section_id = ?6 AND source_id = ?7",
                    params![
                        path_json,
                        path_key,
                        identity.fingerprint,
                        range.start as i64,
                        range.end as i64,
                        prev.as_str(),
                        source_id.as_str()
                    ],
                )
                .map_err(|e| WikiError::Storage(format!("carry section: {e}")))?;
                stats.carried += 1;
            }
            SectionOutcome::Created => {
                let new_id = created_ids.get(&assignment.cur_index).ok_or_else(|| {
                    WikiError::Storage(format!(
                        "section {} is Created but no id was assigned",
                        assignment.cur_index
                    ))
                })?;
                tx.execute(
                    "INSERT INTO source_sections
                     (section_id, source_id, heading_path_json, heading_path_key, content_fingerprint, range_start, range_end, status, created_build_id)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'active', ?8)",
                    params![
                        new_id.as_str(),
                        source_id.as_str(),
                        path_json,
                        path_key,
                        identity.fingerprint,
                        range.start as i64,
                        range.end as i64,
                        build_id
                    ],
                )
                .map_err(|e| WikiError::Storage(format!("insert section: {e}")))?;
                stats.created += 1;
            }
            // Ambiguous sections get no row from this report; the caller must
            // re-analyse the source (PRD §45). They simply do not count here.
            SectionOutcome::Ambiguous => {}
        }
    }

    for retired in &report.retired {
        let changed = tx
            .execute(
                "UPDATE source_sections SET status = 'retired', retired_build_id = ?1
                 WHERE section_id = ?2 AND status = 'active'",
                params![build_id, retired.as_str()],
            )
            .map_err(|e| WikiError::Storage(format!("retire section: {e}")))?;
        stats.retired += changed;
    }

    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit sections: {e}")))?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;
    use crate::sources::upsert_source;
    use llm_wiki_core::matcher::{match_sections, PrevSection, SectionIdentity};

    fn setup() -> (Connection, SourceId) {
        let mut conn = open_in_memory().unwrap();
        let (source_id, _) = upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "a.md"),
            "a.md",
            "hash",
            1,
            None,
        )
        .unwrap();
        (conn, source_id)
    }

    fn identity(path: &[&str], content: &str) -> SectionIdentity {
        SectionIdentity::from_parts(
            &path.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            content,
        )
    }

    #[test]
    fn insert_heading_keeps_neighbor_ids_across_persistence() {
        let (mut conn, source_id) = setup();

        // First build: two sections.
        let first = vec![
            identity(&["Doc", "Intro"], "intro"),
            identity(&["Doc", "Usage"], "usage"),
        ];
        let ranges_first = vec![
            llm_wiki_core::model::SourceRange::new(0, 6),
            llm_wiki_core::model::SourceRange::new(6, 12),
        ];
        let empty_prev: Vec<PrevSection> = Vec::new();
        let report1 = match_sections(&empty_prev, &first);
        let mut ids1 = BTreeMap::new();
        for a in &report1.assignments {
            ids1.insert(a.cur_index, SectionId::generate());
        }
        let stats1 = apply_section_matches(
            &mut conn,
            &source_id,
            &report1,
            &first,
            &ranges_first,
            &ids1,
            Some("b1"),
        )
        .unwrap();
        assert_eq!((stats1.carried, stats1.created, stats1.retired), (0, 2, 0));

        let stored1 = load_active_sections(&conn, &source_id).unwrap();
        let id_intro = stored1[0].section_id.clone();
        let id_usage = stored1[1].section_id.clone();

        // Second build: a heading was inserted in the middle.
        let second = vec![
            identity(&["Doc", "Intro"], "intro"),
            identity(&["Doc", "Middle"], "brand new"),
            identity(&["Doc", "Usage"], "usage"),
        ];
        let ranges_second = vec![
            llm_wiki_core::model::SourceRange::new(0, 6),
            llm_wiki_core::model::SourceRange::new(6, 20),
            llm_wiki_core::model::SourceRange::new(20, 26),
        ];
        let prev_sections: Vec<PrevSection> = stored1
            .iter()
            .map(|s| PrevSection {
                id: s.section_id.clone(),
                identity: SectionIdentity {
                    heading_path: s.heading_path.clone(),
                    fingerprint: s.fingerprint.clone(),
                },
            })
            .collect();
        let report2 = match_sections(&prev_sections, &second);
        assert!(!report2.ambiguous);
        let mut ids2 = BTreeMap::new();
        for a in &report2.assignments {
            if a.outcome == SectionOutcome::Created {
                ids2.insert(a.cur_index, SectionId::generate());
            }
        }
        apply_section_matches(
            &mut conn,
            &source_id,
            &report2,
            &second,
            &ranges_second,
            &ids2,
            Some("b2"),
        )
        .unwrap();

        let stored2 = load_active_sections(&conn, &source_id).unwrap();
        assert_eq!(stored2.len(), 3);
        assert_eq!(
            stored2[0].section_id, id_intro,
            "unrelated section must keep its id"
        );
        assert_eq!(
            stored2[2].section_id, id_usage,
            "unrelated section must keep its id"
        );
        assert_eq!(report2.retired.len(), 0);
    }

    #[test]
    fn retired_sections_leave_the_active_set() {
        let (mut conn, source_id) = setup();
        let first = vec![identity(&["Doc", "A"], "a"), identity(&["Doc", "B"], "b")];
        let ranges = vec![
            llm_wiki_core::model::SourceRange::new(0, 1),
            llm_wiki_core::model::SourceRange::new(1, 2),
        ];
        let report1 = match_sections(&[], &first);
        let mut ids = BTreeMap::new();
        for a in &report1.assignments {
            ids.insert(a.cur_index, SectionId::generate());
        }
        apply_section_matches(&mut conn, &source_id, &report1, &first, &ranges, &ids, None)
            .unwrap();

        let second = vec![identity(&["Doc", "A"], "a")];
        let stored1 = load_active_sections(&conn, &source_id).unwrap();
        let prevs: Vec<PrevSection> = stored1
            .iter()
            .map(|s| PrevSection {
                id: s.section_id.clone(),
                identity: SectionIdentity {
                    heading_path: s.heading_path.clone(),
                    fingerprint: s.fingerprint.clone(),
                },
            })
            .collect();
        let report2 = match_sections(&prevs, &second);
        assert_eq!(report2.retired.len(), 1);
        apply_section_matches(
            &mut conn,
            &source_id,
            &report2,
            &second,
            &[ranges[0]],
            &BTreeMap::new(),
            None,
        )
        .unwrap();

        let stored2 = load_active_sections(&conn, &source_id).unwrap();
        assert_eq!(stored2.len(), 1);
    }
}
