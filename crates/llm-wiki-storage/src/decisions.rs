//! Incremental-build decision records (PRD §19.2, migration 0006): one row
//! per mapping/replan judgment of a build. This is the audit substrate
//! `replan --dry-run` (V0.2) reads to explain WHY a workspace needs a replan
//! and what the last incremental judgment did.
//!
//! Outcomes: `local-update` (mapping succeeded, pages recompiled in place),
//! `replan-required` (mapping failed or the build fingerprint drifted), and
//! `fast-path` (no source changes; the cached full pipeline ran).
//!
//! `trigger` is a SQLite keyword — quoted in every statement below.

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{BuildId, DecisionId, SourceId};

/// Mapping outcome recorded in [`PlanDecision::outcome`].
pub const OUTCOME_LOCAL_UPDATE: &str = "local-update";
pub const OUTCOME_REPLAN_REQUIRED: &str = "replan-required";
pub const OUTCOME_FAST_PATH: &str = "fast-path";

/// Recorded triggers (PRD §19.2; the fingerprint guard adds its own).
pub const TRIGGER_FINGERPRINT_CHANGED: &str = "fingerprint-changed";
pub const TRIGGER_STRUCTURAL_CHANGE: &str = "structural-change";
pub const TRIGGER_UNMAPPABLE_NODE: &str = "unmappable-node";
pub const TRIGGER_PAGE_EMPTIED: &str = "page-emptied";

/// One incremental judgment to persist (PRD §19.2: every judgment is
/// recorded — mapping outcome, trigger, affected pages, notes).
#[derive(Debug, Clone)]
pub struct PlanDecision {
    pub build_id: BuildId,
    /// Source the judgment is about, when it is about one specific source.
    pub source_id: Option<SourceId>,
    /// `local-update` | `replan-required` | `fast-path`.
    pub outcome: String,
    /// `fingerprint-changed` | `structural-change` | `unmappable-node` |
    /// `page-emptied`; `None` for non-replan outcomes.
    pub trigger: Option<String>,
    pub affected_pages: u32,
    pub notes: String,
}

/// A persisted decision row (audit view; includes the record identity).
#[derive(Debug, Clone, PartialEq)]
pub struct PlanDecisionRow {
    pub decision_id: DecisionId,
    pub build_id: BuildId,
    pub source_id: Option<SourceId>,
    pub outcome: String,
    pub trigger: Option<String>,
    pub affected_pages: u32,
    pub notes: String,
    pub created_at: String,
}

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// Persists one decision row in a single transaction and returns its id.
pub fn insert_plan_decision(conn: &mut Connection, decision: &PlanDecision) -> Result<DecisionId> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let decision_id = DecisionId::generate();
    tx.execute(
        "INSERT INTO plan_decisions
         (decision_id, build_id, source_id, outcome, \"trigger\", affected_pages, notes, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            decision_id.as_str(),
            decision.build_id.as_str(),
            decision.source_id.as_ref().map(|s| s.as_str()),
            decision.outcome,
            decision.trigger,
            decision.affected_pages,
            decision.notes,
            chrono::Utc::now().to_rfc3339()
        ],
    )
    .map_err(db)?;
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit plan decision: {e}")))?;
    Ok(decision_id)
}

/// Loads every decision row of one build, oldest first (deterministic).
pub fn list_plan_decisions(conn: &Connection, build_id: &BuildId) -> Result<Vec<PlanDecisionRow>> {
    let mut stmt = conn
        .prepare(
            "SELECT decision_id, build_id, source_id, outcome, \"trigger\", affected_pages, notes, created_at
             FROM plan_decisions WHERE build_id = ?1 ORDER BY created_at, decision_id",
        )
        .map_err(|e| WikiError::Storage(format!("prepare list_plan_decisions: {e}")))?;
    let rows = stmt
        .query_map(params![build_id.as_str()], |row| {
            Ok((
                row.get::<_, String>("decision_id")?,
                row.get::<_, String>("build_id")?,
                row.get::<_, Option<String>>("source_id")?,
                row.get::<_, String>("outcome")?,
                row.get::<_, Option<String>>("trigger")?,
                row.get::<_, i64>("affected_pages")?,
                row.get::<_, String>("notes")?,
                row.get::<_, String>("created_at")?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("list_plan_decisions: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        let (decision_id, build_id, source_id, outcome, trigger, affected_pages, notes, created_at) =
            row.map_err(db)?;
        out.push(PlanDecisionRow {
            decision_id: DecisionId::from_validated(decision_id),
            build_id: BuildId::from_validated(build_id),
            source_id: source_id.map(SourceId::from_validated),
            outcome,
            trigger,
            affected_pages: affected_pages.max(0) as u32,
            notes,
            created_at,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;

    #[test]
    fn decision_roundtrip_preserves_every_field() {
        let mut conn = open_in_memory().unwrap();
        // Decision rows reference real builds and sources (FK).
        let build =
            crate::builds::start_build(&mut conn, &crate::builds::BuildDraft::default()).unwrap();
        let (source, _) = crate::sources::upsert_source(
            &mut conn,
            &llm_wiki_core::ids::SourceLocatorKey::compute("ws", "a.md"),
            "a.md",
            "hash-1",
            1,
            None,
        )
        .unwrap();
        let recorded = insert_plan_decision(
            &mut conn,
            &PlanDecision {
                build_id: build.clone(),
                source_id: Some(source.clone()),
                outcome: OUTCOME_REPLAN_REQUIRED.to_owned(),
                trigger: Some(TRIGGER_UNMAPPABLE_NODE.to_owned()),
                affected_pages: 3,
                notes: "brand-new source has no previous page ownership".into(),
            },
        )
        .unwrap();

        let rows = list_plan_decisions(&conn, &build).unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.decision_id, recorded);
        assert_eq!(row.build_id, build);
        assert_eq!(row.source_id, Some(source));
        assert_eq!(row.outcome, "replan-required");
        assert_eq!(row.trigger.as_deref(), Some("unmappable-node"));
        assert_eq!(row.affected_pages, 3);
        assert!(row.notes.contains("no previous page ownership"));
        assert!(!row.created_at.is_empty());

        // NULL source_id and NULL trigger round-trip.
        insert_plan_decision(
            &mut conn,
            &PlanDecision {
                build_id: build.clone(),
                source_id: None,
                outcome: OUTCOME_FAST_PATH.to_owned(),
                trigger: None,
                affected_pages: 0,
                notes: "no changes".into(),
            },
        )
        .unwrap();
        let rows = list_plan_decisions(&conn, &build).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].source_id, None);
        assert_eq!(rows[1].trigger, None);
        assert_eq!(rows[1].outcome, "fast-path");

        // Decisions are scoped per build.
        assert!(list_plan_decisions(&conn, &BuildId::generate())
            .unwrap()
            .is_empty());
    }
}
