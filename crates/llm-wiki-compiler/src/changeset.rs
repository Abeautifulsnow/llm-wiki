//! Source diff (PRD §19.1) and the BuildFingerprint (PRD §18.1/§19.2).
//!
//! Both steps of the ChangeSet are PURE functions over plain data, so they
//! unit-test without a scanner or a database. The diff MUST run before
//! [`llm_wiki_storage::upsert_sources_batch`] — the upsert overwrites the
//! stored `content_hash`, which would make every modified source look
//! unchanged. `added` sources only receive their opaque `SourceId` when the
//! upsert mints it, so the final ChangeSet is assembled in two steps:
//! [`diff_manifest`] (before the upsert) → [`finalize_change_set`] (after).
//!
//! Locator semantics: identity is the recomputable `SourceLocatorKey`
//! (PRD §8.2). A renamed/moved file is therefore a delete+add pair — rename
//! matching arrives with later V0.2 work (PRD §43).

use serde::{Deserialize, Serialize};

use llm_wiki_core::ids::SourceId;

/// One entry of the scanned manifest (PRD §19.1 left side).
#[derive(Debug, Clone, Copy)]
pub struct ScannedSource<'a> {
    pub locator_key: &'a str,
    pub content_hash: &'a str,
}

/// One ACTIVE row of the Source Registry (PRD §19.1 right side), read BEFORE
/// the upsert refreshes hashes.
#[derive(Debug, Clone, Copy)]
pub struct RegisteredSource<'a> {
    pub source_id: &'a SourceId,
    pub locator_key: &'a str,
    pub content_hash: &'a str,
}

/// Per-scanned-file classification (PRD §19.1). `Added` files receive their
/// `SourceId` from [`finalize_change_set`] once the registry upsert minted it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOutcome {
    Added,
    Modified(SourceId),
    Unchanged(SourceId),
}

/// The diff between the scan and the registry for one build (PRD §19.1).
/// `added`/`modified`/`unchanged` follow the scan order (the scanner's output
/// order is deterministic); `deleted` is sorted by source id.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChangeSet {
    /// New locator keys (no ACTIVE registry row carried them).
    pub added: Vec<SourceId>,
    /// Locator exists and the content hash differs.
    pub modified: Vec<SourceId>,
    /// ACTIVE registry sources absent from the scan.
    pub deleted: Vec<SourceId>,
    /// Locator exists and the content hash matches.
    pub unchanged: Vec<SourceId>,
}

impl ChangeSet {
    /// True when the workspace has any source change at all.
    pub fn has_changes(&self) -> bool {
        !self.added.is_empty() || !self.modified.is_empty() || !self.deleted.is_empty()
    }

    /// added + modified — the sources that must be re-analyzed (§19.2).
    pub fn changed_count(&self) -> usize {
        self.added.len() + self.modified.len()
    }
}

/// Pure diff: scanned manifest vs Source Registry by
/// `locator_key → content_hash` (PRD §19.1). Returns one outcome per scanned
/// file (input order) plus the ACTIVE registry ids missing from the scan.
/// Sources re-appearing after a removal classify as `Added` — the incremental
/// pipeline re-analyzes them, which also re-activates the registry row.
pub fn diff_manifest(
    scanned: &[ScannedSource<'_>],
    registry: &[RegisteredSource<'_>],
) -> (Vec<FileOutcome>, Vec<SourceId>) {
    let by_locator: std::collections::BTreeMap<&str, &RegisteredSource<'_>> = registry
        .iter()
        .map(|source| (source.locator_key, source))
        .collect();

    let mut outcomes = Vec::with_capacity(scanned.len());
    for scanned_source in scanned {
        outcomes.push(match by_locator.get(scanned_source.locator_key) {
            Some(registered) if registered.content_hash == scanned_source.content_hash => {
                FileOutcome::Unchanged(registered.source_id.clone())
            }
            Some(registered) => FileOutcome::Modified(registered.source_id.clone()),
            None => FileOutcome::Added,
        });
    }

    let mut deleted: Vec<SourceId> = registry
        .iter()
        .filter(|registered| {
            !scanned
                .iter()
                .any(|scanned_source| scanned_source.locator_key == registered.locator_key)
        })
        .map(|registered| registered.source_id.clone())
        .collect();
    deleted.sort();
    (outcomes, deleted)
}

/// Assembles the final ChangeSet: pairs the minted ids of the upsert
/// (`upserted` is aligned with the scanned files) with [`diff_manifest`]'s
/// outcomes. Pure.
pub fn finalize_change_set(
    outcomes: &[FileOutcome],
    deleted: Vec<SourceId>,
    upserted: &[(SourceId, bool)],
) -> ChangeSet {
    let mut change_set = ChangeSet {
        deleted,
        ..ChangeSet::default()
    };
    for (outcome, (source_id, _created)) in outcomes.iter().zip(upserted) {
        match outcome {
            FileOutcome::Added => change_set.added.push(source_id.clone()),
            FileOutcome::Modified(id) => change_set.modified.push(id.clone()),
            FileOutcome::Unchanged(id) => change_set.unchanged.push(id.clone()),
        }
    }
    change_set
}

/// The planning-relevant BuildFingerprint (PRD §18.1/§19.2): the fields whose
/// change invalidates the CURRENT plan. The incremental pipeline serializes
/// this onto the `builds.build_fingerprint` column and compares it against
/// the last COMPLETED build — ANY drift forces `REPLAN_REQUIRED` with trigger
/// `fingerprint-changed` (§19.2: planner prompt, schema, rules or fingerprint
/// drift must never silently trigger a global re-plan). The configured model
/// is covered by the effective-config hash; the fingerprint carries the
/// prompt/schema/parser/rule inputs the plan was derived from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildFingerprint {
    /// `document-analysis@N` (analysis prompt fingerprint tag).
    pub analysis_prompt: String,
    /// `wiki-planning@N` (planner prompt fingerprint tag).
    pub planning_prompt: String,
    /// `wiki-compilation@N` (compiler prompt fingerprint tag).
    pub compilation_prompt: String,
    /// Response schema version (§28).
    pub schema_version: String,
    /// Parser/normalizer version (§18.1).
    pub parser_version: String,
    /// Canonical hash of the effective config (§32), which includes the
    /// configured model and every planning-relevant option.
    pub config_hash: String,
}

impl BuildFingerprint {
    /// Canonical JSON serialization persisted on the build row.
    pub fn to_json(&self) -> String {
        // Struct field order is fixed, so the serialization is canonical.
        serde_json::to_string(self).expect("fingerprint serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(tag: &str) -> SourceId {
        // Deterministic stand-ins: the diff never inspects id internals.
        SourceId::from_validated(format!("src_{tag}"))
    }

    fn scanned<'a>(locator: &'a str, hash: &'a str) -> ScannedSource<'a> {
        ScannedSource {
            locator_key: locator,
            content_hash: hash,
        }
    }

    fn registered<'a>(
        source: &'a SourceId,
        locator: &'a str,
        hash: &'a str,
    ) -> RegisteredSource<'a> {
        RegisteredSource {
            source_id: source,
            locator_key: locator,
            content_hash: hash,
        }
    }

    #[test]
    fn empty_inputs_produce_an_empty_changeset() {
        let (outcomes, deleted) = diff_manifest(&[], &[]);
        assert!(outcomes.is_empty());
        assert!(deleted.is_empty());
        let change_set = finalize_change_set(&outcomes, deleted, &[]);
        assert!(!change_set.has_changes());
        assert_eq!(change_set.changed_count(), 0);
    }

    #[test]
    fn classifies_added_modified_deleted_unchanged() {
        let id_a = id("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let id_b = id("01BX5ZZKBKACTAV9WEVGEMMVRZ");
        let id_c = id("01CZZZZZZZZZZZZZZZZZZZZZZZ");
        let id_d = id("01DARZ3NDEKTSV4RRFFQ69G5FAV");

        let scan = vec![
            scanned("loc-a", "hash-a1"), // unchanged
            scanned("loc-b", "hash-b2"), // modified (was hash-b1)
            scanned("loc-d", "hash-d1"), // added
        ];
        let registry = vec![
            registered(&id_a, "loc-a", "hash-a1"),
            registered(&id_b, "loc-b", "hash-b1"),
            registered(&id_c, "loc-c", "hash-c1"), // deleted
        ];

        let (outcomes, deleted) = diff_manifest(&scan, &registry);
        assert_eq!(
            outcomes,
            vec![
                FileOutcome::Unchanged(id_a.clone()),
                FileOutcome::Modified(id_b.clone()),
                FileOutcome::Added,
            ]
        );
        assert_eq!(deleted, vec![id_c.clone()]);

        // The upsert minted id_d for the added file.
        let upserted = vec![
            (id_a.clone(), false),
            (id_b.clone(), false),
            (id_d.clone(), true),
        ];
        let change_set = finalize_change_set(&outcomes, deleted, &upserted);
        assert!(change_set.has_changes());
        assert_eq!(change_set.unchanged, vec![id_a]);
        assert_eq!(change_set.modified, vec![id_b]);
        assert_eq!(change_set.deleted, vec![id_c]);
        assert_eq!(change_set.added, vec![id_d]);
        assert_eq!(change_set.changed_count(), 2, "added + modified");
    }

    #[test]
    fn rename_is_delete_plus_add_by_locator_semantics() {
        // Same content, new path: the locator changed, so the diff sees a
        // deletion and an addition (PRD §43 semantics; documented behavior).
        let id_a = id("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let id_b = id("01BX5ZZKBKACTAV9WEVGEMMVRZ");
        let scan = vec![scanned("loc-b", "same-hash")];
        let registry = vec![registered(&id_a, "loc-a", "same-hash")];

        let (outcomes, deleted) = diff_manifest(&scan, &registry);
        assert_eq!(outcomes, vec![FileOutcome::Added]);
        assert_eq!(deleted, vec![id_a.clone()]);

        let change_set = finalize_change_set(&outcomes, deleted, &[(id_b.clone(), true)]);
        assert_eq!(change_set.deleted, vec![id_a]);
        assert_eq!(change_set.added, vec![id_b]);
        assert!(change_set.modified.is_empty());
        assert!(change_set.unchanged.is_empty());
    }

    #[test]
    fn reappearing_removed_source_classifies_as_added() {
        // A removed source coming back has no ACTIVE registry row, so the
        // diff treats it as new knowledge that must be re-analyzed.
        let id_a = id("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let scan = vec![scanned("loc-a", "hash-a1")];
        let (outcomes, deleted) = diff_manifest(&scan, &[]);
        assert_eq!(outcomes, vec![FileOutcome::Added]);
        assert!(deleted.is_empty());
        let change_set = finalize_change_set(&outcomes, deleted, &[(id_a, false)]);
        assert_eq!(change_set.added.len(), 1);
    }

    #[test]
    fn diff_is_order_independent_for_deleted_and_modified() {
        let id_a = id("01ARZ3NDEKTSV4RRFFQ69G5FAV");
        let id_b = id("01BX5ZZKBKACTAV9WEVGEMMVRZ");
        let id_c = id("01CZZZZZZZZZZZZZZZZZZZZZZZ");

        let scan = vec![
            scanned("loc-c", "h-c2"),
            scanned("loc-b", "h-b2"),
            scanned("loc-a", "h-a"),
        ];
        let registry = vec![
            registered(&id_c, "loc-c", "h-c1"),
            registered(&id_b, "loc-b", "h-b1"),
            registered(&id_a, "loc-a", "h-a"),
        ];

        let (outcomes, deleted) = diff_manifest(&scan, &registry);
        // Every locator was scanned, so nothing is deleted.
        assert!(deleted.is_empty());

        let modified: Vec<SourceId> = outcomes
            .iter()
            .filter_map(|outcome| match outcome {
                FileOutcome::Modified(id) => Some(id.clone()),
                _ => None,
            })
            .collect();
        // `modified` follows the scan order (the scanner's output order is
        // deterministic); `deleted` is sorted by source id.
        assert_eq!(modified, vec![id_c.clone(), id_b.clone()]);

        let mut reversed_scan = scan;
        reversed_scan.reverse();
        let (_, deleted_again) = diff_manifest(&reversed_scan, &registry);
        assert_eq!(deleted, deleted_again);
    }

    #[test]
    fn fingerprint_json_is_canonical_and_sensitive() {
        let fingerprint = BuildFingerprint {
            analysis_prompt: "document-analysis@1".into(),
            planning_prompt: "wiki-planning@1".into(),
            compilation_prompt: "wiki-compilation@1".into(),
            schema_version: "1".into(),
            parser_version: "0.1.0".into(),
            config_hash: "cfg".into(),
        };
        let json = fingerprint.to_json();
        assert_eq!(json, fingerprint.to_json(), "canonical serialization");
        assert!(json.contains("config_hash"), "every field serializes");

        let drifted = BuildFingerprint {
            config_hash: "changed".into(),
            ..fingerprint.clone()
        };
        assert_ne!(json, drifted.to_json());
        assert_ne!(fingerprint, drifted);

        // Round-trips through the stored form.
        let parsed: BuildFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, fingerprint);
    }
}
