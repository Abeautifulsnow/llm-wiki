//! Section Matcher: carries section identity across builds (PRD §45).
//!
//! Matching is deterministic and layered, in fixed order:
//! 1. exact content fingerprint,
//! 2. unique normalized heading path,
//! 3. heading-path ordinal tie-break — used *only* to order genuine duplicate
//!    headings, and never mixed into any ID (PRD §45: ordinal must never enter
//!    the ID hash).
//!
//! Outcomes per current section: `Carried` (reuse previous [`SectionId`]),
//! `Created` (new identity), `Ambiguous` (several near-matches — identity is
//! retired and the source must be re-analysed; citations are never migrated
//! to a guessed candidate). Unmatched previous sections are `Retired`.

use std::collections::HashMap;

use crate::ids::SectionId;

/// Identity inputs of one section, normalized. `heading_path` entries are
/// NFC/trimmed; `fingerprint` is the content fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionIdentity {
    pub heading_path: Vec<String>,
    pub fingerprint: String,
}

impl SectionIdentity {
    pub fn from_parts(heading_path: &[String], raw_content: &str) -> Self {
        Self {
            heading_path: heading_path.iter().map(|h| normalize_heading(h)).collect(),
            fingerprint: crate::hash::content_fingerprint(raw_content),
        }
    }

    /// Key used for heading-path comparison.
    pub fn path_key(&self) -> String {
        self.heading_path.join("\u{1f}")
    }
}

pub fn normalize_heading(heading: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    heading.trim().nfc().collect()
}

/// A previous build's section as stored in the Section Registry.
#[derive(Debug, Clone)]
pub struct PrevSection {
    pub id: SectionId,
    pub identity: SectionIdentity,
}

/// What happened to one current section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionOutcome {
    Carried { prev: SectionId },
    Created,
    Ambiguous,
}

#[derive(Debug, Clone)]
pub struct SectionAssignment {
    /// Index into the current sections slice.
    pub cur_index: usize,
    pub outcome: SectionOutcome,
}

/// Full deterministic match report for one source.
#[derive(Debug, Clone)]
pub struct MatchReport {
    /// One assignment per current section, in current order.
    pub assignments: Vec<SectionAssignment>,
    /// Previous sections that found no successor (PRD §45: retired).
    pub retired: Vec<SectionId>,
    /// True when any section was ambiguous: the whole source requires
    /// re-analysis and no old identity may be silently reused.
    pub ambiguous: bool,
}

/// Extracts the comparable identity from raw stored content, normalizing the
/// same way as [`SectionIdentity::from_parts`].
pub fn stored_identity(heading_path: &[String], stored_content: &str) -> SectionIdentity {
    SectionIdentity {
        heading_path: heading_path.iter().map(|h| normalize_heading(h)).collect(),
        fingerprint: crate::hash::content_fingerprint(stored_content),
    }
}

enum Ambiguity {
    /// Several near-matches: identity must not be continued (PRD §45).
    ConflictingEvidence,
    /// No previous section resembles this one: brand-new identity.
    NoCandidate,
}

pub fn match_sections(prev: &[PrevSection], cur: &[SectionIdentity]) -> MatchReport {
    // Lookup indices built once; iteration order of outputs always follows
    // the input slices, so the report is deterministic.
    let mut by_fingerprint: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut by_path: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, prev_section) in prev.iter().enumerate() {
        by_fingerprint
            .entry(prev_section.identity.fingerprint.as_str())
            .or_default()
            .push(idx);
        by_path
            .entry(prev_section.identity.path_key())
            .or_default()
            .push(idx);
    }

    let mut taken = vec![false; prev.len()];
    let mut assignments = Vec::with_capacity(cur.len());
    let mut ambiguous = false;

    // Previous-side occurrences per heading path, in document order. The
    // duplicate-heading tie-breaker takes the first not-yet-taken occurrence
    // — deterministic, and nothing ordinal ever enters an ID (PRD §45).
    let mut prev_ordinal: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, prev_section) in prev.iter().enumerate() {
        prev_ordinal
            .entry(prev_section.identity.path_key())
            .or_default()
            .push(idx);
    }

    for (i, identity) in cur.iter().enumerate() {
        let path_key = identity.path_key();

        // Unmatched candidates per evidence layer.
        let fp_candidates: Vec<usize> = by_fingerprint
            .get(identity.fingerprint.as_str())
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|idx| !taken[*idx])
            .collect();
        let path_candidates: Vec<usize> = by_path
            .get(&path_key)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|idx| !taken[*idx])
            .collect();

        // Fixed decision order (PRD §45).
        let carried: Result<usize, Ambiguity> = if fp_candidates.len() == 1 {
            // 1. exact fingerprint — strongest evidence, even across a
            //    renamed heading.
            Ok(fp_candidates[0])
        } else if !fp_candidates.is_empty() {
            let same_path: Vec<usize> = fp_candidates
                .iter()
                .copied()
                .filter(|idx| prev[*idx].identity.path_key() == path_key)
                .collect();
            match same_path.len() {
                1 => Ok(same_path[0]),
                // Several indistinguishable copies (same content, same or
                // different headings): conservative ambiguity.
                _ => Err(Ambiguity::ConflictingEvidence),
            }
        } else {
            match path_candidates.len() {
                // 2. unique heading path.
                1 => Ok(path_candidates[0]),
                // 3. ordinal tie-break among genuine duplicate headings —
                //    position decides, nothing ordinal enters any ID.
                _ => prev_ordinal
                    .get(&path_key)
                    .and_then(|occ| occ.iter().find(|idx| !taken[**idx]))
                    .copied()
                    .ok_or(Ambiguity::NoCandidate),
            }
        };

        match carried {
            Ok(idx) => {
                taken[idx] = true;
                assignments.push(SectionAssignment {
                    cur_index: i,
                    outcome: SectionOutcome::Carried {
                        prev: prev[idx].id.clone(),
                    },
                });
            }
            Err(Ambiguity::ConflictingEvidence) => {
                ambiguous = true;
                assignments.push(SectionAssignment {
                    cur_index: i,
                    outcome: SectionOutcome::Ambiguous,
                });
            }
            Err(Ambiguity::NoCandidate) => {
                assignments.push(SectionAssignment {
                    cur_index: i,
                    outcome: SectionOutcome::Created,
                });
            }
        }
    }

    let retired = prev
        .iter()
        .enumerate()
        .filter(|(idx, _)| !taken[*idx])
        .map(|(_, prev_section)| prev_section.id.clone())
        .collect();

    MatchReport {
        assignments,
        retired,
        ambiguous,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prev(id: &str, path: &[&str], content: &str) -> PrevSection {
        PrevSection {
            id: SectionId::parse(id).unwrap(),
            identity: SectionIdentity::from_parts(
                &path.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                content,
            ),
        }
    }

    fn cur(path: &[&str], content: &str) -> SectionIdentity {
        SectionIdentity::from_parts(
            &path.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            content,
        )
    }

    const A: &str = "sec_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    const B: &str = "sec_01BX5ZZKBKACTAV9WEVGEMMVRZ";
    const C: &str = "sec_01C3DZJZRVTQVB3JHU6GVWP1JS";

    #[test]
    fn unchanged_document_carries_every_identity() {
        let prevs = vec![
            prev(A, &["Doc", "Intro"], "intro body"),
            prev(B, &["Doc", "Usage"], "usage body"),
        ];
        let curs = vec![
            cur(&["Doc", "Intro"], "intro body"),
            cur(&["Doc", "Usage"], "usage body"),
        ];
        let report = match_sections(&prevs, &curs);
        assert!(!report.ambiguous);
        assert!(report.retired.is_empty());
        assert!(report
            .assignments
            .iter()
            .all(|a| matches!(a.outcome, SectionOutcome::Carried { .. })));
    }

    #[test]
    fn inserting_a_heading_does_not_disturb_neighbors() {
        // PRD DoD #5: inserting a heading must not change unrelated SectionIds.
        let prevs = vec![
            prev(A, &["Doc", "Intro"], "intro body"),
            prev(B, &["Doc", "Usage"], "usage body"),
        ];
        let curs = vec![
            cur(&["Doc", "Intro"], "intro body"),
            cur(&["Doc", "New Middle"], "brand new"),
            cur(&["Doc", "Usage"], "usage body"),
        ];
        let report = match_sections(&prevs, &curs);
        assert!(!report.ambiguous);
        assert_eq!(
            report.assignments[0].outcome,
            SectionOutcome::Carried {
                prev: SectionId::parse(A).unwrap()
            }
        );
        assert_eq!(report.assignments[1].outcome, SectionOutcome::Created);
        assert_eq!(
            report.assignments[2].outcome,
            SectionOutcome::Carried {
                prev: SectionId::parse(B).unwrap()
            }
        );
        assert!(report.retired.is_empty());
    }

    #[test]
    fn deleting_a_heading_retires_only_it() {
        let prevs = vec![
            prev(A, &["Doc", "Intro"], "intro body"),
            prev(B, &["Doc", "Usage"], "usage body"),
            prev(C, &["Doc", "Legacy"], "legacy body"),
        ];
        let curs = vec![
            cur(&["Doc", "Intro"], "intro body"),
            cur(&["Doc", "Usage"], "usage body"),
        ];
        let report = match_sections(&prevs, &curs);
        assert!(!report.ambiguous);
        assert_eq!(report.retired, vec![SectionId::parse(C).unwrap()]);
    }

    #[test]
    fn renamed_heading_keeps_identity_via_fingerprint() {
        let prevs = vec![prev(A, &["Doc", "Old Name"], "same body")];
        let curs = vec![cur(&["Doc", "New Name"], "same body")];
        let report = match_sections(&prevs, &curs);
        assert!(!report.ambiguous);
        assert_eq!(
            report.assignments[0].outcome,
            SectionOutcome::Carried {
                prev: SectionId::parse(A).unwrap()
            }
        );
    }

    #[test]
    fn duplicate_headings_match_by_ordinal() {
        let prevs = vec![
            prev(A, &["Doc", "Notes"], "first notes"),
            prev(B, &["Doc", "Notes"], "second notes"),
        ];
        let curs = vec![
            cur(&["Doc", "Notes"], "first notes"),
            cur(&["Doc", "Notes"], "second notes"),
        ];
        let report = match_sections(&prevs, &curs);
        assert!(!report.ambiguous);
        assert!(matches!(
            report.assignments[0].outcome,
            SectionOutcome::Carried { .. }
        ));
        assert!(matches!(
            report.assignments[1].outcome,
            SectionOutcome::Carried { .. }
        ));
    }

    #[test]
    fn pure_new_section_is_created_not_ambiguous() {
        let prevs = vec![prev(A, &["Doc", "Only"], "original")];
        let curs = vec![
            cur(&["Doc", "Only"], "original"),
            cur(&["Doc", "Fresh"], "never seen before"),
        ];
        let report = match_sections(&prevs, &curs);
        assert!(!report.ambiguous);
        assert_eq!(report.assignments[1].outcome, SectionOutcome::Created);
        assert!(report.retired.is_empty());
    }

    #[test]
    fn one_section_collapsing_two_same_content_copies_is_ambiguous() {
        // Two previous sections with identical content collapse into one:
        // per PRD §45 the source must be re-analysed instead of guessing.
        let prevs = vec![
            prev(A, &["Doc", "Left"], "twins"),
            prev(B, &["Doc", "Right"], "twins"),
        ];
        let curs = vec![cur(&["Doc", "Merged"], "twins")];
        let report = match_sections(&prevs, &curs);
        assert!(report.ambiguous);
        assert_eq!(report.assignments[0].outcome, SectionOutcome::Ambiguous);
        assert_eq!(report.retired.len(), 2);
    }
}
