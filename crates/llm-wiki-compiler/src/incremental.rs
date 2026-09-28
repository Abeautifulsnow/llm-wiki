//! Deterministic incremental mapping (PRD §19.2/§19.3): after re-analyzing
//! only the changed sources, decide whether the change can be localized onto
//! the CURRENT generation — or must stop at `REPLAN_REQUIRED` (§19.2: never
//! silently publish a wiki that mixes an old plan with new knowledge).
//!
//! Judgment order is fixed (PRD §19.2): first map by Registry ID and the
//! existing `knowledge_refs`; candidate pages come from the pages that own
//! affected nodes plus the pages that previously cited the changed sources
//! (a source's sections' previous page ownership). Every judgment is
//! recorded by the caller in `plan_decisions` (migration 0006).

use std::collections::{BTreeMap, BTreeSet};

use llm_wiki_core::ids::{KnowledgeNodeId, SectionId, SourceId, WikiPageId};
use llm_wiki_core::plan::KnowledgeBase;
use llm_wiki_storage::{
    GenerationPageView, SourceNodeSection, TRIGGER_PAGE_EMPTIED, TRIGGER_STRUCTURAL_CHANGE,
    TRIGGER_UNMAPPABLE_NODE,
};

/// Everything the deterministic mapping needs to judge one incremental build.
#[derive(Debug, Clone)]
pub struct MappingInput {
    /// The currently published generation's pages (with refs, citations and
    /// links) — the mapping target. Deterministic order (slug).
    pub prev_pages: Vec<GenerationPageView>,
    /// Knowledge base BEFORE the build touched anything (previous state).
    pub prev_kb: KnowledgeBase,
    /// Knowledge base AFTER deletion retirement + selective re-analysis.
    pub new_kb: KnowledgeBase,
    /// Nodes whose support vanished because DELETED sources were retired
    /// (§19.3). Losses here may empty a page into `obsolete`.
    pub deletion_gone: BTreeSet<KnowledgeNodeId>,
    /// For each MODIFIED source: the nodes it supported before re-analysis.
    pub modified_pre: BTreeMap<SourceId, Vec<SourceNodeSection>>,
    /// For each ADDED/MODIFIED source: the nodes it supports after
    /// re-analysis.
    pub changed_post: BTreeMap<SourceId, Vec<SourceNodeSection>>,
    /// The MODIFIED source ids (subset of `changed_post` keys).
    pub modified_ids: BTreeSet<SourceId>,
}

/// Outcome of the fixed-order mapping attempt (PRD §19.2).
#[derive(Debug, Clone, PartialEq)]
pub enum MappingDecision {
    /// The change localized onto the current plan: recompile exactly these
    /// pages (their updated `knowledge_refs` are in `updated_refs`), carry
    /// every other surviving page verbatim, and drop the obsolete pages.
    LocalUpdate {
        /// Pages to recompile — every candidate that stays non-empty
        /// (deletion-only emptied pages are `obsolete` instead).
        recompile: BTreeSet<WikiPageId>,
        /// Updated knowledge refs per recompiled page: previous refs minus
        /// nodes that lost support, plus the absorbed new nodes.
        updated_refs: BTreeMap<WikiPageId, Vec<KnowledgeNodeId>>,
        /// Pages left with zero refs after DELETION-only losses (§19.3.5):
        /// excluded from the new generation entirely.
        obsolete: BTreeSet<WikiPageId>,
    },
    /// The change cannot be localized (§19.2): previous generation intact.
    ReplanRequired {
        trigger: &'static str,
        reason: String,
    },
}

/// Runs the fixed-order mapping attempt over the inputs (PRD §19.2).
pub fn map_incremental_change(input: &MappingInput) -> MappingDecision {
    // ---- Fixed order, step 1: registry IDs + existing knowledge_refs. ----
    // Nodes whose support is gone: deletion-retired nodes plus nodes a
    // MODIFIED source no longer supports after re-analysis.
    let mut gone: BTreeSet<KnowledgeNodeId> = input.deletion_gone.clone();
    for (source_id, pre) in &input.modified_pre {
        let Some(post) = input.changed_post.get(source_id) else {
            continue;
        };
        let post_nodes: BTreeSet<&KnowledgeNodeId> = post.iter().map(|ns| &ns.node_id).collect();
        for node in pre {
            if !post_nodes.contains(&node.node_id) {
                gone.insert(node.node_id.clone());
            }
        }
    }

    // ---- Brand-new nodes of changed sources that need a page. A node that
    // already existed in the previous knowledge base keeps its existing page
    // assignment (deterministic Registry-ID mapping); a node that lost support
    // and came back under the same id is handled by the gone/ref logic. ----
    let mut new_nodes: Vec<(SourceId, SourceNodeSection)> = Vec::new();
    for (source_id, post) in &input.changed_post {
        for node_section in post {
            if !input.prev_kb.nodes.contains_key(&node_section.node_id)
                && !new_nodes
                    .iter()
                    .any(|(_, existing)| existing.node_id == node_section.node_id)
            {
                new_nodes.push((source_id.clone(), node_section.clone()));
            }
        }
    }

    // ---- Previous page ownership, from the published citations: a page
    // owns (source, section) when it cites a claim of that source anchored
    // in that section. First page (by id) wins — page assignment of a node is
    // unique, so ties cannot arise from well-formed generations. ----
    let mut pages_by_id: BTreeMap<&WikiPageId, &GenerationPageView> = BTreeMap::new();
    for page in &input.prev_pages {
        pages_by_id.insert(&page.page_id, page);
    }
    let mut section_owner: BTreeMap<(SourceId, SectionId), WikiPageId> = BTreeMap::new();
    let mut source_owner: BTreeMap<SourceId, WikiPageId> = BTreeMap::new();
    for page in pages_by_id.values() {
        for citation in &page.citations {
            source_owner
                .entry(citation.source_id.clone())
                .or_insert_with(|| page.page_id.clone());
            if let Some(section_id) = &citation.section_id {
                section_owner
                    .entry((citation.source_id.clone(), section_id.clone()))
                    .or_insert_with(|| page.page_id.clone());
            }
        }
    }

    // ---- Candidate pages (PRD §19.2): pages owning affected nodes (their
    // refs contain a `gone` node) plus pages that previously cited a MODIFIED
    // source (they must absorb its new nodes). Added sources have no previous
    // owner — their brand-new nodes fail the mapping below. ----
    let mut candidates: BTreeSet<WikiPageId> = BTreeSet::new();
    for page in &input.prev_pages {
        let refs_gone = page
            .knowledge_refs
            .iter()
            .any(|node_id| gone.contains(node_id));
        let cites_modified = page
            .citations
            .iter()
            .any(|citation| input.modified_ids.contains(&citation.source_id));
        if refs_gone || cites_modified {
            candidates.insert(page.page_id.clone());
        }
    }

    // ---- Absorb brand-new nodes by the source's previous section
    // ownership. A brand-new source (or a section without previous
    // ownership) has NO deterministic target: `unmappable-node`. ----
    let mut absorbed: BTreeMap<WikiPageId, Vec<KnowledgeNodeId>> = BTreeMap::new();
    for (source_id, node_section) in &new_nodes {
        let target = node_section
            .section_id
            .as_ref()
            .and_then(|section_id| section_owner.get(&(source_id.clone(), section_id.clone())))
            .or_else(|| source_owner.get(source_id));
        match target {
            Some(page_id) => {
                absorbed
                    .entry(page_id.clone())
                    .or_default()
                    .push(node_section.node_id.clone());
            }
            None => {
                return MappingDecision::ReplanRequired {
                    trigger: TRIGGER_UNMAPPABLE_NODE,
                    reason: format!(
                        "new knowledge node {} from source {source_id} has no previous page ownership; a replan must place it",
                        node_section.node_id
                    ),
                }
            }
        }
    }

    // ---- Per-candidate updated refs: drop unsupported nodes, append the
    // absorbed ones (deterministic order: previous order first, absorbed
    // sorted). ----
    let mut updated_refs: BTreeMap<WikiPageId, Vec<KnowledgeNodeId>> = BTreeMap::new();
    let mut obsolete: BTreeSet<WikiPageId> = BTreeSet::new();
    let mut recompile: BTreeSet<WikiPageId> = BTreeSet::new();
    for page in &input.prev_pages {
        if !candidates.contains(&page.page_id) {
            continue;
        }
        let mut refs: Vec<KnowledgeNodeId> = page
            .knowledge_refs
            .iter()
            .filter(|node_id| !gone.contains(node_id) && input.new_kb.nodes.contains_key(*node_id))
            .cloned()
            .collect();
        let absorbed_refs = absorbed.remove(&page.page_id).unwrap_or_default();
        refs.extend(absorbed_refs);

        if refs.is_empty() {
            // §19.3.5: a page emptied by DELETION-only losses becomes
            // obsolete. A page emptied because a MODIFIED source's knowledge
            // moved away is a structural outcome the plan must own:
            // `page-emptied` (§19.2), never a silent drop.
            let deletion_only = page.knowledge_refs.iter().all(|node_id| {
                if gone.contains(node_id) {
                    // Classify the loss: deleted-source retirement (allowed
                    // to empty a page) vs re-analysis replacement (replan).
                    input.deletion_gone.contains(node_id)
                } else {
                    // Dropped only because the node is missing from the new
                    // knowledge base entirely — an external anomaly, treated
                    // conservatively as NOT deletion-caused.
                    input.new_kb.nodes.contains_key(node_id)
                }
            });
            if deletion_only {
                obsolete.insert(page.page_id.clone());
                continue;
            }
            return MappingDecision::ReplanRequired {
                trigger: TRIGGER_PAGE_EMPTIED,
                reason: format!(
                    "page '{}' (id {}) lost all of its knowledge because a changed source no longer supports it",
                    page.title, page.page_id
                ),
            };
        }

        // ---- Cluster membership (§19.2): every surviving page keeps its
        // previous top-level source directories. ----
        let prev_dirs = top_level_dirs(&page.knowledge_refs, &input.prev_kb);
        let new_dirs = top_level_dirs(&refs, &input.new_kb);
        if !new_dirs.is_subset(&prev_dirs) {
            let intruders: Vec<String> = new_dirs.difference(&prev_dirs).cloned().collect();
            return MappingDecision::ReplanRequired {
                trigger: TRIGGER_STRUCTURAL_CHANGE,
                reason: format!(
                    "page '{}' would start mixing knowledge from source directories {intruders:?} it never covered; cluster boundaries changed",
                    page.title
                ),
            };
        }

        recompile.insert(page.page_id.clone());
        updated_refs.insert(page.page_id.clone(), refs);
    }

    // Absorption targets are always candidate pages (a page can only own a
    // source's sections if it cites that source), so nothing remains here.
    debug_assert!(
        absorbed.is_empty(),
        "absorbed nodes landed on non-candidate pages"
    );

    MappingDecision::LocalUpdate {
        recompile,
        updated_refs,
        obsolete,
    }
}

/// Top-level source directories backing `node_ids` through their claim
/// anchors (the same signal deterministic clustering uses, PRD §14).
fn top_level_dirs(node_ids: &[KnowledgeNodeId], kb: &KnowledgeBase) -> BTreeSet<String> {
    let mut dirs = BTreeSet::new();
    for node_id in node_ids {
        let Some(node) = kb.nodes.get(node_id) else {
            continue;
        };
        for anchor in &node.anchors {
            if let Some((dir, _)) = anchor.rel_path.split_once('/') {
                dirs.insert(dir.to_owned());
            } else {
                dirs.insert(anchor.rel_path.clone());
            }
        }
    }
    dirs
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_core::model::SourceRange;
    use llm_wiki_core::plan::{PlanAnchor, PlanNode};
    use llm_wiki_storage::{PageCitationRecord, PageLinkRecord};

    fn node(id: &str, kind: &str, rel_path: &str) -> (KnowledgeNodeId, PlanNode) {
        let node_id = KnowledgeNodeId::from_validated(format!("kn_{id}"));
        let anchors = if kind == "claim" {
            vec![PlanAnchor {
                source_id: SourceId::from_validated("src_anchor"),
                rel_path: rel_path.to_owned(),
                section_id: None,
                heading_path: vec![],
                range: SourceRange::new(0, 1),
                evidence_digest: "digest".into(),
                source_hash: "hash".into(),
            }]
        } else {
            Vec::new()
        };
        let plan_node = PlanNode {
            id: node_id.clone(),
            kind: kind.to_owned(),
            name: kind.to_owned(),
            entity_type: None,
            description: None,
            statement: (kind == "claim").then(|| format!("statement of {id}")),
            anchors,
        };
        (node_id, plan_node)
    }

    fn base(entries: Vec<(KnowledgeNodeId, PlanNode)>) -> KnowledgeBase {
        KnowledgeBase {
            nodes: entries.into_iter().collect(),
            relations: Vec::new(),
        }
    }

    fn claim_citation(
        claim: &KnowledgeNodeId,
        source: &str,
        section: Option<&str>,
    ) -> PageCitationRecord {
        PageCitationRecord {
            claim_node_id: claim.clone(),
            source_id: SourceId::from_validated(format!("src_{source}")),
            section_id: section.map(|s| SectionId::from_validated(format!("sec_{s}"))),
            range: SourceRange::new(0, 1),
            source_hash: "hash".into(),
            evidence_digest: "digest".into(),
            heading_path: vec![],
        }
    }

    fn page(
        id: &str,
        refs: Vec<KnowledgeNodeId>,
        citations: Vec<PageCitationRecord>,
    ) -> GenerationPageView {
        GenerationPageView {
            page_id: WikiPageId::from_validated(format!("wp_{id}")),
            slug: format!("page-{id}"),
            title: format!("Page {id}"),
            category: "concepts".into(),
            body_hash: "hash".into(),
            content: format!("# Page {id}"),
            knowledge_refs: refs,
            citations,
            links: vec![PageLinkRecord {
                to_page_id: WikiPageId::from_validated("wp_other"),
                target_title: "Other".into(),
            }],
            inbound_links: 0,
        }
    }

    fn node_section(node_id: &KnowledgeNodeId, section: Option<&str>) -> SourceNodeSection {
        SourceNodeSection {
            node_id: node_id.clone(),
            section_id: section.map(|s| SectionId::from_validated(format!("sec_{s}"))),
        }
    }

    const CLAIM_1: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
    const CLAIM_2: &str = "01BX5ZZKBKACTAV9WEVGEMMVRZ";
    const CLAIM_2_NEW: &str = "01CZZZZZZZZZZZZZZZZZZZZZZZ";
    const BRAND_NEW: &str = "01DARZ3NDEKTSV4RRFFQ69G5FAV";

    #[test]
    fn modified_source_remaps_new_claim_by_section_ownership() {
        // Previous page owns claims 1+2 of source guide (section sec_a /
        // sec_b). The modified source keeps claim 1, replaces claim 2 with a
        // brand-new claim anchored in the same section.
        let (claim1, n1) = node(CLAIM_1, "claim", "guide/a.md");
        let (claim2, n2) = node(CLAIM_2, "claim", "guide/a.md");
        let (claim2_new, n2n) = node(CLAIM_2_NEW, "claim", "guide/a.md");
        let source = SourceId::from_validated("src_guide");

        let prev_page = page(
            "p1",
            vec![claim1.clone(), claim2.clone()],
            vec![
                claim_citation(&claim1, "guide", Some("sec_a")),
                claim_citation(&claim2, "guide", Some("sec_b")),
            ],
        );
        let input = MappingInput {
            prev_pages: vec![prev_page],
            prev_kb: base(vec![(claim1.clone(), n1.clone()), (claim2.clone(), n2)]),
            new_kb: base(vec![(claim1.clone(), n1), (claim2_new.clone(), n2n)]),
            deletion_gone: BTreeSet::new(),
            modified_pre: BTreeMap::from([(
                source.clone(),
                vec![
                    node_section(&claim1, Some("sec_a")),
                    node_section(&claim2, Some("sec_b")),
                ],
            )]),
            changed_post: BTreeMap::from([(
                source.clone(),
                vec![
                    node_section(&claim1, Some("sec_a")),
                    node_section(&claim2_new, Some("sec_b")),
                ],
            )]),
            modified_ids: BTreeSet::from([source]),
        };

        let decision = map_incremental_change(&input);
        let MappingDecision::LocalUpdate {
            recompile,
            updated_refs,
            obsolete,
        } = decision
        else {
            panic!("expected a local update, got {decision:?}");
        };
        assert_eq!(obsolete, BTreeSet::new());
        assert_eq!(recompile.len(), 1);
        assert_eq!(
            updated_refs[&WikiPageId::from_validated("wp_p1")],
            vec![claim1, claim2_new]
        );
    }

    #[test]
    fn deletion_only_emptied_page_becomes_obsolete() {
        let (claim1, n1) = node(CLAIM_1, "claim", "guide/a.md");
        let (claim2, n2) = node(CLAIM_2, "claim", "gone/b.md");

        let survivor = page(
            "keep",
            vec![claim1.clone()],
            vec![claim_citation(&claim1, "guide", Some("sec_a"))],
        );
        let doomed = page(
            "drop",
            vec![claim2.clone()],
            vec![claim_citation(&claim2, "gone", Some("sec_z"))],
        );
        let input = MappingInput {
            prev_pages: vec![doomed, survivor],
            prev_kb: base(vec![(claim1.clone(), n1.clone()), (claim2.clone(), n2)]),
            new_kb: base(vec![(claim1.clone(), n1)]),
            deletion_gone: BTreeSet::from([claim2.clone()]),
            modified_pre: BTreeMap::new(),
            changed_post: BTreeMap::new(),
            modified_ids: BTreeSet::new(),
        };

        let decision = map_incremental_change(&input);
        let MappingDecision::LocalUpdate {
            recompile,
            updated_refs,
            obsolete,
        } = decision
        else {
            panic!("expected a local update, got {decision:?}");
        };
        // The emptied page is dropped; the untouched page carries over
        // without a single request (nothing about it changed).
        assert_eq!(
            obsolete,
            BTreeSet::from([WikiPageId::from_validated("wp_drop")])
        );
        assert!(recompile.is_empty());
        assert!(updated_refs.is_empty());
        let _ = claim1;
    }

    #[test]
    fn modified_source_emptied_page_requires_replan() {
        let (claim1, n1) = node(CLAIM_1, "claim", "guide/a.md");
        let source = SourceId::from_validated("src_guide");

        // The page's ONLY claim came from the modified source; re-analysis
        // supports nothing on it anymore.
        let doomed = page(
            "p1",
            vec![claim1.clone()],
            vec![claim_citation(&claim1, "guide", Some("sec_a"))],
        );
        let input = MappingInput {
            prev_pages: vec![doomed],
            prev_kb: base(vec![(claim1.clone(), n1.clone())]),
            new_kb: base(vec![]),
            deletion_gone: BTreeSet::new(),
            modified_pre: BTreeMap::from([(
                source.clone(),
                vec![node_section(&claim1, Some("sec_a"))],
            )]),
            changed_post: BTreeMap::from([(source, vec![])]),
            modified_ids: BTreeSet::from([SourceId::from_validated("src_guide")]),
        };

        let decision = map_incremental_change(&input);
        let MappingDecision::ReplanRequired { trigger, reason } = decision else {
            panic!("expected replan-required, got {decision:?}");
        };
        assert_eq!(trigger, TRIGGER_PAGE_EMPTIED);
        assert!(reason.contains("lost all"), "{reason}");
    }

    #[test]
    fn brand_new_source_nodes_are_unmappable() {
        let (fresh, nf) = node(BRAND_NEW, "claim", "new/c.md");
        let source = SourceId::from_validated("src_new");

        let untouched = page(
            "p1",
            vec![KnowledgeNodeId::from_validated(format!("kn_{CLAIM_1}"))],
            vec![],
        );
        let (claim1, n1) = node(CLAIM_1, "claim", "guide/a.md");
        let input = MappingInput {
            prev_pages: vec![untouched],
            prev_kb: base(vec![(claim1, n1)]),
            new_kb: base(vec![(fresh.clone(), nf)]),
            deletion_gone: BTreeSet::new(),
            modified_pre: BTreeMap::new(),
            changed_post: BTreeMap::from([(source.clone(), vec![node_section(&fresh, None)])]),
            modified_ids: BTreeSet::new(),
        };

        let decision = map_incremental_change(&input);
        let MappingDecision::ReplanRequired { trigger, reason } = decision else {
            panic!("expected replan-required, got {decision:?}");
        };
        assert_eq!(trigger, TRIGGER_UNMAPPABLE_NODE);
        assert!(reason.contains(&format!("kn_{BRAND_NEW}")), "{reason}");
    }

    #[test]
    fn surviving_claim_of_modified_source_stays_on_its_page() {
        // Re-analysis re-derives the SAME claim id (identical text): nothing
        // is gone, but the page still recompiles to refresh its citations.
        let (claim1, n1) = node(CLAIM_1, "claim", "guide/a.md");
        let source = SourceId::from_validated("src_guide");
        let only_page = page(
            "p1",
            vec![claim1.clone()],
            vec![claim_citation(&claim1, "guide", Some("sec_a"))],
        );
        let input = MappingInput {
            prev_pages: vec![only_page],
            prev_kb: base(vec![(claim1.clone(), n1.clone())]),
            new_kb: base(vec![(claim1.clone(), n1)]),
            deletion_gone: BTreeSet::new(),
            modified_pre: BTreeMap::from([(
                source.clone(),
                vec![node_section(&claim1, Some("sec_a"))],
            )]),
            changed_post: BTreeMap::from([(
                source.clone(),
                vec![node_section(&claim1, Some("sec_a"))],
            )]),
            modified_ids: BTreeSet::from([source]),
        };

        let decision = map_incremental_change(&input);
        let MappingDecision::LocalUpdate {
            recompile,
            updated_refs,
            obsolete,
        } = decision
        else {
            panic!("expected a local update, got {decision:?}");
        };
        assert!(obsolete.is_empty());
        assert_eq!(
            recompile,
            BTreeSet::from([WikiPageId::from_validated("wp_p1")])
        );
        assert_eq!(
            updated_refs[&WikiPageId::from_validated("wp_p1")],
            vec![claim1]
        );
    }

    #[test]
    fn cross_directory_absorption_is_a_structural_change() {
        // A brand-new node of a modified source in a DIFFERENT top-level
        // directory would move the page outside its previous cluster.
        let (claim1, n1) = node(CLAIM_1, "claim", "guide/a.md");
        let (fresh, nf) = node(BRAND_NEW, "claim", "elsewhere/c.md");
        let source = SourceId::from_validated("src_guide");

        let target = page(
            "p1",
            vec![claim1.clone()],
            vec![claim_citation(&claim1, "guide", Some("sec_a"))],
        );
        let input = MappingInput {
            prev_pages: vec![target],
            prev_kb: base(vec![(claim1.clone(), n1.clone())]),
            new_kb: base(vec![(claim1.clone(), n1), (fresh.clone(), nf)]),
            deletion_gone: BTreeSet::new(),
            modified_pre: BTreeMap::from([(
                source.clone(),
                vec![node_section(&claim1, Some("sec_a"))],
            )]),
            changed_post: BTreeMap::from([(
                source.clone(),
                vec![
                    node_section(&claim1, Some("sec_a")),
                    node_section(&fresh, Some("sec_x")),
                ],
            )]),
            modified_ids: BTreeSet::from([source]),
        };

        let decision = map_incremental_change(&input);
        let MappingDecision::ReplanRequired { trigger, reason } = decision else {
            panic!("expected replan-required, got {decision:?}");
        };
        assert_eq!(trigger, TRIGGER_STRUCTURAL_CHANGE);
        assert!(reason.contains("elsewhere"), "{reason}");
    }

    #[test]
    fn unrelated_pages_are_never_candidates() {
        let (claim1, n1) = node(CLAIM_1, "claim", "guide/a.md");
        let (other, no) = node(CLAIM_2, "claim", "other/d.md");
        let source = SourceId::from_validated("src_guide");

        let affected = page(
            "p1",
            vec![claim1.clone()],
            vec![claim_citation(&claim1, "guide", Some("sec_a"))],
        );
        let bystander = page(
            "p2",
            vec![other.clone()],
            vec![claim_citation(&other, "other", Some("sec_o"))],
        );
        // Modified source keeps its claim; the bystander must stay carried.
        let input = MappingInput {
            prev_pages: vec![affected, bystander],
            prev_kb: base(vec![
                (claim1.clone(), n1.clone()),
                (other.clone(), no.clone()),
            ]),
            new_kb: base(vec![(claim1.clone(), n1), (other, no)]),
            deletion_gone: BTreeSet::new(),
            modified_pre: BTreeMap::from([(
                source.clone(),
                vec![node_section(&claim1, Some("sec_a"))],
            )]),
            changed_post: BTreeMap::from([(
                source.clone(),
                vec![node_section(&claim1, Some("sec_a"))],
            )]),
            modified_ids: BTreeSet::from([source]),
        };

        let decision = map_incremental_change(&input);
        let MappingDecision::LocalUpdate { recompile, .. } = decision else {
            panic!("expected a local update, got {decision:?}");
        };
        assert_eq!(
            recompile,
            BTreeSet::from([WikiPageId::from_validated("wp_p1")]),
            "only the page owning the changed source's node recompiles"
        );
    }

    #[test]
    fn fingerprint_drift_uses_the_dedicated_trigger_constant() {
        // The guard itself lives in build.rs; this pins the trigger name that
        // lands in plan_decisions (§19.2: observability aggregates by reason).
        assert_eq!(
            llm_wiki_storage::TRIGGER_FINGERPRINT_CHANGED,
            "fingerprint-changed"
        );
    }
}
