//! Planning-domain inputs and deterministic clustering (PRD §14).
//!
//! The planner never sees "all knowledge nodes in one prompt". This module
//! holds the pure, LLM-free half of that contract:
//!
//! - [`KnowledgeBase`] — the planner/compiler view of active knowledge
//!   (registry nodes, active relations, claim citation anchors).
//! - [`cluster_knowledge`] — deterministic clustering (union-find over
//!   relations + claim source-directory groups) with budget-driven
//!   subdivision: over-budget clusters are *split*, never truncated (PRD §14).
//! - Layer cache keys (PRD §14): cluster summary key from the sorted
//!   `node id + content hash` set; local plan key from the summary hash,
//!   planner version and config; reconciliation key from the sorted local
//!   plan hashes, registry revision and config.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::hash::sha256_hex;
use crate::ids::{KnowledgeNodeId, SectionId, SourceId};
use crate::model::SourceRange;

/// One grounded citation anchor behind a claim node (PRD §12.3): where the
/// evidence lives and how to re-verify it. Computed by application code —
/// never taken from LLM output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanAnchor {
    pub source_id: SourceId,
    /// Normalized source-relative path (for frontmatter `sources:`).
    pub rel_path: String,
    pub section_id: Option<SectionId>,
    pub heading_path: Vec<String>,
    pub range: SourceRange,
    pub evidence_digest: String,
    pub source_hash: String,
}

/// A knowledge node as consumed by planning and compilation (PRD §12/§14).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    pub id: KnowledgeNodeId,
    /// `entity | concept | topic | claim`.
    pub kind: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Claim statement text (claim nodes only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub statement: Option<String>,
    /// Grounding anchors (claim nodes only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<PlanAnchor>,
}

/// An active relation between two nodes (PRD §12.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRelation {
    pub source: KnowledgeNodeId,
    pub relation_type: String,
    pub target: KnowledgeNodeId,
}

/// The knowledge view the planner and compiler operate on. Ordering is
/// canonical (BTreeMap + sorted relations) so every derived key is stable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeBase {
    pub nodes: BTreeMap<KnowledgeNodeId, PlanNode>,
    pub relations: Vec<PlanRelation>,
}

impl KnowledgeBase {
    pub fn get(&self, id: &KnowledgeNodeId) -> Option<&PlanNode> {
        self.nodes.get(id)
    }

    /// Content hash of one node: everything the planner/summary consumes.
    /// Claim anchors participate (evidence moves → content changed).
    pub fn node_content_hash(&self, id: &KnowledgeNodeId) -> String {
        let node = &self.nodes[id];
        let mut material = format!("kind={}\nname={}\n", node.kind, node.name);
        if let Some(entity_type) = &node.entity_type {
            material.push_str(&format!("type={entity_type}\n"));
        }
        if let Some(description) = &node.description {
            material.push_str(&format!("desc={description}\n"));
        }
        if let Some(statement) = &node.statement {
            material.push_str(&format!("stmt={statement}\n"));
        }
        for anchor in &node.anchors {
            material.push_str(&format!(
                "anchor={}\u{1f}{}:{}-{}:{}\n",
                anchor.rel_path,
                anchor.range.start,
                anchor.range.end,
                anchor.section_id.as_ref().map(|s| s.as_str()).unwrap_or(""),
                anchor.evidence_digest
            ));
        }
        sha256_hex(material.as_bytes())
    }

    /// Estimated prompt tokens of a payload built from the given nodes
    /// (`chars / 4`, same estimator as §10 segmentation).
    pub fn estimate_payload_tokens(&self, ids: &[KnowledgeNodeId]) -> u64 {
        let mut material = String::new();
        for id in ids {
            let node = &self.nodes[id];
            material.push_str(&node.name);
            if let Some(description) = &node.description {
                material.push_str(description);
            }
            if let Some(statement) = &node.statement {
                material.push_str(statement);
            }
            for anchor in &node.anchors {
                material.push_str(&anchor.heading_path.join(" "));
            }
        }
        crate::plan::estimate_tokens(&material)
    }
}

/// Script-aware text→token estimate, shared with the markdown crate's
/// segmentation (audit FIX-017: chars/4 systematically undercounts CJK —
/// Chinese runs cost roughly 1 token per 1.5 characters even on modern
/// tokenizers, against chars/4's one-per-four). CJK-wide characters weigh a
/// rounded-up 2/3 token each; every other character keeps the chars/4 Latin
/// rule. Both terms round UP: overestimating a payload fails closed early,
/// underestimating it fails at the provider mid-build.
pub fn estimate_tokens(text: &str) -> u64 {
    let mut wide = 0u64;
    let mut narrow = 0u64;
    for c in text.chars() {
        if is_cjk_wide(c) {
            wide += 1;
        } else {
            narrow += 1;
        }
    }
    (wide * 2).div_ceil(3) + narrow.div_ceil(4) + u64::from(text.is_empty())
}

/// CJK-wide scripts that tokenizers charge about one token per character or
/// worse: Han (incl. extensions and compatibility forms), kana, hangul, CJK
/// punctuation and fullwidth forms.
fn is_cjk_wide(c: char) -> bool {
    matches!(
        c as u32,
        0x1100..=0x11FF // Hangul Jamo
            | 0x3000..=0x303F // CJK symbols and punctuation
            | 0x3040..=0x30FF // Hiragana + Katakana
            | 0x3130..=0x318F // Hangul Compatibility Jamo
            | 0x3400..=0x4DBF // CJK Extension A
            | 0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0xAC00..=0xD7AF // Hangul Syllables
            | 0xF900..=0xFAFF // CJK Compatibility Ideographs
            | 0xFF00..=0xFFEF // Fullwidth and halfwidth forms
    )
}

/// A cluster of knowledge nodes handed to one summary/local-plan request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    /// Sorted node ids (stable order inside the payload).
    pub nodes: Vec<KnowledgeNodeId>,
}

/// Deterministic clustering (PRD §14): union-find over active relations,
/// claims grouped by top-level source directory of their anchors. Clusters
/// exceeding `max_cluster_nodes` or the token budget are subdivided —
/// preferring source-directory locality, falling back to the sorted midpoint
/// — never truncated.
pub fn cluster_knowledge(
    base: &KnowledgeBase,
    max_cluster_nodes: usize,
    max_payload_tokens: u64,
) -> Vec<Cluster> {
    let max_cluster_nodes = max_cluster_nodes.max(1);
    let sorted_ids: Vec<KnowledgeNodeId> = base.nodes.keys().cloned().collect();
    let index_of: BTreeMap<&KnowledgeNodeId, usize> = sorted_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();
    // Union-find over indices; the smaller index always becomes the root so
    // component extraction is deterministic.
    let mut parent: Vec<usize> = (0..sorted_ids.len()).collect();

    fn find(parent: &mut [usize], mut current: usize) -> usize {
        while parent[current] != current {
            parent[current] = parent[parent[current]];
            current = parent[current];
        }
        current
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let root_a = find(parent, a);
        let root_b = find(parent, b);
        if root_a != root_b {
            let (keep, attach) = if root_a < root_b {
                (root_a, root_b)
            } else {
                (root_b, root_a)
            };
            parent[attach] = keep;
        }
    }

    for relation in &base.relations {
        if let (Some(&a), Some(&b)) = (
            index_of.get(&relation.source),
            index_of.get(&relation.target),
        ) {
            union(&mut parent, a, b);
        }
    }

    // Claims sharing the top-level source directory of their anchors are
    // clustered together (deterministic "source 目录" signal, PRD §14).
    let mut dir_of_claim: BTreeMap<usize, String> = BTreeMap::new();
    for (index, id) in sorted_ids.iter().enumerate() {
        if base.nodes[id].kind != "claim" {
            continue;
        }
        let mut dirs: Vec<String> = base.nodes[id]
            .anchors
            .iter()
            .map(|anchor| top_level_dir(&anchor.rel_path))
            .collect();
        dirs.sort();
        dirs.dedup();
        if !dirs.is_empty() {
            dir_of_claim.insert(index, dirs.join("\u{1f}"));
        }
    }
    let mut by_dir: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, dir) in &dir_of_claim {
        by_dir.entry(dir.clone()).or_default().push(*index);
    }
    for group in by_dir.into_values() {
        for window in group.windows(2) {
            union(&mut parent, window[0], window[1]);
        }
    }

    let mut components: BTreeMap<usize, Vec<KnowledgeNodeId>> = BTreeMap::new();
    for (index, id) in sorted_ids.iter().enumerate() {
        components
            .entry(find(&mut parent, index))
            .or_default()
            .push(id.clone());
    }

    let mut clusters: Vec<Cluster> = Vec::new();
    for (_, members) in components {
        clusters.extend(subdivide(
            members,
            base,
            max_cluster_nodes,
            max_payload_tokens,
        ));
    }
    clusters.sort_by(|a, b| a.nodes.cmp(&b.nodes));
    clusters
}

/// Splits `members` (sorted) until every cluster fits the node-count and
/// token budgets. Splitting prefers semantic locality (audit FIX-016
/// Level 1): members are first grouped by the top-level source directory of
/// their anchors, so a source-directory community is never cut just because
/// it straddles the ID midpoint. A group that still exceeds the budgets (or
/// an anchor-less / single-directory member set where grouping makes no
/// progress) falls back to the deterministic sorted-midpoint split.
fn subdivide(
    members: Vec<KnowledgeNodeId>,
    base: &KnowledgeBase,
    max_cluster_nodes: usize,
    max_payload_tokens: u64,
) -> Vec<Cluster> {
    let tokens = base.estimate_payload_tokens(&members);
    if members.len() <= max_cluster_nodes && tokens <= max_payload_tokens {
        return vec![Cluster { nodes: members }];
    }
    if members.len() < 2 {
        // A single oversized node cannot be subdivided further; it is handed
        // to the stage alone (the stage fails if it exceeds its own budget).
        return vec![Cluster { nodes: members }];
    }
    // Semantic grouping first. Nodes anchored in several directories keep
    // their joined key (they stay with their own multi-directory group
    // instead of being torn apart); anchor-less nodes form their own group.
    let mut by_dir: BTreeMap<String, Vec<KnowledgeNodeId>> = BTreeMap::new();
    for id in &members {
        let mut dirs: Vec<String> = base.nodes[id]
            .anchors
            .iter()
            .map(|anchor| top_level_dir(&anchor.rel_path))
            .collect();
        dirs.sort();
        dirs.dedup();
        let key = if dirs.is_empty() {
            String::new()
        } else {
            dirs.join("\u{1f}")
        };
        by_dir.entry(key).or_default().push(id.clone());
    }
    if by_dir.len() >= 2 {
        // Groups are built in member order (sorted) per directory key, so
        // each group's internal order stays sorted and deterministic.
        let mut out = Vec::new();
        for (_, group) in by_dir {
            out.extend(subdivide(
                group,
                base,
                max_cluster_nodes,
                max_payload_tokens,
            ));
        }
        return out;
    }
    let mid = members.len().div_ceil(2);
    let mut out = subdivide(
        members[..mid].to_vec(),
        base,
        max_cluster_nodes,
        max_payload_tokens,
    );
    out.extend(subdivide(
        members[mid..].to_vec(),
        base,
        max_cluster_nodes,
        max_payload_tokens,
    ));
    out
}

/// First path segment for `a/b/c.md` → `a`; a root-level file groups by its
/// own name.
fn top_level_dir(rel_path: &str) -> String {
    match rel_path.split_once('/') {
        Some((dir, _)) => dir.to_owned(),
        None => rel_path.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Layer cache keys (PRD §14)
// ---------------------------------------------------------------------------

/// Cluster summary key: sorted `node_id:content_hash` set.
pub fn cluster_summary_key(base: &KnowledgeBase, cluster: &Cluster) -> String {
    let mut lines: Vec<String> = cluster
        .nodes
        .iter()
        .map(|id| format!("{}:{}", id.as_str(), base.node_content_hash(id)))
        .collect();
    lines.sort();
    sha256_hex(lines.join("\n").as_bytes())
}

/// Local plan key: cluster summary hash + planner version + config.
pub fn local_plan_key(summary_key: &str, planner_version: &str, config_tag: &str) -> String {
    sha256_hex(format!("{summary_key}\u{1f}{planner_version}\u{1f}{config_tag}").as_bytes())
}

/// Reconciliation key: sorted local plan hashes + registry revision + config.
pub fn reconciliation_key(
    local_keys: &[String],
    registry_revision: u64,
    config_tag: &str,
) -> String {
    let mut keys = local_keys.to_vec();
    keys.sort();
    sha256_hex(
        format!(
            "{}\u{1f}{registry_revision}\u{1f}{config_tag}",
            keys.join("\n")
        )
        .as_bytes(),
    )
}

/// Canonical config tag baked into every layer key.
pub fn planning_config_tag(hierarchical: bool, max_cluster_nodes: usize) -> String {
    format!("hier={hierarchical}|max_cluster={max_cluster_nodes}")
}

/// Slug for a page title: NFC, lowercased, runs of non-alphanumeric folded to
/// `-`. CJK characters count as alphanumeric and are preserved.
pub fn slugify(title: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let normalized: String = title.nfc().collect();
    let mut slug = String::new();
    let mut last_dash = true; // trim leading dashes
    for ch in normalized.chars() {
        if ch.is_alphanumeric() {
            slug.extend(ch.to_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    slug
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, kind: &str, name: &str) -> PlanNode {
        PlanNode {
            id: KnowledgeNodeId::parse(id).unwrap(),
            kind: kind.to_owned(),
            name: name.to_owned(),
            entity_type: None,
            description: Some(format!("{name} description")),
            statement: None,
            anchors: Vec::new(),
        }
    }

    fn base_with(nodes: Vec<PlanNode>, relations: Vec<PlanRelation>) -> KnowledgeBase {
        let mut map = BTreeMap::new();
        for n in nodes {
            map.insert(n.id.clone(), n);
        }
        KnowledgeBase {
            nodes: map,
            relations,
        }
    }

    const A: &str = "kn_01ARZ3NDEKTSV4RRFFQ69G5FAV";
    const B: &str = "kn_01BX5ZZKBKACTAV9WEVGEMMVRZ";
    const C: &str = "kn_01CZZZZZZZZZZZZZZZZZZZZZZZ";
    const D: &str = "kn_01DZZZZZZZZZZZZZZZZZZZZZZZ";

    #[test]
    fn token_estimate_is_script_aware() {
        // 100 Han chars: ~67 tokens (was 25 under chars/4) — audit FIX-017.
        assert_eq!(estimate_tokens(&"检".repeat(100)), 67);
        // Latin rule unchanged: 400 chars → 100 tokens.
        assert_eq!(estimate_tokens(&"a".repeat(400)), 100);
        // Mixed scripts: both ceiling terms add up (2 + 2).
        assert_eq!(
            estimate_tokens(&format!("{}{}", "检".repeat(3), "a".repeat(8))),
            4
        );
        // Empty text still estimates 1 (rounding floor).
        assert_eq!(estimate_tokens(""), 1);
        // Fullwidth forms and CJK punctuation count as wide: 5 wide chars
        // → ceil(10/3) = 4 tokens.
        assert_eq!(estimate_tokens("ＳＳＯ，。"), 4);
    }

    #[test]
    fn relations_join_nodes_into_one_cluster() {
        let base = base_with(
            vec![node(A, "entity", "Runtime"), node(B, "concept", "Delivery")],
            vec![PlanRelation {
                source: KnowledgeNodeId::parse(A).unwrap(),
                relation_type: "uses".into(),
                target: KnowledgeNodeId::parse(B).unwrap(),
            }],
        );
        let clusters = cluster_knowledge(&base, 10, 100_000);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].nodes.len(), 2);
    }

    #[test]
    fn unconnected_nodes_stay_singletons() {
        let base = base_with(
            vec![
                node(A, "entity", "Runtime"),
                node(B, "entity", "Registry"),
                node(C, "concept", "Delivery"),
            ],
            vec![],
        );
        let clusters = cluster_knowledge(&base, 10, 100_000);
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn oversized_clusters_subdivide_never_truncate() {
        let base = base_with(
            vec![
                node(A, "entity", "Runtime"),
                node(B, "entity", "Registry"),
                node(C, "entity", "Bus"),
            ],
            vec![
                PlanRelation {
                    source: KnowledgeNodeId::parse(A).unwrap(),
                    relation_type: "uses".into(),
                    target: KnowledgeNodeId::parse(B).unwrap(),
                },
                PlanRelation {
                    source: KnowledgeNodeId::parse(B).unwrap(),
                    relation_type: "uses".into(),
                    target: KnowledgeNodeId::parse(C).unwrap(),
                },
            ],
        );
        let clusters = cluster_knowledge(&base, 2, 100_000);
        let total: usize = clusters.iter().map(|c| c.nodes.len()).sum();
        assert_eq!(total, 3, "subdivision keeps every node");
        assert!(clusters.iter().all(|c| c.nodes.len() <= 2));
    }

    #[test]
    fn oversized_clusters_split_by_source_directory_first() {
        // Audit FIX-016 Level 1: an oversized cluster whose members span two
        // top-level source directories splits along directory lines, never
        // across them — even when the ID-sorted midpoint would.
        let anchor_at = |path: &str| PlanAnchor {
            source_id: SourceId::generate(),
            rel_path: path.to_owned(),
            section_id: None,
            heading_path: Vec::new(),
            range: SourceRange { start: 0, end: 10 },
            evidence_digest: "d".into(),
            source_hash: "h".into(),
        };
        // Node ids INTERLEAVE the two directories (A=plugin, B=guides,
        // C=plugin, D=guides), so the ID-sorted midpoint split would cut
        // across both directory communities — the semantic grouping must
        // not.
        let mut a = node(A, "claim", "a");
        a.anchors = vec![anchor_at("plugin/arch.md")];
        let mut b = node(B, "claim", "b");
        b.anchors = vec![anchor_at("guides/intro.md")];
        let mut c = node(C, "claim", "c");
        c.anchors = vec![anchor_at("plugin/security.md")];
        let mut d = node(D, "claim", "d");
        d.anchors = vec![anchor_at("guides/advanced.md")];
        let base = base_with(vec![a, b, c, d], vec![]);

        // max_cluster_nodes = 2: the four-node cluster is over budget and
        // must first split along plugin/ vs guides/, then stay intact.
        let clusters = cluster_knowledge(&base, 2, 1_000_000);
        assert_eq!(clusters.len(), 2, "{clusters:?}");
        for cluster in &clusters {
            let dirs: Vec<&str> = cluster
                .nodes
                .iter()
                .map(|node| match node.as_str() {
                    A | C => "plugin",
                    _ => "guides",
                })
                .collect();
            assert!(
                dirs.windows(2).all(|window| window[0] == window[1]),
                "a split crossed a directory boundary: {clusters:?}"
            );
        }
    }

    #[test]
    fn content_hash_tracks_node_payload_changes() {
        let mut base = base_with(vec![node(A, "claim", "claim a")], vec![]);
        let first = base.node_content_hash(&KnowledgeNodeId::parse(A).unwrap());
        base.nodes
            .get_mut(&KnowledgeNodeId::parse(A).unwrap())
            .unwrap()
            .statement = Some("changed".into());
        let second = base.node_content_hash(&KnowledgeNodeId::parse(A).unwrap());
        assert_ne!(first, second);
    }

    #[test]
    fn layer_keys_are_stable_and_input_sensitive() {
        let base = base_with(
            vec![node(A, "entity", "Runtime"), node(B, "entity", "Bus")],
            vec![],
        );
        let clusters = cluster_knowledge(&base, 10, 100_000);
        let summary = cluster_summary_key(&base, &clusters[0]);
        let config = planning_config_tag(true, 24);
        let local = local_plan_key(&summary, "wiki-planning@1", &config);
        let reconcile = reconciliation_key(std::slice::from_ref(&local), 7, &config);

        assert_eq!(summary, cluster_summary_key(&base, &clusters[0]));
        assert_eq!(local, local_plan_key(&summary, "wiki-planning@1", &config));
        assert_eq!(
            reconcile,
            reconciliation_key(std::slice::from_ref(&local), 7, &config)
        );
        assert_ne!(
            reconcile,
            reconciliation_key(std::slice::from_ref(&local), 8, &config)
        );

        let mut changed = base.clone();
        changed
            .nodes
            .get_mut(&KnowledgeNodeId::parse(A).unwrap())
            .unwrap()
            .description = Some("new".into());
        let changed_clusters = cluster_knowledge(&changed, 10, 100_000);
        assert_ne!(summary, cluster_summary_key(&changed, &changed_clusters[0]));
    }

    #[test]
    fn slugify_folds_case_and_punct_keeps_cjk() {
        assert_eq!(slugify("Plugin System!"), "plugin-system");
        assert_eq!(slugify("  消息总线  设计 "), "消息总线-设计");
        assert_eq!(slugify("---"), "");
    }

    #[test]
    fn claims_group_by_source_top_dir() {
        let mut claim_a = node(A, "claim", "claim a");
        let mut claim_b = node(B, "claim", "claim b");
        let anchor_of = |rel: &str| PlanAnchor {
            source_id: SourceId::generate(),
            rel_path: rel.to_owned(),
            section_id: None,
            heading_path: vec![],
            range: SourceRange::new(0, 1),
            evidence_digest: "d".into(),
            source_hash: "h".into(),
        };
        claim_a.anchors = vec![anchor_of("plugin/security.md")];
        claim_b.anchors = vec![anchor_of("plugin/architecture.md")];
        let claim_c = node(C, "claim", "claim c");
        let mut standalone = claim_c;
        standalone.anchors = vec![anchor_of("overview.md")];

        let base = base_with(vec![claim_a, claim_b, standalone], vec![]);
        let clusters = cluster_knowledge(&base, 10, 100_000);
        assert_eq!(
            clusters.len(),
            2,
            "plugin/* claims merge, overview stays alone"
        );
    }
}
