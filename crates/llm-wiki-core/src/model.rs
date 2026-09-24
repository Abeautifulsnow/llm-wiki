//! Core domain models (PRD §7.1, §8.2, §9, §12, §14.1, §19.1).
//!
//! These structs are the shared vocabulary between crates. Persistence and
//! transport concerns live elsewhere; everything here is plain data.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{BuildId, CitationId, KnowledgeNodeId, SectionId, SourceId, WikiPageId};

/// Byte range into the analyzed document text (PRD §9 `source_range`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRange {
    pub start: usize,
    pub end: usize,
}

impl SourceRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// A source file as recorded by the scanner (PRD §8.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDocument {
    pub source_id: SourceId,
    pub locator_key: crate::ids::SourceLocatorKey,
    /// Normalized relative path (`/` separators, NFC), for humans.
    pub rel_path: String,
    pub content_hash: String,
    pub size: u64,
    pub modified_at: Option<DateTime<Utc>>,
    /// BCP-47-ish language tag (`und` when undetermined, PRD §9).
    pub language: String,
}

/// A heading-delimited document section (PRD §9 `DocumentSection`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    /// Assigned by the Section Registry; `None` until persisted.
    pub id: Option<SectionId>,
    pub heading: Option<String>,
    pub heading_level: u8,
    /// Full heading path, e.g. `["Plugin System", "Architecture", "Runtime"]`.
    pub heading_path: Vec<String>,
    pub content: String,
    pub source_range: Option<SourceRange>,
}

/// Stable knowledge node (PRD §12.1). Identity lives in the Knowledge
/// Registry; this is the resolved view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: KnowledgeNodeId,
    pub canonical_name: String,
    pub entity_type: String,
    pub aliases: Vec<String>,
    pub description: Option<String>,
}

/// Citation binding a claim/page to source evidence (PRD §12.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub id: CitationId,
    pub source_id: SourceId,
    pub section_id: Option<SectionId>,
    pub source_range: Option<SourceRange>,
    pub source_hash: String,
    pub evidence_digest: Option<String>,
    pub heading_path: Vec<String>,
}

/// A falsifiable statement anchored to evidence (PRD §11.1/§12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub statement: String,
    pub citations: Vec<Citation>,
}

/// Directed relation between knowledge nodes (PRD §12.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub source: KnowledgeNodeId,
    pub relation_type: String,
    pub target: KnowledgeNodeId,
    pub citations: Vec<CitationId>,
}

/// One planned wiki page (PRD §14.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiPagePlan {
    pub id: WikiPageId,
    pub slug: String,
    pub title: String,
    pub category: String,
    pub purpose: String,
    pub knowledge_refs: Vec<KnowledgeNodeId>,
    pub source_refs: Vec<SourceId>,
    pub related_pages: Vec<WikiPageId>,
}

/// The global plan (PRD §14.1).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WikiPlan {
    pub pages: Vec<WikiPagePlan>,
}

/// A compiled wiki page (PRD §15).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiPage {
    pub id: WikiPageId,
    pub slug: String,
    pub title: String,
    pub category: String,
    pub body: String,
    pub source_ids: Vec<SourceId>,
    pub build_id: BuildId,
}

/// Result of comparing two source manifests (PRD §19.1).
///
/// Keyed by [`crate::ids::SourceLocatorKey`]: the diff runs before the Source
/// Registry maps locators to persistent `SourceId`s. Per PRD §43, V0.1 treats
/// renames as delete + add.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChangeSet {
    pub added: Vec<crate::ids::SourceLocatorKey>,
    pub modified: Vec<crate::ids::SourceLocatorKey>,
    pub deleted: Vec<crate::ids::SourceLocatorKey>,
    pub unchanged: Vec<crate::ids::SourceLocatorKey>,
}

impl ChangeSet {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.deleted.is_empty()
    }
}

/// Inputs of the build fingerprint (PRD §18.1/§44): everything that must be
/// identical for a rebuild to be a pure cache hit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildFingerprint {
    pub source_snapshot_hash: String,
    pub provider: String,
    pub model: String,
    pub prompt_version: String,
    pub schema_version: String,
    pub parser_version: String,
    pub compiler_version: String,
    pub entity_merge_rules_version: String,
    pub config_hash: String,
}

impl BuildFingerprint {
    /// Canonical, order-stable hash over all fingerprint inputs.
    pub fn fingerprint_hash(&self) -> String {
        // BTreeMap gives a canonical field order for hashing.
        let map: BTreeMap<&str, &str> = BTreeMap::from([
            ("source_snapshot_hash", self.source_snapshot_hash.as_str()),
            ("provider", self.provider.as_str()),
            ("model", self.model.as_str()),
            ("prompt_version", self.prompt_version.as_str()),
            ("schema_version", self.schema_version.as_str()),
            ("parser_version", self.parser_version.as_str()),
            ("compiler_version", self.compiler_version.as_str()),
            (
                "entity_merge_rules_version",
                self.entity_merge_rules_version.as_str(),
            ),
            ("config_hash", self.config_hash.as_str()),
        ]);
        let json = serde_json::to_string(&map).unwrap_or_default();
        crate::hash::sha256_hex(json.as_bytes())
    }
}

/// A retrieval hit (V0.2, PRD §20). Defined here so downstream crates share
/// the shape from day one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub page_slug: String,
    pub heading_path: Vec<String>,
    pub score: f32,
    pub snippet: String,
}

/// Context pack handed to an answering LLM (V0.5, PRD §23).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeContext {
    pub pages: Vec<WikiPage>,
    pub citations: Vec<Citation>,
    pub token_estimate: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_hash_is_stable_across_field_order() {
        let base = BuildFingerprint {
            source_snapshot_hash: "aaa".into(),
            provider: "openai-compatible".into(),
            model: "m1".into(),
            prompt_version: "p1".into(),
            schema_version: "s1".into(),
            parser_version: "parser1".into(),
            compiler_version: "c1".into(),
            entity_merge_rules_version: "e1".into(),
            config_hash: "cfg".into(),
        };
        let mut flipped = base.clone();
        flipped.model = "m1".to_owned();
        assert_eq!(base.fingerprint_hash(), flipped.fingerprint_hash());

        let mut changed = base.clone();
        changed.model = "m2".to_owned();
        assert_ne!(base.fingerprint_hash(), changed.fingerprint_hash());
    }

    #[test]
    fn changeset_empty_detection() {
        let mut cs = ChangeSet::default();
        assert!(cs.is_empty());
        cs.added
            .push(crate::ids::SourceLocatorKey::compute("ws", "a.md"));
        assert!(!cs.is_empty());
    }
}
