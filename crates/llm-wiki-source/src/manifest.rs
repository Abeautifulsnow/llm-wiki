//! Source manifest and change detection (PRD §8.2, §19.1, §43).
//!
//! The manifest is the deterministic snapshot of one scan: entries sorted by
//! normalized rel path, hashed into a snapshot hash. Diffs between two
//! manifests yield the [`ChangeSet`]; per PRD §43, V0.1 treats renames as
//! delete + add.

use serde::{Deserialize, Serialize};

use llm_wiki_core::error::Result;
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::SourceLocatorKey;
use llm_wiki_core::model::ChangeSet;

use crate::scanner::ScannedFile;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    pub rel_path: String,
    pub locator_key: SourceLocatorKey,
    pub content_hash: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct SourceManifest {
    /// Sorted by `rel_path`; do not reorder after construction.
    pub files: Vec<ManifestEntry>,
}

impl SourceManifest {
    pub fn from_scanned(files: &[ScannedFile]) -> Self {
        let mut files: Vec<ManifestEntry> = files
            .iter()
            .map(|f| ManifestEntry {
                rel_path: f.rel_path.clone(),
                locator_key: f.locator_key.clone(),
                content_hash: f.content_hash.clone(),
                size: f.size,
            })
            .collect();
        files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
        SourceManifest { files }
    }

    pub fn get(&self, locator: &SourceLocatorKey) -> Option<&ManifestEntry> {
        self.files.iter().find(|f| &f.locator_key == locator)
    }

    /// Hash over the full manifest: the `source_snapshot_hash` input of the
    /// BuildFingerprint (PRD §18.1).
    pub fn snapshot_hash(&self) -> String {
        let mut buf = String::new();
        for f in &self.files {
            buf.push_str(&f.rel_path);
            buf.push('\u{1f}');
            buf.push_str(f.locator_key.as_str());
            buf.push('\u{1f}');
            buf.push_str(&f.content_hash);
            buf.push('\u{1f}');
            buf.push_str(&f.size.to_string());
            buf.push('\n');
        }
        sha256_hex(buf.as_bytes())
    }

    pub fn diff(&self, current: &SourceManifest) -> ChangeSet {
        let mut change = ChangeSet::default();
        let prev: std::collections::HashMap<&str, &ManifestEntry> = self
            .files
            .iter()
            .map(|f| (f.rel_path.as_str(), f))
            .collect();
        let cur: std::collections::HashMap<&str, &ManifestEntry> = current
            .files
            .iter()
            .map(|f| (f.rel_path.as_str(), f))
            .collect();

        for f in &current.files {
            match prev.get(f.rel_path.as_str()) {
                None => change.added.push(f.locator_key.clone()),
                Some(p) => {
                    if p.content_hash == f.content_hash {
                        change.unchanged.push(f.locator_key.clone());
                    } else {
                        change.modified.push(f.locator_key.clone());
                    }
                }
            }
        }
        for f in &self.files {
            if !cur.contains_key(f.rel_path.as_str()) {
                change.deleted.push(f.locator_key.clone());
            }
        }
        change
    }
}

/// Serializes a manifest canonically (sorted) for storage.
pub fn to_json(manifest: &SourceManifest) -> Result<String> {
    serde_json::to_string(manifest)
        .map_err(|e| llm_wiki_core::error::WikiError::Storage(format!("manifest serialize: {e}")))
}

pub fn from_json(text: &str) -> Result<SourceManifest> {
    serde_json::from_str(text)
        .map_err(|e| llm_wiki_core::error::WikiError::Storage(format!("manifest parse: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn scanned(rel: &str, content: &[u8]) -> ScannedFile {
        ScannedFile {
            rel_path: rel.to_owned(),
            locator_key: SourceLocatorKey::compute("ws", rel),
            content_hash: sha256_hex(content),
            size: content.len() as u64,
            modified_at: Some(Utc::now()),
        }
    }

    #[test]
    fn identical_manifests_have_identical_snapshot_hashes() {
        let a = SourceManifest::from_scanned(&[scanned("a.md", b"1"), scanned("b/c.md", b"2")]);
        let b = SourceManifest::from_scanned(&[scanned("b/c.md", b"2"), scanned("a.md", b"1")]);
        assert_eq!(a.snapshot_hash(), b.snapshot_hash());
    }

    #[test]
    fn diff_reports_add_modify_delete_unchanged() {
        let prev = SourceManifest::from_scanned(&[
            scanned("keep.md", b"same"),
            scanned("changed.md", b"old"),
            scanned("gone.md", b"bye"),
        ]);
        let cur = SourceManifest::from_scanned(&[
            scanned("keep.md", b"same"),
            scanned("changed.md", b"new"),
            scanned("new.md", b"hi"),
        ]);
        let cs = prev.diff(&cur);
        assert_eq!(cs.added.len(), 1);
        assert_eq!(cs.modified.len(), 1);
        assert_eq!(cs.deleted.len(), 1);
        assert_eq!(cs.unchanged.len(), 1);
        assert!(!cs.is_empty());
    }

    #[test]
    fn rename_is_delete_plus_add_in_v01() {
        // PRD §43: V0.1 does not auto-detect renames.
        let prev = SourceManifest::from_scanned(&[scanned("old.md", b"same bytes")]);
        let cur = SourceManifest::from_scanned(&[scanned("new.md", b"same bytes")]);
        let cs = prev.diff(&cur);
        assert_eq!(cs.deleted.len(), 1);
        assert_eq!(cs.added.len(), 1);
        assert_eq!(cs.unchanged.len(), 0);
    }

    #[test]
    fn json_roundtrip() {
        let m = SourceManifest::from_scanned(&[scanned("a.md", b"x")]);
        let text = to_json(&m).unwrap();
        let back = from_json(&text).unwrap();
        assert_eq!(m, back);
    }
}
