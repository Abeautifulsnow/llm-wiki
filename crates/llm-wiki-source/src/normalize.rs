//! Path normalization (PRD §8.3.1, §41).
//!
//! All paths are canonicalized against the source-root boundary, stored with
//! `/` separators in Unicode NFC form. Any path that would escape the source
//! root is rejected. The raw relative path is kept only for human display;
//! locator keys always use the normalized form. Case-folded portable keys
//! detect collisions across platforms so a build never picks different files
//! on different filesystems.

use std::path::{Path, PathBuf};

use unicode_normalization::UnicodeNormalization;

use llm_wiki_core::error::{Result, WikiError};

/// Canonicalizes the source root itself. Fails when it does not exist.
pub fn canonicalize_root(root: &Path) -> Result<PathBuf> {
    root.canonicalize().map_err(|e| {
        WikiError::Source(format!(
            "cannot canonicalize source root {}: {e}",
            root.display()
        ))
    })
}

/// Normalizes one file path found under the canonical root into a stored
/// relative path (`/` separators, NFC). Rejects escapes from the root.
pub fn normalize_rel(root_canonical: &Path, file_path: &Path) -> Result<String> {
    let canonical = file_path.canonicalize().map_err(|e| {
        WikiError::Source(format!("cannot canonicalize {}: {e}", file_path.display()))
    })?;
    let rel = canonical.strip_prefix(root_canonical).map_err(|_| {
        WikiError::Source(format!("path escapes source root: {}", file_path.display()))
    })?;
    Ok(to_stored(rel))
}

/// Converts to the stored form: forward slashes + NFC.
pub fn to_stored(rel: &Path) -> String {
    let joined = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    joined.nfc().collect()
}

/// Case-folded portable key used for collision detection (PRD §8.3.1).
pub fn portable_key(stored_rel: &str) -> String {
    stored_rel
        .chars()
        .flat_map(char::to_lowercase)
        .nfc()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm-wiki-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn stores_forward_slashes_and_nfc() {
        let root = temp_root("normalize");
        let sub = root.join("a b/c.d");
        std::fs::create_dir_all(sub.parent().unwrap()).unwrap();
        std::fs::write(&sub, "x").unwrap();

        let root_canonical = canonicalize_root(&root).unwrap();
        let rel = normalize_rel(&root_canonical, &sub).unwrap();
        assert_eq!(rel, "a b/c.d");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_escape_from_root() {
        let root = temp_root("escape");
        let outside = temp_root("outside");
        std::fs::write(outside.join("x.md"), "x").unwrap();
        let root_canonical = canonicalize_root(&root).unwrap();
        assert!(normalize_rel(&root_canonical, &outside.join("x.md")).is_err());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn portable_key_folds_case_and_composes() {
        assert_eq!(portable_key("Docs/A.md"), portable_key("docs/a.md"));
        // NFC composition: e + combining acute == precomposed é.
        assert_eq!(portable_key("caf\u{e9}.md"), portable_key("cafe\u{301}.md"));
    }
}
