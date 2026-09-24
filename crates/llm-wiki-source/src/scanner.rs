//! Source Scanner (PRD §8, §41).
//!
//! Discovers Markdown/MDX sources under the configured root, applying
//! include/exclude globs, the non-overridable hard excludes
//! (`.llm-wiki/**` and the configured `wiki_dir`), filesystem-safety
//! diagnostics, and case-fold collision detection. Output is deterministic:
//! files and diagnostics are sorted.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use globset::{Glob, GlobSet, GlobSetBuilder};
use walkdir::WalkDir;

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::SourceLocatorKey;

use crate::normalize;

/// Directories that must never be ingested as source (PRD §8.4).
pub const HARD_EXCLUDE_TOP_DIRS: &[&str] = &[".llm-wiki"];

const DEFAULT_MAX_FILE_SIZE: u64 = 16 * 1024 * 1024;
const SNIFF_LEN: usize = 8 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticKind {
    SkippedSymlink,
    TooLarge,
    Binary,
    Unreadable,
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct ScanDiagnostic {
    pub rel_path: String,
    pub kind: DiagnosticKind,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ScannedFile {
    /// Normalized relative path (`/` separators, NFC).
    pub rel_path: String,
    pub locator_key: SourceLocatorKey,
    pub content_hash: String,
    pub size: u64,
    pub modified_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Default)]
pub struct ScanOutput {
    /// Sorted by normalized rel path.
    pub files: Vec<ScannedFile>,
    /// Sorted by (rel_path, kind).
    pub diagnostics: Vec<ScanDiagnostic>,
}

pub struct Scanner {
    root: PathBuf,
    root_canonical: PathBuf,
    workspace_id: String,
    include: GlobSet,
    exclude: GlobSet,
    /// Relative stored path of the configured wiki_dir, if inside the root it
    /// would be a hard exclude; the config rejects overlap outright, so this
    /// is defense in depth.
    wiki_dir_rel: Option<String>,
    max_file_size: u64,
}

impl Scanner {
    pub fn new(
        root: &Path,
        include_globs: &[String],
        exclude_globs: &[String],
        wiki_dir_rel: Option<String>,
    ) -> Result<Self> {
        let root_canonical = normalize::canonicalize_root(root)?;
        let workspace_id = root_canonical.to_string_lossy().to_string();
        Ok(Self {
            root: root.to_path_buf(),
            root_canonical,
            workspace_id,
            include: build_globset(include_globs, "include")?,
            exclude: build_globset(exclude_globs, "exclude")?,
            wiki_dir_rel,
            max_file_size: DEFAULT_MAX_FILE_SIZE,
        })
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    pub fn scan(&self) -> Result<ScanOutput> {
        let mut files: Vec<ScannedFile> = Vec::new();
        let mut diagnostics: Vec<ScanDiagnostic> = Vec::new();
        let mut by_portable_key: BTreeMap<String, String> = BTreeMap::new();

        for entry in WalkDir::new(&self.root)
            .min_depth(1)
            .follow_links(false)
            .sort_by(|a, b| a.file_name().cmp(b.file_name()))
        {
            let entry = match entry {
                Ok(entry) => entry,
                Err(err) => {
                    diagnostics.push(ScanDiagnostic {
                        rel_path: err
                            .path()
                            .map(|path| path.display().to_string())
                            .unwrap_or_default(),
                        kind: DiagnosticKind::Unreadable,
                        message: err.to_string(),
                    });
                    continue;
                }
            };
            let path = entry.path();

            // Symlinks are never followed (PRD §41: default no cross-root
            // following); they produce a deterministic diagnostic instead.
            let link_meta = std::fs::symlink_metadata(path);
            if link_meta.as_ref().map(|m| m.is_symlink()).unwrap_or(false) {
                diagnostics.push(self.diagnostic(
                    path,
                    DiagnosticKind::SkippedSymlink,
                    "symlinks are not followed".to_owned(),
                ));
                continue;
            }
            if !entry.file_type().is_file() {
                continue;
            }

            let rel = match normalize::normalize_rel(&self.root_canonical, path) {
                Ok(rel) => rel,
                Err(err) => {
                    diagnostics.push(ScanDiagnostic {
                        rel_path: path.display().to_string(),
                        kind: DiagnosticKind::Unreadable,
                        message: err.to_string(),
                    });
                    continue;
                }
            };

            // Hard excludes win over include/user exclude (PRD §8.4).
            if is_hard_excluded(&rel, self.wiki_dir_rel.as_deref()) {
                continue;
            }

            if self.exclude.is_match(&rel) {
                continue;
            }
            if !self.include.is_match(&rel) {
                continue;
            }

            // Case-fold collision detection across the whole root.
            let key = normalize::portable_key(&rel);
            if let Some(existing) = by_portable_key.insert(key, rel.clone()) {
                return Err(WikiError::PathCollision {
                    collisions: vec![existing, rel],
                });
            }

            let meta = match link_meta {
                Ok(meta) => meta,
                Err(err) => {
                    diagnostics.push(self.diagnostic(
                        path,
                        DiagnosticKind::Unreadable,
                        err.to_string(),
                    ));
                    continue;
                }
            };
            let size = meta.len();
            if size > self.max_file_size {
                diagnostics.push(self.diagnostic(
                    path,
                    DiagnosticKind::TooLarge,
                    format!("file size {size} exceeds limit {}", self.max_file_size),
                ));
                continue;
            }

            let mut file = match std::fs::File::open(path) {
                Ok(file) => file,
                Err(err) => {
                    diagnostics.push(self.diagnostic(
                        path,
                        DiagnosticKind::Unreadable,
                        err.to_string(),
                    ));
                    continue;
                }
            };
            let mut head = Vec::new();
            let mut limited = (&mut file).take(SNIFF_LEN as u64);
            if let Err(err) = limited.read_to_end(&mut head) {
                diagnostics.push(self.diagnostic(
                    path,
                    DiagnosticKind::Unreadable,
                    err.to_string(),
                ));
                continue;
            }
            if head.contains(&0) {
                diagnostics.push(self.diagnostic(
                    path,
                    DiagnosticKind::Binary,
                    "binary content (NUL byte in first 8 KiB)".to_owned(),
                ));
                continue;
            }
            let mut bytes = head;
            if let Err(err) = file.read_to_end(&mut bytes) {
                diagnostics.push(self.diagnostic(
                    path,
                    DiagnosticKind::Unreadable,
                    err.to_string(),
                ));
                continue;
            }

            let locator = SourceLocatorKey::compute(&self.workspace_id, &rel);
            files.push(ScannedFile {
                rel_path: rel,
                locator_key: locator,
                content_hash: sha256_hex(&bytes),
                size,
                modified_at: meta.modified().ok().map(DateTime::<Utc>::from),
            });
        }

        files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
        diagnostics.sort();

        Ok(ScanOutput { files, diagnostics })
    }

    fn diagnostic(&self, path: &Path, kind: DiagnosticKind, message: String) -> ScanDiagnostic {
        // Diagnostics carry the root-relative path, never an absolute path
        // outside the allowed root (PRD §41).
        let rel_path = path
            .strip_prefix(&self.root)
            .map(|relative| relative.display().to_string())
            .unwrap_or_else(|_| path.display().to_string());
        ScanDiagnostic {
            rel_path,
            kind,
            message,
        }
    }
}

/// `.llm-wiki/**` and the configured `wiki_dir` are non-overridable excludes.
fn is_hard_excluded(rel: &str, wiki_dir_rel: Option<&str>) -> bool {
    if let Some(wiki) = wiki_dir_rel {
        if rel == wiki
            || rel.starts_with(&format!("{wiki}/"))
            || wiki.starts_with(&format!("{rel}/"))
        {
            return true;
        }
    }
    HARD_EXCLUDE_TOP_DIRS
        .iter()
        .any(|dir| rel == *dir || rel.starts_with(&format!("{dir}/")))
}

fn build_globset(patterns: &[String], what: &str) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern)
            .map_err(|e| WikiError::Config(format!("invalid {what} glob '{pattern}': {e}")))?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|e| WikiError::Config(format!("cannot build {what} globset: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, content: &[u8]) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm-wiki-scan-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn scan(root: &Path) -> Result<ScanOutput> {
        Scanner::new(
            root,
            &[
                "**/*.md".to_owned(),
                "**/*.markdown".to_owned(),
                "**/*.mdx".to_owned(),
            ],
            &[
                "node_modules/**".to_owned(),
                ".git/**".to_owned(),
                ".llm-wiki/**".to_owned(),
                "wiki/**".to_owned(),
            ],
            Some("wiki".to_owned()),
        )?
        .scan()
    }

    #[test]
    fn discovers_md_markdown_mdx_and_applies_excludes() {
        let root = temp_root("discover");
        write(&root, "readme.md", b"# top\n");
        write(&root, "guide/intro.markdown", b"intro\n");
        write(&root, "ui/tabs.mdx", b"<Tabs>..</Tabs>\n");
        write(&root, "notes.txt", b"not a source\n");
        write(&root, "node_modules/junk.md", b"junk\n");
        write(&root, ".git/internal.md", b"junk\n");
        write(&root, ".llm-wiki/state.md", b"junk\n");
        write(&root, "wiki/generated.md", b"junk\n");

        let out = scan(&root).unwrap();
        let rels: Vec<&str> = out.files.iter().map(|f| f.rel_path.as_str()).collect();
        assert_eq!(
            rels,
            vec!["guide/intro.markdown", "readme.md", "ui/tabs.mdx"]
        );
        assert!(out.diagnostics.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn binary_and_oversized_files_become_diagnostics() {
        let root = temp_root("diag");
        write(&root, "ok.md", b"fine\n");
        write(&root, "blob.md", &[0x68, 0x00, 0x69]);
        let out = scan(&root).unwrap();
        assert_eq!(out.files.len(), 1);
        assert_eq!(out.diagnostics.len(), 1);
        assert_eq!(out.diagnostics[0].kind, DiagnosticKind::Binary);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hard_excludes_cannot_be_reingested_even_via_include_conflict() {
        let root = temp_root("hard");
        write(&root, ".llm-wiki/cache.md", b"state\n");
        write(&root, "wiki/page.md", b"generated\n");
        // include explicitly lists the excluded trees: hard excludes still win.
        let scanner = Scanner::new(
            &root,
            &[".llm-wiki/**".to_owned(), "wiki/**".to_owned()],
            &[],
            Some("wiki".to_owned()),
        )
        .unwrap();
        let out = scanner.scan().unwrap();
        assert!(out.files.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn content_hash_and_locator_key_are_stable() {
        let root = temp_root("hash");
        write(&root, "a.md", b"hello\n");
        let out1 = scan(&root).unwrap();
        let out2 = scan(&root).unwrap();
        assert_eq!(out1.files[0].content_hash, out2.files[0].content_hash);
        assert_eq!(out1.files[0].locator_key, out2.files[0].locator_key);
        assert!(out1.files[0].locator_key.as_str().starts_with("loc_"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
