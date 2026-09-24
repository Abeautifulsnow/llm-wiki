#![forbid(unsafe_code)]
//! Source scanning layer (PRD §7.2): file discovery, include/exclude,
//! hashing, source manifest and change detection.

pub mod manifest;
pub mod normalize;
pub mod scanner;

pub use manifest::{from_json, to_json, ManifestEntry, SourceManifest};
pub use normalize::{canonicalize_root, normalize_rel, portable_key, to_stored};
pub use scanner::{
    DiagnosticKind, ScanDiagnostic, ScanOutput, ScannedFile, Scanner, HARD_EXCLUDE_TOP_DIRS,
};
