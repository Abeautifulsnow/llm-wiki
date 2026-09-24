#![forbid(unsafe_code)]
//! Markdown/MDX structural parsing (PRD §9).
//!
//! The parser preserves heading paths, source ranges, links and frontmatter
//! as restricted metadata. `.mdx` input first goes through a deterministic
//! JSX component downgrade: text children and readable attribute values are
//! kept, everything unrecognized produces a diagnostic — never a silent drop.
//! The downgrade only affects the text handed to analysis; the Source itself
//! is never rewritten (PRD §9).

pub mod frontmatter;
pub mod language;
pub mod mdx;
pub mod parser;

pub use frontmatter::Frontmatter;
pub use language::detect_language;
pub use mdx::downgrade_jsx;
pub use parser::{
    parse_document, LinkOutput, ParseDiagnostic, ParseDiagnosticKind, ParseOutput, SectionOutput,
};
