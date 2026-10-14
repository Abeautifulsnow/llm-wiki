//! The semantic text of one context section and its identity hash — ONE
//! definition, shared by every party that must agree on a section's key: the
//! search-index writer (which PERSISTS the hash, migration 0015), the embed
//! backfill and the hybrid query path. It lives in `llm-wiki-core` because
//! the writer sits in `llm-wiki-storage`, which cannot depend on
//! `llm-wiki-search` — a private copy there would let the stored key drift
//! from the queried one.

use crate::hash::sha256_hex;

/// The semantic text of one context section — the object the Vector layer
/// embeds and hashes.
pub fn context_section_text(title: &str, heading_path: &[String], body: &str) -> String {
    format!("{title}\n{}\n{body}", heading_path.join(" > "))
}

/// The content hash identifying a context section's text
/// ([`context_section_text`], sha256).
pub fn context_section_hash(title: &str, heading_path: &[String], body: &str) -> String {
    sha256_hex(context_section_text(title, heading_path, body).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::{context_section_hash, context_section_text};

    #[test]
    fn text_joins_title_heading_path_and_body() {
        assert_eq!(
            context_section_text("Title", &["A".into(), "B".into()], "body"),
            "Title\nA > B\nbody"
        );
        assert_eq!(context_section_text("Title", &[], "body"), "Title\n\nbody");
    }

    #[test]
    fn hash_is_content_addressed_over_all_three_parts() {
        let base = context_section_hash("T", &["H".into()], "body");
        assert_eq!(base, context_section_hash("T", &["H".into()], "body"));
        assert_ne!(base, context_section_hash("T", &["H".into()], "other"));
        assert_ne!(base, context_section_hash("T2", &["H".into()], "body"));
        assert_ne!(base, context_section_hash("T", &["H2".into()], "body"));
    }
}
