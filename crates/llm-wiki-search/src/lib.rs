#![forbid(unsafe_code)]
//! Search layer placeholder (PRD §7.6, §20).
//!
//! V0.1 deliberately ships no retrieval runtime (PRD §50: FTS/semantic search
//! are out of scope until V0.2/V0.3). Domain shapes like
//! [`llm_wiki_core::model::SearchResult`] already live in core so downstream
//! consumers can compile against the future surface.

/// Marks crate purpose; replaced by the TextAnalyzer abstraction in V0.2
/// (CJK trigram + Latin word routing per PRD §20).
pub const V0_2_SCOPE: &str = "FTS with language-aware analyzers, rank fusion, graph expansion";
