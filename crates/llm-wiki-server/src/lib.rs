#![forbid(unsafe_code)]
//! HTTP transport placeholder (PRD §7.8, §30).
//!
//! V0.1 has no server (PRD §50). V0.4 introduces Axum as a pure transport
//! adapter: build jobs, search/context/query/pages, local-only default with
//! an explicit remote security boundary. The compiled wiki is already
//! consumable by external LLMs from `wiki_dir` before this crate ships
//! (PRD §24).

pub const V0_4_SCOPE: &str =
    "Axum transport: /v1/health /v1/status /v1/build /v1/jobs /v1/search /v1/context /v1/pages";
