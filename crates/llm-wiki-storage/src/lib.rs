#![forbid(unsafe_code)]
//! SQLite state layer (PRD §7.5, §18, §42): registries, sections, builds.
//!
//! Every query that touches external input uses bound parameters — string
//! assembly of SQL from user-controlled values is forbidden (PRD §42 and the
//! project security constraint).

pub mod analysis;
pub mod builds;
pub mod connection;
pub mod knowledge;
pub mod migrations;
pub mod registry;
pub mod sections;
pub mod sources;
pub mod wiki;

pub use analysis::{
    persist_analysis, AnalysisPersistence, AnalysisRecord, EvidenceRange, PersistedAnalysis,
    PersistedClaim, PersistedRejectedClaim, PersistedRelation,
};
pub use builds::{finish_build, latest_build, start_build, BuildDraft, BuildRecord};
pub use connection::{open, open_in_memory};
pub use knowledge::{load_knowledge_base, load_plan_input};
pub use registry::{
    canonical_key, current_revision, get_entry, get_or_create, get_or_create_batch, merge, resolve,
    retire, NodeDraft, NodeKind, RegistryEntry,
};
pub use sections::{apply_section_matches, load_active_sections, SectionApplyStats, StoredSection};
pub use sources::{
    count_sources, get_by_locator, list_sources, mark_removed, upsert_source, upsert_sources_batch,
    SourceRecord, SourceUpsert,
};
pub use wiki::{
    persist_generation, GenerationStats, PageCitationRecord, PageLinkRecord, WikiPageRecord,
};
