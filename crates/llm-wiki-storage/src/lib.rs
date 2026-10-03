#![forbid(unsafe_code)]
//! SQLite state layer (PRD §7.5, §18, §42): registries, sections, builds.
//!
//! Every query that touches external input uses bound parameters — string
//! assembly of SQL from user-controlled values is forbidden (PRD §42 and the
//! project security constraint).

pub mod analysis;
pub mod builds;
pub mod cache;
pub mod connection;
pub mod decisions;
pub mod graph;
pub mod insights;
pub mod knowledge;
pub mod migrations;
pub mod page_id_map;
pub mod registry;
pub mod search_index;
pub mod sections;
pub mod sources;
pub mod state;
pub mod wiki;

pub use analysis::{
    list_source_active_node_sections, persist_analysis, retire_source_knowledge,
    AnalysisPersistence, AnalysisRecord, EvidenceRange, PersistedAnalysis, PersistedClaim,
    PersistedRejectedClaim, PersistedRelation, RetiredSourceKnowledge, SourceNodeSection,
};
pub use builds::{
    finish_build, latest_build, latest_completed_build, set_build_snapshot_hash, start_build,
    BuildDraft, BuildRecord,
};
pub use cache::{count_cache_entries, get_cached_response, put_cached_response, CacheRow};
pub use connection::{open, open_in_memory, Connection};
pub use decisions::{
    insert_plan_decision, list_plan_decisions, list_recent_plan_decisions_by_outcome, PlanDecision,
    PlanDecisionRow, OUTCOME_FAST_PATH, OUTCOME_LOCAL_UPDATE, OUTCOME_REPLAN_DRY_RUN,
    OUTCOME_REPLAN_EXECUTED, OUTCOME_REPLAN_REQUIRED, TRIGGER_FINGERPRINT_CHANGED,
    TRIGGER_PAGE_EMPTIED, TRIGGER_STRUCTURAL_CHANGE, TRIGGER_UNMAPPABLE_NODE,
};
pub use graph::{
    ensure_graph_matches_active, graph_expand, graph_expand_from_page, page_node_id, rebuild_graph,
    GraphNeighbor, GraphStats, NeighborDirection, EXPAND_MAX_NODES, NODE_TYPE_CONCEPT,
    NODE_TYPE_ENTITY, NODE_TYPE_PAGE, RELATION_LINKS_TO,
};
pub use insights::{insert_insight, list_insights, InsightCitation, InsightRecord};
pub use knowledge::{list_active_relation_pairs, load_knowledge_base, load_plan_input};
pub use page_id_map::{
    insert_page_id_maps, list_page_id_maps_by_predecessor, list_page_id_maps_by_successor,
    list_page_id_maps_for_build, PageIdMapEntry, PageIdMapRow, MAP_KIND_KEEP, MAP_KIND_MERGE,
    MAP_KIND_RETIRE, MAP_KIND_SPLIT,
};
pub use registry::{
    canonical_key, current_revision, get_entry, get_or_create, get_or_create_batch, merge, resolve,
    retire, NodeDraft, NodeKind, RegistryEntry,
};
pub use search_index::{
    activate_build_with_search_index, clear_search_index, default_tokenizer,
    ensure_search_index_matches_active, probe_fts5, rebuild_search_index, search_index,
    DefaultSearchTokenizer, SearchIndexRow, SearchIndexStats, SearchTokenizer, FTS5_UNAVAILABLE,
};
pub use sections::{apply_section_matches, load_active_sections, SectionApplyStats, StoredSection};
pub use sources::{
    count_sources, get_by_locator, list_sources, mark_removed, upsert_source, upsert_sources_batch,
    SourceRecord, SourceUpsert,
};
pub use state::{
    activate_build, get_active_build_id, get_state, list_generation_build_ids,
    mark_stale_builds_interrupted, set_active_build, set_state, update_build_status,
    ACTIVE_BUILD_KEY, BUILD_STATUSES,
};
pub use wiki::{
    generation_stats, load_generation_pages, load_generation_view, persist_generation,
    GenerationPageView, GenerationStats, PageCitationRecord, PageLinkRecord, WikiPageRecord,
};
