//! Query-side vector snapshot and its cache (§19.3 Vector layer, LW-001).
//!
//! Every hybrid retrieval used to re-read the whole vector layer for the
//! active generation: the section hashes, then one blob per section, then a
//! cosine over all of them. The READ is by far the expensive half (measured
//! 293 MiB / ~2 s at 100k sections, LW-001 baseline), and it is identical for
//! every query against the same (generation, model) — so the server loads it
//! ONCE and reuses the immutable snapshot.
//!
//! Invalidation is deliberately not a callback the writers must remember:
//! a cached snapshot is only reused while BOTH the active generation and the
//! embeddings revision
//! ([`llm_wiki_storage::embeddings_revision`]) are unchanged, so a publish
//! switch, a new backfill and an out-of-process `llm-wiki embed` all
//! invalidate it without any cooperation from those paths.
//!
//! Memory is the cost, so the cache is bounded by SECTION COUNT: a snapshot
//! over the cap is returned for that request but never kept.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use llm_wiki_core::error::Result;
use llm_wiki_core::ids::BuildId;
use llm_wiki_search::VectorCandidate;
use llm_wiki_storage::Connection;

/// Top-K returned to fusion, matching the pre-cache retrieval path.
const VECTOR_TOP_K: usize = 8;

/// One immutable read of the vector layer for the ACTIVE generation: the
/// section key set plus the stored vectors for one model. Cheap to share
/// (`Arc`), never mutated after construction.
#[derive(Debug)]
pub struct VectorSnapshot {
    vectors: BTreeMap<String, Vec<f32>>,
}

impl VectorSnapshot {
    /// Loads the snapshot for the ACTIVE generation and `model`. The key set
    /// comes from the persisted section hashes, never from section text.
    pub fn load(conn: &Connection, model: &str) -> Result<Self> {
        let hashes = llm_wiki_storage::active_section_hashes(conn)?;
        let vectors = llm_wiki_storage::section_embeddings_by_hash(conn, model, &hashes)?;
        Ok(Self { vectors })
    }

    /// Whether any vector exists for this model over the active generation —
    /// the coverage pre-check that keeps an uncovered workspace from spending
    /// a query embedding.
    pub fn covered(&self) -> bool {
        !self.is_empty()
    }

    /// Stored vector count (the cache's size measure).
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Approximate resident size: dims are uniform per model, so one vector's
    /// bytes times the count. Reported when a snapshot is too large to keep.
    pub fn resident_bytes(&self) -> usize {
        let per_vector = self
            .vectors
            .values()
            .next()
            .map(|vector| vector.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        per_vector * self.vectors.len()
    }

    /// The top-K cosine candidates, deterministic: score descending, ties
    /// broken by text hash. `None` (a provider that returned no vector) is an
    /// empty result, never a fabricated hit.
    pub fn top_candidates(&self, query_vector: Option<&[f32]>) -> Vec<VectorCandidate> {
        let Some(query_vector) = query_vector else {
            return Vec::new();
        };
        let mut scored: Vec<(f32, &str)> = self
            .vectors
            .iter()
            .map(|(text_hash, vector)| (cosine(query_vector, vector), text_hash.as_str()))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(b.1))
        });
        scored
            .into_iter()
            .take(VECTOR_TOP_K)
            .map(|(score, text_hash)| VectorCandidate {
                text_hash: text_hash.to_owned(),
                score,
            })
            .collect()
    }
}

/// Cosine similarity (vectors of different length score 0 — different
/// embedding dimensions never mix).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

/// The single cached snapshot plus the identity it belongs to.
struct CachedSnapshot {
    build_id: BuildId,
    model: String,
    revision: i64,
    snapshot: Arc<VectorSnapshot>,
}

/// Bounded, self-invalidating cache of the vector layer (single slot: one
/// active generation and, in practice, one embedding model — a second model
/// simply replaces the slot).
pub struct VectorSnapshotCache {
    max_sections: usize,
    slot: Mutex<Option<CachedSnapshot>>,
    /// Over-budget is a CONFIGURATION condition, not per-query data: warn once
    /// per process instead of on every retrieval.
    warned_oversized: std::sync::atomic::AtomicBool,
}

impl VectorSnapshotCache {
    /// `max_sections == 0` disables caching (every call loads).
    pub fn new(max_sections: usize) -> Self {
        Self {
            max_sections,
            slot: Mutex::new(None),
            warned_oversized: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The snapshot for the active generation and `model`, from the cache when
    /// it is still valid, otherwise freshly loaded (and kept when it fits).
    pub fn get_or_load(&self, conn: &Connection, model: &str) -> Result<Arc<VectorSnapshot>> {
        if self.max_sections == 0 {
            return Ok(Arc::new(VectorSnapshot::load(conn, model)?));
        }
        let active = llm_wiki_storage::get_active_build_id(conn)?;
        let revision = llm_wiki_storage::embeddings_revision(conn)?;
        {
            let slot = self.slot.lock().expect("vector cache lock");
            if let Some(cached) = slot.as_ref() {
                if Some(&cached.build_id) == active.as_ref()
                    && cached.model == model
                    && cached.revision == revision
                {
                    return Ok(cached.snapshot.clone());
                }
            }
        }
        let snapshot = Arc::new(VectorSnapshot::load(conn, model)?);
        if snapshot.len() <= self.max_sections {
            let mut slot = self.slot.lock().expect("vector cache lock");
            *slot = active.map(|build_id| CachedSnapshot {
                build_id,
                model: model.to_owned(),
                revision,
                snapshot: snapshot.clone(),
            });
        } else if !self
            .warned_oversized
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!(
                vectors = snapshot.len(),
                resident_mib = snapshot.resident_bytes() / (1024 * 1024),
                max_sections = self.max_sections,
                "vector snapshot exceeds the cache budget; serving it without caching \
                 (raise [embedding] snapshot_cache_sections or leave it uncached)"
            );
        }
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_storage::{activate_build, insert_section_embeddings, BuildDraft, WikiPageRecord};

    /// A published generation with `sections` context sections, no vectors yet.
    fn workspace(sections: usize) -> (Connection, BuildId) {
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        let build_id = llm_wiki_storage::start_build(&mut conn, &BuildDraft::default()).unwrap();
        let content = (0..sections)
            .map(|i| format!("## Section {i}\n\nbody {i}\n\n"))
            .collect::<String>();
        let page = WikiPageRecord {
            page_id: llm_wiki_core::ids::WikiPageId::generate(),
            slug: "page".into(),
            title: "Page".into(),
            category: "concepts".into(),
            language: "en".into(),
            body_hash: llm_wiki_core::hash::sha256_hex(content.as_bytes()),
            content: format!("# Page\n\n{content}"),
            knowledge_refs: Vec::new(),
            citations: Vec::new(),
            links: Vec::new(),
        };
        llm_wiki_storage::persist_generation(&mut conn, &build_id, &[page]).unwrap();
        activate_build(&mut conn, &build_id).unwrap();
        llm_wiki_storage::rebuild_search_index(
            &conn,
            &build_id,
            llm_wiki_storage::default_tokenizer(),
        )
        .unwrap();
        (conn, build_id)
    }

    fn seed(conn: &mut Connection, model: &str, sections: usize) {
        let hashes = llm_wiki_storage::active_section_hashes(conn).unwrap();
        let pairs: Vec<(String, Vec<f32>)> = hashes
            .into_iter()
            .take(sections)
            .enumerate()
            .map(|(i, hash)| (hash, vec![if i % 2 == 0 { 1.0 } else { 0.0 }, 0.5]))
            .collect();
        insert_section_embeddings(conn, model, &pairs).unwrap();
    }

    #[test]
    fn a_valid_entry_is_reused_and_a_new_backfill_invalidates_it() {
        let (mut conn, _) = workspace(4);
        let cache = VectorSnapshotCache::new(100);

        // Nothing embedded yet: an empty snapshot is still cached as valid.
        let empty = cache.get_or_load(&conn, "m").unwrap();
        assert!(!empty.covered());

        seed(&mut conn, "m", 4);
        let after_seed = cache.get_or_load(&conn, "m").unwrap();
        assert!(after_seed.covered());
        assert_eq!(after_seed.len(), 4);

        // Same generation, same revision → the very same snapshot.
        let again = cache.get_or_load(&conn, "m").unwrap();
        assert!(
            Arc::ptr_eq(&after_seed, &again),
            "a valid entry must be reused"
        );

        // A second model is a different vector space: never the cached one.
        let other_model = cache.get_or_load(&conn, "other").unwrap();
        assert!(!Arc::ptr_eq(&after_seed, &other_model));
        assert!(!other_model.covered(), "models never share vectors");
    }

    #[test]
    fn a_new_backfill_between_calls_invalidates_the_cache() {
        let (mut conn, _) = workspace(4);
        let cache = VectorSnapshotCache::new(100);
        seed(&mut conn, "m", 2);
        let first = cache.get_or_load(&conn, "m").unwrap();
        assert_eq!(first.len(), 2);

        // A concurrent backfill (another process, another connection) lands
        // after the fill: the revision check must notice.
        seed(&mut conn, "m", 4);
        let second = cache.get_or_load(&conn, "m").unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "a new vector must invalidate"
        );
        assert_eq!(second.len(), 4);
    }

    #[test]
    fn a_publish_switch_invalidates_the_cache() {
        let (mut conn, first_build) = workspace(2);
        let cache = VectorSnapshotCache::new(100);
        seed(&mut conn, "m", 2);
        let before = cache.get_or_load(&conn, "m").unwrap();
        assert_eq!(before.len(), 2);

        // A new generation with different content: its section hashes differ,
        // so the snapshot must not be reused even though the revision only
        // grew by the new rows.
        let content = "## Other\n\nnew body\n\n".to_owned();
        let page = WikiPageRecord {
            page_id: llm_wiki_core::ids::WikiPageId::generate(),
            slug: "other".into(),
            title: "Other".into(),
            category: "concepts".into(),
            language: "en".into(),
            body_hash: llm_wiki_core::hash::sha256_hex(content.as_bytes()),
            content: format!("# Other\n\n{content}"),
            knowledge_refs: Vec::new(),
            citations: Vec::new(),
            links: Vec::new(),
        };
        let next_build = llm_wiki_storage::start_build(&mut conn, &BuildDraft::default()).unwrap();
        llm_wiki_storage::persist_generation(&mut conn, &next_build, &[page]).unwrap();
        llm_wiki_storage::activate_build_with_search_index(
            &mut conn,
            &next_build,
            llm_wiki_storage::default_tokenizer(),
        )
        .unwrap();
        seed(&mut conn, "m", 1);

        let after = cache.get_or_load(&conn, "m").unwrap();
        assert!(
            !Arc::ptr_eq(&before, &after),
            "a publish switch must invalidate"
        );
        assert_eq!(after.len(), 1, "the new generation's key set");
        assert_ne!(first_build, next_build);
    }

    #[test]
    fn an_oversized_snapshot_is_served_but_not_kept() {
        let (mut conn, _) = workspace(6);
        let cache = VectorSnapshotCache::new(2);
        seed(&mut conn, "m", 6);

        let first = cache.get_or_load(&conn, "m").unwrap();
        assert_eq!(first.len(), 6, "the request still gets its data");
        let second = cache.get_or_load(&conn, "m").unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "over the budget nothing may be retained"
        );
        assert_eq!(second.len(), 6);
    }

    #[test]
    fn a_disabled_cache_never_retains() {
        let (mut conn, _) = workspace(4);
        let cache = VectorSnapshotCache::new(0);
        seed(&mut conn, "m", 4);
        let first = cache.get_or_load(&conn, "m").unwrap();
        let second = cache.get_or_load(&conn, "m").unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
    }

    /// The cached path must produce byte-identical retrieval to the uncached
    /// one — the cache is a read optimisation, never a behavioural change.
    #[test]
    fn cached_and_direct_retrieval_agree() {
        let (mut conn, _) = workspace(5);
        seed(&mut conn, "m", 5);
        let query = vec![1.0_f32, 0.5];
        let cache = VectorSnapshotCache::new(100);

        let direct = VectorSnapshot::load(&conn, "m")
            .unwrap()
            .top_candidates(Some(&query));
        let cached = cache
            .get_or_load(&conn, "m")
            .unwrap()
            .top_candidates(Some(&query));
        assert_eq!(direct, cached);
        assert!(!direct.is_empty());
        assert!(direct.len() <= VECTOR_TOP_K);

        // Determinism: the same query twice, from the cache, is identical.
        let again = cache
            .get_or_load(&conn, "m")
            .unwrap()
            .top_candidates(Some(&query));
        assert_eq!(direct, again);

        // No query vector (provider returned none) → no fabricated hits.
        assert!(cache
            .get_or_load(&conn, "m")
            .unwrap()
            .top_candidates(None)
            .is_empty());
    }
}
