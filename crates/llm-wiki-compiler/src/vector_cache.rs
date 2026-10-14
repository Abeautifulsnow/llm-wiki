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
//!
//! Top-K is answered by the snapshot's index — the exact scan, or an ANN
//! above the configured size (`[embedding] ann`). The ANN is NEVER produced on
//! the query path: the snapshot starts exact, and a detached thread adopts the
//! persisted graph for this revision when there is one and otherwise builds and
//! dumps it. Both halves are far too slow to sit in front of a query (a build
//! is minutes at 100k sections, and reloading a persisted graph is a
//! multi-second disk read), and neither needs the database. Invalidation is the
//! snapshot's own, so a graph is replaced by exactly the events that invalidate
//! the vectors.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use llm_wiki_core::error::Result;
use llm_wiki_core::ids::BuildId;
use llm_wiki_search::VectorCandidate;
use llm_wiki_storage::Connection;

use crate::vector_index::{AnnIndex, ExactIndex, VectorIndex, VectorIndexConfig};

/// Top-K returned to fusion, shared with the index implementations.
pub(crate) const VECTOR_TOP_K: usize = 8;

/// How to load a snapshot: which index policy, and where a persisted ANN
/// graph for this exact revision lives.
pub struct SnapshotLoad<'a> {
    pub config: &'a VectorIndexConfig,
    /// Directory + basename of the persisted graph. `None` means "no graph":
    /// the snapshot scans exactly. A graph is only worth having where it can
    /// be kept, so this doubles as the switch that says a snapshot may carry
    /// one at all — a one-shot snapshot (the CLI) and an uncached one (see
    /// [`VectorSnapshotCache::get_or_load`]) both pass `None`.
    pub persist: Option<(&'a Path, &'a str)>,
}

impl SnapshotLoad<'_> {
    /// Exact scanning, nothing persisted.
    pub fn exact() -> SnapshotLoad<'static> {
        static EXACT: VectorIndexConfig = VectorIndexConfig {
            ann: llm_wiki_core::config::AnnMode::Off,
            min_sections: 50_000,
            ef_search: 128,
        };
        SnapshotLoad {
            config: &EXACT,
            persist: None,
        }
    }
}

/// One read of the vector layer for the ACTIVE generation: the section key
/// set, the stored vectors for one model, and the index that answers top-K
/// over them. Cheap to share (`Arc`); the only thing that changes after
/// construction is the arrival of the ANN.
pub struct VectorSnapshot {
    exact: Arc<ExactIndex>,
    /// The ANN, once a persisted graph is loaded or a background build lands.
    ann: RwLock<Option<Arc<AnnIndex>>>,
    /// One background build at a time.
    building: AtomicBool,
    config: VectorIndexConfig,
    persist: Option<(PathBuf, String)>,
}

impl std::fmt::Debug for VectorSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorSnapshot")
            .field("vectors", &self.exact.len())
            .field("index", &self.index_kind())
            .finish()
    }
}

impl VectorSnapshot {
    /// Loads the snapshot for the ACTIVE generation and `model`, scanning
    /// exactly. The key set comes from the persisted section hashes, never
    /// from section text.
    pub fn load(conn: &Connection, model: &str) -> Result<Arc<Self>> {
        Self::load_with(conn, model, SnapshotLoad::exact())
    }

    /// `load` with an explicit index policy. The returned snapshot is usable
    /// immediately; an ANN lands later if one is wanted.
    pub fn load_with(conn: &Connection, model: &str, load: SnapshotLoad<'_>) -> Result<Arc<Self>> {
        let hashes = llm_wiki_storage::active_section_hashes(conn)?;
        let vectors = llm_wiki_storage::section_embeddings_by_hash(conn, model, &hashes)?;
        Ok(Self::from_vectors(vectors, load))
    }

    fn from_vectors(
        vectors: std::collections::BTreeMap<String, Vec<f32>>,
        load: SnapshotLoad<'_>,
    ) -> Arc<Self> {
        let exact = Arc::new(ExactIndex::build(&vectors));
        drop(vectors);
        let snapshot = Arc::new(Self {
            exact,
            ann: RwLock::new(None),
            building: AtomicBool::new(false),
            config: *load.config,
            persist: load
                .persist
                .map(|(dir, basename)| (dir.to_path_buf(), basename.to_owned())),
        });
        // Driving the ANN off `persist` is deliberate: a graph is only built
        // where it can also be kept, so an index is never built twice for the
        // same vectors.
        if snapshot.persist.is_some() && snapshot.config.wants_ann(snapshot.exact.len()) {
            snapshot.clone().arm_ann();
        }
        snapshot
    }

    /// Produces the ANN without the query path ever waiting for it: a
    /// detached thread adopts the persisted graph when there is one for this
    /// revision, and otherwise builds one and dumps it. Until it lands, queries
    /// are answered by the exact scan that was just loaded.
    ///
    /// Both halves of that work belong off the query path: the build is minutes
    /// at 100k sections, and even the reload of a persisted graph is a
    /// multi-second disk read at that size. The thread touches only the
    /// in-memory snapshot — no database connection, no blocking-pool slot.
    fn arm_ann(self: Arc<Self>) {
        if self.building.swap(true, Ordering::SeqCst) {
            return; // one attempt at a time for this snapshot
        }
        let snapshot = self.clone();
        let spawned = std::thread::Builder::new()
            .name("llm-wiki-ann".to_owned())
            .spawn(move || {
                let started = Instant::now();
                let index = snapshot.adopt_or_build();
                snapshot.building.store(false, Ordering::SeqCst);
                if let Some(index) = index {
                    let installed = Arc::new(index);
                    tracing::info!(
                        vectors = installed.len(),
                        resident_mib = installed.resident_bytes() / (1024 * 1024),
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        "vector index ready (approximate); the exact arena is kept as the \
                         fallback, so this doubles the snapshot's resident size"
                    );
                    snapshot.install(installed);
                }
            });
        if spawned.is_err() {
            self.building.store(false, Ordering::SeqCst);
            tracing::warn!("could not start the vector index builder; staying on the exact scan");
        }
    }

    /// The persisted graph for this revision, or a freshly built one (also
    /// persisted). `None` leaves the exact scan in place. Either way, graphs
    /// for other revisions are dropped: they are ~1 KiB per vector (300 MiB at
    /// 100k) and, being content-addressed, are never looked up again.
    fn adopt_or_build(&self) -> Option<AnnIndex> {
        let index = self.persisted_or_built()?;
        if let Some((dir, basename)) = &self.persist {
            prune_other_graphs(dir, basename);
        }
        Some(index)
    }

    fn persisted_or_built(&self) -> Option<AnnIndex> {
        if let Some((dir, basename)) = &self.persist {
            match AnnIndex::load(dir, basename) {
                // The graph is only reused for the vectors it was built from.
                // The name already encodes (build, model, revision) and the
                // revision only grows because `section_embeddings` is
                // append-only — checked here rather than assumed, so a future
                // delete/cleanup path cannot silently serve stale neighbours.
                Ok(index) if index.hashes() == self.exact.hashes() => {
                    tracing::info!(vectors = index.len(), "vector index loaded from disk");
                    return Some(index);
                }
                Ok(index) => tracing::warn!(
                    graph_vectors = index.len(),
                    current_vectors = self.exact.len(),
                    "the persisted vector index does not match the current vectors; \
                     rebuilding it"
                ),
                Err(err) => tracing::debug!(
                    error = %err,
                    "no usable persisted vector index; building one"
                ),
            }
        }
        match AnnIndex::build(&self.exact, self.config.ef_search) {
            Ok(index) => {
                if let Some((dir, basename)) = &self.persist {
                    if let Err(err) = index.dump(dir, basename) {
                        tracing::warn!(error = %err, "could not persist the vector index");
                    }
                }
                Some(index)
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "vector index build failed; staying on the exact scan"
                );
                None
            }
        }
    }

    fn install(&self, index: Arc<AnnIndex>) {
        *self.ann.write().expect("ann lock") = Some(index);
    }
    /// Whether any vector exists for this model over the active generation —
    /// the coverage pre-check that keeps an uncovered workspace from spending
    /// a query embedding.
    pub fn covered(&self) -> bool {
        !self.is_empty()
    }

    /// Stored vector count (the cache's size measure).
    pub fn len(&self) -> usize {
        self.exact.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exact.is_empty()
    }

    /// Approximate resident size, reported when a snapshot is too large to
    /// keep in the cache. Counts the ANN when one is in use — the exact arena
    /// stays as the fallback, so an ANN snapshot costs both.
    pub fn resident_bytes(&self) -> usize {
        let ann = self.ann.read().expect("ann lock").clone();
        match ann {
            Some(index) => index.resident_bytes() + self.exact.resident_bytes(),
            None => self.exact.resident_bytes(),
        }
    }

    /// Which search path this snapshot uses right now: `"exact"` or `"hnsw"`.
    pub fn index_kind(&self) -> &'static str {
        match self.ann.read().expect("ann lock").as_ref() {
            Some(index) => index.kind(),
            None => self.exact.kind(),
        }
    }

    /// The top-K cosine candidates: score descending, ties broken by text
    /// hash. `None` (a provider that returned no vector) is an empty result,
    /// never a fabricated hit.
    pub fn top_candidates(&self, query_vector: Option<&[f32]>) -> Vec<VectorCandidate> {
        let ann = self.ann.read().expect("ann lock").clone();
        match ann {
            Some(index) => index.top_candidates(query_vector, VECTOR_TOP_K),
            None => self.exact.top_candidates(query_vector, VECTOR_TOP_K),
        }
    }
}

/// Deletes the persisted graphs in `dir` that do not belong to `keep`, so one
/// revision is on disk at a time.
///
/// A file belongs to `keep` when it is `{keep}.…` or a `hnsw_rs` unique-name
/// variant `{keep}-{digits}.…`; the separator is what keeps revision 10's graph
/// from reading as revision 1's (and therefore being kept). A file that cannot
/// be deleted (another process is reading it right now) is left for the next
/// pass.
fn prune_other_graphs(dir: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let ours = name.contains(".hnsw.") || name.contains(crate::vector_index::META_SUFFIX);
        let current =
            name.starts_with(&format!("{keep}.")) || name.starts_with(&format!("{keep}-"));
        if !ours || current {
            continue;
        }
        if let Err(err) = std::fs::remove_file(entry.path()) {
            tracing::debug!(file = %name, error = %err, "could not prune a stale vector index");
        }
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
    index_config: VectorIndexConfig,
    /// Where a persisted ANN graph is looked for and written. `None` keeps
    /// graphs in memory only. The file name carries the identity, so a graph
    /// for a superseded revision is simply never found again.
    persist_dir: Option<PathBuf>,
    slot: Mutex<Option<CachedSnapshot>>,
    /// Over-budget is a CONFIGURATION condition, not per-query data: warn once
    /// per process instead of on every retrieval.
    warned_oversized: std::sync::atomic::AtomicBool,
    /// Same for the ANN that can never be built because nothing is retained.
    warned_uncached_ann: std::sync::atomic::AtomicBool,
}

impl VectorSnapshotCache {
    /// `max_sections == 0` disables caching (every call loads). `index_config`
    /// chooses the search path (exact, or an ANN above the configured size);
    /// `persist_dir` is where a graph is cached across restarts.
    pub fn new(
        max_sections: usize,
        index_config: VectorIndexConfig,
        persist_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            max_sections,
            index_config,
            persist_dir,
            slot: Mutex::new(None),
            warned_oversized: std::sync::atomic::AtomicBool::new(false),
            warned_uncached_ann: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// File name of the graph for one revision. Content-addressed by
    /// `(build_id, model, embeddings_revision)`: the graph is only ever reused
    /// for the exact vectors it was built from.
    fn graph_basename(build_id: &BuildId, model: &str, revision: i64) -> String {
        let model_hash = llm_wiki_core::hash::sha256_hex(model.as_bytes());
        format!("ann-{}-{}-{revision}", build_id.as_str(), &model_hash[..16])
    }

    /// Loads the snapshot for one identity, with the graph location for that
    /// identity when persistence is configured.
    fn load_snapshot(
        &self,
        conn: &Connection,
        model: &str,
        build_id: Option<&BuildId>,
        revision: i64,
    ) -> Result<Arc<VectorSnapshot>> {
        let dir = self.persist_dir.as_deref();
        let basename = match (dir, build_id) {
            (Some(_), Some(build_id)) => Some(Self::graph_basename(build_id, model, revision)),
            _ => None,
        };
        let persist = match (dir, basename.as_deref()) {
            (Some(dir), Some(basename)) => Some((dir, basename)),
            _ => None,
        };
        let load = SnapshotLoad {
            config: &self.index_config,
            persist,
        };
        VectorSnapshot::load_with(conn, model, load)
    }

    /// The snapshot for the active generation and `model`, from the cache when
    /// it is still valid, otherwise freshly loaded (and kept when it fits).
    pub fn get_or_load(&self, conn: &Connection, model: &str) -> Result<Arc<VectorSnapshot>> {
        if self.max_sections == 0 {
            // Nothing is retained, so a graph would be built (and thrown away)
            // on every single query: scan exactly instead. An explicit
            // `ann = "on"` then does nothing, which must not be silent.
            if self.index_config.ann != llm_wiki_core::config::AnnMode::Off
                && !self.warned_uncached_ann.swap(true, Ordering::Relaxed)
            {
                tracing::warn!(
                    "`[embedding] ann` is configured but `snapshot_cache_sections` is 0, so \
                     no snapshot is kept and no index is ever built; raise \
                     snapshot_cache_sections above 0 to enable it"
                );
            }
            return VectorSnapshot::load(conn, model);
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
        let snapshot = self.load_snapshot(conn, model, active.as_ref(), revision)?;
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
        let cache = VectorSnapshotCache::new(100, VectorIndexConfig::exact(), None);

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
        let cache = VectorSnapshotCache::new(100, VectorIndexConfig::exact(), None);
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
        let cache = VectorSnapshotCache::new(100, VectorIndexConfig::exact(), None);
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
        let cache = VectorSnapshotCache::new(2, VectorIndexConfig::exact(), None);
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
        let cache = VectorSnapshotCache::new(0, VectorIndexConfig::exact(), None);
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
        let cache = VectorSnapshotCache::new(100, VectorIndexConfig::exact(), None);

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

    /// A graph persisted for this exact revision is adopted — the same graph,
    /// not a rebuild of it. Proven by the sidecar: a build dumps (and so
    /// rewrites it), an adoption does not even open it for writing.
    #[test]
    fn a_persisted_graph_is_adopted_instead_of_rebuilt() {
        // Distinct, non-degenerate vectors: the flat vectors the other cache
        // tests use make every cosine tie, and a tie is exactly where an
        // approximate index may legitimately answer differently.
        let sections = 40;
        let (mut conn, build_id) = workspace(sections);
        let hashes = llm_wiki_storage::active_section_hashes(&conn).unwrap();
        let expected = hashes.len();
        let dim = 16;
        let pairs: Vec<(String, Vec<f32>)> = hashes
            .iter()
            .enumerate()
            .map(|(slot, hash)| {
                let vector = (0..dim)
                    .map(|axis| ((slot * 17 + axis * 31) % 101) as f32 / 101.0 - 0.5)
                    .collect();
                (hash.clone(), vector)
            })
            .collect();
        insert_section_embeddings(&mut conn, "m", &pairs).unwrap();
        let revision = llm_wiki_storage::embeddings_revision(&conn).unwrap();
        let hashes = llm_wiki_storage::active_section_hashes(&conn).unwrap();
        let vectors = llm_wiki_storage::section_embeddings_by_hash(&conn, "m", &hashes).unwrap();

        // Persist a graph under the name the cache looks for at this identity.
        let dir = std::env::temp_dir().join("llm-wiki-ann-cache-adopt");
        let _ = std::fs::remove_dir_all(&dir);
        let basename = VectorSnapshotCache::graph_basename(&build_id, "m", revision);
        AnnIndex::build(&ExactIndex::build(&vectors), 128)
            .unwrap()
            .dump(&dir, &basename)
            .unwrap();
        let sidecar = crate::vector_index::meta_path(&dir, &basename);
        let dumped_at = std::fs::metadata(&sidecar).unwrap().modified().unwrap();

        let config = VectorIndexConfig {
            ann: llm_wiki_core::config::AnnMode::On,
            min_sections: 0,
            ef_search: 128,
        };
        let cache = VectorSnapshotCache::new(100, config, Some(dir.clone()));
        let adopted = cache.get_or_load(&conn, "m").unwrap();
        assert_eq!(adopted.len(), expected);
        // Answered straight away by the exact scan; the graph arrives on the
        // snapshot's own thread.
        assert!(!adopted.top_candidates(Some(&[0.1_f32; 16])).is_empty());
        await_index(&adopted);

        assert!(
            std::fs::metadata(&sidecar).unwrap().modified().unwrap() <= dumped_at,
            "the sidecar was rewritten: the graph was rebuilt, not adopted"
        );
        let query: Vec<f32> = (0..dim)
            .map(|axis| ((axis * 7 + 11) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let found: Vec<String> = adopted
            .top_candidates(Some(&query))
            .into_iter()
            .map(|candidate| candidate.text_hash)
            .collect();
        let wanted: Vec<String> = VectorSnapshot::load(&conn, "m")
            .unwrap()
            .top_candidates(Some(&query))
            .into_iter()
            .map(|candidate| candidate.text_hash)
            .collect();
        // Hashes only: the ANN's score is `1 - distance`, so it is not
        // bit-identical to the exact dot product even where it agrees.
        assert_eq!(found, wanted, "the adopted graph answers the same query");
    }

    /// Enabling the index must not make a query wait: the snapshot is usable
    /// the moment it is returned, and the graph arrives on its own thread.
    #[test]
    fn an_enabled_index_is_built_off_the_query_path() {
        let (mut conn, _) = workspace(60);
        seed(&mut conn, "m", 60);
        let dir = std::env::temp_dir().join("llm-wiki-ann-background");
        let _ = std::fs::remove_dir_all(&dir);
        let config = VectorIndexConfig {
            ann: llm_wiki_core::config::AnnMode::On,
            min_sections: 0,
            ef_search: 32,
        };
        let cache = VectorSnapshotCache::new(1_000, config, Some(dir.clone()));
        let snapshot = cache.get_or_load(&conn, "m").unwrap();
        let query = vec![1.0_f32, 0.5];

        // Answered immediately, whatever is answering: the exact scan is in
        // place from the start and stays until the graph is complete.
        assert!(!snapshot.top_candidates(Some(&query)).is_empty());
        await_index(&snapshot);

        // ...and it was persisted for the next process (graph + sidecar).
        let written = std::fs::read_dir(&dir).map(|dir| dir.count()).unwrap_or(0);
        assert!(
            written >= 2,
            "expected a dump and a sidecar, found {written}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Waits for the snapshot's index thread to land, bounded so a build that
    /// never finishes fails the test instead of hanging it.
    fn await_index(snapshot: &VectorSnapshot) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while snapshot.index_kind() != "hnsw" && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(
            snapshot.index_kind(),
            "hnsw",
            "the background index must land"
        );
    }

    /// A graph whose keys are not the current vectors is discarded, never
    /// served. Reachable in practice only if `section_embeddings` ever stops
    /// being append-only — which is exactly why the loader checks instead of
    /// assuming it.
    #[test]
    fn a_graph_of_other_vectors_is_discarded_not_served() {
        let (mut conn, build_id) = workspace(12);
        seed(&mut conn, "m", 12);
        let revision = llm_wiki_storage::embeddings_revision(&conn).unwrap();
        let hashes = llm_wiki_storage::active_section_hashes(&conn).unwrap();

        // A same-sized but DIFFERENT key set, persisted under this identity's
        // name — the shape a stale graph would have.
        let stale: std::collections::BTreeMap<String, Vec<f32>> = (0..hashes.len())
            .map(|slot| (format!("stale-{slot:06}"), vec![1.0_f32, 0.5]))
            .collect();
        let dir = std::env::temp_dir().join("llm-wiki-ann-stale");
        let _ = std::fs::remove_dir_all(&dir);
        let basename = VectorSnapshotCache::graph_basename(&build_id, "m", revision);
        AnnIndex::build(&ExactIndex::build(&stale), 32)
            .unwrap()
            .dump(&dir, &basename)
            .unwrap();
        // The stale dump loads and is self-consistent — it is the key check,
        // not the loader, that has to reject it (or this test would pass even
        // with the guard removed).
        let as_dumped = AnnIndex::load(&dir, &basename).expect("the stale dump loads");
        assert_eq!(as_dumped.len(), hashes.len(), "same size, different keys");
        assert!(as_dumped.hashes()[0].starts_with("stale-"));

        let config = VectorIndexConfig {
            ann: llm_wiki_core::config::AnnMode::On,
            min_sections: 0,
            ef_search: 32,
        };
        let cache = VectorSnapshotCache::new(100, config, Some(dir.clone()));
        let snapshot = cache.get_or_load(&conn, "m").unwrap();
        await_index(&snapshot);

        let served: Vec<String> = snapshot
            .top_candidates(Some(&[1.0_f32, 0.5]))
            .into_iter()
            .map(|candidate| candidate.text_hash)
            .collect();
        assert!(
            !served.is_empty() && served.iter().all(|hash| hashes.contains(hash)),
            "a stale graph leaked its own keys: {served:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With caching off there is nothing to keep a graph in, so no graph is
    /// built — and saying so is not optional, or an explicit `ann = "on"`
    /// looks like it took effect.
    #[test]
    fn an_uncached_snapshot_builds_no_index() {
        let (mut conn, _) = workspace(20);
        seed(&mut conn, "m", 20);
        let dir = std::env::temp_dir().join("llm-wiki-ann-uncached");
        let _ = std::fs::remove_dir_all(&dir);
        let config = VectorIndexConfig {
            ann: llm_wiki_core::config::AnnMode::On,
            min_sections: 0,
            ef_search: 32,
        };
        let cache = VectorSnapshotCache::new(0, config, Some(dir.clone()));
        for _ in 0..3 {
            let snapshot = cache.get_or_load(&conn, "m").unwrap();
            assert_eq!(snapshot.index_kind(), "exact");
        }
        assert!(
            !dir.exists(),
            "an uncached snapshot must not leave a graph behind"
        );
        assert!(
            cache.warned_uncached_ann.load(Ordering::Relaxed),
            "the no-op configuration must be reported once"
        );
    }

    /// A corrupt sidecar is an error to rebuild from, through the whole cache
    /// path (not just `AnnIndex::load`): queries keep being answered exactly.
    #[test]
    fn a_corrupt_persisted_graph_still_answers_exactly() {
        let (mut conn, build_id) = workspace(12);
        seed(&mut conn, "m", 12);
        let revision = llm_wiki_storage::embeddings_revision(&conn).unwrap();
        let dir = std::env::temp_dir().join("llm-wiki-ann-corrupt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let basename = VectorSnapshotCache::graph_basename(&build_id, "m", revision);
        std::fs::write(
            crate::vector_index::meta_path(&dir, &basename),
            b"{ truncated",
        )
        .unwrap();

        let config = VectorIndexConfig {
            ann: llm_wiki_core::config::AnnMode::On,
            min_sections: 0,
            ef_search: 32,
        };
        let cache = VectorSnapshotCache::new(100, config, Some(dir.clone()));
        let snapshot = cache.get_or_load(&conn, "m").unwrap();
        let query = vec![1.0_f32, 0.5];
        assert!(!snapshot.top_candidates(Some(&query)).is_empty());
        await_index(&snapshot);
        assert!(!snapshot.top_candidates(Some(&query)).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A directory that was never created (or already cleaned up) is not an
    /// error for the pruning pass.
    #[test]
    fn pruning_a_missing_directory_is_a_no_op() {
        let dir = std::env::temp_dir().join("llm-wiki-ann-prune-missing");
        let _ = std::fs::remove_dir_all(&dir);
        prune_other_graphs(&dir, "ann-a-1");
    }

    /// Exactly one revision's graph is kept. The separator matters: a bare
    /// prefix test would treat revision 10 as revision 1 (and keep it), which
    /// is how a directory of 300 MiB files accumulates.
    #[test]
    fn pruning_keeps_only_the_current_revisions_graph() {
        let dir = std::env::temp_dir().join("llm-wiki-ann-prune");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str| std::fs::write(dir.join(name), b"x").unwrap();
        for name in [
            "ann-a-1.ann.meta.json",
            "ann-a-1-4362.hnsw.data",
            "ann-a-1.hnsw.graph",
            "ann-a-1.hnsw.data",
            "ann-a-10.hnsw.data",
            "ann-a-2.ann.meta.json",
            "ann-a-2.hnsw.data",
            "notes.txt",
        ] {
            write(name);
        }

        prune_other_graphs(&dir, "ann-a-1");

        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(
            left,
            vec![
                "ann-a-1-4362.hnsw.data", // hnsw_rs's unique-name variant
                "ann-a-1.ann.meta.json",
                "ann-a-1.hnsw.data",
                "ann-a-1.hnsw.graph",
                "notes.txt", // never ours in the first place
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
