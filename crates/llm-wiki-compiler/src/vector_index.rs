//! The local VectorStore (PRD §21/§49, audit LW-001 中期 / PR-5): nearest
//! neighbour search over one IMMUTABLE snapshot of the vector layer.
//!
//! SQLite stays the authority — this is a derived, in-memory structure that
//! holds the active generation's vectors for one model and answers top-K.
//! Two implementations sit behind [`VectorIndex`]:
//!
//! - [`ExactIndex`] — the shipped brute-force scan, kept exact, but
//!   contiguous (one arena instead of a `Vec<f32>` per section), with the
//!   per-candidate norms precomputed and the result tuples reduced to
//!   `(score, slot)` so no string comparison runs inside the sort. Scores are
//!   bit-identical to the pre-PR-5 scan.
//! - [`AnnIndex`] — HNSW (`hnsw_rs`) for corpora where the exact scan is the
//!   measured bottleneck: built once per (generation, model, embeddings
//!   revision), inserted in parallel, dumped to disk and reloaded from it on
//!   the next start, and never mutated after it is built.
//!
//! Degradation contract (LW-001 acceptance): an index that is missing,
//! unbuildable or unloadable NEVER fabricates a hit and never silently drops
//! one. The exact scan is always available and always the fallback; the ANN is
//! built OFF the query path (see `vector_cache`) and only replaces the scan
//! once it is complete. A degenerate query (a zero vector, or a width that
//! does not match the corpus) reproduces the exact scan's defined answer
//! instead of an arbitrary neighbour set.
//!
//! Determinism: the exact path is bit-reproducible. The ANN graph is not — the
//! builder assigns levels from an unseeded RNG (and parallel insertion adds
//! small race-dependent differences), so two builds of the same corpus may
//! answer a query slightly differently. That is why the persisted graph is
//! keyed by the embeddings revision: a long-lived server keeps answering with
//! the SAME graph until the vectors change.

use std::collections::BTreeMap;
use std::path::Path;

use hnsw_rs::anndists::dist::distances::DistCosine;
use hnsw_rs::api::AnnT;
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::hnswio::HnswIo;
use llm_wiki_search::VectorCandidate;

use crate::vector_cache::VECTOR_TOP_K;

/// The search-path mode, defined with the rest of the config surface so the
/// accepted values live in exactly one place.
pub use llm_wiki_core::config::AnnMode;

/// `M` from the HNSW paper: links per node. `hnsw_rs` caps this at 256 and
/// exits the process above it, so the value is fixed here, well inside range.
const HNSW_MAX_CONNECTIONS: usize = 32;
/// Layer ceiling; `hnsw_rs` clamps it to its own `NB_LAYER_MAX`.
const HNSW_MAX_LAYER: usize = 16;
/// How hard the builder searches while linking. Higher builds a better graph
/// more slowly.
const HNSW_EF_CONSTRUCTION: usize = 200;
/// Points handed to one parallel insertion batch: big enough to keep the
/// rayon pool busy, small enough that the batch's temporary copies stay a
/// rounding error next to the corpus.
const HNSW_INSERT_BATCH: usize = 4096;
/// Sidecar holding what the graph file cannot: the section hashes, the width
/// and the search breadth. JSON, because it is written once per revision and
/// only has to be readable.
const META_SUFFIX: &str = "ann.meta.json";

/// Resolved index policy: whether to build an ANN, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorIndexConfig {
    pub ann: AnnMode,
    /// `Auto` builds an ANN at or above this many stored vectors.
    pub min_sections: usize,
    /// ANN search breadth: the recall/latency dial.
    pub ef_search: usize,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            ann: AnnMode::Auto,
            min_sections: 50_000,
            ef_search: 128,
        }
    }
}

impl VectorIndexConfig {
    /// Exact scanning only — the CLI's one-shot query path.
    pub fn exact() -> Self {
        Self {
            ann: AnnMode::Off,
            ..Self::default()
        }
    }

    /// Whether this corpus, under this policy, should get an ANN.
    pub fn wants_ann(&self, sections: usize) -> bool {
        match self.ann {
            AnnMode::Off => false,
            AnnMode::On => sections > 0,
            AnnMode::Auto => sections >= self.min_sections,
        }
    }
}

/// Top-K nearest neighbours over one immutable snapshot.
pub trait VectorIndex: Send + Sync + std::fmt::Debug {
    /// `None` (a provider that returned no vector) is an empty result — never
    /// a fabricated hit.
    fn top_candidates(&self, query: Option<&[f32]>, k: usize) -> Vec<VectorCandidate>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Approximate resident size, for the cache's budget and diagnostics.
    fn resident_bytes(&self) -> usize;
    /// `"exact"` or `"hnsw"` — reported so operators can tell which path a
    /// deployment is actually running.
    fn kind(&self) -> &'static str;
}

/// Cosine of `query` against one stored vector, or 0.0 when the widths differ
/// — two models' vectors must never mix, and a width mismatch scores 0 for
/// every candidate exactly as the shipped scan did.
fn cosine(query: &[f32], query_norm_sqrt: f32, vector: &[f32], norm_sqrt: f32) -> f32 {
    if query.len() != vector.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    for (left, right) in query.iter().zip(vector.iter()) {
        dot += left * right;
    }
    let denominator = query_norm_sqrt * norm_sqrt;
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

/// Ranks scored candidates the one way every path in this module does:
/// score descending, ties broken by text hash.
fn rank(scored: &mut Vec<(f32, u32)>, hashes: &[String], k: usize) -> Vec<VectorCandidate> {
    scored.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| hashes[a.1 as usize].cmp(&hashes[b.1 as usize]))
    });
    scored
        .drain(..scored.len().min(k))
        .map(|(score, slot)| VectorCandidate {
            text_hash: hashes[slot as usize].clone(),
            score,
        })
        .collect()
}

/// The answer a zero-width or zero-vector query gets: every cosine is 0, so
/// the top-K is the lowest hashes. Shared by both indexes so the degenerate
/// cases cannot diverge between them.
fn degenerate_top_k(hashes: &[String], k: usize) -> Vec<VectorCandidate> {
    let mut order: Vec<u32> = (0..hashes.len() as u32).collect();
    order.sort_unstable_by(|a, b| hashes[*a as usize].cmp(&hashes[*b as usize]));
    order
        .into_iter()
        .take(k)
        .map(|slot| VectorCandidate {
            text_hash: hashes[slot as usize].clone(),
            score: 0.0,
        })
        .collect()
}

/// The exact scan. Scores are bit-identical to the pre-PR-5 implementation:
/// the same dot product in the same accumulation order, divided by the same
/// `sqrt(sum a²) * sqrt(sum b²)` — only the einsum's *storage* changed.
#[derive(Debug)]
pub struct ExactIndex {
    hashes: Vec<String>,
    /// All vectors back to back: `hashes[i]` owns `arena[i*dim..(i+1)*dim]`.
    arena: Vec<f32>,
    /// Precomputed `sqrt(sum v²)` per vector — computed once at build instead
    /// of once per candidate per query.
    norm_sqrt: Vec<f32>,
    dim: usize,
}

impl ExactIndex {
    pub fn build(vectors: &BTreeMap<String, Vec<f32>>) -> Self {
        let dim = vectors.values().next().map_or(0, Vec::len);
        let count = vectors.len();
        let mut hashes = Vec::with_capacity(count);
        let mut arena = Vec::with_capacity(count * dim);
        let mut norm_sqrt = Vec::with_capacity(count);
        for (hash, vector) in vectors {
            hashes.push(hash.clone());
            norm_sqrt.push(vector.iter().map(|value| value * value).sum::<f32>().sqrt());
            arena.extend_from_slice(vector);
        }
        Self {
            hashes,
            arena,
            norm_sqrt,
            dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn hashes(&self) -> &[String] {
        &self.hashes
    }

    /// One stored vector, for the ANN builder to read without a second copy.
    pub fn point(&self, slot: usize) -> &[f32] {
        &self.arena[slot * self.dim..(slot + 1) * self.dim]
    }
}

impl VectorIndex for ExactIndex {
    fn top_candidates(&self, query: Option<&[f32]>, k: usize) -> Vec<VectorCandidate> {
        let Some(query) = query else {
            return Vec::new();
        };
        let query_norm_sqrt = query.iter().map(|value| value * value).sum::<f32>().sqrt();
        let mut scored: Vec<(f32, u32)> = (0..self.hashes.len())
            .map(|slot| {
                (
                    cosine(
                        query,
                        query_norm_sqrt,
                        self.point(slot),
                        self.norm_sqrt[slot],
                    ),
                    slot as u32,
                )
            })
            .collect();
        rank(&mut scored, &self.hashes, k)
    }

    fn len(&self) -> usize {
        self.hashes.len()
    }

    fn resident_bytes(&self) -> usize {
        self.arena.len() * std::mem::size_of::<f32>()
            + self.norm_sqrt.len() * std::mem::size_of::<f32>()
            + self.hashes.iter().map(String::len).sum::<usize>()
    }

    fn kind(&self) -> &'static str {
        "exact"
    }
}

/// What a persisted graph needs besides the two files `hnsw_rs` writes.
#[derive(serde::Serialize, serde::Deserialize)]
struct AnnMeta {
    dim: usize,
    ef_search: usize,
    hashes: Vec<String>,
    /// The basename `hnsw_rs` actually used for the graph files. It is not
    /// always the requested one: a dump that would overwrite files a live
    /// memory map still holds appends a suffix, so the sidecar — written last,
    /// under the name the caller asked for — is the pointer `load` follows.
    graph_basename: String,
}

/// Approximate nearest neighbours (HNSW). Built once from an immutable
/// snapshot, then read-only: `hnsw_rs` answers from `&self`, so concurrent
/// queries need no lock.
pub struct AnnIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    hashes: Vec<String>,
    dim: usize,
    ef_search: usize,
}

impl std::fmt::Debug for AnnIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnnIndex")
            .field("vectors", &self.hashes.len())
            .field("dim", &self.dim)
            .field("ef_search", &self.ef_search)
            .finish()
    }
}

impl AnnIndex {
    /// Builds the graph from `source` in batches, so the only extra memory is
    /// one batch of copies rather than a second copy of the whole corpus.
    pub fn build(source: &ExactIndex, ef_search: usize) -> Result<Self, String> {
        let count = source.len();
        if count == 0 || source.dim() == 0 {
            return Err("no vectors or zero dimension".to_owned());
        }
        let ef_search = ef_search.max(VECTOR_TOP_K);
        let hnsw = Hnsw::<f32, DistCosine>::new(
            HNSW_MAX_CONNECTIONS,
            count,
            HNSW_MAX_LAYER,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );
        let mut start = 0usize;
        while start < count {
            let end = (start + HNSW_INSERT_BATCH).min(count);
            let batch: Vec<(Vec<f32>, usize)> = (start..end)
                .map(|slot| (source.point(slot).to_vec(), slot))
                .collect();
            let refs: Vec<(&Vec<f32>, usize)> =
                batch.iter().map(|(data, id)| (data, *id)).collect();
            hnsw.parallel_insert(&refs);
            start = end;
        }
        Ok(Self {
            hnsw,
            hashes: source.hashes().to_vec(),
            dim: source.dim(),
            ef_search,
        })
    }

    /// Writes the graph and its sidecar under `dir` with `basename`, replacing
    /// any sidecar already there (a previous graph for the same revision is
    /// the same vectors, so superseding it is free).
    pub fn dump(&self, dir: &Path, basename: &str) -> Result<String, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let written = AnnT::file_dump(&self.hnsw, dir, basename).map_err(|e| e.to_string())?;
        let meta = AnnMeta {
            dim: self.dim,
            ef_search: self.ef_search,
            hashes: self.hashes.clone(),
            graph_basename: written.clone(),
        };
        let encoded = serde_json::to_vec(&meta).map_err(|e| e.to_string())?;
        std::fs::write(meta_path(dir, basename), encoded)
            .map_err(|e| format!("write ann meta: {e}"))?;
        Ok(written)
    }

    /// Loads a graph dumped by [`AnnIndex::dump`] for the SAME revision and
    /// parameters. Any inconsistency (missing file, different width, a
    /// truncated dump from an interrupted write) is an error the caller turns
    /// into "build it again" — never a partially usable index.
    pub fn load(dir: &Path, basename: &str) -> Result<Self, String> {
        let encoded =
            std::fs::read(meta_path(dir, basename)).map_err(|e| format!("read ann meta: {e}"))?;
        let meta: AnnMeta =
            serde_json::from_slice(&encoded).map_err(|e| format!("parse ann meta: {e}"))?;
        if meta.dim == 0 || meta.hashes.is_empty() {
            return Err("ann meta describes an empty corpus".to_owned());
        }
        // `load_hnsw<'b, 'a>(&'a mut self) -> Hnsw<'b>` is bounded `'a: 'b`, so
        // a `Hnsw<'static>` — which is what this struct stores, since the graph
        // outlives the call — requires the loader itself to be borrowed for
        // `'static`. Leaking one small `HnswIo` per loaded revision is the
        // cheapest way to satisfy that; the graph is what holds the memory.
        let io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new(dir, &meta.graph_basename)));
        let hnsw = io
            .load_hnsw::<f32, DistCosine>()
            .map_err(|e| format!("load ann graph: {e}"))?;
        if hnsw.get_nb_point() != meta.hashes.len() {
            return Err(format!(
                "ann graph holds {} points, meta describes {}",
                hnsw.get_nb_point(),
                meta.hashes.len()
            ));
        }
        Ok(Self {
            hnsw,
            hashes: meta.hashes,
            dim: meta.dim,
            ef_search: meta.ef_search.max(VECTOR_TOP_K),
        })
    }
}

/// Sidecar path for one graph basename. `pub(crate)` so the cache's tests can
/// watch whether a load rewrote it (an adoption does not).
pub(crate) fn meta_path(dir: &Path, basename: &str) -> std::path::PathBuf {
    dir.join(format!("{basename}.{META_SUFFIX}"))
}

impl VectorIndex for AnnIndex {
    fn top_candidates(&self, query: Option<&[f32]>, k: usize) -> Vec<VectorCandidate> {
        let Some(query) = query else {
            return Vec::new();
        };
        if query.len() != self.dim || query.iter().all(|value| *value == 0.0) {
            return degenerate_top_k(&self.hashes, k);
        }
        // DistCosine distances are `1 - cosine`, so the score is symmetric.
        let mut scored: Vec<(f32, u32)> = self
            .hnsw
            .search(query, k, self.ef_search)
            .iter()
            .map(|neighbour| (1.0 - neighbour.distance, neighbour.d_id as u32))
            .collect();
        rank(&mut scored, &self.hashes, k)
    }

    fn len(&self) -> usize {
        self.hashes.len()
    }

    fn resident_bytes(&self) -> usize {
        // Points plus the graph's links; both are needed to budget for the
        // resident set, and the sidecar's hashes are the smaller term.
        let points = self.hashes.len() * self.dim * std::mem::size_of::<f32>();
        let links = self.hashes.len() * HNSW_MAX_CONNECTIONS * std::mem::size_of::<u32>() * 2;
        points + links + self.hashes.iter().map(String::len).sum::<usize>()
    }

    fn kind(&self) -> &'static str {
        "hnsw"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vectors (xorshift) — no rng dependency and
    /// reproducible across runs.
    fn synthetic(count: usize, dim: usize, seed: u64) -> BTreeMap<String, Vec<f32>> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1_u64 << 53) as f32
        };
        (0..count)
            .map(|index| {
                let vector: Vec<f32> = (0..dim).map(|_| next() * 2.0 - 1.0).collect();
                (format!("hash-{index:06}"), vector)
            })
            .collect()
    }

    /// The reference implementation this module must reproduce: the shipped
    /// scan, written out naively.
    fn naive_top_k(
        vectors: &BTreeMap<String, Vec<f32>>,
        query: &[f32],
        k: usize,
    ) -> Vec<VectorCandidate> {
        let query_norm = query.iter().map(|value| value * value).sum::<f32>().sqrt();
        let mut scored: Vec<(f32, &str)> = vectors
            .iter()
            .map(|(hash, vector)| {
                let mut dot = 0.0_f32;
                if query.len() == vector.len() {
                    for (left, right) in query.iter().zip(vector.iter()) {
                        dot += left * right;
                    }
                }
                let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
                let denominator = query_norm * norm;
                let score = if query.len() != vector.len() || denominator == 0.0 {
                    0.0
                } else {
                    dot / denominator
                };
                (score, hash.as_str())
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(b.1))
        });
        scored
            .into_iter()
            .take(k)
            .map(|(score, hash)| VectorCandidate {
                text_hash: hash.to_owned(),
                score,
            })
            .collect()
    }

    fn query_for(index: usize, dim: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| ((index * 31 + i * 7) % 97) as f32 / 97.0 - 0.5)
            .collect()
    }

    #[test]
    fn exact_index_is_bit_identical_to_the_shipped_scan() {
        let vectors = synthetic(500, 24, 0x5eed);
        let index = ExactIndex::build(&vectors);
        let query: Vec<f32> = (0..24).map(|i| (i as f32).mul_add(0.31, -1.0)).collect();

        let expected = naive_top_k(&vectors, &query, VECTOR_TOP_K);
        let actual = index.top_candidates(Some(&query), VECTOR_TOP_K);
        assert_eq!(actual.len(), VECTOR_TOP_K);
        assert_eq!(actual, expected, "same hashes AND same f32 scores");
    }

    #[test]
    fn exact_index_handles_queries_the_scan_defines_as_zero() {
        let vectors = synthetic(50, 8, 7);
        let index = ExactIndex::build(&vectors);

        let mismatched = index.top_candidates(Some(&[1.0, 2.0]), 3);
        assert!(mismatched.iter().all(|candidate| candidate.score == 0.0));
        assert_eq!(
            mismatched
                .iter()
                .map(|candidate| candidate.text_hash.clone())
                .collect::<Vec<_>>(),
            vec!["hash-000000", "hash-000001", "hash-000002"]
        );

        let zero_query = vec![0.0_f32; 8];
        assert!(index
            .top_candidates(Some(&zero_query), 4)
            .iter()
            .all(|candidate| candidate.score == 0.0));

        assert!(index.top_candidates(None, 4).is_empty());
    }

    #[test]
    fn ann_mode_is_parsed_by_the_config_layer() {
        let off =
            llm_wiki_core::config::Config::parse_toml("[embedding]\nann = \"off\"\n").unwrap();
        assert_eq!(off.embedding.ann, AnnMode::Off);
        let on = llm_wiki_core::config::Config::parse_toml("[embedding]\nann = \"on\"\n").unwrap();
        assert_eq!(on.embedding.ann, AnnMode::On);
        assert!(
            llm_wiki_core::config::Config::parse_toml("[embedding]\nann = \"sometimes\"\n")
                .is_err()
        );
    }

    #[test]
    fn auto_decides_by_corpus_size_against_the_threshold() {
        let config = VectorIndexConfig {
            ann: AnnMode::Auto,
            min_sections: 200,
            ef_search: 64,
        };
        assert!(config.wants_ann(200));
        assert!(!config.wants_ann(199));
        let off = VectorIndexConfig {
            ann: AnnMode::Off,
            ..config
        };
        assert!(!off.wants_ann(100_000));
        let on = VectorIndexConfig {
            ann: AnnMode::On,
            ..config
        };
        assert!(on.wants_ann(1));
        assert!(!on.wants_ann(0));
    }

    /// The acceptance question for an approximate index: how much of the
    /// exact top-K does it actually return? Measured against the exact scan
    /// over many queries, not asserted by construction.
    #[test]
    fn ann_recalls_the_exact_top_k() {
        let vectors = synthetic(2_000, 32, 0xC0FFEE);
        let exact = ExactIndex::build(&vectors);
        let ann = AnnIndex::build(&exact, 128).unwrap();
        assert_eq!(ann.len(), exact.len());

        let mut hits = 0usize;
        let mut total = 0usize;
        let queries = 40;
        for q in 0..queries {
            let query = query_for(q, 32);
            let expected = exact.top_candidates(Some(&query), VECTOR_TOP_K);
            let actual = ann.top_candidates(Some(&query), VECTOR_TOP_K);
            assert_eq!(actual.len(), VECTOR_TOP_K);
            total += expected.len();
            hits += expected
                .iter()
                .filter(|wanted| actual.iter().any(|got| got.text_hash == wanted.text_hash))
                .count();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.95,
            "recall@{VECTOR_TOP_K} {recall:.3} < 0.95 at ef_search=128 — raise the dial"
        );
        println!("ann recall@{VECTOR_TOP_K} over {queries} queries: {recall:.3}");
    }

    #[test]
    fn ann_answers_are_ordered_and_well_formed() {
        let vectors = synthetic(500, 16, 99);
        let exact = ExactIndex::build(&vectors);
        let ann = AnnIndex::build(&exact, 128).unwrap();
        let query = query_for(7, 16);

        let first = ann.top_candidates(Some(&query), VECTOR_TOP_K);
        let second = ann.top_candidates(Some(&query), VECTOR_TOP_K);
        assert_eq!(first, second, "one graph, one answer");
        for candidate in &first {
            assert!(vectors.contains_key(&candidate.text_hash));
            assert!(
                (-1.0..=1.0).contains(&candidate.score),
                "cosine out of range: {}",
                candidate.score
            );
        }
        for window in first.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    /// Degenerate inputs must reproduce the exact scan's defined answer rather
    /// than an arbitrary neighbour set.
    #[test]
    fn ann_matches_the_exact_answer_for_degenerate_queries() {
        let vectors = synthetic(300, 8, 3);
        let exact = ExactIndex::build(&vectors);
        let ann = AnnIndex::build(&exact, 128).unwrap();

        let mismatched = vec![1.0_f32, 0.0];
        assert_eq!(
            ann.top_candidates(Some(&mismatched), 3),
            exact.top_candidates(Some(&mismatched), 3)
        );
        let zero = vec![0.0_f32; 8];
        assert_eq!(
            ann.top_candidates(Some(&zero), 3),
            exact.top_candidates(Some(&zero), 3)
        );
        assert!(ann.top_candidates(None, 3).is_empty());
    }

    /// A dumped graph must come back answering the same questions — that is
    /// the whole point of persisting it per revision.
    #[test]
    fn dumped_graph_reloads_with_the_same_answers() {
        let dir = std::env::temp_dir().join("llm-wiki-ann-dump-test");
        let _ = std::fs::remove_dir_all(&dir);
        let vectors = synthetic(600, 24, 0xBEEF);
        let exact = ExactIndex::build(&vectors);
        let ann = AnnIndex::build(&exact, 128).unwrap();
        let written = ann.dump(&dir, "ann-test").unwrap();
        assert_eq!(
            written, "ann-test",
            "a fresh directory takes the name as given"
        );

        let loaded = AnnIndex::load(&dir, "ann-test").expect("reload the graph");
        assert_eq!(loaded.len(), ann.len());
        assert_eq!(loaded.dim, ann.dim);
        for q in 0..5 {
            let query = query_for(q, 24);
            assert_eq!(
                ann.top_candidates(Some(&query), VECTOR_TOP_K),
                loaded.top_candidates(Some(&query), VECTOR_TOP_K),
                "the persisted graph must answer identically"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Corrupt or mismatched persistence is an error to rebuild from, never a
    /// silently wrong index.
    #[test]
    fn loading_garbage_is_an_error_not_a_partial_index() {
        let dir = std::env::temp_dir().join("llm-wiki-ann-bad-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(AnnIndex::load(&dir, "never-dumped").is_err());

        std::fs::write(meta_path(&dir, "garbage"), b"{ not json").unwrap();
        assert!(AnnIndex::load(&dir, "garbage").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
