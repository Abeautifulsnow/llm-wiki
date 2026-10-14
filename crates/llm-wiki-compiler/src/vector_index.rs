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
//! - [`HnswIndex`] — an approximate index for corpora where the exact scan is
//!   the measured bottleneck (HNSW via `instant-distance`, built once per
//!   generation/model/embeddings revision, never mutated after).
//!
//! Degradation contract (LW-001 acceptance): a missing, unbuildable or
//! unsearchable index NEVER fabricates a hit and never silently drops one. The
//! ANN path is chosen only when it can actually be built, every build failure
//! falls back to [`ExactIndex`] with a warning, and a degenerate query (a zero
//! vector, or a dimension that does not match the corpus) reproduces the exact
//! scan's defined answer instead of an arbitrary neighbour set.

use std::collections::BTreeMap;
use std::sync::Mutex;

use llm_wiki_search::VectorCandidate;

use crate::vector_cache::VECTOR_TOP_K;

/// The search-path mode, defined with the rest of the config surface so the
/// accepted values live in exactly one place.
pub use llm_wiki_core::config::AnnMode;

/// Resolved index policy: whether to build an ANN, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorIndexConfig {
    pub ann: AnnMode,
    /// `Auto` builds an ANN at or above this many stored vectors. The exact
    /// scan is faster than an index build below roughly this size, and tiny
    /// corpora should never pay ANN's recall risk for nothing.
    pub min_sections: usize,
    /// ANN search breadth: the recall/latency dial. Higher finds more of the
    /// true neighbours and costs more per query.
    pub ef_search: usize,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            ann: AnnMode::Off,
            min_sections: 20_000,
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

    fn wants_ann(&self, sections: usize) -> bool {
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

/// Builds the index for `vectors` under `config`. Any ANN failure degrades to
/// the exact scan: an index that cannot be built must not fail the query, and
/// must not pretend to be approximate-but-fine either.
pub fn build_index(
    vectors: &BTreeMap<String, Vec<f32>>,
    config: &VectorIndexConfig,
) -> Box<dyn VectorIndex> {
    if config.wants_ann(vectors.len()) {
        match HnswIndex::build(vectors, config.ef_search) {
            Ok(index) => {
                tracing::info!(
                    vectors = index.len(),
                    ef_search = config.ef_search,
                    resident_mib = index.resident_bytes() / (1024 * 1024),
                    "vector index built (approximate)"
                );
                return Box::new(index);
            }
            Err(err) => tracing::warn!(
                error = %err,
                vectors = vectors.len(),
                "vector index build failed; falling back to the exact scan"
            ),
        }
    }
    Box::new(ExactIndex::build(vectors))
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

    /// Cosine of `query` against slot `slot`, or 0.0 when the widths differ —
    /// two models' vectors must never mix, and a width mismatch scores 0 for
    /// every candidate exactly as the shipped scan did.
    fn score(&self, query: &[f32], query_norm_sqrt: f32, slot: usize) -> f32 {
        if query.len() != self.dim {
            return 0.0;
        }
        let start = slot * self.dim;
        let mut dot = 0.0_f32;
        for (offset, value) in query.iter().enumerate() {
            dot += value * self.arena[start + offset];
        }
        let denominator = query_norm_sqrt * self.norm_sqrt[slot];
        if denominator == 0.0 {
            0.0
        } else {
            dot / denominator
        }
    }
}

impl VectorIndex for ExactIndex {
    fn top_candidates(&self, query: Option<&[f32]>, k: usize) -> Vec<VectorCandidate> {
        let Some(query) = query else {
            return Vec::new();
        };
        let query_norm_sqrt = query.iter().map(|value| value * value).sum::<f32>().sqrt();
        let mut scored: Vec<(f32, u32)> = (0..self.hashes.len())
            .map(|slot| (self.score(query, query_norm_sqrt, slot), slot as u32))
            .collect();
        // Total order: score descending, ties broken by text hash — the same
        // order the pre-PR-5 scan produced, so results stay comparable.
        scored.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.hashes[a.1 as usize].cmp(&self.hashes[b.1 as usize]))
        });
        scored
            .into_iter()
            .take(k)
            .map(|(score, slot)| VectorCandidate {
                text_hash: self.hashes[slot as usize].clone(),
                score,
            })
            .collect()
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

/// One point of the ANN graph: a unit-normalised vector, so `distance` is
/// `1 - cosine` and the closest neighbours are exactly the highest cosines.
#[derive(Debug, Clone)]
pub struct UnitPoint(Vec<f32>);

impl instant_distance::Point for UnitPoint {
    fn distance(&self, other: &Self) -> f32 {
        let mut dot = 0.0_f32;
        for (left, right) in self.0.iter().zip(other.0.iter()) {
            dot += left * right;
        }
        1.0 - dot
    }
}

/// Approximate nearest neighbours (HNSW). Built once from an immutable
/// snapshot; `search` is the only mutating state and is reused under a lock.
pub struct HnswIndex {
    map: instant_distance::HnswMap<UnitPoint, u32>,
    /// Scratch space the graph search needs; `Search` is sized to the graph,
    /// so it is allocated once and reused rather than per query.
    search: Mutex<instant_distance::Search>,
    hashes: Vec<String>,
    dim: usize,
    ef_search: usize,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("vectors", &self.hashes.len())
            .field("dim", &self.dim)
            .field("ef_search", &self.ef_search)
            .finish()
    }
}

impl HnswIndex {
    /// `Err` for a corpus the index cannot represent (no vectors, or an empty
    /// dimension); the caller falls back to the exact scan.
    pub fn build(vectors: &BTreeMap<String, Vec<f32>>, ef_search: usize) -> Result<Self, String> {
        let dim = vectors.values().next().map_or(0, Vec::len);
        if vectors.is_empty() || dim == 0 {
            return Err("no vectors or zero dimension".to_owned());
        }
        let mut hashes = Vec::with_capacity(vectors.len());
        let mut points = Vec::with_capacity(vectors.len());
        for (hash, vector) in vectors {
            hashes.push(hash.clone());
            points.push(UnitPoint(normalise(vector)));
        }
        let values: Vec<u32> = (0..hashes.len() as u32).collect();
        let map = instant_distance::Hnsw::<UnitPoint>::builder()
            .ef_construction(EF_CONSTRUCTION)
            .ef_search(ef_search.max(VECTOR_TOP_K))
            .seed(HNSW_SEED)
            .build(points, values);
        Ok(Self {
            map,
            search: Mutex::new(instant_distance::Search::default()),
            hashes,
            dim,
            ef_search: ef_search.max(VECTOR_TOP_K),
        })
    }

    /// The exact scan's answer for a query the graph cannot score: a zero
    /// vector or a mismatched width makes every cosine 0, so the top-K is the
    /// lowest hashes. Reproducing that keeps ANN from inventing a different
    /// answer for inputs the exact path defines.
    fn degenerate(&self, k: usize) -> Vec<VectorCandidate> {
        let mut order: Vec<u32> = (0..self.hashes.len() as u32).collect();
        order.sort_unstable_by(|a, b| self.hashes[*a as usize].cmp(&self.hashes[*b as usize]));
        order
            .into_iter()
            .take(k)
            .map(|slot| VectorCandidate {
                text_hash: self.hashes[slot as usize].clone(),
                score: 0.0,
            })
            .collect()
    }
}

/// HNSW build quality. The paper's `M` (links per node) is fixed at 32 inside
/// `instant-distance`; this only sets how hard the builder searches while
/// linking, and [`HNSW_LINKS_M`] mirrors that fixed value for size estimates.
const EF_CONSTRUCTION: usize = 200;
const HNSW_LINKS_M: usize = 32;
/// Fixed graph seed. The crate defaults to an ENTROPY seed, which would make
/// the same corpus produce a slightly different graph (and therefore slightly
/// different neighbours) after every rebuild — the exact scan and every other
/// ranking surface in this project are reproducible, so the index is too:
/// same vectors + same parameters ⇒ same graph ⇒ same answers.
const HNSW_SEED: u64 = 0x11_5EED;

fn normalise(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|value| value / norm).collect()
}

impl VectorIndex for HnswIndex {
    fn top_candidates(&self, query: Option<&[f32]>, k: usize) -> Vec<VectorCandidate> {
        let Some(query) = query else {
            return Vec::new();
        };
        if query.len() != self.dim || query.iter().all(|value| *value == 0.0) {
            return self.degenerate(k);
        }
        let query_point = UnitPoint(normalise(query));
        let mut search = self.search.lock().expect("hnsw search scratch");
        let mut found: Vec<(f32, u32)> = self
            .map
            .search(&query_point, &mut search)
            .map(|item| (1.0 - item.distance, *item.value))
            .collect();
        drop(search);
        found.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.hashes[a.1 as usize].cmp(&self.hashes[b.1 as usize]))
        });
        found
            .into_iter()
            .take(k)
            .map(|(score, slot)| VectorCandidate {
                text_hash: self.hashes[slot as usize].clone(),
                score,
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.hashes.len()
    }

    fn resident_bytes(&self) -> usize {
        // Points plus the graph's links and level tables; the link count is
        // the dominant term and is what an operator needs to budget for.
        let points = self.hashes.len() * self.dim * std::mem::size_of::<f32>();
        let links = self.hashes.len() * HNSW_LINKS_M * std::mem::size_of::<u32>() * 2;
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

    /// The reference implementation this file must reproduce: the shipped
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

        // Width mismatch: every cosine is 0, so the order is by hash.
        let mismatched = index.top_candidates(Some(&[1.0, 2.0]), 3);
        assert!(mismatched.iter().all(|candidate| candidate.score == 0.0));
        assert_eq!(
            mismatched
                .iter()
                .map(|candidate| candidate.text_hash.clone())
                .collect::<Vec<_>>(),
            vec!["hash-000000", "hash-000001", "hash-000002"]
        );

        // A zero query divides by zero in the cosine: also 0 for everyone.
        let zero_query = vec![0.0_f32; 8];
        assert!(index
            .top_candidates(Some(&zero_query), 4)
            .iter()
            .all(|candidate| candidate.score == 0.0));

        // No query vector at all is an empty result, not a fabricated one.
        assert!(index.top_candidates(None, 4).is_empty());
    }

    #[test]
    fn still_exact_when_the_corpus_is_smaller_than_the_ann_threshold() {
        let vectors = synthetic(64, 8, 11);
        let config = VectorIndexConfig::default();
        let index = build_index(&vectors, &config);
        assert_eq!(index.kind(), "exact", "auto stays exact below min_sections");
        assert_eq!(index.len(), 64);
    }

    #[test]
    fn ann_mode_is_parsed_by_the_config_layer() {
        // The accepted spellings live in core (serde), so this pins the
        // contract the compiler relies on rather than re-listing them.
        let off =
            llm_wiki_core::config::Config::parse_toml("[embedding]\nann = \"off\"\n").unwrap();
        assert_eq!(off.embedding.ann, AnnMode::Off);
        let on = llm_wiki_core::config::Config::parse_toml("[embedding]\nann = \"on\"\n").unwrap();
        assert_eq!(on.embedding.ann, AnnMode::On);
        let default = llm_wiki_core::config::Config::default();
        assert_eq!(default.embedding.ann, AnnMode::Off);
        assert!(
            llm_wiki_core::config::Config::parse_toml("[embedding]\nann = \"sometimes\"\n")
                .is_err()
        );
    }

    #[test]
    fn auto_builds_an_ann_once_the_corpus_reaches_the_threshold() {
        let vectors = synthetic(200, 8, 13);
        let config = VectorIndexConfig {
            ann: AnnMode::Auto,
            min_sections: 200,
            ef_search: 64,
        };
        assert_eq!(build_index(&vectors, &config).kind(), "hnsw");

        let off = VectorIndexConfig {
            ann: AnnMode::Off,
            ..config
        };
        assert_eq!(build_index(&vectors, &off).kind(), "exact");
    }

    /// The acceptance question for an approximate index: how much of the
    /// exact top-K does it actually return? Measured against the exact scan
    /// over many queries, not asserted by construction.
    #[test]
    fn ann_recalls_the_exact_top_k() {
        let vectors = synthetic(2_000, 32, 0xC0FFEE);
        let exact = ExactIndex::build(&vectors);
        let ann = HnswIndex::build(&vectors, 128).unwrap();
        assert_eq!(ann.len(), exact.len());

        let mut hits = 0usize;
        let mut total = 0usize;
        let queries = 40;
        for q in 0..queries {
            let query: Vec<f32> = (0..32)
                .map(|i| ((q * 31 + i * 7) % 97) as f32 / 97.0 - 0.5)
                .collect();
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

    /// Approximate is still DETERMINISTIC here: the graph is seeded at build
    /// time, so the same query returns the same neighbours — and so does a
    /// REBUILD of the same corpus, which is what makes a retrieval result
    /// reproducible across server restarts and backfills.
    #[test]
    fn ann_results_are_deterministic_and_well_formed() {
        let vectors = synthetic(500, 16, 99);
        let ann = HnswIndex::build(&vectors, 128).unwrap();
        let query: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();

        let first = ann.top_candidates(Some(&query), VECTOR_TOP_K);
        let second = ann.top_candidates(Some(&query), VECTOR_TOP_K);
        assert_eq!(first, second, "same query, same answer");

        let rebuilt = HnswIndex::build(&vectors, 128).unwrap();
        assert_eq!(
            first,
            rebuilt.top_candidates(Some(&query), VECTOR_TOP_K),
            "a rebuild of the same corpus must answer identically"
        );

        for candidate in &first {
            assert!(vectors.contains_key(&candidate.text_hash));
            assert!(
                (-1.0..=1.0).contains(&candidate.score),
                "cosine out of range: {}",
                candidate.score
            );
        }
        // Sorted by score descending.
        for window in first.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    /// Degenerate inputs must reproduce the exact scan's defined answer
    /// rather than an arbitrary neighbour set.
    #[test]
    fn ann_matches_the_exact_answer_for_degenerate_queries() {
        let vectors = synthetic(300, 8, 3);
        let exact = ExactIndex::build(&vectors);
        let ann = HnswIndex::build(&vectors, 128).unwrap();

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
}
