//! LW-001 acceptance baseline (ignored by default): the hybrid-retrieval read
//! path at N = 1k / 10k / 100k sections — p50/p95 latency, exact SQL
//! statement counts (rusqlite `hooks`), and the vector bytes the read path
//! loads into memory. Also times a harness-side reimplementation of the
//! pre-LW-001 per-hash lookup loop, so the round-trip reduction is measured,
//! not asserted.
//!
//! Run manually (needs `--release` to be meaningful):
//!
//! ```text
//! cargo test --release -p llm-wiki-evals --test embed_baseline -- --ignored --nocapture
//! ```
//!
//! Thresholds are deliberately NOT asserted here: the audit's rule is
//! "measure the baseline first, then agree on targets".

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use llm_wiki_compiler::{embedding_coverage, top_cosine_candidates};
use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::WikiPageId;
use llm_wiki_search::active_context_sections;
use llm_wiki_storage::{
    activate_build, default_tokenizer, insert_section_embeddings, open, persist_generation,
    rebuild_search_index, start_build, BuildDraft, WikiPageRecord,
};

/// Embedding dimension of the synthetic corpus. 768 is a common model width
/// (the read path loads every vector, so this is the memory lever).
const DIM: usize = 768;
/// Talking point: the AUDIT's target sizes.
const SIZES: [usize; 3] = [1_000, 10_000, 100_000];
/// Measurements per size (after a warm-up round).
const REPS: usize = 20;
/// Sections per generated page.
const SECTIONS_PER_PAGE: usize = 1_000;

/// SQL statement counter fed by the rusqlite trace hook (a plain fn
/// pointer — the hook API takes no captures).
///
/// PROCESS-GLOBAL: this file must keep exactly ONE `#[test]` — a second
/// parallel test would interleave into the same counter.
static QUERIES: AtomicUsize = AtomicUsize::new(0);

fn count_sql(_sql: &str) {
    QUERIES.fetch_add(1, Ordering::Relaxed);
}

fn workspace(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-embed-baseline-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join(".llm-wiki")).unwrap();
    dir
}

/// Builds a published generation with ~`target` sections and stores one
/// embedding per section hash. Returns the workspace dir (for cleanup), the
/// DB path, and the ACTIVE generation's section hashes.
fn corpus(tag: &str, target: usize) -> (std::path::PathBuf, std::path::PathBuf, Vec<String>) {
    let dir = workspace(tag);
    let db_path = dir.join(".llm-wiki").join("state.db");
    let mut conn = open(&db_path).unwrap();
    let build_id = start_build(&mut conn, &BuildDraft::default()).unwrap();

    let pages: Vec<WikiPageRecord> = (0..target.div_ceil(SECTIONS_PER_PAGE))
        .map(|page_index| {
            let mut content = format!("# Page {page_index}\n\n");
            for section_index in 0..SECTIONS_PER_PAGE {
                content.push_str(&format!(
                    "## Section {page_index}-{section_index}\n\nSynthetic body \
                     {page_index}-{section_index} carrying token t{page_index}x{section_index}qz.\n\n"
                ));
            }
            WikiPageRecord {
                page_id: WikiPageId::generate(),
                slug: format!("page-{page_index}"),
                title: format!("Page {page_index}"),
                category: "concepts".into(),
                language: "en".into(),
                body_hash: sha256_hex(content.as_bytes()),
                content,
                knowledge_refs: Vec::new(),
                citations: Vec::new(),
                links: Vec::new(),
            }
        })
        .collect();
    persist_generation(&mut conn, &build_id, &pages).unwrap();
    activate_build(&mut conn, &build_id).unwrap();
    rebuild_search_index(&conn, &build_id, default_tokenizer()).unwrap();

    let sections = active_context_sections(&conn).unwrap();
    // Bounded write chunks — and crucially each chunk's vectors are built IN
    // PLACE instead of collecting all of them first: a single 293 MiB buffer
    // would dominate the peak working set and make the retrieval footprint
    // unattributable (the very question this baseline answers).
    for chunk in sections.chunks(5_000) {
        let pairs: Vec<(String, Vec<f32>)> = chunk
            .iter()
            .map(|section| {
                (
                    section.text_hash.clone(),
                    synthetic_vector(&section.text_hash),
                )
            })
            .collect();
        insert_section_embeddings(&mut conn, MODEL, &pairs).unwrap();
    }
    let hashes: Vec<String> = sections.iter().map(|s| s.text_hash.clone()).collect();
    drop(conn);
    (dir, db_path, hashes)
}

/// A deterministic unit vector derived from the section hash (content is
/// irrelevant to the timing; only dim and count matter).
fn synthetic_vector(seed: &str) -> Vec<f32> {
    let digest = sha256_hex(seed.as_bytes());
    let mut vector = vec![0.0_f32; DIM];
    for (index, byte) in digest.bytes().cycle().take(DIM).enumerate() {
        vector[index] = f32::from(byte) - 128.0;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

const MODEL: &str = "baseline-embedding-model";

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let index = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[index]
}

/// Times one closure `REPS` times and returns (p50, p95, SQL statements per
/// run). The warm-up round is excluded from BOTH latency and statement
/// counts.
fn measure(reps: usize, mut run: impl FnMut()) -> (Duration, Duration, usize) {
    run(); // warm-up
    QUERIES.store(0, Ordering::Relaxed);
    let mut samples: Vec<Duration> = (0..reps)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed()
        })
        .collect();
    let stmts_per_run = QUERIES.load(Ordering::Relaxed) / reps.max(1);
    samples.sort_unstable();
    (
        percentile(&samples, 0.50),
        percentile(&samples, 0.95),
        stmts_per_run,
    )
}

/// PR-5 evidence (LW-001 中期): the same vectors, in memory, scanned exactly
/// versus indexed approximately. Build cost, query latency, resident size and
/// — the number that decides whether the default should be `auto` — how much
/// of the exact top-K the index actually returns.
fn index_comparison(n: usize, hashes: &[String]) {
    use llm_wiki_compiler::{AnnIndex, ExactIndex, VectorIndex};

    const SCAN_REPS: usize = 5;
    const RECALL_QUERIES: usize = 20;
    const K: usize = 8;
    const EF_SEARCH: usize = 128;

    let vectors: std::collections::BTreeMap<String, Vec<f32>> = hashes
        .iter()
        .map(|hash| (hash.clone(), synthetic_vector(hash)))
        .collect();

    let started = Instant::now();
    let exact = ExactIndex::build(&vectors);
    let exact_build = started.elapsed();
    drop(vectors);
    let started = Instant::now();
    let ann = match AnnIndex::build(&exact, EF_SEARCH) {
        Ok(index) => index,
        Err(err) => {
            println!("    ↳ index: HNSW unavailable ({err})");
            return;
        }
    };
    let ann_build = started.elapsed();

    // The restart path: a graph persisted once is adopted instead of rebuilt.
    // This is what makes the ANN safe to enable — a query never waits.
    let dir = std::env::temp_dir().join(format!("llm-wiki-pr5-{n}"));
    let started = Instant::now();
    let dumped = ann.dump(&dir, "graph").is_ok();
    let dump_time = started.elapsed();
    let started = Instant::now();
    let reload = if dumped {
        AnnIndex::load(&dir, "graph").ok()
    } else {
        None
    };
    let load_time = started.elapsed();

    let query = synthetic_vector("pr5 query vector");
    let (exact_p50, exact_p95, _) = measure(SCAN_REPS, || {
        let _ = exact.top_candidates(Some(&query), K);
    });
    let (ann_p50, ann_p95, _) = measure(SCAN_REPS, || {
        let _ = ann.top_candidates(Some(&query), K);
    });
    let reloaded_p50 = reload.as_ref().map(|index| {
        measure(SCAN_REPS, || {
            let _ = index.top_candidates(Some(&query), K);
        })
        .0
    });

    let mut hits = 0usize;
    let mut total = 0usize;
    for index in 0..RECALL_QUERIES {
        let probe = synthetic_vector(&format!("pr5 probe {index}"));
        let wanted = exact.top_candidates(Some(&probe), K);
        let found = ann.top_candidates(Some(&probe), K);
        total += wanted.len();
        hits += wanted
            .iter()
            .filter(|want| found.iter().any(|got| got.text_hash == want.text_hash))
            .count();
    }

    println!(
        "    ↳ index: exact build {:.0}ms / p50 {:.1}ms / p95 {:.1}ms / {:.0} MiB  vs  \
         hnsw build {:.0}ms / p50 {:.1}ms / p95 {:.1}ms / {:.0} MiB (ef={EF_SEARCH})  →  \
         {:.1}× faster, recall@{K} {:.3}",
        exact_build.as_secs_f64() * 1e3,
        exact_p50.as_secs_f64() * 1e3,
        exact_p95.as_secs_f64() * 1e3,
        exact.resident_bytes() as f64 / (1024.0 * 1024.0),
        ann_build.as_secs_f64() * 1e3,
        ann_p50.as_secs_f64() * 1e3,
        ann_p95.as_secs_f64() * 1e3,
        ann.resident_bytes() as f64 / (1024.0 * 1024.0),
        exact_p50.as_secs_f64() / ann_p50.as_secs_f64().max(1e-9),
        hits as f64 / total.max(1) as f64,
    );
    match reloaded_p50 {
        Some(p50) => println!(
            "    ↳ persisted: dump {:.0}ms / reload {:.0}ms / reloaded p50 {:.1}ms  →  \
             a restart skips the {:.1}s build",
            dump_time.as_secs_f64() * 1e3,
            load_time.as_secs_f64() * 1e3,
            p50.as_secs_f64() * 1e3,
            ann_build.as_secs_f64(),
        ),
        None => println!("    ↳ persisted: unavailable (dump failed)"),
    }
    let _ = n;
}

/// The pre-LW-001 lookup: one point query per hash (harness-side
/// reimplementation, kept as the "before" number).
///
/// The DB call inside the loop is INTENTIONAL and load-bearing: it is the
/// exact per-hash shape LW-001 replaced, and reproducing its statement count
/// (N vs 1 batched) is the whole point of this comparison. Not a pattern to
/// copy into product code.
fn legacy_per_hash_lookup(conn: &llm_wiki_storage::Connection, hashes: &[String]) -> usize {
    let mut stmt = conn
        .prepare("SELECT text_hash, embedding FROM section_embeddings WHERE model = ?1 AND text_hash = ?2")
        .unwrap();
    let mut found = 0usize;
    for hash in hashes {
        let rows = stmt
            .query_map((MODEL, hash.as_str()), |row| row.get::<_, Vec<u8>>(1))
            .unwrap();
        for row in rows {
            let _ = row.unwrap();
            found += 1;
        }
    }
    found
}

/// Peak working set of THIS process, in bytes (Windows only; `None` when the
/// query is unavailable).
fn peak_working_set() -> Option<u64> {
    let pid = std::process::id();
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!("(Get-Process -Id {pid}).PeakWorkingSet64"),
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

/// Length note: ~75 lines — one linear report flow (header → per-size
/// corpus+measure → lookup comparison); the per-size work is already
/// factored into `corpus`/`measure`/`legacy_per_hash_lookup`.
#[test]
#[ignore = "manual LW-001 baseline; run with --release -- --ignored --nocapture"]
fn embed_retrieval_baseline() {
    println!("\nLW-001 read-path baseline   dim={DIM}   reps={REPS}   (wall-clock, release)\n");
    println!(
        "| N sections | coverage p50/p95 | top-K p50/p95 | SQL stmts coverage/top-K | \
         vectors loaded (derived) | peak WS (process high-water) |"
    );
    println!("|---|---|---|---|---|---|");

    for target in SIZES {
        let (dir, db_path, hashes) = corpus(&format!("n{target}"), target);
        let n = hashes.len();
        let mut conn = open(&db_path).unwrap();
        conn.trace(Some(count_sql));

        let query = synthetic_vector("baseline query vector");

        // Coverage probe + the full top-K scan, each measured separately.
        let (coverage_p50, coverage_p95, coverage_stmts) = measure(REPS, || {
            assert!(embedding_coverage(&conn, MODEL).unwrap());
        });

        let (topk_p50, topk_p95, topk_stmts) = measure(REPS, || {
            let candidates = top_cosine_candidates(&conn, MODEL, Some(&query)).unwrap();
            assert!(!candidates.is_empty());
        });

        let loaded_bytes = n as u64 * DIM as u64 * 4;

        println!(
            "| {n} | {:.1}ms / {:.1}ms | {:.1}ms / {:.1}ms | {} / {} | {:.0} MiB | {} |",
            coverage_p50.as_secs_f64() * 1e3,
            coverage_p95.as_secs_f64() * 1e3,
            topk_p50.as_secs_f64() * 1e3,
            topk_p95.as_secs_f64() * 1e3,
            coverage_stmts,
            topk_stmts,
            loaded_bytes as f64 / (1024.0 * 1024.0),
            peak_working_set()
                .map(|bytes| format!("{:.0} MiB", bytes as f64 / (1024.0 * 1024.0)))
                .unwrap_or_else(|| "n/a".to_owned()),
        );

        if n >= 10_000 {
            // Fair lookup-only comparison: the shipped batched read vs the
            // pre-LW-001 per-hash loop — BOTH do nothing but the lookup, so
            // the statement count and wall clock are directly comparable.
            QUERIES.store(0, Ordering::Relaxed);
            let started = Instant::now();
            let batched = llm_wiki_storage::section_embeddings_by_hash(&conn, MODEL, &hashes)
                .unwrap()
                .len();
            let batched_time = started.elapsed();
            let batched_stmts = QUERIES.load(Ordering::Relaxed);

            QUERIES.store(0, Ordering::Relaxed);
            let started = Instant::now();
            let legacy = legacy_per_hash_lookup(&conn, &hashes);
            let legacy_time = started.elapsed();
            let legacy_stmts = QUERIES.load(Ordering::Relaxed);

            println!(
                "    ↳ lookup-only: batched {batched_stmts} stmts / {:.1}ms ({batched} rows)  vs  \
                 per-hash {legacy_stmts} stmts / {:.1}ms ({legacy} rows)  →  {:.0}× fewer stmts, \
                 {:.1}× faster",
                batched_time.as_secs_f64() * 1e3,
                legacy_time.as_secs_f64() * 1e3,
                legacy_stmts as f64 / batched_stmts.max(1) as f64,
                legacy_time.as_secs_f64() / batched_time.as_secs_f64().max(1e-9),
            );
        }

        // PR-5 evidence: once the vectors are in memory, what does an
        // approximate index actually buy — and what does it cost in recall?
        index_comparison(n, &hashes);

        // Temp corpora reach hundreds of MB; reclaim each size before building
        // the next one. The connection must be closed first or the delete of
        // state.db fails on Windows.
        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
