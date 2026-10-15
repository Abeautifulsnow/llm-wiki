//! LW-002 evidence: what `embedding.max_concurrency` actually buys.
//!
//! The audit's acceptance for the bounded-parallel backfill asks for a
//! measured throughput gain on top of the behavioural tests
//! (`llm-wiki-compiler/tests/vector.rs` proves overlap happens and stays under
//! the cap; this measures the wall clock that overlap is worth). A provider
//! with a fixed per-request latency makes the arithmetic checkable: with
//! `requests` batches of `latency` each, a run cannot finish faster than
//! `requests * latency / concurrency`, and the gap between that floor and the
//! measured time is the part that is still serial — the single-writer SQLite
//! path, which is a design choice (no lock contention, no busy/locked churn)
//! rather than an oversight.
//!
//! Manual, release-only: `cargo test --release -p llm-wiki-evals --test
//! embed_concurrency -- --ignored --nocapture`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use llm_wiki_compiler::{run_embed, EmbedOptions};
use llm_wiki_core::config::Config;
use llm_wiki_llm::{EmbeddingProvider, LlmError};

/// Sections in the fixture (the page's intro is one more — see `sections()`).
const SECTIONS: usize = 2_000;
/// Sections per request. 2000 / 8 = 250 requests.
const BATCH_SIZE: usize = 8;
/// Simulated provider round-trip per request.
const LATENCY: Duration = Duration::from_millis(20);
/// Concurrency levels to compare against the serial baseline.
const LEVELS: [usize; 3] = [1, 4, 8];

/// The providers are deterministic: same text → same vector, so the stored
/// vectors can be recomputed and compared after a concurrent run.
fn vector_for(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; 64];
    for bigram in text.as_bytes().windows(2) {
        let index = (bigram[0] as usize * 31 + bigram[1] as usize) % vector.len();
        vector[index] += 1.0;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

/// A provider that spends `LATENCY` per request and records how many requests
/// it served and how many ran at once.
#[derive(Clone, Default)]
struct SlowEmbeddings {
    requests: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

impl SlowEmbeddings {
    fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for SlowEmbeddings {
    async fn embed(&self, _model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.requests.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(LATENCY).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(texts.iter().map(|text| vector_for(text)).collect())
    }
}

/// A published generation with `headings + 1` context sections, built the same
/// way the integration tests build theirs.
fn workspace(tag: &str, headings: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-embed-throughput-{tag}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    std::fs::create_dir_all(dir.join(".llm-wiki")).unwrap();
    std::fs::write(
        dir.join(".llm-wiki").join("config.toml"),
        "[project]\nname = \"fixture\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n\
         [llm]\nmodel = \"fake-throughput\"\n",
    )
    .unwrap();
    let mut conn = llm_wiki_storage::open(&dir.join(".llm-wiki").join("state.db")).unwrap();
    let build_id =
        llm_wiki_storage::start_build(&mut conn, &llm_wiki_storage::BuildDraft::default()).unwrap();
    let mut content = String::from("# Throughput Guide\n\nIntro body.\n\n");
    for index in 0..headings {
        content.push_str(&format!(
            "## Part {index}\n\nBody {index} carries marker tp{index}qz.\n\n"
        ));
    }
    let page = llm_wiki_storage::WikiPageRecord {
        page_id: llm_wiki_core::ids::WikiPageId::generate(),
        slug: "throughput-guide".into(),
        title: "Throughput Guide".into(),
        category: "concepts".into(),
        language: "en".into(),
        body_hash: llm_wiki_core::hash::sha256_hex(content.as_bytes()),
        content,
        knowledge_refs: Vec::new(),
        citations: Vec::new(),
        links: Vec::new(),
    };
    llm_wiki_storage::persist_generation(&mut conn, &build_id, &[page]).unwrap();
    llm_wiki_storage::activate_build_with_search_index(
        &mut conn,
        &build_id,
        llm_wiki_storage::default_tokenizer(),
    )
    .unwrap();
    std::fs::create_dir_all(dir.join("wiki")).unwrap();
    llm_wiki_compiler::write_current_pointer(
        &llm_wiki_compiler::PublishPaths::new(&dir.join("wiki")),
        &build_id,
    )
    .unwrap();
    dir
}

struct Measurement {
    concurrency: usize,
    sections: usize,
    requests: usize,
    peak: usize,
    elapsed: Duration,
    /// Every stored vector equals the one its own section's text produces —
    /// the "no mis-binding under concurrency" check.
    all_vectors_match: bool,
}

/// Runs one backfill at `concurrency` under a fresh model, so it always starts
/// from zero coverage (the store is keyed by hash AND model).
async fn measure(workspace: &Path, concurrency: usize) -> Measurement {
    use llm_wiki_search::active_context_sections;

    let config = Config::load(workspace).unwrap();
    let model = format!("throughput-c{concurrency}");
    let provider = SlowEmbeddings::default();
    let started = Instant::now();
    let report = run_embed(
        workspace,
        &config,
        Arc::new(provider.clone()),
        &model,
        EmbedOptions {
            batch_size: BATCH_SIZE,
            max_concurrency: concurrency,
            ..EmbedOptions::serial(BATCH_SIZE)
        },
    )
    .await
    .unwrap()
    .expect("a generation is published");
    let elapsed = started.elapsed();

    // Read the vectors back and re-derive them from each section's own text:
    // a batch landing under another batch's hash would show up here.
    let conn = llm_wiki_storage::open(&workspace.join(".llm-wiki").join("state.db")).unwrap();
    let sections = active_context_sections(&conn).unwrap();
    let hashes: Vec<String> = sections.iter().map(|s| s.text_hash.clone()).collect();
    let stored = llm_wiki_storage::section_embeddings_by_hash(&conn, &model, &hashes).unwrap();
    let all_vectors_match = sections.iter().all(|section| {
        let text = llm_wiki_search::context_section_text(
            &section.title,
            &section.heading_path,
            &section.body,
        );
        stored.get(&section.text_hash) == Some(&vector_for(&text))
    });

    assert_eq!(
        report.embedded + report.covered_before,
        sections.len(),
        "every section must be accounted for"
    );
    Measurement {
        concurrency,
        sections: report.total_sections,
        requests: provider.requests(),
        peak: provider.peak(),
        elapsed,
        all_vectors_match,
    }
}

/// Length note: ~80 lines — one linear report flow (header → per-level measure
/// → table → verdict); the per-level work lives in `measure`.
#[test]
#[ignore = "manual LW-002 measurement; run with --release -- --ignored --nocapture"]
fn embed_backfill_throughput_by_concurrency() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let workspace = workspace("main", SECTIONS);

    println!(
        "\nLW-002 backfill throughput   sections≈{SECTIONS}   batch={BATCH_SIZE}   \
              provider latency={}ms   (wall clock, release)",
        LATENCY.as_millis()
    );

    let mut measurements = Vec::new();
    for concurrency in LEVELS {
        let measurement = runtime.block_on(measure(&workspace, concurrency));
        println!(
            "    ↳ concurrency {concurrency}: {:.2}s for {} requests (peak in flight {}), \
             vectors correct: {}",
            measurement.elapsed.as_secs_f64(),
            measurement.requests,
            measurement.peak,
            measurement.all_vectors_match
        );
        measurements.push(measurement);
    }

    let serial = &measurements[0];
    println!(
        "\n| concurrency | requests | peak in flight | wall clock | speedup vs serial | \
         ideal floor |"
    );
    println!("|---|---|---|---|---|---|");
    for measurement in &measurements {
        let floor = serial.requests as f64 * LATENCY.as_secs_f64() / measurement.concurrency as f64;
        println!(
            "| {} | {} | {} | {:.2}s | {:.2}× | {:.2}s |",
            measurement.concurrency,
            measurement.requests,
            measurement.peak,
            measurement.elapsed.as_secs_f64(),
            serial.elapsed.as_secs_f64() / measurement.elapsed.as_secs_f64(),
            floor,
        );
    }

    // Same work, same spend: concurrency must not change the request count.
    for measurement in &measurements {
        assert_eq!(
            measurement.requests, serial.requests,
            "concurrency {} issued a different number of provider requests",
            measurement.concurrency
        );
        assert!(
            measurement.peak <= measurement.concurrency,
            "concurrency {} exceeded its own cap (peak {})",
            measurement.concurrency,
            measurement.peak
        );
        assert!(
            measurement.all_vectors_match,
            "concurrency {} mis-bound at least one vector to the wrong hash",
            measurement.concurrency
        );
    }
    let parallel = measurements.last().unwrap();
    assert!(
        parallel.elapsed < serial.elapsed,
        "concurrency {} was not faster than serial ({:.2}s vs {:.2}s)",
        parallel.concurrency,
        parallel.elapsed.as_secs_f64(),
        serial.elapsed.as_secs_f64()
    );
    println!(
        "\n    each run: {} sections, {} requests, no extra spend, every vector keyed to its \
         own section.\n",
        serial.sections, serial.requests
    );
    let _ = std::fs::remove_dir_all(&workspace);
}
