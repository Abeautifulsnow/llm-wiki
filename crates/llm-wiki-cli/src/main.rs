#![forbid(unsafe_code)]
//! `llm-wiki` CLI. Per PRD §7.7 this crate only parses arguments and calls
//! application services; all logic lives in the library crates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};

use llm_wiki_core::config::{lexical_absolute, Config, LlmConfig};
use llm_wiki_core::error::WikiError;
use llm_wiki_llm::{
    CohereCompatibleReranker, OpenAiCompatibleEmbeddings, OpenAiCompatibleProvider,
};
use llm_wiki_search::FullTextSearch;
use llm_wiki_source::{ScanDiagnostic, Scanner, SourceManifest};

#[derive(Parser)]
#[command(
    name = "llm-wiki",
    version,
    about = "Knowledge Compiler: compile docs into a cited, maintainable wiki"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create `.llm-wiki/config.toml` with safe defaults.
    Init,
    /// Scan the source tree, build the manifest and persist the Source Registry.
    Scan {
        /// Source root override (defaults to config `source.root`).
        root: Option<String>,
    },
    /// Run the full pipeline: scan → analyze → plan → compile → publish (§35).
    Build {
        /// Source root override (defaults to config `source.root`).
        root: Option<String>,
    },
    /// Explicit global re-plan (PRD §19.2/§29): fresh planning + stable-ID
    /// plan diff. `--dry-run` audits the change and cost WITHOUT compiling or
    /// publishing; the bare command executes and republishes — it IS the user
    /// confirmation, so audit with `--dry-run` first.
    Replan {
        /// Audit only: no compiler calls, no generation, no publish.
        #[arg(long)]
        dry_run: bool,
    },
    /// Show source counts and the latest build.
    Status,
    /// Full-text search over the published wiki (PRD §5.5/§20): returns
    /// page sections, not answers.
    Search {
        /// The query text (中文、English or mixed — the shared TextAnalyzer
        /// handles all three).
        query: String,
    },
    /// Check configuration, filesystem layout, state db and provider env.
    ///
    /// Offline by default. `--live` additionally makes one real request per
    /// configured endpoint (chat, and embedding/rerank when configured) — the
    /// only way to tell a working endpoint from a configured-but-dead one.
    Doctor {
        /// Probe the LLM / embedding / rerank endpoints with real requests.
        #[arg(long)]
        live: bool,
    },
    /// Ask the published wiki a question: retrieval-grounded synthesis with
    /// verified citations (audit FIX-020). Every cited claim must exist in
    /// the retrieved context; citations expand from stored anchors.
    Ask {
        /// The question (中文、English or mixed).
        query: String,
        /// Persist the verified insight with provenance into the
        /// `wiki_insights` layer (the generated wiki itself is never
        /// hand-modified).
        #[arg(long)]
        write_back: bool,
        /// Add Vector-layer candidates to the retrieval fusion (§19.3).
        /// Requires `llm-wiki embed` coverage; degrades to lexical-only
        /// with a warning when absent.
        #[arg(long)]
        hybrid: bool,
        /// Embedding model for --hybrid (falls back to
        /// $LLM_WIKI_EMBEDDING_MODEL).
        #[arg(long)]
        embedding_model: Option<String>,
    },
    /// Backfill section embeddings for the ACTIVE generation (§19.3 Vector
    /// layer). Incremental: fully-covered generations issue zero requests.
    Embed {
        /// Embedding model (falls back to $LLM_WIKI_EMBEDDING_MODEL).
        #[arg(long)]
        model: Option<String>,
        /// Sections per embedding request.
        #[arg(long, default_value_t = 16)]
        batch: usize,
    },
    /// Lint the currently published generation (PRD §36): citation integrity,
    /// links, orphans, unsupported sections, duplicates, hand edits.
    Lint {
        /// Also run the LLM-judged semantic review (audit FIX-019):
        /// contradictions, superseded facts, weak synthesis and coverage
        /// gaps. Advisory only — findings never change the exit code. Costs
        /// model calls (§28-cached).
        #[arg(long)]
        semantic: bool,
    },
    /// Serve the HTTP API (PRD §5.6/§30): health/status/build jobs/search/
    /// context/query/pages. Local-only unless `server.remote_enabled` (which
    /// requires the auth token env); the server refuses to start otherwise.
    Serve {
        /// Bind host override (defaults to config `server.bind`).
        #[arg(long)]
        host: Option<String>,
        /// Bind port.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    let code = match run(cli.command) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("error: {err}");
            err.exit_code()
        }
    };
    std::process::exit(code);
}

fn run(command: Command) -> Result<(), WikiError> {
    let workspace = std::env::current_dir()
        .map_err(|e| WikiError::Source(format!("cannot determine cwd: {e}")))?;
    match command {
        Command::Init => init(&workspace),
        Command::Scan { root } => scan(&workspace, root.as_deref()),
        Command::Build { root } => build(&workspace, root.as_deref()),
        Command::Replan { dry_run } => replan(&workspace, dry_run),
        Command::Status => status(&workspace),
        Command::Search { query } => search(&workspace, &query),
        Command::Doctor { live } => doctor(&workspace, live),
        Command::Ask {
            query,
            write_back,
            hybrid,
            embedding_model,
        } => ask(&workspace, &query, write_back, hybrid, embedding_model),
        Command::Embed { model, batch } => embed(&workspace, model.as_deref(), batch),
        Command::Lint { semantic } => lint(&workspace, semantic),
        Command::Serve { host, port } => serve(&workspace, host, port),
    }
}

fn state_dir(workspace: &Path) -> PathBuf {
    workspace.join(".llm-wiki")
}

fn state_db(workspace: &Path) -> PathBuf {
    state_dir(workspace).join("state.db")
}

fn load_config(workspace: &Path) -> Result<Config, WikiError> {
    Config::load(workspace)
}

fn init(workspace: &Path) -> Result<(), WikiError> {
    let dir = state_dir(workspace);
    let config_path = dir.join("config.toml");
    if config_path.exists() {
        return Err(WikiError::Config(format!(
            "{} already exists; refusing to overwrite",
            config_path.display()
        )));
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| WikiError::Source(format!("cannot create {}: {e}", dir.display())))?;
    let toml_body = toml_string(&Config::default())?;
    std::fs::write(&config_path, toml_body)
        .map_err(|e| WikiError::Source(format!("cannot write config: {e}")))?;
    println!("created {}", config_path.display());
    println!("next: edit source.root / llm settings, then run `llm-wiki scan`");
    Ok(())
}

/// Serializes the default config without pulling `toml` into the CLI's
/// dependency list twice — core already depends on it.
fn toml_string(config: &Config) -> Result<String, WikiError> {
    // A hand-rolled emitter keeps the generated file documented and ordered;
    // core's serde round-trip is the machine-checked path.
    let _ = config;
    Ok(
        r#"# LLM-Wiki configuration (PRD §32). Priority: CLI args > env (secrets only) > this file > defaults.

[project]
name = "llm-wiki"
wiki_dir = "./wiki"

[source]
root = "./docs"
include = ["**/*.md", "**/*.markdown", "**/*.mdx"]
exclude = ["wiki/**", ".llm-wiki/**", ".git/**", "node_modules/**"]

[llm]
provider = "openai-compatible"
base_url = "http://localhost:8000/v1"
model = ""
api_key_env = "LLM_WIKI_API_KEY"
# Thinking models spend reasoning tokens from this budget — raise for
# deepseek-style models (4096 exhausts before any visible output).
max_output_tokens = 4096
max_concurrency = 4
timeout_seconds = 120

# Chat, embedding and rerank may live at THREE different providers: every
# empty/0 field below inherits the [llm] value, so fill in only what
# differs (a different base_url, API key env, model or timeout).

[embedding]
# provider = "openai-compatible"
# base_url = "https://embeddings.example.com/v1"
# api_key_env = "LLM_WIKI_EMBEDDING_API_KEY"
# Model for hybrid retrieval / `llm-wiki embed` (falls back to the
# --embedding-model flag or $LLM_WIKI_EMBEDDING_MODEL).
# model = ""
# timeout_seconds = 0

# Used when [search] rerank = "cohere-compatible" (Cohere/Jina-style
# POST {base_url}/rerank — vLLM, Jina, SiliconFlow, Voyage, …).
[rerank]
# base_url = "https://rerank.example.com/v1"
# api_key_env = "LLM_WIKI_RERANK_API_KEY"
# model = ""
# timeout_seconds = 0

[analysis]
max_input_tokens = 32000
section_target_tokens = 6000
max_plan_input_tokens = 32000
max_rejected_claim_ratio = 0.10

[search]
full_text = true
vector = false
graph = true
# Rerank strategy applied after rank fusion (V0.5): "none" keeps the
# deterministic fused order; "cohere-compatible" calls the endpoint from
# the [rerank] section.
rerank = "none"

[build]
incremental = true
keep_generations = 3

[server]
bind = "127.0.0.1"
remote_enabled = false
auth_token_env = "LLM_WIKI_SERVER_TOKEN"
# Remote-mode guardrails (PRD §30): job queue depth, request body size and
# per-caller request budget per minute (0 disables the limiter).
max_queued_jobs = 8
max_body_bytes = 1048576
rate_limit_per_minute = 120
# Re-run build jobs interrupted by a server restart (spends LLM budget on
# restart — enable deliberately).
resume_interrupted_jobs = false
"#
        .to_owned(),
    )
}

fn scan(workspace: &Path, root_override: Option<&str>) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    let root = match root_override {
        Some(root) => PathBuf::from(root),
        None => workspace.join(&config.source.root),
    };

    let wiki_dir_rel = normalized_rel_of(&root, workspace, &config.project.wiki_dir);
    let scanner = Scanner::new(
        &root,
        &config.source.include,
        &config.source.exclude,
        wiki_dir_rel,
    )?;
    let output = scanner.scan()?;

    for diagnostic in &output.diagnostics {
        warn_diagnostic(diagnostic);
    }

    let manifest = SourceManifest::from_scanned(&output.files);
    println!(
        "scanned {} file(s) under {} (snapshot {})",
        output.files.len(),
        root.display(),
        &manifest.snapshot_hash()[..16]
    );

    let db_path = state_db(workspace);
    std::fs::create_dir_all(state_dir(workspace))
        .map_err(|e| WikiError::Source(format!("cannot create state dir: {e}")))?;
    let mut conn = llm_wiki_storage::open(&db_path)?;

    // One transaction for the whole scan — no per-file commits.
    let batch: Vec<llm_wiki_storage::SourceUpsert> = output
        .files
        .iter()
        .map(|file| llm_wiki_storage::SourceUpsert {
            locator_key: &file.locator_key,
            rel_path: &file.rel_path,
            content_hash: &file.content_hash,
            size: file.size as i64,
        })
        .collect();
    let _results = llm_wiki_storage::upsert_sources_batch(&mut conn, &batch, None)?;
    for file in &output.files {
        println!(
            "  {:<60} {} bytes  {}",
            file.rel_path,
            file.size,
            &file.content_hash[..12]
        );
    }
    println!(
        "source registry: {} file(s) up to date in {}",
        output.files.len(),
        db_path.display()
    );
    Ok(())
}

fn normalized_rel_of(root: &Path, workspace: &Path, wiki_dir: &Path) -> Option<String> {
    let abs = llm_wiki_core::config::lexical_absolute(workspace, root);
    let wiki_abs = llm_wiki_core::config::lexical_absolute(workspace, wiki_dir);
    wiki_abs.strip_prefix(&abs).ok().map(|rel| {
        rel.components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
    })
}

/// Builds the configured LLM provider. The API key is read from the env var
/// *named* by the config (PRD §32) at first request.
fn build_provider(llm: &LlmConfig) -> Result<Arc<dyn llm_wiki_llm::LlmProvider>, WikiError> {
    if llm.model.trim().is_empty() {
        return Err(WikiError::Config(
            "llm.model must be set in .llm-wiki/config.toml before building".into(),
        ));
    }
    match llm.provider.as_str() {
        "openai-compatible" => {
            let provider = OpenAiCompatibleProvider::new(
                &llm.base_url,
                &llm.model,
                &llm.api_key_env,
                llm.timeout_seconds,
                2,
            )
            .map_err(|e| WikiError::Llm(e.to_string()))?
            .with_thinking(&llm.thinking, &llm.thinking_effort);
            Ok(Arc::new(provider))
        }
        other => Err(WikiError::Config(format!(
            "unsupported llm.provider '{other}' (supported: openai-compatible)"
        ))),
    }
}

/// `llm-wiki build [root]` (PRD §29): thin transport over
/// `llm_wiki_compiler::run_build`. `ReplanRequired` propagates with its exit
/// code (7) and trigger reason — the CLI never auto-replans.
fn build(workspace: &Path, root_override: Option<&str>) -> Result<(), WikiError> {
    let mut config = load_config(workspace)?;
    if let Some(root) = root_override {
        config.source.root = PathBuf::from(root);
    }
    config.validate()?;
    let provider = build_provider(&config.llm)?;

    let wiki_abs = lexical_absolute(workspace, &config.project.wiki_dir);
    println!(
        "build: sources {} → wiki {}",
        config.source.root.display(),
        wiki_abs.display()
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| WikiError::Llm(format!("cannot start async runtime: {e}")))?;
    let report = runtime.block_on(llm_wiki_compiler::run_build(workspace, &config, provider))?;

    println!("build {} completed", report.build_id);
    println!("  sources: {}", report.sources);
    println!(
        "  pages: {}  citations: {}  links: {}",
        report.pages, report.citations, report.links
    );
    println!("  llm requests: {}", report.llm_request_count);
    println!(
        "  llm cache: {} hit(s), {} miss(es)",
        report.cache.hits, report.cache.misses
    );
    if let Some(incremental) = &report.incremental {
        // §19 incremental summary: what changed and what the pipeline did.
        println!(
            "  incremental: {} changed, {} recompiled, {} carried, {} obsolete",
            incremental.changed, incremental.recompiled, incremental.carried, incremental.obsolete
        );
        if incremental.deleted > 0 {
            println!(
                "  incremental: {} deleted source(s) retired",
                incremental.deleted
            );
        }
    }
    println!("  published: {}", report.published_path.display());
    if let Some(recovery) = &report.recovery {
        println!("  recovered publish: {recovery}");
    }
    Ok(())
}

/// Length note: ~91 lines — report formatting only (linear println branches over ReplanReport fields), no nesting.
/// `llm-wiki replan [--dry-run]` (PRD §29/§19.2): thin transport over the
/// compiler's replan service.
fn replan(workspace: &Path, dry_run: bool) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    config.validate()?;
    let provider = build_provider(&config.llm)?;

    if dry_run {
        println!("replan --dry-run: auditing the global plan change (no compile, no publish)");
        println!("  note: pending source changes are analyzed and the knowledge registry is updated; the published wiki is not touched");
    } else {
        println!("replan: fresh planning + stable-ID plan diff + recompile of every changed page");
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| WikiError::Llm(format!("cannot start async runtime: {e}")))?;
    let report = runtime.block_on(llm_wiki_compiler::replan(
        workspace, &config, provider, dry_run,
    ))?;

    for trigger in &report.triggers {
        println!("  trigger: {trigger}");
    }
    println!(
        "  diff: {} unchanged, {} modified, {} merged, {} split, {} created, {} retired",
        report.unchanged,
        report.modified,
        report.merged,
        report.split,
        report.created,
        report.retired
    );
    for (slug, predecessors) in &report.merged_pages {
        println!("    merged: {slug} ← {}", predecessors.join(", "));
    }
    for (slug, successors) in &report.split_pages {
        println!("    split: {slug} → {}", successors.join(", "));
    }
    for (slug, lost) in &report.retired_pages {
        if lost.is_empty() {
            println!("    retired: {slug}");
        } else {
            println!(
                "    retired: {slug} ({} ref(s) no new page covers)",
                lost.len()
            );
        }
    }
    for warning in &report.knowledge_loss_warnings {
        println!("  warning: {warning}");
    }
    if report.dry_run {
        println!(
            "  estimate: {} compile llm call(s), cost upper bound {} output tokens",
            report.estimated_compile_llm_calls, report.estimated_max_output_tokens
        );
        println!(
            "  spent on this audit: {} llm request(s) (cache: {} hit(s), {} miss(es))",
            report.llm_request_count, report.cache.hits, report.cache.misses
        );
        if report.no_replan_needed {
            println!("replan: nothing to change — the fresh plan matches the current generation");
        } else {
            println!("next: review the diff, then run `llm-wiki replan` to execute");
        }
        return Ok(());
    }

    if report.no_replan_needed {
        println!("replan: nothing to change — the fresh plan matches the current generation");
        return Ok(());
    }
    if let Some(build_id) = &report.build_id {
        println!("replan {} completed", build_id);
    }
    println!(
        "  pages: {}  citations: {}  links: {}  ({} recompiled, {} carried)",
        report.pages, report.citations, report.links, report.recompiled, report.carried
    );
    println!("  llm requests: {}", report.llm_request_count);
    println!(
        "  llm cache: {} hit(s), {} miss(es)",
        report.cache.hits, report.cache.misses
    );
    if let Some(path) = &report.published_path {
        println!("  published: {}", path.display());
    }
    if let Some(recovery) = &report.recovery {
        println!("  recovered publish: {recovery}");
    }
    Ok(())
}

/// `llm-wiki search <query>` (PRD §5.5/§20): thin transport over
/// `llm_wiki_search::FullTextSearch`. Prints ranked page sections with body
/// snippets; a never-built workspace is not an error. With
/// `config.search.graph = true` the top hit gains a one-hop "related" line
/// from the §17 Wiki Graph (§22 limits: depth 1, max 10 nodes).
fn search(workspace: &Path, query: &str) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    let db_path = state_db(workspace);
    if !db_path.exists() {
        println!("nothing published; run build first");
        return Ok(());
    }
    let conn = llm_wiki_storage::open(&db_path)?;
    let fts = llm_wiki_search::SqliteFullTextSearch::new(conn, config.search.full_text);
    if !fts.has_published()? {
        println!("nothing published; run build first");
        return Ok(());
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| WikiError::Llm(format!("cannot start async runtime: {e}")))?;
    let hits = runtime.block_on(fts.search(query, 10))?;

    if hits.is_empty() {
        println!("no results for {query:?}");
        return Ok(());
    }
    for (rank_index, hit) in hits.iter().enumerate() {
        println!(
            "{}. {} [{}] @ {}",
            rank_index + 1,
            hit.title,
            hit.slug,
            hit.heading_path.join(" > ")
        );
        let snippet = hit.snippet.replace('\n', " ");
        println!("   {snippet}");
    }
    print_related_pages(&db_path, &config, &hits[0], &runtime)?;
    Ok(())
}

/// One-hop graph expansion of the top hit (PRD §17/§22), gated by
/// `search.graph`. Outgoing links render as-is; incoming ones are marked as
/// backlinks. A page without neighbors prints nothing — a disabled flag and
/// an empty graph are indistinguishable from the outside, both are "no
/// related line". A graph failure is downgraded to a warning: the FTS
/// results above already delivered the search's value, and a supplementary
/// line must not fail an otherwise successful command (success-shape
/// principle for recoverable conditions).
fn print_related_pages(
    db_path: &Path,
    config: &Config,
    top: &llm_wiki_search::SearchHit,
    runtime: &tokio::runtime::Runtime,
) -> Result<(), WikiError> {
    if !config.search.graph {
        return Ok(());
    }
    let graph = match llm_wiki_storage::open(db_path) {
        Ok(conn) => llm_wiki_search::SqliteGraphExploration::new(conn),
        Err(error) => {
            eprintln!("  ! related pages unavailable: {error}");
            return Ok(());
        }
    };
    let neighbors = match runtime
        .block_on(graph.neighbors_of_page(&top.page_id, llm_wiki_storage::EXPAND_MAX_NODES))
    {
        Ok(neighbors) => neighbors,
        Err(error) => {
            eprintln!("  ! related pages unavailable: {error}");
            return Ok(());
        }
    };
    let related: Vec<String> = neighbors
        .iter()
        .filter(|n| n.node_type == llm_wiki_storage::NODE_TYPE_PAGE)
        .map(|n| match n.direction {
            llm_wiki_storage::NeighborDirection::Outgoing => n.label.clone(),
            llm_wiki_storage::NeighborDirection::Incoming => format!("{} (backlink)", n.label),
        })
        .collect();
    if !related.is_empty() {
        println!("   related: {}", related.join("; "));
    }
    Ok(())
}

fn warn_diagnostic(diagnostic: &ScanDiagnostic) {
    println!(
        "  ! {} {:?}: {}",
        diagnostic.rel_path, diagnostic.kind, diagnostic.message
    );
}

/// Resolves the embedding model: flag → $LLM_WIKI_EMBEDDING_MODEL →
/// `[embedding] model` → error (explicit args beat env beats project config).
fn embedding_model(flag: Option<&str>, config: &Config) -> Result<String, WikiError> {
    let from_config = config.embedding.model.trim();
    flag.map(str::to_owned)
        .or_else(|| std::env::var("LLM_WIKI_EMBEDDING_MODEL").ok())
        .filter(|model| !model.trim().is_empty())
        .or_else(|| {
            (!from_config.is_empty()).then(|| from_config.to_owned())
        })
        .ok_or_else(|| {
            WikiError::Config(
                "no embedding model: pass --embedding-model <model>, set LLM_WIKI_EMBEDDING_MODEL, or set [embedding] model in .llm-wiki/config.toml"
                    .into(),
            )
        })
}

/// Builds an embedding provider from the `[embedding]` config section —
/// empty fields inherit `[llm]`, so the embedding model may live at its own
/// provider (own base_url / API key / timeout) without reconfiguring chat.
fn build_embedding_provider(
    config: &Config,
) -> Result<Arc<dyn llm_wiki_llm::EmbeddingProvider>, WikiError> {
    let llm = &config.llm;
    let endpoint = config.embedding.endpoint(llm);
    match config.embedding.resolved_provider(llm) {
        "openai-compatible" => {
            let provider = OpenAiCompatibleEmbeddings::new(
                &endpoint.base_url,
                &endpoint.api_key_env,
                endpoint.timeout_seconds,
                2,
            )
            .map_err(|e| WikiError::Llm(e.to_string()))?;
            Ok(Arc::new(provider))
        }
        other => Err(WikiError::Config(format!(
            "unsupported embedding.provider '{other}' (supported: openai-compatible)"
        ))),
    }
}

/// `llm-wiki ask` (audit FIX-020): grounded answer + verified citations;
/// `--write-back` persists the insight with provenance; `--hybrid` adds
/// Vector-layer candidates (§19.3).
fn ask(
    workspace: &Path,
    query: &str,
    write_back: bool,
    hybrid: bool,
    embedding_model_flag: Option<String>,
) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    let provider = build_provider(&config.llm)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| WikiError::Llm(format!("cannot start async runtime: {e}")))?;
    // Owned bindings so the HybridContext borrow outlives the call.
    let embedding_provider = if hybrid {
        Some(build_embedding_provider(&config)?)
    } else {
        None
    };
    let hybrid_context = if hybrid {
        let model = embedding_model(embedding_model_flag.as_deref(), &config)?;
        Some(llm_wiki_compiler::HybridContext {
            provider: embedding_provider.as_ref().expect("checked above"),
            model,
        })
    } else {
        None
    };
    let Some(report) = runtime.block_on(llm_wiki_compiler::run_ask(
        workspace,
        &config,
        provider,
        query,
        write_back,
        hybrid_context,
    ))?
    else {
        println!("nothing published — build the wiki first");
        return Ok(());
    };
    println!("{}", report.answer);
    if !report.sources.is_empty() {
        println!("\nsources:");
        for source in &report.sources {
            println!("  - {source}");
        }
    }
    match &report.insight_id {
        Some(id) => println!("insight {id} written back"),
        None => println!("(dry run — pass --write-back to persist this insight)"),
    }
    Ok(())
}

/// `llm-wiki embed` (§19.3): incremental embedding backfill.
fn embed(workspace: &Path, model: Option<&str>, batch: usize) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    let model = embedding_model(model, &config)?;
    let embedding_provider = build_embedding_provider(&config)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| WikiError::Llm(format!("cannot start async runtime: {e}")))?;
    let Some(report) = runtime.block_on(llm_wiki_compiler::run_embed(
        workspace,
        &config,
        embedding_provider,
        &model,
        batch,
    ))?
    else {
        println!("nothing published — build the wiki first");
        return Ok(());
    };
    println!(
        "embeddings [{}]: {} section(s), {} already covered, {} embedded",
        report.model, report.total_sections, report.covered_before, report.embedded
    );
    Ok(())
}

/// `llm-wiki lint` (PRD §29/§36): thin transport over
/// `llm_wiki_compiler::run_lint`. Findings print grouped by check in
/// deterministic order; any Error-severity finding exits with the dedicated
/// lint code (11), warnings alone exit 0. A never-built workspace is not an
/// error.
fn lint(workspace: &Path, semantic: bool) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    let Some(report) = llm_wiki_compiler::run_lint(workspace, &config)? else {
        println!("nothing published — nothing to lint");
        return Ok(());
    };

    let mut current: Option<llm_wiki_compiler::LintCheck> = None;
    for finding in &report.findings {
        if current != Some(finding.check) {
            println!("{}:", finding.check.label());
            current = Some(finding.check);
        }
        println!(
            "  [{}] {}: {}",
            finding.severity.label(),
            finding.page_slug,
            finding.message
        );
    }

    let (errors, warnings) = (report.errors(), report.warnings());
    println!("{errors} error(s), {warnings} warning(s)");
    let exit = if errors > 0 {
        Err(WikiError::Lint { errors, warnings })
    } else {
        Ok(())
    };

    // Semantic review (audit FIX-019): advisory, printed after the
    // structural report, never affecting the exit code.
    if semantic {
        let provider = build_provider(&config.llm)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| WikiError::Llm(format!("cannot start async runtime: {e}")))?;
        let semantic_report = runtime.block_on(llm_wiki_compiler::run_semantic_lint(
            workspace, &config, provider,
        ))?;
        if let Some(semantic_report) = semantic_report {
            println!("semantic review (advisory):");
            let mut current_kind: Option<llm_wiki_compiler::SemanticFindingKind> = None;
            for finding in &semantic_report.findings {
                if current_kind != Some(finding.kind) {
                    println!("  {}:", finding.kind.label());
                    current_kind = Some(finding.kind);
                }
                let claims = if finding.claim_ids.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", finding.claim_ids.join(", "))
                };
                println!("    {}{claims}: {}", finding.page_slug, finding.message);
            }
            println!(
                "  {} page(s) reviewed, {} finding(s), skipped: {}",
                semantic_report.pages_reviewed,
                semantic_report.findings.len(),
                if semantic_report.skipped_pages.is_empty() {
                    "none".to_owned()
                } else {
                    semantic_report.skipped_pages.join(", ")
                }
            );
        }
    }
    exit
}

/// `llm-wiki serve` (PRD §29/§30): thin transport over the llm-wiki-server
/// crate. The LLM provider is optional — build/query endpoints report a
/// clear config error when `[llm]` is unconfigured; search/context/pages
/// keep working.
fn serve(workspace: &Path, host: Option<String>, port: u16) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    config.validate()?;
    let provider = if config.llm.model.trim().is_empty() {
        eprintln!(
            "note: llm.model is not configured — /v1/build and /v1/query will return config errors"
        );
        None
    } else {
        Some(build_provider(&config.llm)?)
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| WikiError::Llm(format!("cannot start async runtime: {e}")))?;
    runtime.block_on(llm_wiki_server::serve(llm_wiki_server::ServeOptions {
        workspace: workspace.to_path_buf(),
        config,
        host,
        port,
        provider,
    }))?;
    Ok(())
}

fn status(workspace: &Path) -> Result<(), WikiError> {
    let db_path = state_db(workspace);
    if !db_path.exists() {
        println!(
            "no state db at {} — run `llm-wiki scan` first",
            db_path.display()
        );
        return Ok(());
    }
    let conn = llm_wiki_storage::open(&db_path)?;
    println!("sources: {}", llm_wiki_storage::count_sources(&conn)?);
    match llm_wiki_storage::latest_build(&conn)? {
        Some(build) => println!(
            "latest build: {} [{}] started {}",
            build.build_id,
            build.status,
            build.started_at.to_rfc3339()
        ),
        None => println!("no builds recorded yet"),
    }
    Ok(())
}

/// A failed doctor check: what is wrong, and the command that fixes it. Every
/// FAIL must carry a remedy that actually works — a bare "FAIL" left the V1.0
/// field test guessing (and in the publish-wedge case there WAS no command,
/// which is a defect in the recovery state machine, not a missing hint).
#[derive(Debug)]
struct Gap {
    detail: String,
    remedy: String,
}

impl Gap {
    fn new(detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }
}

fn report_gap(label: &str, result: Result<String, Gap>) -> bool {
    match result {
        Ok(msg) => {
            println!("ok   {label} {msg}");
            true
        }
        Err(gap) => {
            println!("FAIL {label} {}", gap.detail);
            println!("     fix: {}", gap.remedy);
            false
        }
    }
}

/// Publish-integrity check for `doctor` (PRD §29/§35): un-recovered publish
/// journal and `current.json` vs DB `active_build_id` consistency. Reports
/// the finding with a remedy; never repairs anything itself.
fn check_publish_state(
    conn: &llm_wiki_storage::Connection,
    wiki_dir: &Path,
) -> Result<String, Gap> {
    let paths = llm_wiki_compiler::PublishPaths::new(wiki_dir);
    let journal = paths.journal_path();
    if llm_wiki_compiler::journal_exists(&paths) {
        return Err(Gap::new(
            format!(
                "an un-recovered publish journal is present at {} — the last publish was interrupted",
                journal.display()
            ),
            "run `llm-wiki build`: publish recovery resolves the journal before the build starts, \
             completing the new generation or rolling back to the previous one (never leaves the \
             workspace wedged)",
        ));
    }
    let pointer = llm_wiki_compiler::read_current_pointer(&paths).map_err(|err| {
        Gap::new(
            err.to_string(),
            format!(
                "{} is unreadable or malformed; restore it from a backup (it is a one-line \
                 build-id pointer into wiki/generations/), or delete the wiki/ directory \
                 and run `llm-wiki build` to publish a fresh generation",
                paths.pointer_path().display()
            ),
        )
    })?;
    let db_active = llm_wiki_storage::get_active_build_id(conn).map_err(|err| {
        Gap::new(
            err.to_string(),
            "re-run `llm-wiki scan` to re-open the state db",
        )
    })?;
    match (pointer, db_active) {
        (None, None) => Ok("no wiki published yet (run `llm-wiki build`)".to_owned()),
        (Some(pointer), Some(active)) if pointer.build_id == active.as_str() => {
            Ok(format!("consistent (current generation {})", active))
        }
        (Some(pointer), Some(active)) => Err(Gap::new(
            format!(
                "{} points at {} but the database active_build_id is {} (no publish journal), so \
                 search/pages/ask would serve a different generation than the files on disk",
                paths.pointer_path().display(),
                pointer.build_id,
                active
            ),
            format!(
                "the two must be reconciled by hand — publish recovery refuses to guess which one \
                 is right: either edit {} back to \"{}\", or rebuild the wiki (delete wiki/ and \
                 .llm-wiki/state.db, then `llm-wiki build`)",
                paths.pointer_path().display(),
                active
            ),
        )),
        (Some(pointer), None) => Err(Gap::new(
            format!(
                "{} points at {} but the database has no active generation",
                paths.pointer_path().display(),
                pointer.build_id
            ),
            "the database lost its active pointer while the wiki is still published: rebuild with \
             `llm-wiki build` (the pipeline re-activates the published generation when it still \
             validates)",
        )),
        (None, Some(active)) => Err(Gap::new(
            format!(
                "the database active_build_id is {} but {} is missing",
                active,
                paths.pointer_path().display()
            ),
            "re-publish with `llm-wiki build`; without the pointer the published files cannot be \
             identified, so search/pages/ask have no generation to serve",
        )),
    }
}

/// Doctor endpoint probes must not inherit the build's per-request budget: a
/// build may legitimately wait 120s for one completion, but a connectivity
/// check that hangs that long on a dead host is indistinguishable from a hung
/// `doctor`.
const PROBE_TIMEOUT_SECONDS: u64 = 15;

/// One live chat round trip (注意点: `/v1/models` alone is not evidence — a
/// gateway can list a model it cannot serve, and serve one it does not list).
fn probe_chat_endpoint(config: &Config) -> Result<String, Gap> {
    let provider = OpenAiCompatibleProvider::new(
        &config.llm.base_url,
        &config.llm.model,
        &config.llm.api_key_env,
        config.llm.timeout_seconds.min(PROBE_TIMEOUT_SECONDS),
        0,
    )
    .map_err(|err| {
        Gap::new(
            err.to_string(),
            "check llm.base_url in .llm-wiki/config.toml",
        )
    })?;
    let runtime = tokio_runtime()?;
    let outcome = runtime.block_on(provider.probe()).map_err(|err| {
        Gap::new(
            format!(
                "{} did not answer a chat request for model '{}': {err}",
                config.llm.base_url, config.llm.model
            ),
            format!(
                "check llm.base_url (must end at the API root, e.g. http://host/v1), \
                 llm.model, and export {} with the endpoint's key",
                config.llm.api_key_env
            ),
        )
    })?;
    let catalog = match outcome.model_listed {
        Some(true) => "listed by /models".to_owned(),
        Some(false) => {
            "NOT in /models (the endpoint still served it — the catalog is advisory)".to_owned()
        }
        None => "/models unavailable (not checked)".to_owned(),
    };
    let served = match outcome.served_model {
        Some(served) if served != config.llm.model => {
            format!("served as '{served}' (different revision than requested)")
        }
        _ => "served the requested model".to_owned(),
    };
    Ok(format!("{}ms, {served}, {catalog}", outcome.latency_ms))
}

/// Live embedding probe. Runs whenever an embedding model resolves — the
/// Vector layer is opt-in per command, so a configured-but-broken endpoint
/// would otherwise stay invisible until the first `--hybrid` query.
fn probe_embedding_endpoint(config: &Config) -> Result<String, Gap> {
    let model =
        match embedding_model(None, config) {
            Ok(model) => model,
            Err(_) => return Ok(
                "not configured (set [embedding] model or $LLM_WIKI_EMBEDDING_MODEL to enable the \
                 Vector layer)"
                    .to_owned(),
            ),
        };
    let endpoint = config.embedding.endpoint(&config.llm);
    let provider = OpenAiCompatibleEmbeddings::new(
        &endpoint.base_url,
        &endpoint.api_key_env,
        endpoint.timeout_seconds.min(PROBE_TIMEOUT_SECONDS),
        0,
    )
    .map_err(|err| {
        Gap::new(
            err.to_string(),
            "check [embedding] base_url in .llm-wiki/config.toml",
        )
    })?;
    let runtime = tokio_runtime()?;
    let outcome = runtime.block_on(provider.probe(&model)).map_err(|err| {
        Gap::new(
            format!(
                "{} did not answer an embeddings request for model '{model}': {err}",
                endpoint.base_url
            ),
            format!(
                "check [embedding] base_url / model and export {}; run `llm-wiki embed` to verify",
                endpoint.api_key_env
            ),
        )
    })?;
    Ok(format!(
        "{}ms, model '{model}', {} dimensions",
        outcome.latency_ms, outcome.dimensions
    ))
}

/// Live rerank probe: only when the strategy is enabled — a rerank endpoint
/// that is configured but unreachable silently degrades search to the fused
/// order, which is exactly the kind of quiet failure `doctor` exists to name.
fn probe_rerank_endpoint(config: &Config) -> Result<String, Gap> {
    if config.search.rerank != "cohere-compatible" {
        return Ok(format!(
            "not configured (search.rerank = \"{}\")",
            config.search.rerank
        ));
    }
    let endpoint = config.rerank.endpoint(&config.llm);
    let reranker = CohereCompatibleReranker::new(
        &endpoint.base_url,
        &endpoint.api_key_env,
        endpoint.timeout_seconds.min(PROBE_TIMEOUT_SECONDS),
        0,
    )
    .map_err(|err| {
        Gap::new(
            err.to_string(),
            "check [rerank] base_url in .llm-wiki/config.toml",
        )
    })?;
    let runtime = tokio_runtime()?;
    let outcome = runtime
        .block_on(reranker.probe(&config.rerank.model))
        .map_err(|err| {
            Gap::new(
                format!(
                    "{} did not answer a /rerank request for model '{}': {err}",
                    endpoint.base_url, config.rerank.model
                ),
                format!(
                    "check [rerank] base_url / model and export {}; the endpoint must speak the \
                     Cohere /rerank wire format (vLLM, Jina, SiliconFlow, Voyage)",
                    endpoint.api_key_env
                ),
            )
        })?;
    Ok(format!(
        "{}ms, model '{}', top score {:?}",
        outcome.latency_ms, config.rerank.model, outcome.top_relevance
    ))
}

fn tokio_runtime() -> Result<tokio::runtime::Runtime, Gap> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            Gap::new(
                format!("cannot start async runtime: {err}"),
                "this is an environment problem, not a config one",
            )
        })
}

/// Length note: ~150 lines — a flat list of checks, one report per line, each
/// FAIL carrying its own remedy; the helpers above keep every branch shallow.
fn doctor(workspace: &Path, live: bool) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    let config_path = state_dir(workspace).join("config.toml");
    let mut failures = 0usize;

    let valid = config.validate().map(|_| {
        format!(
            "valid (source root {}, wiki_dir {})",
            config.source.root.display(),
            config.project.wiki_dir.display()
        )
    });
    let valid = valid.map_err(|err| {
        Gap::new(
            err.to_string(),
            format!(
                "fix the reported key in {} (delete it and re-run `llm-wiki init` for a documented \
                 reference file)",
                config_path.display()
            ),
        )
    });
    if !report_gap("config", valid) {
        failures += 1;
    }

    let root = workspace.join(&config.source.root);
    let root_exists = if root.is_dir() {
        Ok(format!("exists ({})", root.display()))
    } else {
        Err(Gap::new(
            format!("missing ({})", root.display()),
            format!(
                "create that directory and drop your docs in, or point source.root (in {}) at the \
                 right tree",
                config_path.display()
            ),
        ))
    };
    if !report_gap("source root", root_exists) {
        failures += 1;
    }

    let db_path = state_db(workspace);
    let db = std::fs::create_dir_all(state_dir(workspace))
        .map_err(|e| e.to_string())
        .and_then(|_| llm_wiki_storage::open(&db_path).map_err(|e| e.to_string()));
    let conn = match db {
        Ok(conn) => {
            let count = llm_wiki_storage::count_sources(&conn);
            let count = count.map_err(|err| {
                Gap::new(
                    err.to_string(),
                    "the state db exists but is not readable as a wiki state db",
                )
            });
            if !report_gap(
                "state db",
                count.map(|count| format!("open + migrated ({count} sources)")),
            ) {
                failures += 1;
            }
            Some(conn)
        }
        Err(err) => {
            report_gap(
                "state db",
                Err(Gap::new(
                    format!("cannot open {}: {err}", db_path.display()),
                    format!(
                        "move {} aside and run `llm-wiki scan` to rebuild it (build state, cache \
                         and index are all derived — nothing in it is hand-authored)",
                        db_path.display()
                    ),
                )),
            );
            failures += 1;
            None
        }
    };

    // Publish integrity (PRD §35): report, never auto-fix.
    if let Some(conn) = conn {
        let wiki_abs = lexical_absolute(workspace, &config.project.wiki_dir);
        let publish_state = check_publish_state(&conn, &wiki_abs);
        if !report_gap("publish state", publish_state) {
            failures += 1;
        }

        // FTS5 availability (PRD §20): with the bundled SQLite this always
        // passes; a custom SQLite build without FTS5 must surface here as a
        // reported degradation — search refuses to run rather than silently
        // returning unusable results.
        let fts_state = if llm_wiki_storage::probe_fts5(&conn) {
            Ok("fts5 available".to_owned())
        } else if config.search.full_text {
            Err(Gap::new(
                llm_wiki_storage::FTS5_UNAVAILABLE.to_owned(),
                "this binary was linked against a SQLite without FTS5: rebuild with the default \
                 `bundled` feature, or set search.full_text = false to accept lexical-less search",
            ))
        } else {
            Ok("fts5 unavailable (full-text search is disabled in config)".to_owned())
        };
        if !report_gap("fts", fts_state) {
            failures += 1;
        }
    }

    let api_key = match std::env::var(&config.llm.api_key_env) {
        Ok(_) => Ok(format!("env {} is set", config.llm.api_key_env)),
        Err(_) => Err(Gap::new(
            format!("env {} is not set", config.llm.api_key_env),
            format!(
                "export {}=<key> before `llm-wiki build` (scan/doctor never call the model; a local \
                 endpoint that needs no key can ignore this)",
                config.llm.api_key_env
            ),
        )),
    };
    if !report_gap("llm api key", api_key) {
        failures += 1;
    }

    // Endpoint connectivity: the checks above prove the config is COMPLETE,
    // not that the endpoints are ALIVE. Opt-in because each probe is a real
    // (billed) request.
    if live {
        if config.llm.model.trim().is_empty() {
            report_gap(
                "llm endpoint",
                Err(Gap::new(
                    "llm.model is empty, nothing to probe".to_owned(),
                    format!("set llm.model in {}", config_path.display()),
                )),
            );
            failures += 1;
        } else {
            let chat = probe_chat_endpoint(&config);
            if !report_gap("llm endpoint", chat) {
                failures += 1;
            }
        }
        let embedding = probe_embedding_endpoint(&config);
        if !report_gap("embedding endpoint", embedding) {
            failures += 1;
        }
        let rerank = probe_rerank_endpoint(&config);
        if !report_gap("rerank endpoint", rerank) {
            failures += 1;
        }
    } else {
        println!("skip endpoint connectivity (pass --live to probe llm/embedding/rerank for real)");
    }

    if failures == 0 {
        println!("doctor: all checks passed");
        Ok(())
    } else {
        Err(WikiError::Config(format!("{failures} check(s) failed")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_compiler::{write_current_pointer, write_journal, PublishPaths};
    use llm_wiki_core::ids::BuildId;

    /// V1.0 field test: `doctor` reported FAIL with no command that fixed it.
    /// A wedge caused by the publish journal must name the re-entrant command
    /// (`build` — recovery resolves the journal before the build starts), and
    /// the pointer/DB mismatch must name the two hand-reconcilable options
    /// (recovery deliberately refuses to guess).
    #[test]
    fn publish_state_failures_name_a_remedy_that_can_actually_run() {
        let conn = llm_wiki_storage::open_in_memory().unwrap();
        let build = BuildId::generate();

        // Consistent state: nothing published, nothing to remedy.
        let empty = doctor_temp_dir("doctor-clean");
        assert!(check_publish_state(&conn, &empty).is_ok());

        // A leftover journal is the wedge that used to dead-end. The remedy
        // must be a command, not a description of the problem.
        let wedged = doctor_temp_dir("doctor-journal");
        let paths = PublishPaths::new(&wedged);
        write_journal(&paths, None, &build).unwrap();
        let gap = check_publish_state(&conn, &wedged).unwrap_err();
        assert!(gap.detail.contains("journal"), "{}", gap.detail);
        assert!(
            gap.remedy.contains("llm-wiki build"),
            "the remedy must be the command that resolves it: {}",
            gap.remedy
        );

        // Pointer without a database generation: the pointer cannot be
        // trusted alone, so the remedy is a re-publish.
        let pointer_only = doctor_temp_dir("doctor-pointer");
        let paths = PublishPaths::new(&pointer_only);
        write_current_pointer(&paths, &build).unwrap();
        let gap = check_publish_state(&conn, &pointer_only).unwrap_err();
        assert!(gap.detail.contains(&build.to_string()), "{}", gap.detail);
        assert!(gap.remedy.contains("llm-wiki build"), "{}", gap.remedy);
    }

    /// The live chat probe is the point of 注意点 3: an endpoint that lists a
    /// model but cannot serve it, or serves a different revision, must be
    /// distinguishable in `doctor` output — and a working endpoint must pass.
    #[test]
    fn live_chat_probe_passes_on_a_real_endpoint_and_names_a_revision_swap() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            // probe: GET /models then POST /chat/completions.
            for path in ["/models", "/chat/completions"] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65536];
                let _ = stream.read(&mut buf).unwrap();
                let body = if path == "/models" {
                    r#"{"data":[{"id":"tiny-thinker"}]}"#.to_owned()
                } else {
                    r#"{"model":"tiny-thinker-2026","choices":[{"message":{"content":""},"finish_reason":"length"}]}"#
                        .to_owned()
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let mut config = Config::default();
        config.llm.base_url = format!("http://127.0.0.1:{port}");
        config.llm.model = "tiny-thinker".to_owned();
        config.llm.api_key_env = "UNUSED_VAR".to_owned();
        config.llm.timeout_seconds = 10;

        let verdict = probe_chat_endpoint(&config).unwrap();
        assert!(
            verdict.contains("served as 'tiny-thinker-2026'"),
            "{verdict}"
        );
        assert!(verdict.contains("listed by /models"), "{verdict}");
        server.join().unwrap();
    }

    /// A dead endpoint must FAIL the live probe with an actionable remedy —
    /// this is the case `/v1/models` alone used to hide.
    #[test]
    fn live_chat_probe_fails_with_a_remedy_when_the_endpoint_is_dead() {
        // Bind and drop: the port is (almost certainly) closed afterwards.
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };

        let mut config = Config::default();
        config.llm.base_url = format!("http://127.0.0.1:{port}");
        config.llm.model = "tiny-thinker".to_owned();
        config.llm.api_key_env = "UNUSED_VAR".to_owned();
        config.llm.timeout_seconds = 2;

        let gap = probe_chat_endpoint(&config).unwrap_err();
        assert!(gap.detail.contains("did not answer"), "{}", gap.detail);
        assert!(gap.remedy.contains("base_url"), "{}", gap.remedy);
    }

    /// Optional layers that are OFF must be reported, not failed — doctor
    /// only red-flags what the user actually turned on.
    #[test]
    fn a_disabled_rerank_layer_is_reported_not_failed() {
        let config = Config::default();
        let rerank = probe_rerank_endpoint(&config).unwrap();
        assert!(rerank.contains("not configured"), "{rerank}");
    }

    /// An embedding endpoint that is configured but dead must fail the live
    /// probe with a remedy naming the config keys to check. (The embedding
    /// model may also come from `LLM_WIKI_EMBEDDING_MODEL`, so the probe takes
    /// the endpoint from config and the model from wherever it resolves.)
    #[test]
    fn live_embedding_probe_fails_with_a_remedy_on_a_dead_endpoint() {
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };

        let mut config = Config::default();
        config.embedding.base_url = format!("http://127.0.0.1:{port}");
        config.embedding.model = "probe-embed".to_owned();
        config.embedding.api_key_env = "UNUSED_VAR".to_owned();
        config.embedding.timeout_seconds = 2;

        let gap = probe_embedding_endpoint(&config).unwrap_err();
        assert!(gap.detail.contains("did not answer"), "{}", gap.detail);
        assert!(gap.remedy.contains("[embedding]"), "{}", gap.remedy);
    }

    fn doctor_temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llm-wiki-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
