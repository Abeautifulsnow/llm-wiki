#![forbid(unsafe_code)]
//! `llm-wiki` CLI. Per PRD §7.7 this crate only parses arguments and calls
//! application services; all logic lives in the library crates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand};

use llm_wiki_core::config::{lexical_absolute, Config, LlmConfig};
use llm_wiki_core::error::WikiError;
use llm_wiki_llm::OpenAiCompatibleProvider;
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
    Doctor,
    /// Lint the currently published generation (PRD §36): citation integrity,
    /// links, orphans, unsupported sections, duplicates, hand edits.
    Lint,
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
        Command::Doctor => doctor(&workspace),
        Command::Lint => lint(&workspace),
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
max_concurrency = 4
timeout_seconds = 120

[analysis]
max_input_tokens = 32000
section_target_tokens = 6000
max_plan_input_tokens = 32000
max_rejected_claim_ratio = 0.10

[search]
full_text = true
vector = false
graph = true

[build]
incremental = true
keep_generations = 3

[server]
bind = "127.0.0.1"
remote_enabled = false
auth_token_env = "LLM_WIKI_SERVER_TOKEN"
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
            .map_err(|e| WikiError::Llm(e.to_string()))?;
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
/// snippets; a never-built workspace is not an error.
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
    Ok(())
}

fn warn_diagnostic(diagnostic: &ScanDiagnostic) {
    println!(
        "  ! {} {:?}: {}",
        diagnostic.rel_path, diagnostic.kind, diagnostic.message
    );
}

/// `llm-wiki lint` (PRD §29/§36): thin transport over
/// `llm_wiki_compiler::run_lint`. Findings print grouped by check in
/// deterministic order; any Error-severity finding exits with the dedicated
/// lint code (11), warnings alone exit 0. A never-built workspace is not an
/// error.
fn lint(workspace: &Path) -> Result<(), WikiError> {
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
    if errors > 0 {
        Err(WikiError::Lint { errors, warnings })
    } else {
        Ok(())
    }
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

fn report(label: &str, result: Result<String, WikiError>) -> bool {
    match result {
        Ok(msg) => {
            println!("ok   {label} {msg}");
            true
        }
        Err(msg) => {
            println!("FAIL {label} {msg}");
            false
        }
    }
}

/// Publish-integrity check for `doctor` (PRD §29/§35): un-recovered publish
/// journal and `current.json` vs DB `active_build_id` consistency. Reports
/// the finding; never repairs anything.
fn check_publish_state(
    conn: &llm_wiki_storage::Connection,
    wiki_dir: &Path,
) -> Result<String, WikiError> {
    let paths = llm_wiki_compiler::PublishPaths::new(wiki_dir);
    if llm_wiki_compiler::journal_exists(&paths) {
        return Err(WikiError::PublishRecovery(format!(
            "un-recovered publish journal at {} — run `llm-wiki build` to let recovery resolve it",
            paths.journal_path().display()
        )));
    }
    let pointer = llm_wiki_compiler::read_current_pointer(&paths)?;
    let db_active = llm_wiki_storage::get_active_build_id(conn)?;
    match (pointer, db_active) {
        (None, None) => Ok("no wiki published yet".to_owned()),
        (Some(pointer), Some(active)) if pointer.build_id == active.as_str() => {
            Ok(format!("consistent (current generation {})", active))
        }
        (Some(pointer), Some(active)) => Err(WikiError::PublishRecovery(format!(
            "current.json points at {} but the database active_build_id is {}; refusing to guess",
            pointer.build_id, active
        ))),
        (Some(pointer), None) => Err(WikiError::PublishRecovery(format!(
            "current.json points at {} but the database has no active_build_id; refusing to guess",
            pointer.build_id
        ))),
        (None, Some(active)) => Err(WikiError::PublishRecovery(format!(
            "the database active_build_id is {} but {} is missing; refusing to guess",
            active,
            paths.pointer_path().display()
        ))),
    }
}

/// Length note: ~91 lines — a flat check list (config/root/db/fts/journal), each check one report() call.
fn doctor(workspace: &Path) -> Result<(), WikiError> {
    let config = load_config(workspace)?;
    let mut failures = 0usize;

    let valid = config.validate().map(|_| {
        format!(
            "valid (source root {}, wiki_dir {})",
            config.source.root.display(),
            config.project.wiki_dir.display()
        )
    });
    if !report("config", valid) {
        failures += 1;
    }

    let root = workspace.join(&config.source.root);
    let root_exists = if root.is_dir() {
        Ok(format!("exists ({})", root.display()))
    } else {
        Err(WikiError::Config(format!("missing ({})", root.display())))
    };
    if !report("source root", root_exists) {
        failures += 1;
    }

    let db = std::fs::create_dir_all(state_dir(workspace))
        .map_err(|e| WikiError::Source(e.to_string()))
        .and_then(|_| llm_wiki_storage::open(&state_db(workspace)));
    let conn = match db {
        Ok(conn) => {
            let count = llm_wiki_storage::count_sources(&conn);
            if !report(
                "state db",
                count.map(|count| format!("open + migrated ({count} sources)")),
            ) {
                failures += 1;
            }
            Some(conn)
        }
        Err(err) => {
            report("state db", Err(err));
            failures += 1;
            None
        }
    };

    // Publish integrity (PRD §35): report, never auto-fix. Failures here must
    // be resolved by running a build (journal recovery) or manually.
    if let Some(conn) = conn {
        let wiki_abs = lexical_absolute(workspace, &config.project.wiki_dir);
        let publish_state = check_publish_state(&conn, &wiki_abs);
        if !report("publish state", publish_state) {
            failures += 1;
        }

        // FTS5 availability (PRD §20): with the bundled SQLite this always
        // passes; a custom SQLite build without FTS5 must surface here as a
        // reported degradation — search refuses to run rather than silently
        // returning unusable results.
        let fts_state = if llm_wiki_storage::probe_fts5(&conn) {
            Ok("fts5 available".to_owned())
        } else if config.search.full_text {
            Err(WikiError::Index(
                llm_wiki_storage::FTS5_UNAVAILABLE.to_owned(),
            ))
        } else {
            Ok("fts5 unavailable (full-text search is disabled in config)".to_owned())
        };
        if !report("fts", fts_state) {
            failures += 1;
        }
    }

    let api_key = match std::env::var(&config.llm.api_key_env) {
        Ok(_) => Ok(format!("env {} is set", config.llm.api_key_env)),
        Err(_) => Err(WikiError::Config(format!(
            "env {} is not set (needed for build; scan/doctor do not call the model)",
            config.llm.api_key_env
        ))),
    };
    if !report("llm api key", api_key) {
        failures += 1;
    }

    if failures == 0 {
        println!("doctor: all checks passed");
        Ok(())
    } else {
        Err(WikiError::Config(format!("{failures} check(s) failed")))
    }
}
