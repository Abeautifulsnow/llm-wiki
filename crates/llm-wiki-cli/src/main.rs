#![forbid(unsafe_code)]
//! `llm-wiki` CLI. Per PRD §7.7 this crate only parses arguments and calls
//! application services; all logic lives in the library crates.

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use llm_wiki_core::config::Config;
use llm_wiki_core::error::WikiError;
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
    /// Show source counts and the latest build.
    Status,
    /// Check configuration, filesystem layout, state db and provider env.
    Doctor,
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
        Command::Status => status(&workspace),
        Command::Doctor => doctor(&workspace),
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

fn warn_diagnostic(diagnostic: &ScanDiagnostic) {
    println!(
        "  ! {} {:?}: {}",
        diagnostic.rel_path, diagnostic.kind, diagnostic.message
    );
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
        .and_then(|_| llm_wiki_storage::open(&state_db(workspace)))
        .and_then(|conn| {
            llm_wiki_storage::count_sources(&conn)
                .map(|count| format!("open + migrated ({count} sources)"))
        });
    if !report("state db", db) {
        failures += 1;
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
