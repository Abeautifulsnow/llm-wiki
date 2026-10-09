//! Project configuration (PRD §32): `.llm-wiki/config.toml`.
//!
//! Priority is fixed: explicit CLI args > environment (secrets/deployment
//! only) > project config > safe defaults. The effective config must be
//! printable without leaking secrets.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, WikiError};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub project: ProjectConfig,
    #[serde(default)]
    pub source: SourceConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub rerank: RerankConfig,
    #[serde(default)]
    pub analysis: AnalysisConfig,
    #[serde(default)]
    pub planning: PlanningConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub build: BuildConfig,
    #[serde(default)]
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    #[serde(default = "default_project_name")]
    pub name: String,
    /// Canonical managed generations + `current` pointer (PRD §32).
    #[serde(default = "default_wiki_dir")]
    pub wiki_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    #[serde(default = "default_source_root")]
    pub root: PathBuf,
    #[serde(default = "default_include")]
    pub include: Vec<String>,
    #[serde(default = "default_exclude")]
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub model: String,
    /// Name of the env var holding the API key. The key itself never lives in
    /// project config (PRD §32 / security constraint).
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    /// Per-request output ceiling for the analysis and planning stages (the
    /// compile stage scales its own estimate but never goes below this).
    /// Thinking models spend chain-of-thought tokens from the SAME budget —
    /// a reasoning-heavy response can exhaust 4096 before any visible JSON
    /// is emitted (observed: 10-13K chars of reasoning, content_len 0). Raise
    /// this when pointing at a thinking model.
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
    /// Thinking control for reasoning-capable models: `auto` (default — send
    /// nothing), `on`, `off`. Providers implement thinking differently; a
    /// rejected parameter downgrades to auto for the whole build with a
    /// warning in the log, never an error (T1 finding #5/#6).
    #[serde(default)]
    pub thinking: String,
    /// Optional reasoning-effort hint (`low`/`medium`/`high`, OpenAI-style
    /// `reasoning_effort`). Empty sends nothing. Same tolerant downgrade.
    #[serde(default)]
    pub thinking_effort: String,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: u32,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

/// Connection parameters shared by the chat, embedding and rerank endpoint
/// families: chat/embedding/rerank may live at THREE different providers, so
/// each stage carries its own endpoint instead of assuming `[llm]`'s.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub base_url: String,
    /// Name of the env var holding the API key — never the key itself.
    pub api_key_env: String,
    pub timeout_seconds: u64,
}

/// Vector-layer provider overrides (§19.3). Every field left empty/0
/// INHERITS the corresponding `[llm]` value: embedding may point at a
/// different provider (different base_url / key / model) by filling in only
/// what differs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Endpoint family. Empty inherits `llm.provider`; currently only
    /// `"openai-compatible"` is supported.
    #[serde(default)]
    pub provider: String,
    /// Empty inherits `llm.base_url`.
    #[serde(default)]
    pub base_url: String,
    /// Name of the env var holding the API key. Empty inherits
    /// `llm.api_key_env`.
    #[serde(default)]
    pub api_key_env: String,
    /// Embedding model. Empty falls back to `--embedding-model` /
    /// `$LLM_WIKI_EMBEDDING_MODEL` at resolution time.
    #[serde(default)]
    pub model: String,
    /// 0 inherits `llm.timeout_seconds`.
    #[serde(default)]
    pub timeout_seconds: u64,
}

/// Rerank-stage provider overrides (PRD §50 V0.5), used when
/// `[search] rerank = "cohere-compatible"`. Every field left empty/0
/// INHERITS the corresponding `[llm]` value — the rerank model may live at
/// its own provider. There is deliberately no `provider` field: the wire
/// protocol is named by the `[search] rerank` strategy itself.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Empty inherits `llm.base_url`.
    #[serde(default)]
    pub base_url: String,
    /// Name of the env var holding the API key. Empty inherits
    /// `llm.api_key_env`.
    #[serde(default)]
    pub api_key_env: String,
    /// Rerank model (required when the strategy is enabled).
    #[serde(default)]
    pub model: String,
    /// 0 inherits `llm.timeout_seconds`.
    #[serde(default)]
    pub timeout_seconds: u64,
}

impl LlmConfig {
    /// The chat endpoint as the inheritance root for embedding/rerank.
    pub fn endpoint(&self) -> EndpointConfig {
        EndpointConfig {
            base_url: self.base_url.clone(),
            api_key_env: self.api_key_env.clone(),
            timeout_seconds: self.timeout_seconds,
        }
    }
}

impl EmbeddingConfig {
    /// Resolves the embedding endpoint: explicit values win, empty/0
    /// inherits from the `[llm]` section.
    pub fn endpoint(&self, llm: &LlmConfig) -> EndpointConfig {
        let root = llm.endpoint();
        EndpointConfig {
            base_url: if self.base_url.trim().is_empty() {
                root.base_url
            } else {
                self.base_url.clone()
            },
            api_key_env: if self.api_key_env.trim().is_empty() {
                root.api_key_env
            } else {
                self.api_key_env.clone()
            },
            timeout_seconds: if self.timeout_seconds == 0 {
                root.timeout_seconds
            } else {
                self.timeout_seconds
            },
        }
    }

    /// Resolved endpoint family; empty inherits `llm.provider`.
    pub fn resolved_provider<'a>(&'a self, llm: &'a LlmConfig) -> &'a str {
        if self.provider.trim().is_empty() {
            &llm.provider
        } else {
            &self.provider
        }
    }
}

impl RerankConfig {
    /// Resolves the rerank endpoint: explicit values win, empty/0 inherits
    /// from the `[llm]` section.
    pub fn endpoint(&self, llm: &LlmConfig) -> EndpointConfig {
        let root = llm.endpoint();
        EndpointConfig {
            base_url: if self.base_url.trim().is_empty() {
                root.base_url
            } else {
                self.base_url.clone()
            },
            api_key_env: if self.api_key_env.trim().is_empty() {
                root.api_key_env
            } else {
                self.api_key_env.clone()
            },
            timeout_seconds: if self.timeout_seconds == 0 {
                root.timeout_seconds
            } else {
                self.timeout_seconds
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    #[serde(default = "default_max_input_tokens")]
    pub max_input_tokens: u32,
    #[serde(default = "default_section_target_tokens")]
    pub section_target_tokens: u32,
    #[serde(default = "default_max_plan_input_tokens")]
    pub max_plan_input_tokens: u32,
    #[serde(default = "default_max_rejected_claim_ratio")]
    pub max_rejected_claim_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConfig {
    /// Hierarchical planning (PRD §14). When disabled, planning must fail
    /// with an actionable error instead of executing an incomplete global
    /// plan if the estimated input exceeds `max_plan_input_tokens`.
    #[serde(default = "default_true")]
    pub hierarchical: bool,
    /// Subdivide clusters beyond this node count (never truncate).
    #[serde(default = "default_max_cluster_nodes")]
    pub max_cluster_nodes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_true")]
    pub full_text: bool,
    #[serde(default)]
    pub vector: bool,
    /// Gates the query-side consumption of the §17 Wiki Graph (one-hop
    /// related pages in `search` output). The stored graph itself is always
    /// rebuilt with the active generation, so this flag never makes the
    /// config promise a capability that does not exist.
    #[serde(default = "default_true")]
    pub graph: bool,
    /// Rerank strategy applied after rank fusion (PRD §50 V0.5). `"none"`
    /// keeps the deterministic fused order; `"cohere-compatible"` calls the
    /// Cohere/Jina-style `/rerank` endpoint described by the `[rerank]`
    /// section (empty fields inherit `[llm]`). Unknown values are config
    /// errors.
    #[serde(default = "default_rerank")]
    pub rerank: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    #[serde(default = "default_true")]
    pub incremental: bool,
    /// How many recent generations cleanup retains after a successful publish
    /// (PRD §35). The currently published generation is always kept on top of
    /// this window.
    #[serde(default = "default_keep_generations")]
    pub keep_generations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_bind")]
    pub bind: String,
    #[serde(default)]
    pub remote_enabled: bool,
    #[serde(default = "default_auth_token_env")]
    pub auth_token_env: String,
    /// Maximum jobs waiting in the queue before new build requests are
    /// rejected (PRD §30: job queue cap). Running jobs don't count.
    #[serde(default = "default_max_queued_jobs")]
    pub max_queued_jobs: u32,
    /// Request body size cap in bytes (PRD §30: request size limit).
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: u32,
    /// Per-caller request budget per minute in remote mode (PRD §30:
    /// per-caller rate limits). 0 disables the limiter.
    #[serde(default = "default_rate_limit_per_minute")]
    pub rate_limit_per_minute: u32,
    /// Re-enqueue build jobs that were interrupted by a server restart
    /// (PRD §31 "根据未来策略恢复"). Off by default: resuming spends LLM
    /// budget on restart, which must never come as a surprise.
    #[serde(default)]
    pub resume_interrupted_jobs: bool,
}

fn default_max_queued_jobs() -> u32 {
    8
}
fn default_max_body_bytes() -> u32 {
    1_048_576
}
fn default_rate_limit_per_minute() -> u32 {
    120
}

fn default_project_name() -> String {
    "llm-wiki".to_owned()
}
fn default_wiki_dir() -> PathBuf {
    PathBuf::from("./wiki")
}
fn default_source_root() -> PathBuf {
    PathBuf::from("./docs")
}
fn default_include() -> Vec<String> {
    vec![
        "**/*.md".to_owned(),
        "**/*.markdown".to_owned(),
        "**/*.mdx".to_owned(),
    ]
}
fn default_exclude() -> Vec<String> {
    vec![
        "wiki/**".to_owned(),
        ".llm-wiki/**".to_owned(),
        ".git/**".to_owned(),
        "node_modules/**".to_owned(),
    ]
}
fn default_provider() -> String {
    "openai-compatible".to_owned()
}
fn default_base_url() -> String {
    "http://localhost:8000/v1".to_owned()
}
fn default_api_key_env() -> String {
    "LLM_WIKI_API_KEY".to_owned()
}
fn default_max_output_tokens() -> u32 {
    4096
}
fn default_max_concurrency() -> u32 {
    4
}
fn default_timeout_seconds() -> u64 {
    120
}
fn default_max_input_tokens() -> u32 {
    32_000
}
fn default_section_target_tokens() -> u32 {
    6_000
}
fn default_max_plan_input_tokens() -> u32 {
    32_000
}
fn default_max_rejected_claim_ratio() -> f32 {
    0.10
}
fn default_max_cluster_nodes() -> u32 {
    24
}
fn default_keep_generations() -> u32 {
    3
}
fn default_true() -> bool {
    true
}
fn default_bind() -> String {
    "127.0.0.1".to_owned()
}
fn default_auth_token_env() -> String {
    "LLM_WIKI_SERVER_TOKEN".to_owned()
}
fn default_rerank() -> String {
    "none".to_owned()
}

impl Default for ProjectConfig {
    fn default() -> Self {
        ProjectConfig {
            name: default_project_name(),
            wiki_dir: default_wiki_dir(),
        }
    }
}

impl Default for SourceConfig {
    fn default() -> Self {
        SourceConfig {
            root: default_source_root(),
            include: default_include(),
            exclude: default_exclude(),
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig {
            provider: default_provider(),
            base_url: default_base_url(),
            model: String::new(),
            api_key_env: default_api_key_env(),
            max_output_tokens: default_max_output_tokens(),
            thinking: String::new(),
            thinking_effort: String::new(),
            max_concurrency: default_max_concurrency(),
            timeout_seconds: default_timeout_seconds(),
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        AnalysisConfig {
            max_input_tokens: default_max_input_tokens(),
            section_target_tokens: default_section_target_tokens(),
            max_plan_input_tokens: default_max_plan_input_tokens(),
            max_rejected_claim_ratio: default_max_rejected_claim_ratio(),
        }
    }
}

impl Default for PlanningConfig {
    fn default() -> Self {
        PlanningConfig {
            hierarchical: true,
            max_cluster_nodes: default_max_cluster_nodes(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            full_text: true,
            vector: false,
            graph: true,
            rerank: default_rerank(),
        }
    }
}

impl Default for BuildConfig {
    fn default() -> Self {
        BuildConfig {
            incremental: true,
            keep_generations: default_keep_generations(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            bind: default_bind(),
            remote_enabled: false,
            auth_token_env: default_auth_token_env(),
            max_queued_jobs: default_max_queued_jobs(),
            max_body_bytes: default_max_body_bytes(),
            rate_limit_per_minute: default_rate_limit_per_minute(),
            resume_interrupted_jobs: false,
        }
    }
}

impl Config {
    /// Parses a config file body. Missing fields fall back to safe defaults.
    pub fn parse_toml(text: &str) -> Result<Self> {
        let config: Config =
            toml::from_str(text).map_err(|e| WikiError::Config(format!("invalid TOML: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Loads `.llm-wiki/config.toml` under `workspace_root`, or returns the
    /// safe defaults when the file does not exist.
    pub fn load(workspace_root: &Path) -> Result<Self> {
        let path = workspace_root.join(".llm-wiki").join("config.toml");
        if !path.exists() {
            let config = Config::default();
            config.validate()?;
            return Ok(config);
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| WikiError::Config(format!("cannot read {}: {e}", path.display())))?;
        Config::parse_toml(&text)
    }

    /// Structural validation (PRD §8.4, §32):
    /// - source root and `wiki_dir` must not overlap — this is invalid config
    ///   and must be rejected before any build;
    /// - numeric bounds.
    pub fn validate(&self) -> Result<()> {
        if self.llm.max_concurrency == 0 {
            return Err(WikiError::Config("llm.max_concurrency must be >= 1".into()));
        }
        if self.llm.timeout_seconds == 0 {
            return Err(WikiError::Config("llm.timeout_seconds must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.analysis.max_rejected_claim_ratio) {
            return Err(WikiError::Config(
                "analysis.max_rejected_claim_ratio must be within [0, 1]".into(),
            ));
        }
        if self.analysis.max_plan_input_tokens == 0 || self.analysis.max_input_tokens == 0 {
            return Err(WikiError::Config(
                "analysis token budgets must be > 0".into(),
            ));
        }
        if self.planning.max_cluster_nodes < 2 {
            return Err(WikiError::Config(
                "planning.max_cluster_nodes must be >= 2".into(),
            ));
        }
        if self.build.keep_generations < 1 {
            return Err(WikiError::Config(
                "build.keep_generations must be >= 1 (the current generation is always kept)"
                    .into(),
            ));
        }
        match self.search.rerank.as_str() {
            "none" => {}
            "cohere-compatible" => {
                if self.rerank.model.trim().is_empty() {
                    return Err(WikiError::Config(
                        "[rerank] model must be set when search.rerank = \"cohere-compatible\""
                            .into(),
                    ));
                }
            }
            other => {
                return Err(WikiError::Config(format!(
                    "unsupported search.rerank strategy '{other}' (supported: none, cohere-compatible)"
                )));
            }
        }

        let root = lexical_absolute(Path::new(""), &self.source.root);
        let wiki = lexical_absolute(Path::new(""), &self.project.wiki_dir);
        if paths_overlap(&root, &wiki) {
            return Err(WikiError::Config(format!(
                "source root ({}) and wiki_dir ({}) overlap; wiki_dir must live outside the source tree",
                root.display(),
                wiki.display()
            )));
        }
        Ok(())
    }

    /// Human-readable effective config. Secret *values* never appear here by
    /// construction: config only stores env-var names (PRD §32).
    pub fn effective_summary(&self) -> String {
        format!(
            "project.name = {}\nproject.wiki_dir = {}\nsource.root = {}\nsource.include = {:?}\nsource.exclude = {:?}\nllm.provider = {}\nllm.base_url = {}\nllm.model = {}\nllm.api_key_env = <env name: redacted>\nllm.max_concurrency = {}\nembedding.provider = {}\nembedding.base_url = {}\nembedding.model = {}\nembedding.api_key_env = <env name: redacted>\nsearch.rerank = {}\nrerank.base_url = {}\nrerank.model = {}\nrerank.api_key_env = <env name: redacted>\nanalysis.max_input_tokens = {}\nanalysis.max_plan_input_tokens = {}\nanalysis.max_rejected_claim_ratio = {}\nplanning.hierarchical = {}\nplanning.max_cluster_nodes = {}\nbuild.incremental = {}\nbuild.keep_generations = {}\nserver.bind = {}\nserver.remote_enabled = {}",
            self.project.name,
            self.project.wiki_dir.display(),
            self.source.root.display(),
            self.source.include,
            self.source.exclude,
            self.llm.provider,
            self.llm.base_url,
            if self.llm.model.is_empty() { "<unset>" } else { &self.llm.model },
            self.llm.max_concurrency,
            self.embedding.resolved_provider(&self.llm),
            if self.embedding.base_url.is_empty() { &self.llm.base_url } else { &self.embedding.base_url },
            if self.embedding.model.is_empty() { "<unset>" } else { &self.embedding.model },
            self.search.rerank,
            if self.rerank.base_url.is_empty() { &self.llm.base_url } else { &self.rerank.base_url },
            if self.rerank.model.is_empty() { "<unset>" } else { &self.rerank.model },
            self.analysis.max_input_tokens,
            self.analysis.max_plan_input_tokens,
            self.analysis.max_rejected_claim_ratio,
            self.planning.hierarchical,
            self.planning.max_cluster_nodes,
            self.build.incremental,
            self.build.keep_generations,
            self.server.bind,
            self.server.remote_enabled,
        )
    }
}

/// Absolutizes `p` against `base` lexically (no filesystem round-trip), so it
/// works for not-yet-created directories such as a fresh `wiki_dir`.
pub fn lexical_absolute(base: &Path, p: &Path) -> PathBuf {
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    };
    let mut out = PathBuf::new();
    for comp in joined.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Overlap in both directions after case-folding components (Windows/macOS
/// delivery, PRD §8.3.1).
fn paths_overlap(a: &Path, b: &Path) -> bool {
    let a = fold_components(a);
    let b = fold_components(b);
    let (short, long) = if a.len() <= b.len() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    long.starts_with(short.as_slice())
}

fn fold_components(p: &Path) -> Vec<String> {
    p.components()
        .filter_map(|c| c.as_os_str().to_str().map(|s| s.nf_lowercase()))
        .collect()
}

trait NfLowercase {
    fn nf_lowercase(&self) -> String;
}
impl NfLowercase for str {
    fn nf_lowercase(&self) -> String {
        use unicode_normalization::UnicodeNormalization;
        self.chars().flat_map(char::to_lowercase).nfc().collect()
    }
}

#[cfg(test)]
fn overlap(a: &str, b: &str) -> bool {
    paths_overlap(Path::new(a), Path::new(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planning_config_defaults_and_bounds() {
        let cfg = Config::default();
        assert!(cfg.planning.hierarchical);
        assert_eq!(cfg.planning.max_cluster_nodes, 24);
        let mut bad = Config::default();
        bad.planning.max_cluster_nodes = 1;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn keep_generations_defaults_and_bounds() {
        let cfg = Config::default();
        assert_eq!(cfg.build.keep_generations, 3);
        let mut bad = Config::default();
        bad.build.keep_generations = 0;
        assert!(bad.validate().is_err());
        let mut minimal = Config::default();
        minimal.build.keep_generations = 1;
        minimal.validate().unwrap();
    }

    #[test]
    fn defaults_validate_cleanly() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn wiki_dir_inside_source_root_is_rejected() {
        let mut cfg = Config::default();
        cfg.source.root = PathBuf::from("./docs");
        cfg.project.wiki_dir = PathBuf::from("./docs/wiki");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn source_root_inside_wiki_dir_is_rejected() {
        let mut cfg = Config::default();
        cfg.source.root = PathBuf::from("./wiki/docs");
        cfg.project.wiki_dir = PathBuf::from("./wiki");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn sibling_dirs_are_fine() {
        let mut cfg = Config::default();
        cfg.source.root = PathBuf::from("./docs");
        cfg.project.wiki_dir = PathBuf::from("./wiki-out");
        cfg.validate().unwrap();
    }

    #[test]
    fn case_folded_overlap_is_rejected() {
        assert!(overlap("/repo/Docs", "/repo/docs/wiki"));
    }

    #[test]
    fn parse_toml_example_from_prd() {
        let cfg = Config::parse_toml(
            r#"
[project]
name = "example"
wiki_dir = "./wiki"

[source]
root = "./docs"

[llm]
provider = "openai-compatible"
base_url = "http://localhost:8000/v1"
model = "model-name"
api_key_env = "LLM_WIKI_API_KEY"

[analysis]
max_rejected_claim_ratio = 0.10
"#,
        )
        .unwrap();
        assert_eq!(cfg.llm.model, "model-name");
        assert_eq!(cfg.analysis.max_plan_input_tokens, 32_000);
        assert!(cfg.source.include.contains(&"**/*.mdx".to_owned()));
        assert!(cfg.source.exclude.contains(&".llm-wiki/**".to_owned()));
    }

    #[test]
    fn effective_summary_has_no_secret_material() {
        let summary = Config::default().effective_summary();
        assert!(!summary.contains("api_key ="));
        assert!(summary.contains("api_key_env = <env name: redacted>"));
        assert!(summary.contains("embedding.provider = openai-compatible"));
        assert!(summary.contains("rerank.model = <unset>"));
    }

    #[test]
    fn embedding_and_rerank_inherit_llm_endpoint_when_unset() {
        let mut cfg = Config::default();
        cfg.llm.base_url = "https://chat.example.com/v1".into();
        cfg.llm.api_key_env = "CHAT_KEY".into();
        cfg.llm.timeout_seconds = 77;
        // All-empty sections resolve to the [llm] endpoint.
        assert_eq!(cfg.embedding.endpoint(&cfg.llm).base_url, cfg.llm.base_url);
        assert_eq!(
            cfg.embedding.endpoint(&cfg.llm).api_key_env,
            cfg.llm.api_key_env
        );
        assert_eq!(cfg.rerank.endpoint(&cfg.llm).timeout_seconds, 77);
        assert_eq!(
            cfg.embedding.resolved_provider(&cfg.llm),
            "openai-compatible"
        );
    }

    #[test]
    fn embedding_and_rerank_can_point_at_their_own_providers() {
        let defaults = Config::default();
        let mut cfg = Config {
            embedding: EmbeddingConfig {
                provider: "openai-compatible".into(),
                base_url: "https://embed.example.com/v1".into(),
                api_key_env: "EMBED_KEY".into(),
                model: "bge-m3".into(),
                timeout_seconds: 30,
            },
            rerank: RerankConfig {
                base_url: "https://rerank.example.com/v1".into(),
                api_key_env: "RERANK_KEY".into(),
                model: "bge-reranker-v2-m3".into(),
                timeout_seconds: 15,
            },
            search: SearchConfig {
                rerank: "cohere-compatible".into(),
                ..SearchConfig::default()
            },
            ..defaults
        };
        cfg.validate().unwrap();
        let embedding = cfg.embedding.endpoint(&cfg.llm);
        assert_eq!(embedding.base_url, "https://embed.example.com/v1");
        assert_eq!(embedding.api_key_env, "EMBED_KEY");
        assert_eq!(embedding.timeout_seconds, 30);
        let rerank = cfg.rerank.endpoint(&cfg.llm);
        assert_eq!(rerank.base_url, "https://rerank.example.com/v1");
        assert_eq!(rerank.api_key_env, "RERANK_KEY");
        // Partial override: only base_url set, the rest inherits [llm].
        cfg.embedding.base_url = "https://only-embed.example.com/v1".into();
        cfg.embedding.api_key_env = String::new();
        let partial = cfg.embedding.endpoint(&cfg.llm);
        assert_eq!(partial.base_url, "https://only-embed.example.com/v1");
        assert_eq!(partial.api_key_env, "LLM_WIKI_API_KEY");
    }

    #[test]
    fn rerank_strategy_validation() {
        let config_with = |strategy: &str, model: &str| Config {
            search: SearchConfig {
                rerank: strategy.to_owned(),
                ..SearchConfig::default()
            },
            rerank: RerankConfig {
                model: model.to_owned(),
                ..RerankConfig::default()
            },
            ..Config::default()
        };

        let err = config_with("cross-encoder", "any")
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported search.rerank strategy"), "{err}");

        let err = config_with("cohere-compatible", "")
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("[rerank] model must be set"), "{err}");

        config_with("cohere-compatible", "bge-reranker-v2-m3")
            .validate()
            .unwrap();
    }

    #[test]
    fn lexical_absolute_resolves_dots() {
        let p = lexical_absolute(Path::new("/repo"), Path::new("./docs/../wiki"));
        assert_eq!(p, PathBuf::from("/repo/wiki"));
    }
}
