//! Unified error model (PRD §34).
//!
//! Every error the system produces maps to one of the PRD error categories;
//! the CLI derives its non-zero exit code from the category.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WikiError {
    #[error("source error: {0}")]
    Source(String),

    #[error("path collision detected (paths fold to the same portable key): {collisions:?}")]
    PathCollision { collisions: Vec<String> },

    #[error("parse error at {location}: {message}")]
    Parse { location: String, message: String },

    #[error("storage error: {0}")]
    Storage(String),

    #[error("llm error: {0}")]
    Llm(String),

    #[error("schema validation failed: {0}")]
    SchemaValidation(String),

    #[error("evidence validation failed: {0}")]
    EvidenceValidation(String),

    #[error("planning error: {0}")]
    Planning(String),

    #[error("replan required: {reason} (run `llm-wiki replan --dry-run`)")]
    ReplanRequired { reason: String },

    #[error("compilation error: {0}")]
    Compilation(String),

    #[error("index error: {0}")]
    Index(String),

    #[error("budget exceeded: {0}")]
    BudgetExceeded(String),

    #[error(
        "publish recovery required: {0}; the last consistent generation is retained and must not be guessed"
    )]
    PublishRecovery(String),

    /// `llm-wiki lint` found at least one Error-severity finding (PRD §34/§36).
    /// Carries the counts so the CLI can report them; warnings alone exit 0.
    #[error("lint found {errors} error(s) and {warnings} warning(s)")]
    Lint { errors: u32, warnings: u32 },

    /// Cooperative cancellation (PRD §31): stop starting new LLM requests and
    /// abandon the unpublished generation. The previous generation stays
    /// visible.
    #[error("cancelled")]
    Cancelled,

    #[error("config error: {0}")]
    Config(String),

    #[error("invalid id '{value}': {reason}")]
    InvalidId { value: String, reason: String },
}

impl WikiError {
    /// Non-zero CLI exit code per category (PRD §34).
    pub fn exit_code(&self) -> i32 {
        match self {
            WikiError::Config(_) | WikiError::InvalidId { .. } => 2,
            WikiError::Source(_) | WikiError::PathCollision { .. } => 3,
            WikiError::Parse { .. } => 4,
            WikiError::Storage(_) => 5,
            WikiError::Llm(_)
            | WikiError::SchemaValidation(_)
            | WikiError::EvidenceValidation(_) => 6,
            WikiError::Planning(_) | WikiError::ReplanRequired { .. } => 7,
            WikiError::Compilation(_) | WikiError::Index(_) => 8,
            WikiError::BudgetExceeded(_) => 9,
            WikiError::PublishRecovery(_) => 10,
            WikiError::Lint { .. } => 11,
            WikiError::Cancelled => 12,
        }
    }
}

pub type Result<T> = std::result::Result<T, WikiError>;
