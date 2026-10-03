#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

pub mod analysis;
pub mod cancel;
pub mod config;
pub mod error;
pub mod hash;
pub mod ids;
pub mod matcher;
pub mod model;
pub mod plan;

pub use cancel::CancelFlag;

pub use error::{Result, WikiError};
pub use ids::{
    AnalysisId, BuildId, CitationId, ClaimRowId, InsightId, JobId, KnowledgeNodeId, LinkRowId,
    RejectedClaimId, RelationRowId, SectionId, SourceId, SourceLocatorKey, WikiPageId,
};
