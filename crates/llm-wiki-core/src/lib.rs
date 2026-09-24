#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

pub mod config;
pub mod error;
pub mod hash;
pub mod ids;
pub mod matcher;
pub mod model;

pub use error::{Result, WikiError};
pub use ids::{
    BuildId, CitationId, KnowledgeNodeId, SectionId, SourceId, SourceLocatorKey, WikiPageId,
};
