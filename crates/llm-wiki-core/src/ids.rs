//! Opaque, registry-assigned identifiers (PRD §8.2, §12.1.1, §45).
//!
//! Rules encoded here:
//! - IDs are opaque strings (`<prefix>_<ULID>`); they are never derived from
//!   names or content, and LLM output may only *reference* them.
//! - ULID strings are fixed-width Crockford base32, so lexicographic order is
//!   also chronological order; derived `Ord` is therefore stable.
//! - `SourceLocatorKey` is the one recomputable identity
//!   (`hash(workspace + normalized relative path)`); every other ID is
//!   assigned once by a registry and then immutable.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::WikiError;

macro_rules! opaque_id {
    ($(#[$doc:meta])* $name:ident, $prefix:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub const PREFIX: &'static str = $prefix;

            /// Assigns a fresh opaque identity. Only registries call this.
            pub fn generate() -> Self {
                Self(format!("{}_{}", $prefix, ulid::Ulid::new()))
            }

            /// Wraps an existing ID string, validating the prefix.
            pub fn parse(value: impl Into<String>) -> Result<Self, WikiError> {
                let value = value.into();
                let reason = match Self::validate(&value) {
                    Ok(()) => return Ok(Self(value)),
                    Err(reason) => reason,
                };
                Err(WikiError::InvalidId {
                    value,
                    reason: reason.to_owned(),
                })
            }

            fn validate(value: &str) -> Result<(), &'static str> {
                let suffix = value
                    .strip_prefix($prefix)
                    .ok_or("missing prefix")?
                    .strip_prefix('_')
                    .ok_or("missing separator after prefix")?;
                if suffix.is_empty() {
                    return Err("empty id body");
                }
                if !suffix
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric())
                {
                    return Err("id body must be alphanumeric");
                }
                Ok(())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Wraps a value that is already known to be valid (e.g. read
            /// back from storage). Never use this for unvalidated input;
            /// prefer [`Self::parse`].
            pub fn from_validated(value: impl Into<String>) -> Self {
                Self(value.into())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

opaque_id!(
    /// Persistent source identity assigned by the Source Registry (PRD §8.2).
    SourceId,
    "src"
);
opaque_id!(
    /// Source-local section identity assigned by the Section Registry and
    /// carried across builds by the Section Matcher (PRD §45).
    SectionId,
    "sec"
);
opaque_id!(
    /// Stable knowledge identity assigned by the Knowledge Registry (PRD §12.1.1).
    KnowledgeNodeId,
    "kn"
);
opaque_id!(
    /// Wiki page identity created by the planner and persisted (PRD §45).
    WikiPageId,
    "wp"
);
opaque_id!(
    /// Citation identity (PRD §12.3).
    CitationId,
    "cit"
);
opaque_id!(
    /// Build identity; also the immutable generation directory name (PRD §35).
    BuildId,
    "bld"
);
opaque_id!(
    /// Stage-one analysis record identity (PRD §11).
    AnalysisId,
    "an"
);
opaque_id!(
    /// Claim row identity; the claim's *knowledge* identity is the registry
    /// `KnowledgeNodeId` it references (PRD §12.1.1).
    ClaimRowId,
    "cl"
);
opaque_id!(
    /// Relation row identity.
    RelationRowId,
    "rel"
);
opaque_id!(
    /// Auditable rejected-claim record identity (PRD §11.1).
    RejectedClaimId,
    "rj"
);

opaque_id!(
    /// Recomputable source locator: `hash(workspace + normalized relative path)`
    /// (PRD §8.2). Not an identity — the Source Registry maps it to a
    /// [`SourceId`].
    SourceLocatorKey,
    "loc"
);

impl SourceLocatorKey {
    /// Computes the locator key from the workspace id (canonical source root)
    /// and the normalized (`/`-separated, NFC) relative path.
    pub fn compute(workspace_id: &str, normalized_rel_path: &str) -> Self {
        let digest = crate::hash::sha256_hex(
            format!("{}\u{0}{}", workspace_id, normalized_rel_path).as_bytes(),
        );
        Self(format!("loc_{}", digest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_ids_have_prefix_and_are_unique() {
        let a = KnowledgeNodeId::generate();
        let b = KnowledgeNodeId::generate();
        assert!(a.as_str().starts_with("kn_"));
        assert_ne!(a, b);
        assert_eq!(a.as_str().len(), "kn_".len() + 26);
    }

    #[test]
    fn parse_rejects_wrong_prefix_and_garbage() {
        assert!(KnowledgeNodeId::parse(SourceId::generate().as_str()).is_err());
        assert!(KnowledgeNodeId::parse("kn_").is_err());
        assert!(KnowledgeNodeId::parse("kn_has space").is_err());
        assert!(KnowledgeNodeId::parse("plain").is_err());
        assert!(KnowledgeNodeId::parse(KnowledgeNodeId::generate().as_str()).is_ok());
    }

    #[test]
    fn ids_sort_chronologically() {
        // Fixed ULIDs from the spec with increasing timestamps: lexicographic
        // order of the Crockford base32 body is chronological order, which is
        // what the derived `Ord` relies on.
        let a = KnowledgeNodeId::parse("kn_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let b = KnowledgeNodeId::parse("kn_01BX5ZZKBKACTAV9WEVGEMMVRZ").unwrap();
        assert!(a < b);
    }

    #[test]
    fn locator_key_is_recomputable_and_path_sensitive() {
        let a = SourceLocatorKey::compute("ws", "docs/a.md");
        let b = SourceLocatorKey::compute("ws", "docs/a.md");
        let c = SourceLocatorKey::compute("ws", "docs/b.md");
        let d = SourceLocatorKey::compute("other", "docs/a.md");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
}
