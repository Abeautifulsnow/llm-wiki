//! Restricted YAML frontmatter handling (PRD §9).
//!
//! Frontmatter is metadata only: it must never override system fields
//! (`id`, `generated`, `schema_version`, citations). V0.1 parses the flat
//! `key: value` subset; anything else is a deterministic diagnostic, not a
//! hard failure.

use std::collections::BTreeMap;

use crate::parser::{ParseDiagnostic, ParseDiagnosticKind};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Frontmatter {
    map: BTreeMap<String, String>,
}

impl Frontmatter {
    /// Case-insensitive lookup.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(&key.to_ascii_lowercase()).map(|s| s.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Detects and strips a leading `---` frontmatter block.
///
/// Returns `(frontmatter, body, diagnostic)`. A malformed block (missing
/// closing fence) yields a diagnostic and the raw text is parsed as body.
pub fn strip_and_parse(raw: &str) -> (Option<Frontmatter>, String, Option<ParseDiagnostic>) {
    let Some(first_line_end) = raw.find('\n') else {
        return (None, raw.to_owned(), None);
    };
    let first = raw[..first_line_end].trim_end_matches('\r');
    if first != "---" {
        return (None, raw.to_owned(), None);
    }

    let mut fm = Frontmatter::default();
    let mut malformed: Option<String> = None;
    let mut closed = false;
    let mut body_start = raw.len();

    // split_inclusive keeps every byte (including '\r' before '\n') inside
    // one chunk, so the offset bookkeeping is exact for CRLF documents too —
    // `.lines()` would silently drop '\r' and shift body_start left.
    let mut consumed = first_line_end + 1;
    for chunk in raw[first_line_end + 1..].split_inclusive('\n') {
        consumed += chunk.len();
        let trimmed = chunk.trim_end_matches(['\r', '\n']);
        if trimmed == "---" || trimmed == "..." {
            body_start = consumed.min(raw.len());
            closed = true;
            break;
        }
        if trimmed.is_empty() {
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            if malformed.is_none() {
                malformed = Some(format!("frontmatter line is not `key: value`: {trimmed:?}"));
            }
            continue;
        };
        let key = key.trim();
        let mut value = value.trim();
        if value.len() >= 2
            && ((value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\'')))
        {
            value = &value[1..value.len() - 1];
        }
        if key.is_empty() {
            if malformed.is_none() {
                malformed = Some(format!("frontmatter line has empty key: {trimmed:?}"));
            }
            continue;
        }
        fm.map.insert(key.to_ascii_lowercase(), value.to_owned());
    }

    if malformed.is_none() && body_start == raw.len() {
        // Ran off the end without a closing fence.
        malformed = Some("frontmatter block is not closed with `---`".to_owned());
    }

    let unterminated = !closed;
    let diagnostic = malformed.map(|message| ParseDiagnostic {
        kind: ParseDiagnosticKind::MalformedFrontmatter,
        message,
        range: None,
    });

    // An unterminated block is not stripped: its lines stay in the body as
    // plain text and the diagnostic above marks them. A closed block with a
    // malformed line is still stripped.
    if unterminated {
        return (None, raw.to_owned(), diagnostic);
    }
    (Some(fm), raw[body_start..].to_owned(), diagnostic)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_keys_and_strips_quotes() {
        let raw = "---\ntitle: My Doc\nlang: \"zh-CN\"\nignore_system_id: x\n---\n\n# Body\n";
        let (fm, body, diag) = strip_and_parse(raw);
        let fm = fm.unwrap();
        assert!(diag.is_none());
        assert_eq!(fm.get("title"), Some("My Doc"));
        assert_eq!(fm.get("LANG"), Some("zh-CN"));
        assert_eq!(body, "\n# Body\n");
    }

    #[test]
    fn missing_closing_fence_is_a_diagnostic_and_body_is_raw() {
        let raw = "---\ntitle: Broken\n\n# Body";
        let (fm, body, diag) = strip_and_parse(raw);
        assert!(fm.is_none());
        assert_eq!(body, raw);
        assert_eq!(
            diag.unwrap().kind,
            ParseDiagnosticKind::MalformedFrontmatter
        );
    }

    #[test]
    fn non_key_value_line_is_a_diagnostic_but_block_still_parses() {
        let raw = "---\ntitle: Ok\njusttext\n---\nbody";
        let (fm, body, diag) = strip_and_parse(raw);
        assert_eq!(fm.unwrap().get("title"), Some("Ok"));
        assert_eq!(body, "body");
        assert!(diag.is_some());
    }

    #[test]
    fn crlf_frontmatter_strips_cleanly() {
        let raw = "---\r\ntitle: My Doc\r\nlang: en\r\n---\r\n\r\n# Body\r\n\r\ntext\r\n";
        let (fm, body, diag) = strip_and_parse(raw);
        let fm = fm.unwrap();
        assert!(diag.is_none());
        assert_eq!(fm.get("title"), Some("My Doc"));
        assert_eq!(fm.get("lang"), Some("en"));
        assert!(
            !body.starts_with('-'),
            "fence must not leak into body: {body:?}"
        );
        assert_eq!(body, "\r\n# Body\r\n\r\ntext\r\n");
    }

    #[test]
    fn no_frontmatter_passes_through() {
        let (fm, body, diag) = strip_and_parse("# Just a doc\n");
        assert!(fm.is_none());
        assert_eq!(body, "# Just a doc\n");
        assert!(diag.is_none());
    }
}
