//! Deterministic MDX JSX component downgrade (PRD §9).
//!
//! `.mdx` files are parsed as Markdown *after* their embedded JSX components
//! are downgraded. Rules:
//! - JSX components (capitalized or namespaced tag names, e.g. `<Callout>`,
//!   `<Tabs>`, `</Tabs>`, `<Badge label="x" />`) are unwrapped: text children
//!   remain, string attribute values are kept as readable text.
//! - Standard HTML tags (`<div>`, `<br>`, …) are left as-is.
//! - JSX comments (`{/* ... */}`) are dropped — they are recognized markup,
//!   not content.
//! - Anything unrecognized (unterminated tags, non-literal expressions) is
//!   **never silently dropped**: the raw text is kept and a diagnostic is
//!   produced.
//!
//! The downgrade only affects the text handed to analysis; the Source itself
//! is never rewritten (PRD §9). Note that `source_range`s of the parsed
//! document refer to the *downgraded* text returned by [`downgrade_jsx`].

use pulldown_cmark::{Event, Options, Parser};

use crate::parser::{ParseDiagnostic, ParseDiagnosticKind};

/// Downgrades JSX in a document body. Returns the downgraded text and the
/// list of diagnostics (empty when the downgrade was lossless).
pub fn downgrade_jsx(body: &str) -> (String, Vec<ParseDiagnostic>) {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);

    let mut replacements: Vec<(std::ops::Range<usize>, String)> = Vec::new();
    let mut diagnostics: Vec<ParseDiagnostic> = Vec::new();

    for (event, range) in Parser::new_ext(body, options).into_offset_iter() {
        let html = match event {
            Event::Html(h) | Event::InlineHtml(h) => h,
            _ => continue,
        };
        match downgrade_chunk(&html) {
            ChunkResult::Replace(new_text) => {
                // A block-level chunk usually ends with the line's newline;
                // keep it so the following line does not glue onto this one.
                let mut new_text = new_text;
                if html.ends_with('\n') && !new_text.ends_with('\n') {
                    new_text.push('\n');
                }
                replacements.push((range, new_text));
            }
            ChunkResult::KeepRaw => {}
            ChunkResult::KeepWithDiagnostic(message) => {
                diagnostics.push(ParseDiagnostic {
                    kind: ParseDiagnosticKind::UnrecognizedJsx,
                    message,
                    range: Some(llm_wiki_core::model::SourceRange::new(
                        range.start,
                        range.end,
                    )),
                });
            }
        }
    }

    // Genuinely broken JSX never reaches the parser as HTML — CommonMark
    // passes it through as plain text. A raw line scan (fence-aware) makes
    // sure such constructs still produce a diagnostic instead of a silent
    // passthrough.
    diagnostics.extend(raw_diagnostics(body));

    let mut out = body.to_owned();
    // Apply from the end so earlier offsets stay valid.
    for (range, new_text) in replacements.into_iter().rev() {
        out.replace_range(range, &new_text);
    }
    out = drop_jsx_comment_lines(&out);
    (out, diagnostics)
}

/// Removes single-line `{/* … */}` JSX comment lines (fence-aware). They are
/// recognized markup, not content; multi-line comments are left as text.
fn drop_jsx_comment_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_fence: Option<&str> = None;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if let Some(marker) = in_fence {
            out.push_str(line);
            if trimmed.starts_with(marker) {
                in_fence = None;
            }
            continue;
        }
        if trimmed.starts_with("```") {
            in_fence = Some("```");
            out.push_str(line);
            continue;
        }
        if trimmed.starts_with("~~~") {
            in_fence = Some("~~~");
            out.push_str(line);
            continue;
        }
        if trimmed.starts_with("{/*") && trimmed.ends_with("*/}") {
            continue;
        }
        out.push_str(line);
    }
    out
}

/// Fence-aware scan for JSX-looking tags that the CommonMark layer treated
/// as plain text (i.e. syntactically broken ones).
fn raw_diagnostics(body: &str) -> Vec<ParseDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut in_fence: Option<String> = None;
    let mut offset = 0usize;
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim();
        let line_start = offset;
        offset += line.len();

        if let Some(marker) = in_fence.as_ref() {
            if trimmed.starts_with(marker.as_str()) {
                in_fence = None;
            }
            continue;
        }
        if trimmed.starts_with("```") {
            in_fence = Some("```".to_owned());
            continue;
        }
        if trimmed.starts_with("~~~") {
            in_fence = Some("~~~".to_owned());
            continue;
        }

        for (pos, _) in trimmed.match_indices('<') {
            let rest = &trimmed[pos + 1..];
            let looks_like_component = rest.starts_with('/')
                && rest
                    .chars()
                    .nth(1)
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
                || rest
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false);
            if !looks_like_component {
                continue;
            }
            // A well-formed component tag on this line has a closing '>'
            // with balanced quotes before it.
            let tag_part = &trimmed[pos..];
            if !has_complete_tag_end(tag_part) {
                diagnostics.push(ParseDiagnostic {
                    kind: ParseDiagnosticKind::UnrecognizedJsx,
                    message: format!(
                        "JSX-looking tag without a well-formed closing '>' kept verbatim: {tag_part}"
                    ),
                    range: Some(llm_wiki_core::model::SourceRange::new(
                        line_start,
                        line_start + line.len(),
                    )),
                });
                break;
            }
        }
    }
    diagnostics
}

fn has_complete_tag_end(tag_part: &str) -> bool {
    find_tag_end(tag_part, 1).is_some()
}

enum ChunkResult {
    Replace(String),
    KeepRaw,
    KeepWithDiagnostic(String),
}

/// Downgrades one HTML event chunk.
fn downgrade_chunk(html: &str) -> ChunkResult {
    let trimmed = html.trim_start();
    if trimmed.starts_with("<!--") {
        return ChunkResult::Replace(String::new()); // HTML comment
    }
    if trimmed.starts_with('{') {
        let inner = trimmed.trim_start_matches('{').trim_start();
        if inner.starts_with("/*") {
            return ChunkResult::Replace(String::new()); // JSX comment
        }
        return ChunkResult::KeepWithDiagnostic(format!(
            "unrecognized JSX expression kept verbatim: {}",
            first_line(html)
        ));
    }
    if !trimmed.starts_with('<') {
        return ChunkResult::KeepRaw; // plain content that pulldown passed through
    }

    let closing = trimmed.starts_with("</");
    let (name, name_end) = match parse_tag_name(trimmed, if closing { 2 } else { 1 }) {
        Some(parsed) => parsed,
        None => {
            return ChunkResult::KeepWithDiagnostic(format!(
                "unterminated JSX tag kept verbatim: {}",
                first_line(html)
            ))
        }
    };

    if !is_component(&name) {
        return ChunkResult::KeepRaw; // standard HTML tag stays as-is
    }

    if closing {
        return ChunkResult::Replace(String::new());
    }

    // Opening tag: scan attributes up to the matching '>'.
    match find_tag_end(trimmed, name_end) {
        None => ChunkResult::KeepWithDiagnostic(format!(
            "unterminated JSX tag kept verbatim: {}",
            first_line(html)
        )),
        Some(gt) => {
            let self_closing = trimmed[..gt].trim_end().ends_with('/');
            let attrs = &trimmed[name_end..if self_closing { gt - 1 } else { gt }];
            match collect_readable_attrs(attrs) {
                AttrScan::Readable(values) => {
                    let mut new_text = values.join(" ");
                    if !new_text.is_empty() {
                        new_text.push(' ');
                    }
                    if !self_closing {
                        // Same-chunk content (no blank line between the open
                        // tag and the block end): keep it, and strip a closing
                        // tag if it lives in this chunk.
                        let rest = &trimmed[gt + 1..];
                        let rest = strip_closing_tag(rest, &name);
                        new_text.push_str(rest);
                    }
                    ChunkResult::Replace(new_text)
                }
                AttrScan::Unrecognized(what) => ChunkResult::KeepWithDiagnostic(format!(
                    "unrecognized JSX attribute ({what}) kept verbatim: {}",
                    first_line(html)
                )),
            }
        }
    }
}

fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("").trim();
    if line.chars().count() > 80 {
        let cut: String = line.chars().take(77).collect();
        format!("{cut}...")
    } else {
        line.to_owned()
    }
}

fn strip_closing_tag<'a>(rest: &'a str, name: &str) -> &'a str {
    let needle = format!("</{name}");
    match rest.rfind(&needle) {
        // Keep the children (everything before the closing tag); the tag
        // itself and anything after it inside this chunk is dropped.
        Some(pos) => rest[..pos].trim_end(),
        None => rest,
    }
}

fn parse_tag_name(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    let mut end = start;
    while end < bytes.len() {
        let b = bytes[end];
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || b == b'-' || b == b':' {
            end += 1;
        } else {
            break;
        }
    }
    if end == start {
        return None;
    }
    Some((s[start..end].to_owned(), end))
}

fn is_component(name: &str) -> bool {
    name.chars()
        .next()
        .map(|c| c.is_ascii_uppercase())
        .unwrap_or(false)
        || name.contains(':')
}

/// Finds the `>` closing the tag, honoring double/single-quoted strings and
/// brace expressions.
fn find_tag_end(s: &str, from: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = from;
    let mut quote: Option<u8> = None;
    let mut brace_depth = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];
        if let Some(active_quote) = quote {
            if byte == b'\\' {
                i += 2;
                continue;
            }
            if byte == active_quote {
                quote = None;
            }
        } else if brace_depth > 0 {
            if byte == b'{' {
                brace_depth += 1;
            } else if byte == b'}' {
                brace_depth -= 1;
            }
        } else {
            match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'{' => brace_depth += 1,
                b'>' => return Some(i),
                _ => {}
            }
        }
        i += 1;
    }
    None
}

enum AttrScan {
    Readable(Vec<String>),
    Unrecognized(String),
}

fn collect_readable_attrs(attrs: &str) -> AttrScan {
    let mut values = Vec::new();
    let bytes = attrs.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'{' => {
                // Spread/expression attribute: JSX comment allowed, anything
                // else is unreadable.
                let inner_start = i + 1;
                let mut depth = 1usize;
                let mut j = inner_start;
                let mut quote: Option<u8> = None;
                while j < bytes.len() && depth > 0 {
                    let byte = bytes[j];
                    if let Some(active_quote) = quote {
                        if byte == b'\\' {
                            j += 2;
                            continue;
                        }
                        if byte == active_quote {
                            quote = None;
                        }
                    } else {
                        match byte {
                            b'"' | b'\'' => quote = Some(byte),
                            b'{' => depth += 1,
                            b'}' => depth -= 1,
                            _ => {}
                        }
                    }
                    j += 1;
                }
                if depth != 0 {
                    return AttrScan::Unrecognized("unterminated expression".to_owned());
                }
                let inner = attrs[inner_start..j - 1].trim();
                if !inner.starts_with("/*") {
                    return AttrScan::Unrecognized(format!("expression: {{ {inner} }}"));
                }
                i = j;
            }
            _ => {
                // key = value | key
                let key_start = i;
                while i < bytes.len()
                    && !bytes[i].is_ascii_whitespace()
                    && bytes[i] != b'='
                    && bytes[i] != b'>'
                {
                    i += 1;
                }
                let key = &attrs[key_start..i];
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'=' {
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                    if i >= bytes.len() {
                        return AttrScan::Unrecognized(format!("unterminated value for '{key}'"));
                    }
                    match bytes[i] {
                        quote_char @ (b'"' | b'\'') => {
                            i += 1;
                            let val_start = i;
                            while i < bytes.len() && bytes[i] != quote_char {
                                if bytes[i] == b'\\' {
                                    i += 1;
                                }
                                i += 1;
                            }
                            if i >= bytes.len() {
                                return AttrScan::Unrecognized(format!(
                                    "unterminated string for '{key}'"
                                ));
                            }
                            values.push(attrs[val_start..i].to_owned());
                            i += 1;
                        }
                        b'{' => {
                            return AttrScan::Unrecognized(format!(
                                "non-literal value for '{key}'"
                            ));
                        }
                        _ => {
                            let val_start = i;
                            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                                i += 1;
                            }
                            values.push(attrs[val_start..i].to_owned());
                        }
                    }
                }
                // bare key: no readable value, ignored
            }
        }
    }
    AttrScan::Readable(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn downgrade(body: &str) -> String {
        downgrade_jsx(body).0
    }

    #[test]
    fn standalone_components_are_unwrapped() {
        let out = downgrade("<Callout>\n\nSome text here.\n\n</Callout>\n");
        assert_eq!(out, "\n\nSome text here.\n\n\n");
        assert!(
            downgrade_jsx("<Callout>\n\nSome text here.\n\n</Callout>\n")
                .1
                .is_empty()
        );
    }

    #[test]
    fn string_attributes_survive_as_text() {
        let out = downgrade("<Badge label=\"Stable\" tone=\"green\" />\n");
        assert_eq!(out, "Stable green \n");
    }

    #[test]
    fn same_chunk_children_survive() {
        let out = downgrade("<Tabs>Tab body text</Tabs>\n");
        assert_eq!(out, "Tab body text\n");
    }

    #[test]
    fn standard_html_is_untouched() {
        let raw = "<div class=\"x\">html stays</div>\n";
        assert_eq!(downgrade(raw), raw);
    }

    #[test]
    fn code_blocks_are_untouched() {
        let raw = "```mdx\n<Tabs>not real jsx</Tabs>\n```\n";
        assert_eq!(downgrade(raw), raw);
    }

    #[test]
    fn jsx_comments_are_dropped() {
        let out = downgrade("{/* a comment */}\nVisible text\n");
        assert_eq!(out, "Visible text\n");
    }

    #[test]
    fn unterminated_tag_is_kept_with_diagnostic() {
        let body = "<Callout title=\"broken\n";
        let (out, diags) = downgrade_jsx(body);
        assert_eq!(out, body, "unrecognized JSX must not be silently dropped");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].kind, ParseDiagnosticKind::UnrecognizedJsx);
    }

    #[test]
    fn non_literal_attribute_is_kept_with_diagnostic() {
        let body = "<Badge count={n} />\n";
        let (out, diags) = downgrade_jsx(body);
        assert_eq!(out, body);
        assert_eq!(diags.len(), 1);
    }
}
