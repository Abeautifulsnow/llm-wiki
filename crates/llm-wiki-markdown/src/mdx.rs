//! Deterministic MDX JSX component downgrade (PRD §9, EPIC C).
//!
//! `.mdx` files are parsed as Markdown *after* their embedded JSX components
//! are downgraded. Rules:
//! - **Structural components** — `Callout`, `Steps`/`Step`, `Cards`/`Card`,
//!   `Accordions`/`Accordion`, `Tabs`/`Tab` — are downgraded to Markdown that
//!   *carries their semantics*, never leaking a bare attribute keyword:
//!   - `Callout` → a readable label (`<Callout type="warn">` → `**Warning:**`),
//!     with the type → label map below;
//!   - `Step` → an ordered-list marker (`1.`, `2.`, …), so step order survives
//!     and the step body (indented under the marker) becomes the list item;
//!   - `Card` → a real link `[title](href)` (so `href`s count as links);
//!   - `Accordion` → its `title` as a bold label, body kept in order;
//!   - `Tab` → its `label` as a bold label, each branch kept;
//!   - containers (`Steps`, `Cards`, `Accordions`, `Tabs`) are unwrapped — the
//!     ordered/labelled children carry the structure.
//! - **Other JSX components** (e.g. `Badge`) keep the legacy behavior:
//!   unwrapped with string attribute values kept as readable text.
//! - Standard HTML tags (`<div>`, `<br>`, …) are left as-is.
//! - JSX comments (`{/* ... */}`) are dropped — they are recognized markup,
//!   not content.
//! - Anything unrecognized (unterminated tags) or **unrepresentable** (a
//!   structural component whose needed attributes are non-literal, e.g.
//!   `href={expr}`) is **never silently dropped**: the raw text is kept and a
//!   diagnostic is produced.
//!
//! Everything is deterministic: the same body yields the same text, the same
//! diagnostics and the same records.
//!
//! The downgrade only affects the text handed to analysis; the Source itself
//! is never rewritten (PRD §9). Note that `source_range`s of the parsed
//! document refer to the *downgraded* text returned by [`downgrade_jsx`].

use std::ops::Range;

use pulldown_cmark::{Event, Options, Parser};

use crate::parser::{ParseDiagnostic, ParseDiagnosticKind};

/// One structural component that was downgraded to Markdown. Recorded so
/// callers (and the fidelity audit) can account for every component occurrence
/// instead of having to guess from the downgraded text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsxDowngradeRecord {
    /// The JSX component name, e.g. `"Callout"` or `"Step"`.
    pub component: String,
}

/// Result of a full JSX downgrade pass.
#[derive(Debug, Clone)]
pub struct DowngradeOutcome {
    /// The downgraded text (all downstream ranges refer to this).
    pub text: String,
    pub diagnostics: Vec<ParseDiagnostic>,
    /// One entry per structural component occurrence that was downgraded, in
    /// document order.
    pub records: Vec<JsxDowngradeRecord>,
}

/// Downgrades JSX in a document body. Returns the downgraded text and the
/// list of diagnostics (empty when the downgrade was lossless).
pub fn downgrade_jsx(body: &str) -> (String, Vec<ParseDiagnostic>) {
    let outcome = downgrade_jsx_detailed(body);
    (outcome.text, outcome.diagnostics)
}

/// Full downgrade result, including the record of every structural component
/// that was downgraded (EPIC C fidelity accounting).
pub fn downgrade_jsx_detailed(body: &str) -> DowngradeOutcome {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);

    let mut replacements: Vec<(Range<usize>, String)> = Vec::new();
    let mut diagnostics: Vec<ParseDiagnostic> = Vec::new();
    let mut records: Vec<JsxDowngradeRecord> = Vec::new();
    let mut state = State::default();

    for (event, range) in Parser::new_ext(body, options).into_offset_iter() {
        let html = match event {
            Event::Html(h) | Event::InlineHtml(h) => h,
            _ => continue,
        };
        match downgrade_chunk(&html, range.clone(), &mut state) {
            ChunkResult::Replace(new_text) => {
                replacements.push((range, with_trailing_newline(&html, new_text)));
            }
            ChunkResult::Structured { text, component } => {
                replacements.push((range, with_trailing_newline(&html, text)));
                records.push(JsxDowngradeRecord { component });
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

    // Structural components that were opened but never closed are unbalanced:
    // surface them instead of letting the tag vanish silently (PRD §2.1).
    for (name, at) in state.open_stack.drain(..) {
        diagnostics.push(ParseDiagnostic {
            kind: ParseDiagnosticKind::UnrecognizedJsx,
            message: format!("unbalanced JSX component without a closing tag: <{name}>"),
            range: Some(llm_wiki_core::model::SourceRange::new(at, at)),
        });
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
    DowngradeOutcome {
        text: out,
        diagnostics,
        records,
    }
}

/// A block-level chunk usually ends with the line's newline; keep it so the
/// following line does not glue onto this one.
fn with_trailing_newline(html: &str, mut new_text: String) -> String {
    if html.ends_with('\n') && !new_text.ends_with('\n') {
        new_text.push('\n');
    }
    new_text
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
    /// A structural component was downgraded; `component` is recorded.
    Structured {
        text: String,
        component: String,
    },
    KeepRaw,
    KeepWithDiagnostic(String),
}

/// Mutable state threaded across HTML chunks: the ordered-list counter for
/// `Steps`/`Step`, plus the stack of structural components that have been
/// opened but not yet closed (used to diagnose unbalanced tags).
#[derive(Default)]
struct State {
    step_counter: u32,
    /// `(component name, byte offset of the opening tag)`.
    open_stack: Vec<(String, usize)>,
}

/// Downgrades one HTML event chunk. `range` locates the chunk in the body.
fn downgrade_chunk(html: &str, range: Range<usize>, state: &mut State) -> ChunkResult {
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
        if is_structural(&name) && !pop_matching(&mut state.open_stack, &name) {
            // An orphan closing tag for a structural component: keep it and
            // say so, rather than dropping the tag with no signal.
            return ChunkResult::KeepWithDiagnostic(format!(
                "unbalanced closing JSX tag kept verbatim: {}",
                first_line(html)
            ));
        }
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

            if is_structural(&name) {
                return downgrade_structural(&name, attrs, gt, self_closing, trimmed, state, range);
            }

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

/// Pops the innermost open component if it matches `name`. Returns whether a
/// matching open tag was found.
fn pop_matching(stack: &mut Vec<(String, usize)>, name: &str) -> bool {
    if stack.last().map(|(open, _)| open.as_str()) == Some(name) {
        stack.pop();
        true
    } else {
        false
    }
}

/// Components with a dedicated structural downgrade (EPIC C). Anything else
/// keeps the legacy "unwrap + readable values" behavior.
fn is_structural(name: &str) -> bool {
    matches!(
        name,
        "Callout"
            | "Steps"
            | "Step"
            | "Cards"
            | "Card"
            | "Accordions"
            | "Accordion"
            | "Tabs"
            | "Tab"
    )
}

/// Builds the Markdown for one structural component. `gt` is the index of the
/// tag's closing `>` within `trimmed`; `trimmed` is the chunk with leading
/// whitespace removed.
fn downgrade_structural(
    name: &str,
    attrs: &str,
    gt: usize,
    self_closing: bool,
    trimmed: &str,
    state: &mut State,
    range: Range<usize>,
) -> ChunkResult {
    let pairs = match collect_named_attrs(attrs) {
        Ok(pairs) => pairs,
        Err(what) => {
            return ChunkResult::KeepWithDiagnostic(format!(
                "unrecognized JSX attribute ({what}) kept verbatim: {name}"
            ))
        }
    };

    let marker = match structural_marker(name, &pairs, state) {
        Ok(marker) => marker,
        Err(message) => return ChunkResult::KeepWithDiagnostic(message),
    };

    let mut text = marker;
    if self_closing {
        if !text.is_empty() {
            text.push(' ');
        }
    } else {
        let after = &trimmed[gt + 1..];
        // A same-chunk closing tag means the component is balanced right here;
        // otherwise remember that it is still open so an unmatched close (or a
        // missing close) can be diagnosed.
        if !after.contains(&format!("</{name}")) {
            state.open_stack.push((name.to_owned(), range.start));
        }
        let rest = strip_closing_tag(after, name);
        // Keep the body in place. A body that starts on the next line
        // (`\n…`) is written verbatim; an inline body gets one separating
        // space so `**Warning:**` and the text do not glue together.
        if !text.is_empty() && !rest.starts_with('\n') {
            text.push(' ');
        }
        text.push_str(rest);
    }

    ChunkResult::Structured {
        text,
        component: name.to_owned(),
    }
}

/// The Markdown marker that replaces a structural component's tag.
fn structural_marker(
    name: &str,
    pairs: &[(String, String)],
    state: &mut State,
) -> Result<String, String> {
    match name {
        "Callout" => Ok(match attr(pairs, "type") {
            Some(kind) => format!("**{}:**", callout_label(kind)),
            None => String::new(),
        }),
        "Steps" => {
            state.step_counter = 0;
            Ok(String::new())
        }
        "Step" => {
            state.step_counter += 1;
            Ok(format!("{}.", state.step_counter))
        }
        "Card" => match (attr(pairs, "title"), attr(pairs, "href")) {
            (Some(title), Some(href)) => Ok(format!("[{title}]({href})")),
            (Some(title), None) => Ok(format!("**{title}**")),
            _ => Err(
                "structural JSX component without a literal title/href kept verbatim: <Card>"
                    .to_string(),
            ),
        },
        "Accordion" => Ok(match attr(pairs, "title") {
            Some(title) => format!("**{title}**"),
            None => String::new(),
        }),
        "Tab" => Ok(
            match attr(pairs, "label").or_else(|| attr(pairs, "title")) {
                Some(label) => format!("**{label}**"),
                None => String::new(),
            },
        ),
        // Containers carry no text of their own; their children hold the
        // structure (ordered items / labelled cards / …).
        "Cards" | "Accordions" | "Tabs" => Ok(String::new()),
        _ => Ok(String::new()),
    }
}

/// Maps a `Callout type` value to a readable label (never the bare keyword).
fn callout_label(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "info" | "note" => "Note".to_owned(),
        "warn" | "warning" => "Warning".to_owned(),
        "tip" => "Tip".to_owned(),
        "danger" => "Danger".to_owned(),
        "error" => "Error".to_owned(),
        "success" => "Success".to_owned(),
        "caution" => "Caution".to_owned(),
        other => capitalize(other),
    }
}

fn capitalize(value: &str) -> String {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// First literal value of `key`, if present.
fn attr<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
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

/// Legacy behavior for non-structural components: readable string attribute
/// values, in order (bare keys ignored, expressions rejected).
fn collect_readable_attrs(attrs: &str) -> AttrScan {
    match collect_named_attrs(attrs) {
        Ok(pairs) => AttrScan::Readable(pairs.into_iter().map(|(_, value)| value).collect()),
        Err(what) => AttrScan::Unrecognized(what),
    }
}

/// Parses attributes into literal `key = value` pairs. Bare keys are ignored
/// (no readable value); JSX comments are allowed; any expression value or
/// unterminated string/expression is an error.
fn collect_named_attrs(attrs: &str) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
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
                    return Err("unterminated expression".to_owned());
                }
                let inner = attrs[inner_start..j - 1].trim();
                if !inner.starts_with("/*") {
                    return Err(format!("expression: {{ {inner} }}"));
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
                        return Err(format!("unterminated value for '{key}'"));
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
                                return Err(format!("unterminated string for '{key}'"));
                            }
                            pairs.push((key.to_owned(), attrs[val_start..i].to_owned()));
                            i += 1;
                        }
                        b'{' => {
                            return Err(format!("non-literal value for '{key}'"));
                        }
                        _ => {
                            let val_start = i;
                            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                                i += 1;
                            }
                            pairs.push((key.to_owned(), attrs[val_start..i].to_owned()));
                        }
                    }
                }
                // bare key: no readable value, ignored
            }
        }
    }
    Ok(pairs)
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

    // --- EPIC C: structural downgrade -------------------------------------

    #[test]
    fn callout_type_becomes_a_readable_label() {
        let outcome = downgrade_jsx_detailed("<Callout type=\"warn\">Watch out.</Callout>\n");
        assert_eq!(outcome.text, "**Warning:** Watch out.\n");
        assert!(
            !outcome.text.contains("warn"),
            "bare type leaked: {}",
            outcome.text
        );
        assert_eq!(outcome.records.len(), 1);
        assert_eq!(outcome.records[0].component, "Callout");
    }

    #[test]
    fn callout_type_map_covers_common_values() {
        assert_eq!(callout_label("info"), "Note");
        assert_eq!(callout_label("warn"), "Warning");
        assert_eq!(callout_label("tip"), "Tip");
        assert_eq!(callout_label("danger"), "Danger");
        assert_eq!(callout_label("error"), "Error");
    }

    #[test]
    fn step_becomes_an_ordered_item_and_keeps_order() {
        let body = "<Steps>\n  <Step>\n    ### First\n\n    alpha\n\n  </Step>\n  <Step>\n    ### Second\n\n    beta\n\n  </Step>\n</Steps>\n";
        let outcome = downgrade_jsx_detailed(body);
        assert!(
            outcome.text.contains("1.\n    ### First"),
            "{}",
            outcome.text
        );
        assert!(
            outcome.text.contains("2.\n    ### Second"),
            "{}",
            outcome.text
        );
        assert!(outcome.diagnostics.is_empty(), "{:?}", outcome.diagnostics);
        let components: Vec<&str> = outcome
            .records
            .iter()
            .map(|r| r.component.as_str())
            .collect();
        assert_eq!(components, vec!["Steps", "Step", "Step"]);
    }

    #[test]
    fn card_href_becomes_a_real_link() {
        let body = "<Cards>\n  <Card title=\"Prod\" href=\"/p\">\n    guard\n  </Card>\n</Cards>\n";
        let outcome = downgrade_jsx_detailed(body);
        assert!(outcome.text.contains("[Prod](/p)"), "{}", outcome.text);
        assert!(outcome.diagnostics.is_empty());
        let components: Vec<&str> = outcome
            .records
            .iter()
            .map(|r| r.component.as_str())
            .collect();
        assert_eq!(components, vec!["Cards", "Card"]);
    }

    #[test]
    fn accordion_title_is_preserved() {
        let outcome = downgrade_jsx_detailed(
            "<Accordions>\n  <Accordion title=\"Q1\">A1</Accordion>\n</Accordions>\n",
        );
        assert!(outcome.text.contains("**Q1**"), "{}", outcome.text);
        assert!(outcome.text.contains("A1"), "{}", outcome.text);
    }

    #[test]
    fn tab_label_is_preserved() {
        let outcome = downgrade_jsx_detailed(
            "<Tabs>\n  <Tab label=\"Node\">n</Tab>\n  <Tab label=\"CLI\">c</Tab>\n</Tabs>\n",
        );
        assert!(outcome.text.contains("**Node**"), "{}", outcome.text);
        assert!(outcome.text.contains("**CLI**"), "{}", outcome.text);
    }

    #[test]
    fn nested_components_keep_their_labels() {
        // A nested component is only real JSX to the CommonMark layer when it
        // is indented fewer than four spaces (four spaces is an indented code
        // block, which the downgrade must not touch).
        let body = "<Steps>\n  <Step>\n    ### Do it\n\n  <Callout type=\"warn\">Careful.</Callout>\n\n  </Step>\n</Steps>\n";
        let outcome = downgrade_jsx_detailed(body);
        assert!(outcome.text.contains("1."), "{}", outcome.text);
        assert!(
            outcome.text.contains("**Warning:** Careful."),
            "{}",
            outcome.text
        );
        assert!(outcome.diagnostics.is_empty(), "{:?}", outcome.diagnostics);
    }

    #[test]
    fn non_literal_structural_attribute_keeps_raw_and_diagnoses() {
        let body = "<Card title=\"T\" href={link} />\n";
        let (out, diags) = downgrade_jsx(body);
        assert_eq!(
            out, body,
            "unrepresentable structural JSX must not be dropped"
        );
        assert_eq!(diags.len(), 1, "{diags:?}");
    }

    #[test]
    fn unbalanced_tags_are_kept_with_diagnostics() {
        // Orphan closing tag: kept verbatim and diagnosed.
        let (out, diags) = downgrade_jsx("</Callout>\n");
        assert_eq!(out, "</Callout>\n");
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(diags[0].message.contains("unbalanced closing"), "{diags:?}");

        // Unclosed opening tag: downgraded, but the imbalance is diagnosed.
        let (out, diags) = downgrade_jsx("<Callout type=\"warn\">\nbody\n");
        assert!(out.contains("**Warning:**"), "{out}");
        assert_eq!(diags.len(), 1, "{diags:?}");
        assert!(
            diags[0].message.contains("without a closing tag"),
            "{diags:?}"
        );

        // Balanced input stays diagnostic-free.
        let (_, diags) = downgrade_jsx("<Callout type=\"warn\">ok</Callout>\n");
        assert!(diags.is_empty(), "{diags:?}");
    }

    #[test]
    fn structural_downgrade_is_deterministic() {
        let body = "<Steps>\n  <Step>\n    a\n\n  </Step>\n</Steps>\n\n<Callout type=\"info\">x</Callout>\n";
        let first = downgrade_jsx_detailed(body);
        let second = downgrade_jsx_detailed(body);
        assert_eq!(first.text, second.text);
        assert_eq!(first.records, second.records);
    }
}
