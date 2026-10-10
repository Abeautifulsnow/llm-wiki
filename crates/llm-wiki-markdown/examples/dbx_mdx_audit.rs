//! DBX MDX fidelity audit — M0 read-only parse report.
//!
//! Task: `10-10-m0-dbx-fidelity-audit`. Purpose: measure how much real DBX
//! documentation survives `llm_wiki_markdown::parse_document` today, so EPIC C
//! (high-fidelity MDX import) has a data-driven baseline.
//!
//! ```text
//! Usage: dbx_mdx_audit <file-or-dir>...
//! ```
//!
//! A directory argument contributes its top-level `*.mdx` files only
//! (**NON-recursive**; chosen for determinism, and because the DBX goldens live
//! in one flat directory). Inputs are opened READ-ONLY; the tool writes nothing
//! but a Markdown report to stdout.
//!
//! Every fact below comes from the real `parse_document` output — there are no
//! hand-written expectations. Per file the report prints:
//!
//! 1. file stats: line count, detected language, section count, heading tree,
//!    code blocks grouped by fence language (with a fenced-content non-empty
//!    check) and link count;
//! 2. every `parse_document` diagnostic verbatim (kind + message + byte range);
//! 3. a dependency-free, fence-aware census of the JSX components
//!    `Tabs/Tab/Steps/Step/Callout/Cards/Card/Accordion/Accordions`, each
//!    classified as structurally-downgraded / downgraded-with-diagnostic /
//!    preserved-verbatim / **SILENT LOSS**.
//!
//! `SILENT LOSS` means the component appears in the raw text (outside fenced
//! code blocks) yet the parse produced none of: a diagnostic, a surviving
//! occurrence of its tag, or a recorded structural downgrade. In other words
//! the tag and the component's boundary/semantics were dropped with no signal.
//!
//! Since EPIC C the parser downgrades structural JSX to Markdown that carries
//! its semantics (`Callout` → a label, `Card` → a link, `Step` → an ordered
//! item …). Such components leave no tag behind, so the census detects them
//! through [`ParseOutput::jsx_downgrades`] rather than by tag survival.
//!
//! Only `std` and `llm-wiki-markdown` are used (no regex, no walkdir).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use llm_wiki_markdown::{parse_document, ParseDiagnostic, ParseOutput};

/// The JSX components the audit censuses (PRD §1.1).
const COMPONENTS: &[&str] = &[
    "Tabs",
    "Tab",
    "Steps",
    "Step",
    "Callout",
    "Cards",
    "Card",
    "Accordion",
    "Accordions",
];

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: dbx_mdx_audit <file-or-dir>...");
        std::process::exit(2);
    }

    let files = collect_inputs(&args);
    if files.is_empty() {
        eprintln!("ERROR: no .mdx input files found");
        std::process::exit(1);
    }

    let mut out = String::new();
    out.push_str("# DBX MDX fidelity audit (llm-wiki-markdown parse report)\n\n");
    out.push_str(&format!(
        "Tool: `dbx_mdx_audit` · files audited: {}\n\n",
        files.len()
    ));
    out.push_str(
        "Legend: **SILENT LOSS** = the component is present in the raw text (outside\n\
         fenced code blocks) but the parse produced no diagnostic, no surviving tag and\n\
         no recorded structural downgrade — dropped with no signal.\n\n---\n\n",
    );

    for file in &files {
        let display = file.display().to_string();
        match fs::read_to_string(file) {
            Ok(raw) => audit_file(&raw, &display, &mut out),
            Err(err) => {
                eprintln!("ERROR: cannot read {display}: {err}");
                out.push_str(&format!(
                    "## `{display}`\n\nERROR: cannot read file: {err}\n\n---\n\n"
                ));
            }
        }
    }

    print!("{out}");
}

/// Expands CLI arguments into a sorted, de-duplicated list of `.mdx` files.
/// Directories contribute their top-level `*.mdx` entries (non-recursive).
fn collect_inputs(args: &[String]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for arg in args {
        let path = PathBuf::from(arg);
        if path.is_dir() {
            match fs::read_dir(&path) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        let candidate = entry.path();
                        let is_mdx = candidate
                            .extension()
                            .map(|ext| ext.eq_ignore_ascii_case("mdx"))
                            .unwrap_or(false);
                        if candidate.is_file() && is_mdx {
                            files.push(candidate);
                        }
                    }
                }
                Err(err) => eprintln!("ERROR: cannot read directory {}: {err}", path.display()),
            }
        } else if path.is_file() {
            files.push(path);
        } else {
            eprintln!("ERROR: no such file or directory: {}", path.display());
        }
    }
    files.sort();
    files.dedup();
    files
}

/// Emits the full Markdown section for one document.
fn audit_file(raw: &str, display_path: &str, out: &mut String) {
    let file_name = Path::new(display_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(display_path);
    let doc = parse_document(raw, file_name);

    out.push_str(&format!("## `{display_path}`\n\n"));
    out.push_str(&format!("- lines: **{}**\n", raw.lines().count()));
    out.push_str(&format!("- detected language: `{}`\n", doc.language));
    if let Some(title) = &doc.title {
        out.push_str(&format!("- title: {title}\n"));
    }
    out.push_str(&format!("- sections: **{}**\n", doc.sections.len()));
    out.push_str(&format!("- links: **{}**\n\n", doc.links.len()));

    render_heading_tree(&doc, out);
    render_code_blocks(&doc, out);
    render_diagnostics(&doc, out);
    render_jsx_census(raw, &doc, out);

    out.push_str("---\n\n");
}

fn render_heading_tree(doc: &ParseOutput, out: &mut String) {
    out.push_str("### Heading tree\n\n");
    if doc.sections.is_empty() {
        out.push_str("_(no sections)_\n\n");
        return;
    }
    for section in &doc.sections {
        let indent = "  ".repeat(section.heading_level.saturating_sub(1) as usize);
        match &section.heading {
            Some(heading) => out.push_str(&format!(
                "{indent}- [H{}] {heading}\n",
                section.heading_level
            )),
            None => out.push_str(&format!(
                "{indent}- (preamble, {})\n",
                section.heading_level
            )),
        }
    }
    out.push('\n');
}

struct CodeBlock {
    lang: String,
    nonempty: bool,
}

/// Fence-aware scan of the analyzed text for fenced code blocks.
fn scan_code_blocks(text: &str) -> Vec<CodeBlock> {
    let mut blocks: Vec<CodeBlock> = Vec::new();
    let mut in_fence: Option<String> = None;
    let mut lang = String::new();
    let mut nonempty = false;

    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(marker) = in_fence.clone() {
            if trimmed.starts_with(marker.as_str()) {
                blocks.push(CodeBlock {
                    lang: std::mem::take(&mut lang),
                    nonempty,
                });
                in_fence = None;
            } else if !trimmed.is_empty() {
                nonempty = true;
            }
            continue;
        }
        if let Some(rest) = fence_info(trimmed, "```") {
            in_fence = Some("```".to_owned());
            lang = fence_language(rest);
            nonempty = false;
        } else if let Some(rest) = fence_info(trimmed, "~~~") {
            in_fence = Some("~~~".to_owned());
            lang = fence_language(rest);
            nonempty = false;
        }
    }
    if in_fence.is_some() {
        // Unterminated fence: report what was collected rather than dropping it.
        blocks.push(CodeBlock { lang, nonempty });
    }
    blocks
}

/// Returns the text after an opening fence marker, or `None` if `line` is not
/// an opening fence.
fn fence_info<'a>(line: &'a str, marker: &str) -> Option<&'a str> {
    if line.starts_with(marker) && !line.starts_with(&format!("{marker}{marker}")) {
        Some(&line[marker.len()..])
    } else {
        None
    }
}

fn fence_language(info: &str) -> String {
    info.split_whitespace().next().unwrap_or("").to_owned()
}

fn render_code_blocks(doc: &ParseOutput, out: &mut String) {
    let blocks = scan_code_blocks(&doc.analyzed_text);
    let total = blocks.len();
    let empty = blocks.iter().filter(|b| !b.nonempty).count();

    out.push_str(&format!("### Code blocks — {total} total\n\n"));
    if total == 0 {
        out.push_str("_(no fenced code blocks)_\n\n");
        return;
    }

    // Group by fence language, preserving first-seen order.
    let mut order: Vec<String> = Vec::new();
    let mut counts: Vec<(usize, usize)> = Vec::new(); // (total, nonempty)
    for block in &blocks {
        match order.iter().position(|l| l == &block.lang) {
            Some(idx) => {
                counts[idx].0 += 1;
                if block.nonempty {
                    counts[idx].1 += 1;
                }
            }
            None => {
                order.push(block.lang.clone());
                counts.push((1, usize::from(block.nonempty)));
            }
        }
    }
    for (lang, (count, nonempty)) in order.iter().zip(counts.iter()) {
        let label = if lang.is_empty() {
            "(no language)".to_owned()
        } else {
            format!("`{lang}`")
        };
        out.push_str(&format!("- {label}: {count} (non-empty: {nonempty})\n"));
    }
    if empty == 0 {
        out.push_str(&format!(
            "- fenced-content check: **PASS** (all {total} blocks non-empty)\n\n"
        ));
    } else {
        out.push_str(&format!(
            "- fenced-content check: **FAIL** ({empty} of {total} blocks are empty)\n\n"
        ));
    }
}

fn render_diagnostics(doc: &ParseOutput, out: &mut String) {
    out.push_str(&format!("### Diagnostics — {}\n\n", doc.diagnostics.len()));
    if doc.diagnostics.is_empty() {
        out.push_str("_(none)_\n\n");
        return;
    }
    for diagnostic in &doc.diagnostics {
        out.push_str(&format_diagnostic(diagnostic));
        out.push('\n');
    }
    out.push('\n');
}

fn format_diagnostic(diagnostic: &ParseDiagnostic) -> String {
    let location = match &diagnostic.range {
        Some(range) => format!("bytes {}..{}", range.start, range.end),
        None => "no location".to_owned(),
    };
    format!(
        "- [{:?}] {} ({location})",
        diagnostic.kind, diagnostic.message
    )
}

/// Raw, fence-aware JSX census compared against the parse product.
fn render_jsx_census(raw: &str, doc: &ParseOutput, out: &mut String) {
    let raw_scan = strip_fenced_lines(raw);
    let analyzed_scan = strip_fenced_lines(&doc.analyzed_text);

    out.push_str("### JSX census (raw text, fenced code blocks excluded)\n\n");
    out.push_str(
        "| component | raw open | raw close | diagnostics | structured | tag survived | disposition |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---|---|\n");

    let mut silent: Vec<String> = Vec::new();
    for name in COMPONENTS {
        let (raw_open, raw_close) = count_tags(&raw_scan, name);
        if raw_open == 0 && raw_close == 0 {
            continue;
        }
        let diag_mentions = doc
            .diagnostics
            .iter()
            .filter(|d| mentions_component(&d.message, name))
            .count();
        let structured = doc
            .jsx_downgrades
            .iter()
            .filter(|record| record.component == *name)
            .count();
        let survived = count_tags(&analyzed_scan, name).0 > 0;

        let disposition = if structured > 0 {
            "downgraded (structured)"
        } else if diag_mentions > 0 {
            "downgraded (diagnostic)"
        } else if survived {
            "preserved verbatim"
        } else {
            "SILENT LOSS"
        };
        if disposition == "SILENT LOSS" {
            silent.push((*name).to_owned());
        }

        out.push_str(&format!(
            "| `{name}` | {raw_open} | {raw_close} | {diag_mentions} | {structured} | {survived} | {disposition} |\n"
        ));
    }

    out.push('\n');
    if silent.is_empty() {
        out.push_str("**SILENT LOSS: none** — every census component was accounted for by a structured downgrade, a diagnostic, or a surviving tag.\n\n");
    } else {
        out.push_str(&format!(
            "**SILENT LOSS: {} component(s)** — present in raw text, yet the parse produced no structured downgrade, no diagnostic and no surviving tag: {}.\n\n",
            silent.len(),
            silent
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
}

/// Removes lines inside fenced code blocks (``` / ~~~), fence-aware. Used so the
/// census and the "survived" check measure real JSX, not fenced examples.
fn strip_fenced_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_fence: Option<&str> = None;
    for line in text.lines() {
        let trimmed = line.trim_start();
        match in_fence {
            Some(marker) => {
                if trimmed.starts_with(marker) {
                    in_fence = None;
                }
            }
            None => {
                if trimmed.starts_with("```") {
                    in_fence = Some("```");
                } else if trimmed.starts_with("~~~") {
                    in_fence = Some("~~~");
                } else {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
    }
    out
}

/// Counts opening (`<Name`) and closing (`</Name`) tags of exactly `name` in
/// `text`, with a maximal-name boundary check so `<TabsList>` never matches
/// `Tabs`.
fn count_tags(text: &str, name: &str) -> (usize, usize) {
    let bytes = text.as_bytes();
    let mut open = 0usize;
    let mut close = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        let closing = bytes.get(i + 1) == Some(&b'/');
        let name_start = if closing { i + 2 } else { i + 1 };
        let Some(name_end) = tag_name_end(text, name_start) else {
            i += 1;
            continue;
        };
        if &text[name_start..name_end] == name {
            if closing {
                close += 1;
            } else {
                open += 1;
            }
        }
        i = name_end;
    }
    (open, close)
}

/// Finds the end (exclusive) of a JSX tag name starting at `start`.
fn tag_name_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut end = start;
    while end < bytes.len() {
        let byte = bytes[end];
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-' | b':') {
            end += 1;
        } else {
            break;
        }
    }
    if end == start {
        None
    } else {
        Some(end)
    }
}

/// Whether a diagnostic message references `<name` (used as its own text scan,
/// boundary-checked).
fn mentions_component(message: &str, name: &str) -> bool {
    let (open, close) = count_tags(message, name);
    open > 0 || close > 0
}
