//! Structural Markdown parsing (PRD §9).
//!
//! Produces `ParsedDocument`-shaped output: sections delimited by headings,
//! each with heading path, source range and content; links; restricted
//! frontmatter; and deterministic diagnostics. Heading paths
//! (`Plugin System > Architecture > Runtime`) are the backbone of citations
//! and of the Section Matcher.

use serde::{Deserialize, Serialize};

use llm_wiki_core::model::SourceRange;

use crate::frontmatter::{strip_and_parse, Frontmatter};
use crate::language::detect_language;
use crate::mdx::downgrade_jsx;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ParseDiagnosticKind {
    MalformedFrontmatter,
    UnrecognizedJsx,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParseDiagnostic {
    pub kind: ParseDiagnosticKind,
    pub message: String,
    /// Offsets into the analyzed text.
    pub range: Option<SourceRange>,
}

#[derive(Debug, Clone)]
pub struct SectionOutput {
    pub heading: Option<String>,
    pub heading_level: u8,
    pub heading_path: Vec<String>,
    pub content: String,
    /// Offsets into the analyzed text (frontmatter-stripped, MDX-downgraded).
    pub source_range: SourceRange,
}

#[derive(Debug, Clone)]
pub struct LinkOutput {
    pub url: String,
    pub text: String,
    pub source_range: SourceRange,
}

#[derive(Debug, Clone)]
pub struct ParseOutput {
    /// The text all ranges refer to: frontmatter-stripped and, for `.mdx`,
    /// JSX-downgraded. Evidence digests are computed over this text.
    pub analyzed_text: String,
    pub title: Option<String>,
    pub language: String,
    pub frontmatter: Frontmatter,
    pub sections: Vec<SectionOutput>,
    pub links: Vec<LinkOutput>,
    pub diagnostics: Vec<ParseDiagnostic>,
}

/// Parses one document.
///
/// `file_name` is used for MDX detection and language heuristics; it may be a
/// full path or bare file name.
pub fn parse_document(raw: &str, file_name: &str) -> ParseOutput {
    let is_mdx = file_name.to_ascii_lowercase().ends_with(".mdx");

    let (frontmatter, body, fm_diagnostic) = strip_and_parse(raw);

    let (analyzed_text, diagnostics) = if is_mdx {
        let (downgraded, mut diags) = downgrade_jsx(&body);
        if let Some(d) = fm_diagnostic {
            diags.push(d);
        }
        (downgraded, diags)
    } else {
        let mut diags = Vec::new();
        if let Some(d) = fm_diagnostic {
            diags.push(d);
        }
        (body, diags)
    };

    let title = frontmatter
        .as_ref()
        .and_then(|fm| fm.get("title").map(str::to_owned))
        .or_else(|| analyzed_text.lines().find_map(first_heading_line));

    let language = detect_language(frontmatter.as_ref(), file_name, &analyzed_text);

    let (sections, links) = extract_structure(&analyzed_text);

    ParseOutput {
        analyzed_text,
        title,
        language,
        frontmatter: frontmatter.unwrap_or_default(),
        sections,
        links,
        diagnostics,
    }
}

fn first_heading_line(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) {
        let rest = trimmed[hashes..].trim();
        if !rest.is_empty() {
            return Some(rest.to_owned());
        }
    }
    None
}

fn extract_structure(text: &str) -> (Vec<SectionOutput>, Vec<LinkOutput>) {
    use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_HEADING_ATTRIBUTES);

    let mut sections: Vec<SectionOutput> = Vec::new();
    let mut links: Vec<LinkOutput> = Vec::new();

    // Open section state. `None` heading = preamble.
    let mut open_heading: Option<String> = None;
    let mut open_level: u8 = 0;
    let mut open_path: Vec<String> = Vec::new();
    let mut open_start: usize = 0;
    let mut has_open = true; // implicit preamble

    let mut heading_stack: Vec<(u8, String)> = Vec::new();

    let mut in_heading = false;
    let mut heading_text = String::new();
    let mut heading_level: u8 = 0;

    let mut open_link: Option<(String, usize)> = None;
    let mut link_text = String::new();

    let close_section = |sections: &mut Vec<SectionOutput>,
                         open_heading: &mut Option<String>,
                         open_level: &mut u8,
                         open_path: &mut Vec<String>,
                         open_start: &mut usize,
                         has_open: &mut bool,
                         end: usize| {
        if *has_open {
            let content = text[*open_start..end].trim();
            // An empty preamble carries no knowledge; skip it.
            if open_heading.is_some() || !content.is_empty() {
                sections.push(SectionOutput {
                    heading: open_heading.take(),
                    heading_level: *open_level,
                    heading_path: std::mem::take(open_path),
                    content: content.to_owned(),
                    source_range: SourceRange::new(*open_start, end),
                });
            }
        }
        *has_open = false;
    };

    for (event, range) in Parser::new_ext(text, options).into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                close_section(
                    &mut sections,
                    &mut open_heading,
                    &mut open_level,
                    &mut open_path,
                    &mut open_start,
                    &mut has_open,
                    range.start,
                );
                in_heading = true;
                heading_text.clear();
                heading_level = match level {
                    pulldown_cmark::HeadingLevel::H1 => 1,
                    pulldown_cmark::HeadingLevel::H2 => 2,
                    pulldown_cmark::HeadingLevel::H3 => 3,
                    pulldown_cmark::HeadingLevel::H4 => 4,
                    pulldown_cmark::HeadingLevel::H5 => 5,
                    pulldown_cmark::HeadingLevel::H6 => 6,
                };
            }
            Event::Text(t) if in_heading => heading_text.push_str(&t),
            Event::Code(t) if in_heading => heading_text.push_str(&t),
            Event::End(TagEnd::Heading(_)) => {
                in_heading = false;
                let text_of_heading = heading_text.trim().to_owned();
                while heading_stack
                    .last()
                    .map(|(lvl, _)| *lvl >= heading_level)
                    .unwrap_or(false)
                {
                    heading_stack.pop();
                }
                heading_stack.push((heading_level, text_of_heading.clone()));

                open_heading = Some(text_of_heading);
                open_level = heading_level;
                open_path = heading_stack.iter().map(|(_, t)| t.clone()).collect();
                open_start = range.end;
                has_open = true;
            }
            Event::Start(Tag::Link { dest_url, .. }) => {
                open_link = Some((dest_url.to_string(), range.start));
                link_text.clear();
            }
            Event::Text(t) if open_link.is_some() => link_text.push_str(&t),
            Event::Code(t) if open_link.is_some() => link_text.push_str(&t),
            Event::End(TagEnd::Link) => {
                if let Some((url, start)) = open_link.take() {
                    links.push(LinkOutput {
                        url,
                        text: link_text.trim().to_owned(),
                        source_range: SourceRange::new(start, range.end),
                    });
                }
            }
            _ => {}
        }
    }

    close_section(
        &mut sections,
        &mut open_heading,
        &mut open_level,
        &mut open_path,
        &mut open_start,
        &mut has_open,
        text.len(),
    );

    (sections, links)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_sections_with_heading_paths_and_ranges() {
        let raw = "# Plugin System\n\nIntro text.\n\n## Architecture\n\nArch body.\n\n### Runtime\n\nRuntime body.\n";
        let out = parse_document(raw, "doc.md");
        assert_eq!(out.title.as_deref(), Some("Plugin System"));
        assert_eq!(out.language, "en");
        assert_eq!(out.sections.len(), 3);

        let s0 = &out.sections[0];
        assert_eq!(s0.heading_path, vec!["Plugin System"]);
        assert_eq!(s0.heading_level, 1);
        assert_eq!(s0.content, "Intro text.");

        let s2 = &out.sections[2];
        assert_eq!(
            s2.heading_path,
            vec!["Plugin System", "Architecture", "Runtime"]
        );
        assert_eq!(s2.heading_level, 3);
        assert_eq!(s2.content, "Runtime body.");

        // Ranges round-trip to the content.
        for s in &out.sections {
            assert_eq!(
                out.analyzed_text[s.source_range.start..s.source_range.end].trim(),
                s.content
            );
        }
    }

    #[test]
    fn preamble_before_first_heading_is_a_section() {
        let raw = "Preamble only.\n\n# Heading\n\nBody.\n";
        let out = parse_document(raw, "doc.md");
        assert_eq!(out.sections[0].heading, None);
        assert_eq!(out.sections[0].content, "Preamble only.");
        assert!(out.sections[0].heading_path.is_empty());
    }

    #[test]
    fn headings_after_skipped_levels_nest_correctly() {
        let raw = "# A\n\n### C\n\n## B\n\nb body\n";
        let out = parse_document(raw, "doc.md");
        assert_eq!(out.sections[1].heading_path, vec!["A", "C"]);
        assert_eq!(out.sections[2].heading_path, vec!["A", "B"]);
    }

    #[test]
    fn duplicate_headings_are_kept_as_distinct_sections() {
        let raw = "## Notes\n\nfirst\n\n## Notes\n\nsecond\n";
        let out = parse_document(raw, "doc.md");
        assert_eq!(out.sections.len(), 2);
        assert_eq!(out.sections[0].content, "first");
        assert_eq!(out.sections[1].content, "second");
    }

    #[test]
    fn links_are_extracted_with_ranges() {
        let raw = "# T\n\nsee [docs](./other.md) for more\n";
        let out = parse_document(raw, "doc.md");
        assert_eq!(out.links.len(), 1);
        let link = &out.links[0];
        assert_eq!(link.url, "./other.md");
        assert_eq!(link.text, "docs");
        assert_eq!(
            &out.analyzed_text[link.source_range.start..link.source_range.end],
            "[docs](./other.md)"
        );
    }

    #[test]
    fn frontmatter_feeds_title_and_language_without_polluting_body() {
        let raw = "---\ntitle: 插件系统\nlang: zh-CN\n---\n\n# 插件系统\n\n正文。\n";
        let out = parse_document(raw, "plugin.cn.md");
        assert_eq!(out.title.as_deref(), Some("插件系统"));
        assert_eq!(out.language, "zh-CN");
        assert!(out.sections.iter().all(|s| !s.content.contains("title:")));
    }
    #[test]
    fn mdx_components_are_downgraded_and_ranges_match() {
        let raw = "---\ntitle: UI\n---\n\n# Components\n\n<Callout>\n\nWatch out.\n\n</Callout>\n\n<Badge label=\"Stable\" />\n";
        let out = parse_document(raw, "components.mdx");
        assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
        let section = &out.sections[0];
        assert!(section.content.contains("Watch out."));
        assert!(section.content.contains("Stable"));
        assert!(!section.content.contains("<Callout>"));
        for s in &out.sections {
            assert_eq!(
                out.analyzed_text[s.source_range.start..s.source_range.end].trim(),
                s.content
            );
        }
    }

    #[test]
    fn code_blocks_keep_jsx_examples_intact() {
        let raw = "# Examples\n\n```mdx\n<Tabs>demo</Tabs>\n```\n";
        let out = parse_document(raw, "examples.mdx");
        assert!(out.sections[0].content.contains("<Tabs>demo</Tabs>"));
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn setext_headings_are_supported() {
        let raw = "Title Line\n==========\n\nbody\n";
        let out = parse_document(raw, "doc.md");
        assert_eq!(out.sections[0].heading.as_deref(), Some("Title Line"));
        assert_eq!(out.sections[0].heading_level, 1);
    }
}
