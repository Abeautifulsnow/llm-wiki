//! EPIC C — synthetic MDX fidelity fixtures.
//!
//! These tests pin the *structured* JSX downgrade against repo-committed
//! minimal fixtures (`tests/fixtures/*.mdx`) — no real DBX content is vendored.
//! Each component class from the PRD table, plus nesting and the "keep raw +
//! diagnose" edge cases, is covered.

use std::fs;
use std::path::PathBuf;

use llm_wiki_markdown::{parse_document, ParseDiagnosticKind, ParseOutput};

fn fixture(name: &str) -> ParseOutput {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cannot read fixture {}: {err}", path.display()));
    parse_document(&raw, name)
}

/// Word tokens (alphanumeric runs, lowercased) — used to prove no bare
/// attribute keyword leaks into the downgraded body.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn trimmed_lines(text: &str) -> Vec<String> {
    text.lines().map(|line| line.trim().to_owned()).collect()
}

/// Index of the first line whose trimmed content equals `needle`.
fn line_at(lines: &[String], needle: &str) -> usize {
    lines
        .iter()
        .position(|line| line == needle)
        .unwrap_or_else(|| panic!("no line equal to {needle:?} in\n{}", lines.join("\n")))
}

fn components(doc: &ParseOutput) -> Vec<&str> {
    doc.jsx_downgrades
        .iter()
        .map(|record| record.component.as_str())
        .collect()
}

/// Every section/link range must round-trip to the downgraded text.
fn assert_ranges_refer_to_downgraded_text(doc: &ParseOutput) {
    for section in &doc.sections {
        assert_eq!(
            doc.analyzed_text[section.source_range.start..section.source_range.end].trim(),
            section.content,
            "section range drifted from the downgraded text"
        );
    }
    for link in &doc.links {
        assert!(
            doc.analyzed_text[link.source_range.start..link.source_range.end].contains(&link.text),
            "link range drifted from the downgraded text"
        );
    }
}

#[test]
fn callouts_get_readable_labels_and_leak_no_type_keyword() {
    let doc = fixture("callouts.mdx");
    let text = &doc.analyzed_text;

    assert!(
        text.contains("**Note:** Read the introduction before you start."),
        "{text}"
    );
    assert!(
        text.contains("**Warning:** Production writes need a read-only account."),
        "{text}"
    );
    assert!(text.contains("**Tip:**"), "{text}");
    assert!(
        text.contains("**Danger:** Flushing the database cannot be undone."),
        "{text}"
    );

    // The `type` value must never survive as a meaningless bare keyword. The
    // values DBX uses (`warn`, `info`) map to different labels (`Warning`,
    // `Note`); `tip`/`danger` are labels in their own right (`Tip`, `Danger`).
    let tokens = words(text);
    for keyword in ["warn", "info", "type"] {
        assert!(
            !tokens.iter().any(|token| token == keyword),
            "bare attribute keyword {keyword:?} leaked into:\n{text}"
        );
    }
    assert!(!text.contains("type="), "{text}");

    assert!(doc.diagnostics.is_empty(), "{:?}", doc.diagnostics);
    assert_eq!(
        components(&doc),
        vec!["Callout", "Callout", "Callout", "Callout"]
    );
    assert_ranges_refer_to_downgraded_text(&doc);
}

#[test]
fn card_hrefs_become_real_links() {
    let doc = fixture("cards.mdx");

    assert_eq!(doc.links.len(), 2, "{:?}", doc.links);
    assert_eq!(doc.links[0].text, "Production safety");
    assert_eq!(doc.links[0].url, "/en/docs/production-safety");
    assert_eq!(doc.links[1].text, "Connections");
    assert_eq!(doc.links[1].url, "/en/docs/connections");

    assert!(
        doc.analyzed_text
            .contains("[Production safety](/en/docs/production-safety)"),
        "{}",
        doc.analyzed_text
    );
    // The href must not survive as a bare attribute.
    assert!(
        !doc.analyzed_text.contains("href="),
        "{}",
        doc.analyzed_text
    );
    // Card bodies are preserved.
    assert!(
        doc.analyzed_text
            .contains("Configure read-only connections"),
        "{}",
        doc.analyzed_text
    );

    assert!(doc.diagnostics.is_empty(), "{:?}", doc.diagnostics);
    assert_eq!(components(&doc), vec!["Cards", "Card", "Card"]);
    assert_ranges_refer_to_downgraded_text(&doc);
}

#[test]
fn steps_become_an_ordered_list_in_order() {
    let doc = fixture("steps.mdx");
    let text = &doc.analyzed_text;
    let lines = trimmed_lines(text);

    let first_ordinal = line_at(&lines, "1.");
    let second_ordinal = line_at(&lines, "2.");
    let third_ordinal = line_at(&lines, "3.");

    let first_step = line_at(&lines, "### Create the project");
    let second_step = line_at(&lines, "### Configure it");
    let third_step = line_at(&lines, "### Publish it");

    assert!(first_ordinal < first_step, "{text}");
    assert!(first_step < second_ordinal, "{text}");
    assert!(second_ordinal < second_step, "{text}");
    assert!(second_step < third_ordinal, "{text}");
    assert!(third_ordinal < third_step, "{text}");

    // A Callout nested inside a Step still carries its label.
    assert!(
        text.contains("**Warning:** Check the configuration before publishing."),
        "{text}"
    );

    let names = components(&doc);
    assert_eq!(names.iter().filter(|n| **n == "Step").count(), 3);
    assert_eq!(names.iter().filter(|n| **n == "Steps").count(), 1);
    assert_eq!(names.iter().filter(|n| **n == "Callout").count(), 1);
    assert!(doc.diagnostics.is_empty(), "{:?}", doc.diagnostics);
    assert_ranges_refer_to_downgraded_text(&doc);
}

#[test]
fn accordions_keep_their_titles_and_bodies() {
    let doc = fixture("accordions.mdx");
    let text = &doc.analyzed_text;

    assert!(text.contains("**Why is the package missing?**"), "{text}");
    assert!(
        text.contains("Reinstall without the optional-dependency flag."),
        "{text}"
    );
    assert!(text.contains("**Where is the data directory?**"), "{text}");
    assert!(
        text.contains("Point the data directory at a writable folder."),
        "{text}"
    );

    assert!(doc.diagnostics.is_empty(), "{:?}", doc.diagnostics);
    assert_eq!(
        components(&doc),
        vec!["Accordions", "Accordion", "Accordion"]
    );
}

#[test]
fn tabs_keep_each_branch_with_its_label() {
    let doc = fixture("tabs.mdx");
    let text = &doc.analyzed_text;

    assert!(text.contains("**Node**"), "{text}");
    assert!(text.contains("Use the Node client."), "{text}");
    assert!(text.contains("**CLI**"), "{text}");
    assert!(text.contains("Use the command line client."), "{text}");

    assert!(doc.diagnostics.is_empty(), "{:?}", doc.diagnostics);
    assert_eq!(components(&doc), vec!["Tabs", "Tab", "Tab"]);
}

#[test]
fn unrepresentable_and_unrecognized_jsx_is_kept_with_diagnostics() {
    let doc = fixture("edge-cases.mdx");
    let text = &doc.analyzed_text;

    // A non-literal structural attribute and an unterminated tag are both
    // kept verbatim and diagnosed — never silently dropped.
    assert!(
        text.contains("<Card title=\"Dynamic\" href={link} />"),
        "{text}"
    );
    assert!(text.contains("<Callout title=\"unterminated"), "{text}");
    assert_eq!(doc.diagnostics.len(), 2, "{:?}", doc.diagnostics);
    assert!(doc
        .diagnostics
        .iter()
        .all(|d| d.kind == ParseDiagnosticKind::UnrecognizedJsx));
    // Nothing was pseudo-downgraded.
    assert!(doc.jsx_downgrades.is_empty(), "{:?}", doc.jsx_downgrades);

    // A non-structural component keeps its readable attribute values.
    assert!(text.contains("Stable green"), "{text}");
}

#[test]
fn fenced_code_is_never_touched_by_the_downgrade() {
    let doc = fixture("edge-cases.mdx");
    let text = &doc.analyzed_text;

    assert!(text.contains("```mdx"), "{text}");
    assert!(
        text.contains("<Callout type=\"warn\">code sample, not real JSX</Callout>"),
        "{text}"
    );
    // No structural downgrade happened inside the fence.
    assert!(!doc.jsx_downgrades.iter().any(|r| r.component == "Callout"));
}

#[test]
fn unbalanced_tags_are_never_silent() {
    let doc = fixture("unbalanced.mdx");

    // One orphan closing tag kept verbatim, plus two unclosed opens — three
    // diagnostics, no silent drop.
    assert_eq!(doc.diagnostics.len(), 3, "{:?}", doc.diagnostics);
    assert!(doc
        .diagnostics
        .iter()
        .all(|d| d.kind == ParseDiagnosticKind::UnrecognizedJsx));
    assert!(
        doc.analyzed_text.contains("</Callout>"),
        "{}",
        doc.analyzed_text
    );
    assert!(
        doc.diagnostics
            .iter()
            .any(|d| d.message.contains("unbalanced closing")),
        "{:?}",
        doc.diagnostics
    );
    assert!(
        doc.diagnostics
            .iter()
            .any(|d| d.message.contains("without a closing tag")),
        "{:?}",
        doc.diagnostics
    );
}

#[test]
fn downgrade_is_deterministic() {
    for name in ["callouts.mdx", "steps.mdx", "cards.mdx", "edge-cases.mdx"] {
        let first = fixture(name);
        let second = fixture(name);
        assert_eq!(first.analyzed_text, second.analyzed_text, "{name}");
        assert_eq!(first.jsx_downgrades, second.jsx_downgrades, "{name}");
        assert_eq!(first.links.len(), second.links.len(), "{name}");
    }
}
