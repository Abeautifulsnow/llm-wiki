//! Single-file, dependency-free browser for the published wiki (V0.6).
//!
//! `llm-wiki browse` inlines the ACTIVE generation into one HTML file, so the
//! wiki can be read by double-clicking it — no server, no CDN, no network.
//! The same file, when served over HTTP, upgrades its search box to
//! `/v1/search`; every other feature works identically offline.
//!
//! Why a generated file and not a server route: a workspace is a directory
//! pair (`.llm-wiki/` + `wiki/`), and readers often have the directory without
//! a running process. Inlining the generation makes the wiki portable — the
//! HTML can be copied next to the Markdown, archived, or attached to a ticket.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use llm_wiki_core::config::Config;
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::BuildId;
use llm_wiki_storage::{list_sources, load_generation_view, open};

const TEMPLATE: &str = include_str!("browse.html");
const DATA_PLACEHOLDER: &str = "__WIKI_DATA__";
const WORKSPACE_PLACEHOLDER: &str = "__WORKSPACE__";
/// Aggregate pages carry a reserved category; domains additionally encode their
/// source category in the slug, resolved through `llm_wiki_compiler` so the
/// format has exactly one definition.
const CATEGORY_DOMAIN: &str = "domain";

/// Generates the browse page and prints a one-line summary.
pub fn run(workspace: &Path, output: Option<&str>, json: bool) -> Result<()> {
    let config = Config::load(workspace)?;
    let wiki_root = workspace.join(&config.project.wiki_dir);
    let (build_id, payload, page_count) = load_payload(workspace, &wiki_root, &config)?;

    let json_text = serde_json::to_string(&payload)
        .map_err(|error| WikiError::Storage(format!("serialize browse payload: {error}")))?;
    let html = render(&json_text, &config.project.name)?;

    let target = output
        .map(PathBuf::from)
        .unwrap_or_else(|| wiki_root.join("browse.html"));
    write_page(&target, &html)?;
    report(&target, &build_id, page_count, html.len(), json);
    Ok(())
}

/// Reads the pointer, the active generation and the source paths, and turns
/// them into the payload the browser consumes.
fn load_payload(
    workspace: &Path,
    wiki_root: &Path,
    config: &Config,
) -> Result<(BuildId, serde_json::Value, usize)> {
    let pointer_path = wiki_root.join("current.json");
    let raw = fs::read_to_string(&pointer_path).map_err(|error| {
        WikiError::Storage(format!(
            "cannot read {}: {error} — run `llm-wiki build` first",
            pointer_path.display()
        ))
    })?;
    let pointer: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|error| WikiError::Storage(format!("corrupt current.json: {error}")))?;
    let build_id_text = pointer["build_id"]
        .as_str()
        .ok_or_else(|| WikiError::Storage("current.json has no build_id".to_owned()))?;
    let published_at = pointer["published_at"].as_str().unwrap_or_default().to_owned();
    let build_id = BuildId::from_validated(build_id_text.to_owned());

    let conn = open(&workspace.join(".llm-wiki").join("state.db"))?;
    let pages = load_generation_view(&conn, &build_id)?;
    if pages.is_empty() {
        return Err(WikiError::Storage(
            "the active generation has no pages — rebuild before browsing".to_owned(),
        ));
    }
    let paths = source_paths(&conn)?;
    let page_count = pages.len();

    let payload = serde_json::json!({
        "build_id": build_id.as_str(),
        "published_at": published_at,
        "workspace": config.project.name,
        "pages": pages
            .iter()
            .map(|page| page_payload(page, &paths))
            .collect::<Vec<_>>(),
    });
    Ok((build_id, payload, page_count))
}

/// One page as the browser consumes it.
fn page_payload(
    page: &llm_wiki_storage::GenerationPageView,
    paths: &BTreeMap<String, String>,
) -> serde_json::Value {
    serde_json::json!({
        "id": page.page_id.as_str(),
        "slug": page.slug,
        "title": page.title,
        "category": page.category,
        "domain": domain_of(&page.category, &page.slug),
        "language": page.language,
        "inbound": page.inbound_links,
        "content": page.content,
        "links": page
            .links
            .iter()
            .map(|link| {
                serde_json::json!({
                    "id": link.to_page_id.as_str(),
                    "title": link.target_title,
                })
            })
            .collect::<Vec<_>>(),
        "citations": page
            .citations
            .iter()
            .map(|citation| {
                serde_json::json!({
                    "claim": citation.claim_node_id.as_str(),
                    "source": citation.source_id.as_str(),
                    "path": paths
                        .get(citation.source_id.as_str())
                        .cloned()
                        .unwrap_or_else(|| citation.source_id.as_str().to_owned()),
                    "heading": citation.heading_path,
                    "range": format!("{}-{}", citation.range.start, citation.range.end),
                })
            })
            .collect::<Vec<_>>(),
    })
}

fn write_page(target: &Path, html: &str) -> Result<()> {
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                WikiError::Storage(format!("cannot create {}: {error}", parent.display()))
            })?;
        }
    }
    fs::write(target, html)
        .map_err(|error| WikiError::Storage(format!("cannot write {}: {error}", target.display())))
}

fn report(target: &Path, build_id: &BuildId, pages: usize, bytes: usize, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::json!({
                "path": target.display().to_string(),
                "build_id": build_id.as_str(),
                "pages": pages,
                "bytes": bytes,
            })
        );
    } else {
        println!(
            "browse page: {} ({} pages, {} KiB)",
            target.display(),
            pages,
            bytes / 1024
        );
        println!("open it with a browser — no server needed");
    }
}

/// Substitutes the two placeholders. Kept separate so the escaping rules are
/// testable without touching the filesystem.
fn render(payload_json: &str, workspace_name: &str) -> Result<String> {
    if !TEMPLATE.contains(DATA_PLACEHOLDER) || !TEMPLATE.contains(WORKSPACE_PLACEHOLDER) {
        return Err(WikiError::Storage(
            "browse template is missing a placeholder (corrupted build?)".to_owned(),
        ));
    }
    // A page body containing `</script>` would close the data block early and
    // turn the rest of the JSON into markup. Escaping the slash is valid JSON
    // and inert HTML.
    let safe_json = payload_json.replace("</", "<\\/");
    // Order matters: substitute the workspace name FIRST, then splice in the
    // payload. Reversed, the second replace walks the freshly inlined page
    // bodies and rewrites any literal `__WORKSPACE__` a page happens to hold.
    Ok(TEMPLATE
        .replace(WORKSPACE_PLACEHOLDER, &escape_html(workspace_name))
        .replace(DATA_PLACEHOLDER, &safe_json))
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `source_id → relative path`, so citations show the document a reader knows.
fn source_paths(conn: &llm_wiki_storage::Connection) -> Result<BTreeMap<String, String>> {
    Ok(list_sources(conn)?
        .into_iter()
        .map(|source| (source.source_id.as_str().to_owned(), source.rel_path))
        .collect())
}

/// Domain pages encode their category in the slug; nothing else does.
fn domain_of(category: &str, slug: &str) -> Option<String> {
    if category != CATEGORY_DOMAIN {
        return None;
    }
    llm_wiki_compiler::domain_category(slug).map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_close_inside_a_page_body_cannot_break_out() {
        let payload = r#"{"pages":[{"content":"before </script><img src=x onerror=alert(1)>"}]}"#;
        let html = render(payload, "kb").unwrap();
        assert!(
            !html.contains("</script><img"),
            "the closing tag must be escaped inside the data block"
        );
        assert!(html.contains("<\\/script>"));
    }

    #[test]
    fn placeholders_are_all_replaced() {
        let html = render("{}", "kb").unwrap();
        assert!(!html.contains(DATA_PLACEHOLDER));
        assert!(!html.contains(WORKSPACE_PLACEHOLDER));
        assert!(html.contains("<title>kb · 知识浏览</title>"));
    }

    #[test]
    fn literal_placeholders_in_a_page_body_survive() {
        let payload =
            r#"{"pages":[{"content":"literal __WORKSPACE__ and __WIKI_DATA__ inside a page"}]}"#;
        let html = render(payload, "kb").unwrap();
        assert!(
            html.contains("literal __WORKSPACE__ and __WIKI_DATA__ inside a page"),
            "page content must not be rewritten by the template substitution"
        );
    }

    #[test]
    fn workspace_name_is_html_escaped() {
        let html = render("{}", r#"a<b&"c"#).unwrap();
        assert!(html.contains("<title>a&lt;b&amp;&quot;c · 知识浏览</title>"));
    }

    #[test]
    fn only_domain_pages_carry_a_domain_key() {
        assert_eq!(
            domain_of(CATEGORY_DOMAIN, "_agg-domain-concepts").as_deref(),
            Some("concepts")
        );
        assert_eq!(domain_of("concepts", "auth-flow"), None);
        assert_eq!(domain_of("domain", "unexpected-slug"), None);
    }
}
