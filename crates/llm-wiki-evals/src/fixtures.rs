//! Loaders for the checked-in fixture files (`evals/dataset.yaml`,
//! `evals/expected/pages.yaml`). The files use a small, fixed YAML shape that
//! the pipeline itself never mutates, so a purpose-built parser keeps the
//! evals crate free of a YAML dependency (same approach as
//! `evals/check_fixtures.py`).

use std::path::{Path, PathBuf};

use llm_wiki_core::error::WikiError;

/// §37.3 minimums, mirrored from `evals/README.md` — asserted by the loader
/// so a shrunken fixture fails loudly instead of deflating the gates.
pub const MIN_DOCS: usize = 30;
pub const MIN_MDX: usize = 2;
pub const MIN_QUESTIONS: usize = 20;
pub const MIN_CJK: usize = 2;

#[derive(Debug, Clone)]
pub struct Fact {
    pub id: String,
    /// Verbatim sentence from the source doc (annotation contract).
    pub span: String,
    pub importance: String,
    /// Corpus-relative path of the doc this fact is annotated in.
    pub doc_path: String,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub facts: Vec<Fact>,
    pub doc_paths: Vec<String>,
}

impl Dataset {
    /// The `importance: high` facts — the Source Coverage denominator.
    pub fn high_facts(&self) -> impl Iterator<Item = &Fact> {
        self.facts.iter().filter(|f| f.importance == "high")
    }
}

#[derive(Debug, Clone)]
pub struct ExpectedPage {
    pub title: String,
    pub min_sources_merged: usize,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ExpectedPages {
    pub pages: Vec<ExpectedPage>,
}

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("fixture {path}: {problem}")]
    Shape { path: PathBuf, problem: String },
    #[error("fixture minimums violated: {0}")]
    Minimums(String),
}

impl From<FixtureError> for WikiError {
    fn from(err: FixtureError) -> Self {
        WikiError::Config(err.to_string())
    }
}

fn shape(path: &Path, problem: impl Into<String>) -> FixtureError {
    FixtureError::Shape {
        path: path.to_path_buf(),
        problem: problem.into(),
    }
}

/// A scalar-or-container tree for the fixture files' fixed shape.
#[derive(Debug, Clone)]
enum Node {
    Scalar(String),
    Seq(Vec<Node>),
    Map(Vec<(String, Node)>),
}

impl Node {
    fn get(&self, key: &str, path: &Path) -> Result<&Node, FixtureError> {
        match self {
            Node::Map(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .ok_or_else(|| shape(path, format!("missing key '{key}'"))),
            _ => Err(shape(path, format!("expected a map to read '{key}'"))),
        }
    }

    fn as_seq(&self, path: &Path) -> Result<&[Node], FixtureError> {
        match self {
            Node::Seq(items) => Ok(items),
            _ => Err(shape(path, "expected a list")),
        }
    }

    fn as_scalar(&self, path: &Path) -> Result<&str, FixtureError> {
        match self {
            Node::Scalar(s) => Ok(s),
            _ => Err(shape(path, "expected a scalar")),
        }
    }
}

/// True when the next meaningful line after `line_index` (0-based) starts a
/// list item — used to decide whether `key:` opens a list or a map.
fn next_line_is_item(lines: &[&str], line_index: usize) -> bool {
    lines
        .iter()
        .skip(line_index + 1)
        .find(|l| {
            let trimmed = l.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .is_some_and(|l| l.trim_start().starts_with("- "))
}

/// Parses the fixture YAML shape: indentation-nested maps/lists, `- ` items,
/// quoted or bare scalars, `#` comments. Frames hold the container being
/// filled plus the key under which it attaches to its parent on pop.
fn parse_yaml(text: &str, path: &Path) -> Result<Node, FixtureError> {
    let lines: Vec<&str> = text.lines().collect();
    // (children_min_indent, container, pending_key_in_parent)
    let mut frames: Vec<(usize, Node, Option<String>)> =
        vec![(usize::MAX, Node::Map(Vec::new()), None)];

    // The map that receives `key:` lines for the top frame: the frame's own
    // map, or — inside a list — the map of its last item.
    fn current_map<'a>(
        frame: &'a mut Node,
        path: &Path,
        lineno: usize,
    ) -> Result<&'a mut Vec<(String, Node)>, FixtureError> {
        match frame {
            Node::Map(entries) => Ok(entries),
            Node::Seq(items) => match items.last_mut() {
                Some(Node::Map(entry)) => Ok(entry),
                _ => Err(shape(
                    path,
                    format!("line {lineno}: 'key:' with no enclosing map"),
                )),
            },
            Node::Scalar(_) => Err(shape(
                path,
                format!("line {lineno}: 'key:' inside a scalar"),
            )),
        }
    }

    // Attaches the popped frame's container under its pending key in the
    // parent's current map.
    fn attach_child(frames: &mut Vec<(usize, Node, Option<String>)>) {
        let (_, container, key) = frames.pop().expect("frame");
        let parent = frames.last_mut().expect("root frame");
        if let (Ok(entries), Some(key)) = (
            current_map(&mut parent.1, Path::new("yaml"), 0),
            key.as_ref(),
        ) {
            if let Some(slot) = entries.iter_mut().find(|(k, _)| k == key) {
                slot.1 = container;
            }
        }
    }

    for (idx, raw) in lines.iter().enumerate() {
        let lineno = idx + 1;
        let line = raw.trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let indent = line.len() - trimmed.len();

        // Pop frames this line exits, attaching their containers inward.
        while frames.len() > 1 && indent < frames.last().unwrap().0 {
            attach_child(&mut frames);
        }

        let body = trimmed.strip_prefix("- ").unwrap_or(trimmed).trim();

        if trimmed.starts_with("- ") {
            let frame = frames.last_mut().unwrap();
            let seq = match &mut frame.1 {
                Node::Seq(items) => items,
                _ => {
                    return Err(shape(
                        path,
                        format!("line {lineno}: list item outside a list"),
                    ))
                }
            };
            match split_kv(body) {
                // `- key: value` opens a one-entry map that deeper-indented
                // `key: value` lines extend (the fixture item shape).
                Some((key, Some(value))) => seq.push(Node::Map(vec![(key, Node::Scalar(value))])),
                // Pure scalar item (e.g. `- auth/authentication.md`).
                None => seq.push(Node::Scalar(unquote(body))),
                // `- key:` with an empty value never occurs in the fixtures;
                // reject rather than guess.
                Some((key, None)) => {
                    return Err(shape(
                        path,
                        format!("line {lineno}: '- {key}:' with empty value is not fixture YAML"),
                    ));
                }
            }
            continue;
        }

        // Plain `key: value` or `key:` line.
        let (key, value) = split_kv(body)
            .ok_or_else(|| shape(path, format!("line {lineno}: expected 'key: value'")))?;
        match value {
            Some(v) => {
                let frame = frames.last_mut().unwrap();
                current_map(&mut frame.1, path, lineno)?.push((key, Node::Scalar(v)));
            }
            None => {
                let child = if next_line_is_item(&lines, idx) {
                    Node::Seq(Vec::new())
                } else {
                    Node::Map(Vec::new())
                };
                {
                    let frame = frames.last_mut().unwrap();
                    current_map(&mut frame.1, path, lineno)?.push((key.clone(), child.clone()));
                }
                frames.push((indent + 1, child, Some(key)));
            }
        }
    }

    while frames.len() > 1 {
        attach_child(&mut frames);
    }
    Ok(frames.pop().unwrap().1)
}

fn split_kv(body: &str) -> Option<(String, Option<String>)> {
    let sep = find_kv_colon(body)?;
    let key = unquote(body[..sep].trim());
    let value_raw = body[sep + 1..].trim();
    let value = if value_raw.is_empty() {
        None
    } else {
        Some(unquote(value_raw))
    };
    Some((key, value))
}

/// Colon that separates key from value: not inside quotes, not a URL scheme.
fn find_kv_colon(body: &str) -> Option<usize> {
    let bytes = body.as_bytes();
    let mut in_quotes = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => in_quotes = !in_quotes,
            b':' if !in_quotes => {
                if body[i..].starts_with("://") {
                    i += 3;
                    continue;
                }
                return Some(i);
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn unquote(text: &str) -> String {
    let text = text.trim();
    if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
        text[1..text.len() - 1]
            .replace("\\\\", "\\")
            .replace("\\\"", "\"")
    } else {
        text.to_owned()
    }
}

/// Loads `evals/dataset.yaml`, asserting every annotated span appears
/// verbatim in its corpus doc and the §37.3 minimums hold.
pub fn load_dataset(evals_dir: &Path) -> Result<Dataset, FixtureError> {
    let path = evals_dir.join("dataset.yaml");
    let text = std::fs::read_to_string(&path).map_err(|source| FixtureError::Io {
        path: path.clone(),
        source,
    })?;
    let root = parse_yaml(&text, &path)?;
    let docs = root.get("docs", &path)?.as_seq(&path)?;

    let mut facts = Vec::new();
    let mut doc_paths = Vec::new();
    let mut mdx_count = 0usize;
    let mut cjk_count = 0usize;
    for doc in docs {
        let doc_path = doc.get("path", &path)?.as_scalar(&path)?.to_owned();
        let language = doc.get("language", &path)?.as_scalar(&path)?.to_owned();
        if doc_path.ends_with(".mdx") {
            mdx_count += 1;
        }
        if language.starts_with("zh") {
            cjk_count += 1;
        }
        let corpus_file = evals_dir.join("corpus").join(&doc_path);
        let body = std::fs::read_to_string(&corpus_file).map_err(|source| FixtureError::Io {
            path: corpus_file,
            source,
        })?;
        for fact in doc.get("facts", &path)?.as_seq(&path)? {
            let id = fact.get("id", &path)?.as_scalar(&path)?.to_owned();
            let span = fact.get("span", &path)?.as_scalar(&path)?.to_owned();
            let importance = fact.get("importance", &path)?.as_scalar(&path)?.to_owned();
            if !body.contains(&span) {
                return Err(shape(
                    &path,
                    format!("fact {id}: span not verbatim in {doc_path}"),
                ));
            }
            facts.push(Fact {
                id,
                span,
                importance,
                doc_path: doc_path.clone(),
            });
        }
        doc_paths.push(doc_path);
    }

    if doc_paths.len() < MIN_DOCS {
        return Err(FixtureError::Minimums(format!(
            "{} docs < {MIN_DOCS}",
            doc_paths.len()
        )));
    }
    if mdx_count < MIN_MDX {
        return Err(FixtureError::Minimums(format!(
            "{mdx_count} mdx < {MIN_MDX}"
        )));
    }
    if cjk_count < MIN_CJK {
        return Err(FixtureError::Minimums(format!(
            "{cjk_count} cjk < {MIN_CJK}"
        )));
    }
    Ok(Dataset { facts, doc_paths })
}

/// Loads `evals/expected/pages.yaml` (the required page subset).
pub fn load_expected_pages(evals_dir: &Path) -> Result<ExpectedPages, FixtureError> {
    let path = evals_dir.join("expected").join("pages.yaml");
    let text = std::fs::read_to_string(&path).map_err(|source| FixtureError::Io {
        path: path.clone(),
        source,
    })?;
    let root = parse_yaml(&text, &path)?;
    let mut pages = Vec::new();
    for page in root.get("pages", &path)?.as_seq(&path)? {
        let title = page.get("title", &path)?.as_scalar(&path)?.to_owned();
        let min_sources_merged = page
            .get("min_sources_merged", &path)?
            .as_scalar(&path)?
            .parse::<usize>()
            .map_err(|e| shape(&path, format!("page {title}: min_sources_merged: {e}")))?;
        let sources = page
            .get("sources", &path)?
            .as_seq(&path)?
            .iter()
            .map(|s| s.as_scalar(&path).map(str::to_owned))
            .collect::<Result<Vec<_>, _>>()?;
        if min_sources_merged > sources.len() {
            return Err(shape(
                &path,
                format!("page {title}: min_sources_merged exceeds listed sources"),
            ));
        }
        pages.push(ExpectedPage {
            title,
            min_sources_merged,
            sources,
        });
    }
    Ok(ExpectedPages { pages })
}

/// Loads both fixtures plus `questions.yaml`, enforcing all minimums.
pub fn load_fixtures(evals_dir: &Path) -> Result<(Dataset, ExpectedPages), FixtureError> {
    let dataset = load_dataset(evals_dir)?;
    let pages = load_expected_pages(evals_dir)?;
    let questions_path = evals_dir.join("questions.yaml");
    let questions =
        std::fs::read_to_string(&questions_path).map_err(|source| FixtureError::Io {
            path: questions_path.clone(),
            source,
        })?;
    let question_count = questions
        .lines()
        .filter(|line| line.trim_start().starts_with("- id:"))
        .count();
    if question_count < MIN_QUESTIONS {
        return Err(FixtureError::Minimums(format!(
            "{question_count} questions < {MIN_QUESTIONS}"
        )));
    }
    Ok((dataset, pages))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_parser_handles_fixture_shape() {
        let text = "version: 1\ndocs:\n  - path: a.md\n    language: en\n    facts:\n      - id: F-1\n        span: \"quoted: colon and \\\"escapes\\\"\"\n        importance: high\n      - id: F-2\n        span: \"second\"\n        importance: normal\n  - path: b.mdx\n    language: zh-CN\n    facts: []\n";
        let path = Path::new("sample.yaml");
        let root = parse_yaml(text, path).unwrap();
        let docs = root.get("docs", path).unwrap().as_seq(path).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(
            docs[0].get("path", path).unwrap().as_scalar(path).unwrap(),
            "a.md"
        );
        let facts = docs[0].get("facts", path).unwrap().as_seq(path).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(
            facts[0].get("span", path).unwrap().as_scalar(path).unwrap(),
            "quoted: colon and \"escapes\""
        );
        assert_eq!(
            facts[1]
                .get("importance", path)
                .unwrap()
                .as_scalar(path)
                .unwrap(),
            "normal"
        );
        assert_eq!(
            docs[1]
                .get("language", path)
                .unwrap()
                .as_scalar(path)
                .unwrap(),
            "zh-CN"
        );
    }

    #[test]
    fn yaml_parser_handles_expected_pages_shape() {
        let text = "version: 1\npages:\n  - title: \"Security & Compliance\"\n    min_sources_merged: 4\n    sources:\n      - auth/authentication.md\n      - storage/encryption.md\n  - title: \"API Access\"\n    min_sources_merged: 3\n    sources:\n      - api/api-auth.md\n";
        let path = Path::new("pages.yaml");
        let root = parse_yaml(text, path).unwrap();
        let pages = root.get("pages", path).unwrap().as_seq(path).unwrap();
        assert_eq!(pages.len(), 2);
        let sources = pages[0].get("sources", path).unwrap().as_seq(path).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[1].as_scalar(path).unwrap(), "storage/encryption.md");
        assert_eq!(
            pages[1]
                .get("min_sources_merged", path)
                .unwrap()
                .as_scalar(path)
                .unwrap(),
            "3"
        );
    }
}
