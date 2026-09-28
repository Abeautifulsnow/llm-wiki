//! Prompt loading and versioning (PRD §26): prompts live in `prompts/*.md`
//! with a `name`/`version` header, never hardcoded in Rust source. The build
//! records the prompt version (BuildFingerprint `prompt_version`).

use std::path::Path;

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_markdown::frontmatter::strip_and_parse;

const EMBEDDED_DOCUMENT_ANALYSIS: &str = include_str!("../../../prompts/document-analysis.md");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptDocument {
    pub name: String,
    pub version: u32,
    pub body: String,
}

impl PromptDocument {
    /// Parses a prompt file body: frontmatter `name`/`version` + template
    /// body with `{{VAR}}` placeholders.
    pub fn parse(text: &str) -> Result<Self> {
        let (frontmatter, body, diagnostic) = strip_and_parse(text);
        if let Some(diagnostic) = diagnostic {
            return Err(WikiError::Config(format!(
                "prompt header invalid: {}",
                diagnostic.message
            )));
        }
        let frontmatter = frontmatter.ok_or_else(|| {
            WikiError::Config("prompt file must start with a `---` name/version header".to_owned())
        })?;
        let name = frontmatter
            .get("name")
            .ok_or_else(|| WikiError::Config("prompt header missing `name`".to_owned()))?
            .to_owned();
        let version: u32 = frontmatter
            .get("version")
            .ok_or_else(|| WikiError::Config("prompt header missing `version`".to_owned()))?
            .parse()
            .map_err(|_| WikiError::Config("prompt header `version` is not a number".to_owned()))?;
        Ok(PromptDocument {
            name,
            version,
            body: body.trim().to_owned(),
        })
    }

    /// Loads a prompt by name: `prompts_dir/{name}.md` when a directory is
    /// given (workspace override), otherwise the version pinned in this
    /// repository.
    pub fn load(name: &str, prompts_dir: Option<&Path>) -> Result<Self> {
        let path = prompts_dir.map(|dir| dir.join(format!("{name}.md")));
        match path {
            Some(path) if path.exists() => {
                let text = std::fs::read_to_string(&path).map_err(|e| {
                    WikiError::Config(format!("cannot read {}: {e}", path.display()))
                })?;
                let parsed = Self::parse(&text)?;
                if parsed.name != name {
                    return Err(WikiError::Config(format!(
                        "prompt file {} declares name '{}' != '{name}'",
                        path.display(),
                        parsed.name
                    )));
                }
                Ok(parsed)
            }
            _ => match name {
                "document-analysis" => Self::parse(EMBEDDED_DOCUMENT_ANALYSIS),
                other => Err(WikiError::Config(format!(
                    "unknown prompt '{other}' and no file override provided"
                ))),
            },
        }
    }

    /// Substitutes `{{VAR}}` placeholders.
    pub fn render(&self, vars: &[(&str, &str)]) -> String {
        let mut out = self.body.clone();
        for (key, value) in vars {
            out = out.replace(&format!("{{{{{key}}}}}"), value);
        }
        out
    }

    /// `name@version`, the BuildFingerprint `prompt_version` value.
    pub fn fingerprint_tag(&self) -> String {
        format!("{}@{}", self.name, self.version)
    }
}

pub fn load_prompt(name: &str, prompts_dir: Option<&Path>) -> Result<PromptDocument> {
    PromptDocument::load(name, prompts_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_document_analysis_prompt_is_v1() {
        let prompt = load_prompt("document-analysis", None).unwrap();
        assert_eq!(prompt.name, "document-analysis");
        assert_eq!(prompt.version, 1);
        assert!(prompt.body.contains("{{SECTIONS}}"));
        assert!(prompt.body.contains("{{LANGUAGE}}"));
        assert!(prompt.body.contains("{{REPAIR_NOTES}}"));
        assert_eq!(prompt.fingerprint_tag(), "document-analysis@1");
    }

    #[test]
    fn render_substitutes_all_placeholders() {
        let prompt = PromptDocument::parse(
            "---\nname: t\nversion: 3\n---\nHello {{A}} and {{B}} and {{A}}!",
        )
        .unwrap();
        assert_eq!(prompt.version, 3);
        assert_eq!(
            prompt.render(&[("A", "x"), ("B", "y")]),
            "Hello x and y and x!"
        );
    }

    #[test]
    fn header_errors_are_deterministic() {
        assert!(PromptDocument::parse("no header at all").is_err());
        assert!(PromptDocument::parse("---\nname: x\n---\nbody").is_err());
        assert!(PromptDocument::parse("---\nname: x\nversion: zero\n---\nbody").is_err());
    }
}
