//! Stage-1 structured output handling (PRD §28): extract and parse the JSON
//! payload from an LLM response.
//!
//! Stage 1 is *shape only*: it turns response text into a typed value or a
//! machine-readable reason. Referential and semantic validation (stages 2/3)
//! live in the compiler crate because they need registry/section context.
//! A failed schema validation is repaired once by the caller (PRD §11); it is
//! never silently dropped.

use serde::de::DeserializeOwned;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StructuredOutputError {
    #[error("no JSON object found in response")]
    NoJson,
    #[error("response is not valid JSON: {message}")]
    InvalidJson { message: String },
    #[error("response does not match the expected schema: {message}")]
    SchemaMismatch { message: String },
}

impl StructuredOutputError {
    /// Machine-readable one-line reason, fed back to the model for the
    /// single repair attempt (PRD §11.1).
    pub fn machine_reason(&self) -> String {
        match self {
            StructuredOutputError::NoJson => {
                "NO_JSON: the response contained no JSON object; reply with the JSON object only"
                    .to_owned()
            }
            StructuredOutputError::InvalidJson { message } => {
                format!("INVALID_JSON: {message}")
            }
            StructuredOutputError::SchemaMismatch { message } => {
                format!("SCHEMA_MISMATCH: {message}")
            }
        }
    }
}

/// Extracts the outermost JSON object/array from response text, tolerating
/// markdown code fences and surrounding prose.
pub fn extract_json_payload(text: &str) -> Result<&str, StructuredOutputError> {
    let bytes = text.as_bytes();
    let mut start: Option<usize> = None;
    let mut open_byte = b'{';
    for (idx, &b) in bytes.iter().enumerate() {
        if b == b'{' || b == b'[' {
            start = Some(idx);
            open_byte = b;
            break;
        }
    }
    let Some(start) = start else {
        return Err(StructuredOutputError::NoJson);
    };
    let close_byte = match open_byte {
        b'{' => b'}',
        _ => b']',
    };

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, &b) in bytes[start..].iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            _ if b == open_byte => depth += 1,
            _ if b == close_byte => {
                depth -= 1;
                if depth == 0 {
                    return Ok(&text[start..=start + offset]);
                }
            }
            _ => {}
        }
    }
    Err(StructuredOutputError::NoJson)
}

/// Stage 1: response text → typed value, or a machine-readable reason.
pub fn parse_json<T: DeserializeOwned>(response_text: &str) -> Result<T, StructuredOutputError> {
    let payload = extract_json_payload(response_text)?;
    match serde_json::from_str::<T>(payload) {
        Ok(value) => Ok(value),
        Err(err) => Err(StructuredOutputError::SchemaMismatch {
            message: err.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Probe {
        summary: String,
        count: u32,
    }

    #[test]
    fn parses_plain_and_fenced_json_with_prose() {
        let probe = parse_json::<Probe>(r#"{"summary": "ok", "count": 3}"#).unwrap();
        assert_eq!(
            probe,
            Probe {
                summary: "ok".into(),
                count: 3
            }
        );

        let fenced = "Here you go:\n```json\n{\"summary\": \"fenced\", \"count\": 1}\n```\n";
        let probe = parse_json::<Probe>(fenced).unwrap();
        assert_eq!(probe.summary, "fenced");
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_extraction() {
        let text = r#"prefix {"summary": "has } brace", "count": 2} suffix"#;
        let probe = parse_json::<Probe>(text).unwrap();
        assert_eq!(probe.count, 2);
    }

    #[test]
    fn errors_are_machine_readable() {
        let err = parse_json::<Probe>("no json here at all").unwrap_err();
        assert_eq!(err, StructuredOutputError::NoJson);
        assert!(err.machine_reason().starts_with("NO_JSON"));

        let err = parse_json::<Probe>(r#"{"summary": 1, "count": 2}"#).unwrap_err();
        assert!(matches!(err, StructuredOutputError::SchemaMismatch { .. }));
        assert!(err.machine_reason().starts_with("SCHEMA_MISMATCH"));

        let err = parse_json::<Probe>(r#"{broken"#).unwrap_err();
        assert!(matches!(err, StructuredOutputError::NoJson));
    }
}
