//! Language tagging (PRD §9): priority is
//! frontmatter `lang`/`language` > filename suffix (`.cn.`, `.zh-cn.` …) >
//! script heuristic > `und`. The tag feeds the V0.2 TextAnalyzer.

use crate::frontmatter::Frontmatter;

/// Detects the document language as a BCP-47-ish tag.
pub fn detect_language(frontmatter: Option<&Frontmatter>, file_name: &str, body: &str) -> String {
    if let Some(fm) = frontmatter {
        for key in ["lang", "language"] {
            if let Some(v) = fm.get(key) {
                let v = v.trim();
                if !v.is_empty() {
                    return normalize_tag(v);
                }
            }
        }
    }

    let lower = file_name.to_ascii_lowercase();
    for (needle, tag) in [
        (".zh-cn.", "zh-CN"),
        (".zh_cn.", "zh-CN"),
        (".cn.", "zh-CN"),
        (".zh.", "zh-CN"),
        (".en.", "en"),
    ] {
        if lower.contains(needle) {
            return tag.to_owned();
        }
    }

    match script_heuristic(body) {
        Some(tag) => tag.to_owned(),
        None => "und".to_owned(),
    }
}

fn normalize_tag(tag: &str) -> String {
    let mut parts = tag.split('-');
    let primary = parts.next().unwrap_or("").to_ascii_lowercase();
    match parts.next() {
        Some(region) => format!("{primary}-{}", region.to_ascii_uppercase()),
        None => primary,
    }
}

/// Ratio-based script heuristic over a sample of the body.
fn script_heuristic(body: &str) -> Option<&'static str> {
    let sample: String = body.chars().take(4096).collect();
    let mut cjk = 0usize;
    let mut latin = 0usize;
    for ch in sample.chars() {
        if is_cjk(ch) {
            cjk += 1;
        } else if ch.is_ascii_alphabetic() {
            latin += 1;
        }
    }
    if cjk + latin < 16 {
        return None;
    }
    if cjk * 20 > (cjk + latin) {
        Some("zh")
    } else {
        Some("en")
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3400..=0x4DBF   // CJK ext A
        | 0x4E00..=0x9FFF // CJK unified
        | 0xF900..=0xFAFF // compatibility ideographs
        | 0x3040..=0x30FF // kana
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontmatter::strip_and_parse;

    fn fm_of(raw: &str) -> Frontmatter {
        strip_and_parse(raw).0.unwrap()
    }

    #[test]
    fn frontmatter_wins() {
        let fm = fm_of("---\nlang: zh-CN\n---\nEnglish text only here.");
        assert_eq!(
            detect_language(Some(&fm), "doc.en.md", "English text only here."),
            "zh-CN"
        );
    }

    #[test]
    fn filename_suffix_is_second() {
        assert_eq!(
            detect_language(None, "plugin-development.cn.mdx", "Plug-in development"),
            "zh-CN"
        );
        assert_eq!(
            detect_language(None, "plugin-development.en.md", "English"),
            "en"
        );
    }

    #[test]
    fn script_heuristic_detects_chinese_and_english() {
        assert_eq!(
            detect_language(
                None,
                "doc.md",
                "本插件系统提供统一的权限模型，支持多种运行时配置。"
            ),
            "zh"
        );
        assert_eq!(
            detect_language(
                None,
                "doc.md",
                "The plugin system provides a unified permission model for all runtimes."
            ),
            "en"
        );
    }

    #[test]
    fn falls_back_to_und() {
        assert_eq!(detect_language(None, "doc.md", "123456 +++ ---"), "und");
    }
}
