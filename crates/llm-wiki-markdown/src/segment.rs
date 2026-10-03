//! Section segmentation for oversized sections (PRD §10).
//!
//! A section larger than `section_target_tokens` is split at *block
//! boundaries* into `SectionSegment`s: same section identity, contiguous
//! source ranges, stable ordinal. Truncation is never allowed; a single block
//! larger than the target becomes one oversized segment rather than being cut.

use llm_wiki_core::model::SourceRange;

use crate::parser::SectionOutput;

/// Rough text→token estimate used for packing decisions only (never for
/// citation semantics). Single implementation lives in the core crate so the
/// segmentation and the planning budgets can never drift apart (audit
/// FIX-017): script-aware, CJK charged ~1 token per 1.5 chars.
pub use llm_wiki_core::plan::estimate_tokens;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionSegment {
    /// Absolute offsets into the analyzed document text; segments of one
    /// section are contiguous and ordered.
    pub source_range: SourceRange,
    pub content: String,
    /// Stable 0-based ordinal within the section.
    pub ordinal: usize,
}

/// Splits a section into segments of at most `max_tokens` each, at blank-line
/// block boundaries. A section within budget yields a single segment (the
/// whole section, ordinal 0).
pub fn split_section(section: &SectionOutput, max_tokens: u64) -> Vec<SectionSegment> {
    let section_start = section.source_range.start;
    let body = section.content.as_str();
    if estimate_tokens(body) <= max_tokens {
        return vec![SectionSegment {
            source_range: section.source_range,
            content: body.to_owned(),
            ordinal: 0,
        }];
    }

    // Block boundaries: blank lines inside the section content.
    let mut boundaries: Vec<usize> = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = body[search_from..].find("\n\n") {
        boundaries.push(search_from + rel);
        search_from += rel + 2;
    }

    let mut segments: Vec<SectionSegment> = Vec::new();
    let mut block_ranges: Vec<(usize, usize)> = Vec::new();
    let mut prev = 0usize;
    for boundary in boundaries.iter().chain(std::iter::once(&body.len())) {
        let end = (*boundary).min(body.len());
        if end > prev {
            block_ranges.push((prev, end));
        }
        prev = end + 2.min(body.len() - end);
    }
    if block_ranges.is_empty() {
        // No blank lines at all: one oversized segment (no truncation).
        block_ranges.push((0, body.len()));
    }

    let mut current_start = block_ranges[0].0;
    let mut current_end = block_ranges[0].0;
    for (block_start, block_end) in block_ranges {
        let candidate_end = block_end;
        let candidate = &body[current_start..candidate_end];
        if current_end > current_start && estimate_tokens(candidate) > max_tokens {
            segments.push(flush_segment(
                body,
                current_start,
                current_end,
                section_start,
                segments.len(),
            ));
            current_start = block_start;
        }
        current_end = candidate_end;
    }
    if current_end > current_start {
        segments.push(flush_segment(
            body,
            current_start,
            current_end,
            section_start,
            segments.len(),
        ));
    }

    segments
}

fn flush_segment(
    body: &str,
    start: usize,
    end: usize,
    section_start: usize,
    ordinal: usize,
) -> SectionSegment {
    let content = body[start..end].trim();
    // Trimmed content keeps absolute offsets via the trim prefix.
    let trim_prefix = body[start..end].len() - body[start..end].trim_start().len();
    SectionSegment {
        source_range: SourceRange::new(
            section_start + start + trim_prefix,
            section_start + start + trim_prefix + content.len(),
        ),
        content: content.to_owned(),
        ordinal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(content: &str, start: usize) -> SectionOutput {
        SectionOutput {
            heading: Some("H".into()),
            heading_level: 2,
            heading_path: vec!["H".into()],
            content: content.to_owned(),
            source_range: SourceRange::new(start, start + content.len()),
        }
    }

    #[test]
    fn small_section_is_one_segment() {
        let s = section("short body", 10);
        let segments = split_section(&s, 100);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].ordinal, 0);
        assert_eq!(segments[0].content, "short body");
        assert_eq!(segments[0].source_range, SourceRange::new(10, 20));
    }

    #[test]
    fn oversized_section_splits_at_blank_lines_with_contiguous_ranges() {
        let para = "word ".repeat(40).trim_end().to_owned(); // ~50 tokens
        let content = format!("{para}\n\n{para}\n\n{para}\n\n{para}\n");
        let s = section(&content, 0);
        let segments = split_section(&s, 60);
        assert!(
            segments.len() >= 2,
            "expected split, got {}",
            segments.len()
        );
        for (i, seg) in segments.iter().enumerate() {
            assert_eq!(seg.ordinal, i);
            assert!(
                estimate_tokens(&seg.content) <= 60,
                "segment {i} exceeds budget"
            );
        }
        // Contiguity: each segment starts after the previous one ends.
        for pair in segments.windows(2) {
            assert!(pair[1].source_range.start >= pair[0].source_range.end);
        }
        // Ranges round-trip into the source content.
        for seg in &segments {
            assert_eq!(
                &content[seg.source_range.start..seg.source_range.end],
                seg.content
            );
        }
    }

    #[test]
    fn monolithic_block_is_never_truncated() {
        let huge = "x".repeat(10_000);
        let s = section(&huge, 0);
        let segments = split_section(&s, 100);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].content, huge, "no truncation allowed");
    }
}
