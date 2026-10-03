//! Retrieval eval (PRD §50 V0.5 tail): aggregate Recall@K / MRR over every
//! `evals/questions.yaml` query against the REAL FTS index of the published
//! generation.
//!
//! Where `tests/search_gates.rs` asserts per-category Top-K presence (a
//! floor per query class), this module scores the WHOLE question set as one
//! ranked-retrieval benchmark — the metric a rerank strategy has to beat.

use std::collections::BTreeMap;

pub use crate::fixtures::RetrievalQuery;

#[derive(Debug, Clone)]
pub struct RetrievalQuestionResult {
    pub id: String,
    /// 1-based rank of the first hit whose page cites an expected source
    /// (0 = not found within the candidate list).
    pub first_relevant_rank: usize,
}

#[derive(Debug, Clone)]
pub struct RetrievalReport {
    pub questions: usize,
    pub k: usize,
    /// Fraction of questions with ≥1 relevant hit in the top K.
    pub recall_at_k: f64,
    /// Mean reciprocal rank of the first relevant hit (0 when missed).
    pub mrr: f64,
    pub per_question: Vec<RetrievalQuestionResult>,
}

/// Scores one query's ranked hit slugs against the page-slugs-per-corpus-doc
/// mapping of the active generation.
///
/// A hit is relevant when its page cites ANY of the query's expected
/// sources. Deterministic; no LLM.
pub fn score_query(
    hit_slugs: &[String],
    expected_sources: &[String],
    pages_by_source: &BTreeMap<String, Vec<String>>,
) -> usize {
    let relevant: std::collections::BTreeSet<&str> = expected_sources
        .iter()
        .filter_map(|source| pages_by_source.get(source))
        .flat_map(|slugs| slugs.iter().map(String::as_str))
        .collect();
    hit_slugs
        .iter()
        .position(|slug| relevant.contains(slug.as_str()))
        .map_or(0, |index| index + 1)
}

/// Aggregates per-query ranks into the retrieval report.
pub fn aggregate(results: Vec<RetrievalQuestionResult>, k: usize) -> RetrievalReport {
    let questions = results.len();
    let hits_in_k = results
        .iter()
        .filter(|r| r.first_relevant_rank > 0 && r.first_relevant_rank <= k)
        .count();
    let reciprocal: f64 = results
        .iter()
        .filter(|r| r.first_relevant_rank > 0)
        .map(|r| 1.0 / r.first_relevant_rank as f64)
        .sum();
    RetrievalReport {
        questions,
        k,
        recall_at_k: if questions == 0 {
            0.0
        } else {
            hits_in_k as f64 / questions as f64
        },
        mrr: if questions == 0 {
            0.0
        } else {
            reciprocal / questions as f64
        },
        per_question: results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slugs(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).to_owned()).collect()
    }

    fn mapping() -> BTreeMap<String, Vec<String>> {
        BTreeMap::from([
            ("auth.md".to_owned(), slugs(&["identity"])),
            ("queue.md".to_owned(), slugs(&["messaging", "retries"])),
        ])
    }

    #[test]
    fn relevance_is_any_expected_source_covered_by_the_page() {
        let hits = slugs(&["unrelated", "messaging", "identity"]);
        let rank = score_query(&hits, &["queue.md".to_owned()], &mapping());
        assert_eq!(rank, 2);

        let rank = score_query(&hits, &["auth.md".to_owned()], &mapping());
        assert_eq!(rank, 3);

        assert_eq!(score_query(&hits, &["nope.md".to_owned()], &mapping()), 0);
    }

    #[test]
    fn aggregate_computes_recall_and_mrr() {
        let results = vec![
            RetrievalQuestionResult {
                id: "Q-1".into(),
                first_relevant_rank: 1,
            },
            RetrievalQuestionResult {
                id: "Q-2".into(),
                first_relevant_rank: 3,
            },
            RetrievalQuestionResult {
                id: "Q-3".into(),
                first_relevant_rank: 0,
            },
        ];
        let report = aggregate(results, 5);
        assert_eq!(report.questions, 3);
        assert!((report.recall_at_k - 2.0 / 3.0).abs() < 1e-9);
        assert!((report.mrr - (1.0 + 1.0 / 3.0) / 3.0).abs() < 1e-9);
    }

    #[test]
    fn ranks_beyond_k_count_for_mrr_but_not_recall() {
        let results = vec![RetrievalQuestionResult {
            id: "Q-1".into(),
            first_relevant_rank: 7,
        }];
        let report = aggregate(results, 5);
        assert!(report.recall_at_k.abs() < 1e-9);
        assert!((report.mrr - 1.0 / 7.0).abs() < 1e-9);
    }
}
