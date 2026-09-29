//! Integration tests for the stage-one analysis pipeline (PRD §10/§11/§28),
//! driven by FakeLlmProvider — no real model in CI (PRD §54).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use llm_wiki_compiler::{
    persist_outcome, AnalysisSection, AnalyzedDocument, DocumentAnalyzer, PersistOptions,
};
use llm_wiki_core::ids::{SectionId, SourceLocatorKey};
use llm_wiki_core::model::SourceRange;
use llm_wiki_llm::{FakeLlmProvider, LlmProvider, LlmRequest};
use llm_wiki_storage::open_in_memory;

/// Fake provider that replays scripted responses and records every prompt.
struct ScriptedLlm {
    inner: FakeLlmProvider,
    prompts: Arc<Mutex<Vec<String>>>,
}

impl ScriptedLlm {
    fn new(responses: Vec<String>) -> Self {
        let prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let prompt_log = prompts.clone();
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let inner = FakeLlmProvider::new(
            "fake-analysis",
            Arc::new(move |_request: &LlmRequest| {
                prompt_log.lock().unwrap().push(_request.prompt.clone());
                queue
                    .lock()
                    .unwrap()
                    .pop_front()
                    .ok_or_else(|| llm_wiki_llm::LlmError::Api {
                        code: 500,
                        message: "script exhausted".into(),
                    })
            }),
        );
        Self { inner, prompts }
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl LlmProvider for ScriptedLlm {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    async fn generate(
        &self,
        request: LlmRequest,
    ) -> Result<llm_wiki_llm::LlmResponse, llm_wiki_llm::LlmError> {
        self.inner.generate(request).await
    }
}

const SECTION_A_TEXT: &str = "The plugin runtime retries the resolved to active transition up to three times.\n\nDelivery is at-least-once and handlers must be idempotent.";
const SECTION_B_TEXT: &str = "Permissions are enforced at the message bus boundary.";

const FIXTURE_SECTION_A: &str = "sec_01ARZ3NDEKTSV4RRFFQ69G5FAV";
const FIXTURE_SECTION_B: &str = "sec_01BX5ZZKBKACTAV9WEVGEMMVRZ";

fn fixture_doc() -> AnalyzedDocument {
    let len_a = SECTION_A_TEXT.len();
    let sections = vec![
        AnalysisSection {
            section_id: SectionId::parse(FIXTURE_SECTION_A).unwrap(),
            heading_path: vec!["Plugin Architecture".into(), "Lifecycle".into()],
            range: SourceRange::new(0, len_a),
            content: SECTION_A_TEXT.to_owned(),
        },
        AnalysisSection {
            section_id: SectionId::parse(FIXTURE_SECTION_B).unwrap(),
            heading_path: vec!["Plugin Security".into()],
            range: SourceRange::new(len_a + 1, len_a + 1 + SECTION_B_TEXT.len()),
            content: SECTION_B_TEXT.to_owned(),
        },
    ];
    AnalyzedDocument {
        source_id: llm_wiki_core::ids::SourceId::generate(),
        rel_path: "plugin/architecture.md".into(),
        content_hash: "hash-1".into(),
        language: "en".into(),
        sections,
    }
}

fn analysis_response(
    summary: &str,
    claims: serde_json::Value,
    relations: serde_json::Value,
) -> String {
    serde_json::json!({
        "summary": summary,
        "topics": ["lifecycle"],
        "entities": [{ "name": "Plugin Runtime", "entity_type": "component", "description": "hosts plugins" }],
        "concepts": [{ "name": "At-Least-Once Delivery", "description": "delivery guarantee" }],
        "claims": claims,
        "relations": relations,
    })
    .to_string()
}

fn valid_claim(section_id: &SectionId, quote: &str) -> serde_json::Value {
    serde_json::json!({
        "text": "The runtime retries up to three times.",
        "section_id": section_id.as_str(),
        "evidence_text": quote,
        "evidence_start": 4,
        "confidence": 0.9,
    })
}

/// Inserts the fixture source + sections so claim FKs resolve.
fn seed_db(conn: &mut rusqlite::Connection, rel: &str) -> llm_wiki_core::ids::SourceId {
    let (source_id, _) = llm_wiki_storage::upsert_source(
        conn,
        &SourceLocatorKey::compute("ws", rel),
        rel,
        "hash-1",
        10,
        None,
    )
    .unwrap();
    for (sid, path) in [
        (FIXTURE_SECTION_A, vec!["Plugin Architecture", "Lifecycle"]),
        (FIXTURE_SECTION_B, vec!["Plugin Security"]),
    ] {
        conn.execute(
            "INSERT INTO source_sections (section_id, source_id, heading_path_json, heading_path_key, content_fingerprint, range_start, range_end, status)
             VALUES (?1, ?2, ?3, ?4, 'fp', 0, 100, 'active')",
            rusqlite::params![
                sid,
                source_id.as_str(),
                serde_json::to_string(&path).unwrap(),
                path.join("\u{1f}")
            ],
        )
        .unwrap();
    }
    source_id
}

fn analyzer(provider: Arc<dyn LlmProvider>) -> DocumentAnalyzer {
    DocumentAnalyzer::new(
        provider,
        llm_wiki_compiler::load_prompt("document-analysis", None).unwrap(),
        6000,
        0.10,
        4096,
    )
}

/// Analyzer with a lenient ratio gate, for tests that exercise rejected-
/// record mechanics rather than the threshold gate itself.
fn lenient_analyzer(provider: Arc<dyn LlmProvider>) -> DocumentAnalyzer {
    DocumentAnalyzer::new(
        provider,
        llm_wiki_compiler::load_prompt("document-analysis", None).unwrap(),
        6000,
        0.50,
        4096,
    )
}

#[tokio::test]
async fn happy_path_verifies_claims_and_persists_knowledge() {
    let doc = fixture_doc();
    let id_a = doc.sections[0].section_id.clone();
    let id_b = doc.sections[1].section_id.clone();

    // Evidence quote with different whitespace runs: must still locate.
    let quote = "Permissions are enforced at the message bus
boundary";
    let response = analysis_response(
        "Two facts about the plugin runtime.",
        serde_json::json!([
            valid_claim(
                &id_a,
                "retries the resolved to active transition up to three times"
            ),
            valid_claim(&id_b, quote),
        ]),
        serde_json::json!([
            { "source": "Plugin Runtime", "relation_type": "contains", "target": "At-Least-Once Delivery",
              "section_id": id_b.as_str(), "evidence_text": "Permissions are enforced at the message bus boundary" },
        ]),
    );
    let llm = Arc::new(ScriptedLlm::new(vec![response]));
    let outcome = analyzer(llm.clone())
        .analyze_document(&doc, None)
        .await
        .unwrap();

    assert_eq!(outcome.llm_request_count, 1);
    assert!(
        outcome.rejected_claims.is_empty(),
        "{:?}",
        outcome.rejected_claims
    );
    assert_eq!(outcome.analysis.claims.len(), 2);
    assert_eq!(outcome.verified_relations.len(), 1);

    // Claims cite absolute ranges inside the analyzed text.
    let first = &outcome.analysis.claims[0];
    let range = first.evidence_ranges[0];
    assert!(range.start >= doc.sections[0].range.start && range.end <= doc.sections[0].range.end);

    // ---- Persistence ----
    let mut conn = open_in_memory().unwrap();
    let source_id = seed_db(&mut conn, "plugin/architecture.md");
    let doc_with_source = AnalyzedDocument {
        source_id: source_id.clone(),
        rel_path: doc.rel_path.clone(),
        content_hash: doc.content_hash.clone(),
        language: doc.language.clone(),
        sections: doc.sections.clone(),
    };
    let before_revision = llm_wiki_storage::current_revision(&conn).unwrap();
    let report = persist_outcome(
        &mut conn,
        &doc_with_source,
        &outcome,
        &PersistOptions {
            prompt_version: Some("document-analysis@1".into()),
            replace_source: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(report.persisted.claim_count, 2);
    assert_eq!(report.persisted.citation_count, 3); // 2 claims + 1 relation
    assert_eq!(report.registry_nodes_created, 4); // entity + concept + 2 claims (relation endpoints dedupe into entity)
    assert!(report.registry_revision > before_revision);

    let (active_claims, citations, registry_rows): (i64, i64, i64) = conn
        .query_row(
            "SELECT (SELECT COUNT(*) FROM claims WHERE status='active'),
                    (SELECT COUNT(*) FROM citations),
                    (SELECT COUNT(*) FROM knowledge_registry)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!((active_claims, citations, registry_rows), (2, 3, 4));
}

#[tokio::test]
async fn hallucinated_section_id_is_repaired_once() {
    let doc = fixture_doc();
    let id_a = doc.sections[0].section_id.clone();
    let bogus = serde_json::json!({
        "text": "Fabricated.",
        "section_id": "sec_01AAAAAAAAAAAAAAAAAAAAAAAA",
        "evidence_text": "The plugin runtime retries",
        "evidence_start": 0,
    });
    let good = analysis_response(
        "ok",
        serde_json::json!([bogus, valid_claim(&id_a, "up to three times")]),
        serde_json::json!([]),
    );
    // First response references a hallucinated id; the repair resends it
    // unchanged — the claim must become a RejectedClaim, never knowledge,
    // while the sibling claim stays verified.
    let llm = Arc::new(ScriptedLlm::new(vec![good.clone(), good]));
    let outcome = lenient_analyzer(llm.clone())
        .analyze_document(&doc, None)
        .await
        .unwrap();

    assert_eq!(
        outcome.llm_request_count, 1,
        "hallucinated ids are stage-2, not stage-1: no LLM repair, just a rejected record"
    );
    assert_eq!(outcome.analysis.claims.len(), 1);
    assert_eq!(outcome.rejected_claims.len(), 1);
    assert!(outcome.rejected_claims[0]
        .reason
        .contains("SECTION_NOT_FOUND"));

    // A stage-1 (shape) failure DOES repair once.
    let llm2 = Arc::new(ScriptedLlm::new(vec![
        "this is not json at all".to_owned(),
        analysis_response(
            "recovered",
            serde_json::json!([valid_claim(&id_a, "up to three times")]),
            serde_json::json!([]),
        ),
    ]));
    let outcome2 = analyzer(llm2.clone())
        .analyze_document(&doc, None)
        .await
        .unwrap();
    assert_eq!(outcome2.llm_request_count, 2);
    assert!(outcome2.analysis.claims.len() == 1);
    let repair_prompt = &llm2.prompts()[1];
    assert!(
        repair_prompt.contains("NO_JSON:"),
        "repair must carry the machine reason"
    );
    assert!(repair_prompt.contains("Previous attempt rejected"));
}

#[tokio::test]
async fn double_schema_failure_fails_the_task() {
    let doc = fixture_doc();
    let llm = Arc::new(ScriptedLlm::new(vec![
        "not json".to_owned(),
        "still not json".to_owned(),
    ]));
    let err = analyzer(llm.clone())
        .analyze_document(&doc, None)
        .await
        .unwrap_err();
    assert!(
        matches!(err, llm_wiki_core::WikiError::SchemaValidation(_)),
        "{err}"
    );
}

#[tokio::test]
async fn unverifiable_evidence_becomes_rejected_record() {
    let doc = fixture_doc();
    let id_a = doc.sections[0].section_id.clone();
    let response = analysis_response(
        "one good, one unverifiable",
        serde_json::json!([
            valid_claim(&id_a, "up to three times"),
            valid_claim(&id_a, "this sentence does not exist in the section at all"),
        ]),
        serde_json::json!([]),
    );
    let llm = Arc::new(ScriptedLlm::new(vec![response]));
    let outcome = lenient_analyzer(llm.clone())
        .analyze_document(&doc, None)
        .await
        .unwrap();

    assert_eq!(outcome.analysis.claims.len(), 1);
    assert_eq!(outcome.rejected_claims.len(), 1);
    assert!(outcome.rejected_claims[0]
        .reason
        .contains("EVIDENCE_NOT_IN_SECTION"));

    // Persisted: the rejected candidate is auditable, the verified one is not
    // polluted by it.
    let mut conn = open_in_memory().unwrap();
    let source_id = seed_db(&mut conn, "a.md");
    let doc = AnalyzedDocument { source_id, ..doc };
    let report = persist_outcome(&mut conn, &doc, &outcome, &PersistOptions::default()).unwrap();
    assert_eq!(report.persisted.claim_count, 1);
    assert_eq!(report.persisted.rejected_claim_count, 1);
    let rejected_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM rejected_claims", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rejected_rows, 1);
}

#[tokio::test]
async fn rejected_ratio_over_threshold_fails_the_unit() {
    let doc = fixture_doc();
    let id_a = doc.sections[0].section_id.clone();
    let response = analysis_response(
        "mostly fabricated",
        serde_json::json!([
            valid_claim(&id_a, "up to three times"),
            valid_claim(&id_a, "fabricated one"),
            valid_claim(&id_a, "fabricated two"),
            valid_claim(&id_a, "fabricated three"),
        ]),
        serde_json::json!([]),
    );
    let llm = Arc::new(ScriptedLlm::new(vec![response]));
    let err = analyzer(llm.clone())
        .analyze_document(&doc, None)
        .await
        .unwrap_err();
    assert!(
        matches!(err, llm_wiki_core::WikiError::EvidenceValidation(_)),
        "{err}"
    );
    assert!(err.to_string().contains("75%"));
}

#[tokio::test]
async fn oversized_section_is_segmented_and_evidence_spans_segments() {
    let para =
        "Handlers must remain idempotent under retry delivery at-least-once semantics always. "
            .repeat(8);
    let long_section =
        format!("{para}\n\n{para}\n\n{para}\n\nFinal sentence about the audit log entries.\n");
    let section_id = SectionId::generate();
    let doc = AnalyzedDocument {
        source_id: llm_wiki_core::ids::SourceId::generate(),
        rel_path: "big.md".into(),
        content_hash: "h".into(),
        language: "en".into(),
        sections: vec![AnalysisSection {
            section_id: section_id.clone(),
            heading_path: vec!["Big".into()],
            range: SourceRange::new(0, long_section.len()),
            content: long_section.clone(),
        }],
    };

    // Target small enough to force segmentation. Units arrive in document
    // order; only the final segment holds the quoted sentence. Earlier units
    // return empty extractions.
    let empty = analysis_response("part", serde_json::json!([]), serde_json::json!([]));
    let claim = analysis_response(
        "spans",
        serde_json::json!([valid_claim(
            &section_id,
            "Final sentence about the audit log entries"
        )]),
        serde_json::json!([]),
    );
    let llm = Arc::new(ScriptedLlm::new(vec![
        empty.clone(),
        empty.clone(),
        empty,
        claim,
    ]));
    let analyzer = DocumentAnalyzer::new(
        llm.clone(),
        llm_wiki_compiler::load_prompt("document-analysis", None).unwrap(),
        64,
        0.10,
        4096,
    );
    let outcome = analyzer.analyze_document(&doc, None).await.unwrap();

    assert!(outcome.unit_count > 1, "expected segmented units");
    assert_eq!(outcome.analysis.claims.len(), 1);
    let range = outcome.analysis.claims[0].evidence_ranges[0];
    // The quote lives in the last segment: its absolute range must sit inside
    // the original section and locate the actual text.
    assert!(range.end <= doc.sections[0].range.end);
    assert_eq!(
        &long_section[range.start..range.end],
        "Final sentence about the audit log entries."
    );
}
