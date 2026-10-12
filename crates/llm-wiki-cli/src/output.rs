//! Machine-readable (`--json`) renderings of CLI command results.
//!
//! One function per command shapes exactly the data its human-readable path
//! already prints — the CLI never invents fields it did not compute. Field
//! names mirror the frozen HTTP protocol v1 payloads (`docs/agent-protocol.md`,
//! built in `llm-wiki-server/src/routes.rs`) wherever the same concept exists,
//! so a consumer can reuse one parser across `llm-wiki --json` and `/v1/*`.
//!
//! Commands that need a published wiki carry a top-level `published` flag:
//! `false` is the JSON form of the human "nothing published" line (still exit
//! 0 — a never-built workspace is not an error).

use std::path::Path;

use llm_wiki_compiler::{
    AskReport, BuildReport, EmbedReport, LintReport, ReplanReport, SemanticReport,
};
use llm_wiki_core::ids::BuildId;
use llm_wiki_search::SearchHit;
use llm_wiki_source::{ScanDiagnostic, ScannedFile};
use llm_wiki_storage::BuildRecord;
use serde_json::{json, Value};

/// One `doctor` check: `status` is `ok` / `fail` / `skip`, and a `fail`
/// always carries the `remedy` command that fixes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Ok,
    Fail,
    Skip,
}

impl CheckStatus {
    fn label(self) -> &'static str {
        match self {
            CheckStatus::Ok => "ok",
            CheckStatus::Fail => "fail",
            CheckStatus::Skip => "skip",
        }
    }
}

#[derive(Debug)]
pub struct Check {
    pub label: String,
    pub status: CheckStatus,
    pub detail: String,
    pub remedy: Option<String>,
}

/// `llm-wiki init --json`.
pub fn init(config_path: &Path) -> Value {
    json!({ "config_path": config_path.display().to_string() })
}

/// `llm-wiki scan --json`. The per-file list mirrors the text path (it prints
/// one line per file); `rel_path` / `size` / `content_hash` are the same fields.
pub fn scan(
    root: &Path,
    files: &[ScannedFile],
    snapshot: &str,
    state_db: &Path,
    diagnostics: &[ScanDiagnostic],
) -> Value {
    json!({
        "source_root": root.display().to_string(),
        "snapshot": snapshot,
        "state_db": state_db.display().to_string(),
        "files": files.iter().map(|file| json!({
            "rel_path": file.rel_path,
            "size": file.size,
            "content_hash": file.content_hash,
        })).collect::<Vec<_>>(),
        "diagnostics": diagnostics
            .iter()
            .map(|d| json!({
                "rel_path": d.rel_path,
                "kind": format!("{:?}", d.kind),
                "message": d.message,
            }))
            .collect::<Vec<_>>(),
    })
}

/// `llm-wiki build --json`.
pub fn build(source_root: &Path, wiki_dir: &Path, report: &BuildReport) -> Value {
    json!({
        "source_root": source_root.display().to_string(),
        "wiki_dir": wiki_dir.display().to_string(),
        "build_id": report.build_id.as_str(),
        "sources": report.sources,
        "pages": report.pages,
        "citations": report.citations,
        "links": report.links,
        "llm_request_count": report.llm_request_count,
        "cache": { "hits": report.cache.hits, "misses": report.cache.misses },
        "incremental": report.incremental.as_ref().map(|i| json!({
            "changed": i.changed,
            "deleted": i.deleted,
            "recompiled": i.recompiled,
            "carried": i.carried,
            "obsolete": i.obsolete,
        })),
        "published_path": report.published_path.display().to_string(),
        "recovery": report.recovery,
    })
}

/// `llm-wiki replan --json`. The dry-run and execute shapes share one object:
/// `estimate` is present only on a dry run, `build_id`/`published_path` only
/// after an executed replan.
pub fn replan(report: &ReplanReport) -> Value {
    json!({
        "dry_run": report.dry_run,
        "no_replan_needed": report.no_replan_needed,
        "triggers": report.triggers,
        "diff": {
            "unchanged": report.unchanged,
            "modified": report.modified,
            "merged": report.merged,
            "split": report.split,
            "created": report.created,
            "retired": report.retired,
        },
        "unchanged_pages": report.unchanged_pages,
        "modified_pages": report.modified_pages,
        "merged_pages": report.merged_pages.iter().map(|(slug, predecessors)| json!({
            "slug": slug,
            "predecessors": predecessors,
        })).collect::<Vec<_>>(),
        "split_pages": report.split_pages.iter().map(|(slug, successors)| json!({
            "slug": slug,
            "successors": successors,
        })).collect::<Vec<_>>(),
        "created_pages": report.created_pages,
        "retired_pages": report.retired_pages.iter().map(|(slug, uncovered_refs)| json!({
            "slug": slug,
            "uncovered_refs": uncovered_refs,
        })).collect::<Vec<_>>(),
        "knowledge_loss_warnings": report.knowledge_loss_warnings,
        "estimate": report.dry_run.then(|| json!({
            "compile_llm_calls": report.estimated_compile_llm_calls,
            "max_output_tokens": report.estimated_max_output_tokens,
        })),
        "build_id": report.build_id.as_ref().map(|b| b.as_str()),
        "sources": report.sources,
        "pages": report.pages,
        "citations": report.citations,
        "links": report.links,
        "recompiled": report.recompiled,
        "carried": report.carried,
        "llm_request_count": report.llm_request_count,
        "cache": { "hits": report.cache.hits, "misses": report.cache.misses },
        "published_path": report.published_path.as_ref().map(|p| p.display().to_string()),
        "recovery": report.recovery,
    })
}

/// `llm-wiki status --json`. Source counts and the latest build only — the CLI
/// has no job queue, so unlike `/v1/status` there is no `jobs` field.
/// `sources` is `null` when no state db exists yet (nothing scanned).
pub fn status(
    sources: Option<u64>,
    latest: Option<&BuildRecord>,
    active: Option<&BuildId>,
) -> Value {
    json!({
        "sources": sources,
        "latest_build": latest.map(|build| json!({
            "build_id": build.build_id.as_str(),
            "status": build.status,
            "started_at": build.started_at.to_rfc3339(),
        })),
        "active_build_id": active.map(|b| b.as_str()),
    })
}

/// `llm-wiki search --json`. The CLI's `search` is wiki-only FTS, so hits
/// always carry `evidence_kind: "wiki"`; `citation_count` is a server-side
/// enrichment the CLI does not compute and therefore does not report.
/// `truncated` matches `/v1/search`: true when more hits existed than the
/// returned `hits`.
pub fn search(hits: &[SearchHit], truncated: bool) -> Value {
    json!({
        "published": true,
        "hits": hits.iter().map(|hit| json!({
            "evidence_kind": "wiki",
            "page_id": hit.page_id.as_str(),
            "slug": hit.slug,
            "title": hit.title,
            "heading_path": hit.heading_path,
            "snippet": hit.snippet,
            "rank": hit.rank,
        })).collect::<Vec<_>>(),
        "truncated": truncated,
    })
}

/// `llm-wiki ask --json` — mirrors `POST /v1/query`.
pub fn ask(report: &AskReport) -> Value {
    json!({
        "published": true,
        "answer": report.answer,
        "citations": report.citations.iter().map(|c| json!({
            "claim_node_id": c.claim_node_id.as_str(),
            "source_id": c.source_id.as_str(),
            "section_id": c.section_id.as_ref().map(|s| s.as_str()),
            "source_hash": c.source_hash,
            "evidence_digest": c.evidence_digest,
            "heading_path": c.heading_path,
        })).collect::<Vec<_>>(),
        "sources": report.sources,
        "llm_request_count": report.llm_request_count,
        "insight_id": report.insight_id,
    })
}

/// `llm-wiki embed --json` — mirrors `POST /v1/embed` (minus the transport
/// fields).
pub fn embed(report: &EmbedReport) -> Value {
    json!({
        "published": true,
        "model": report.model,
        "total_sections": report.total_sections,
        "covered_before": report.covered_before,
        "embedded": report.embedded,
    })
}

/// `llm-wiki lint --json`. `semantic` is present only when `--semantic` ran
/// (it costs model calls and is advisory — it never affects the exit code,
/// which stays 11 when `errors > 0`).
pub fn lint(report: &LintReport, semantic: Option<&SemanticReport>) -> Value {
    json!({
        "published": true,
        "findings": report.findings.iter().map(|f| json!({
            "check": f.check.label(),
            "severity": f.severity.label(),
            "page_slug": f.page_slug,
            "message": f.message,
        })).collect::<Vec<_>>(),
        "errors": report.errors(),
        "warnings": report.warnings(),
        "semantic": semantic.map(|s| json!({
            "findings": s.findings.iter().map(|f| json!({
                "kind": f.kind.label(),
                "page_slug": f.page_slug,
                "claim_ids": f.claim_ids,
                "message": f.message,
            })).collect::<Vec<_>>(),
            "pages_reviewed": s.pages_reviewed,
            "skipped_pages": s.skipped_pages,
        })),
    })
}

/// `llm-wiki doctor --json`. `failures` / `ok` are derived from the checks so
/// the count can never disagree with the list.
pub fn doctor(checks: &[Check]) -> Value {
    let failures = checks
        .iter()
        .filter(|check| check.status == CheckStatus::Fail)
        .count();
    json!({
        "ok": failures == 0,
        "failures": failures,
        "checks": checks.iter().map(|c| json!({
            "label": c.label,
            "status": c.status.label(),
            "detail": c.detail,
            "remedy": c.remedy,
        })).collect::<Vec<_>>(),
    })
}

/// `llm-wiki serve --json`: the address the server is about to bind. Printed
/// before the listener starts, so a failed bind still surfaces as a
/// non-zero exit rather than a false "listening" claim.
pub fn serve(bind: &str, remote_enabled: bool) -> Value {
    json!({
        "status": "starting",
        "bind": bind,
        "remote_enabled": remote_enabled,
    })
}

/// The JSON form of the human "nothing published" line (exit 0).
pub fn not_published() -> Value {
    json!({ "published": false })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_compiler::{
        LintCheck, LintFinding, LintReport, LintSeverity, SemanticFinding, SemanticFindingKind,
        SemanticReport,
    };

    fn finding(check: LintCheck, severity: LintSeverity) -> LintFinding {
        LintFinding {
            check,
            severity,
            page_slug: "auth".to_owned(),
            message: "boom".to_owned(),
        }
    }

    #[test]
    fn lint_json_carries_labels_counts_and_the_advisory_review() {
        let report = LintReport {
            findings: vec![
                finding(LintCheck::MissingSource, LintSeverity::Error),
                finding(LintCheck::OrphanPage, LintSeverity::Warning),
            ],
        };
        let semantic = SemanticReport {
            findings: vec![SemanticFinding {
                kind: SemanticFindingKind::Contradiction,
                page_slug: "auth".to_owned(),
                claim_ids: vec!["claim-1".to_owned()],
                message: "conflict".to_owned(),
            }],
            skipped_pages: vec!["billing".to_owned()],
            pages_reviewed: 3,
        };

        let value = lint(&report, Some(&semantic));
        assert_eq!(value["published"], json!(true));
        assert_eq!(value["errors"], json!(1));
        assert_eq!(value["warnings"], json!(1));
        assert_eq!(value["findings"][0]["check"], json!("missing-source"));
        assert_eq!(value["findings"][0]["severity"], json!("error"));
        assert_eq!(value["findings"][1]["severity"], json!("warning"));
        assert_eq!(
            value["semantic"]["findings"][0]["kind"],
            json!("contradictory-claims")
        );
        assert_eq!(
            value["semantic"]["findings"][0]["claim_ids"][0],
            json!("claim-1")
        );
        assert_eq!(value["semantic"]["pages_reviewed"], json!(3));
        assert_eq!(value["semantic"]["skipped_pages"][0], json!("billing"));
    }

    #[test]
    fn lint_json_omits_semantic_when_it_did_not_run() {
        let value = lint(&LintReport { findings: vec![] }, None);
        assert!(value["semantic"].is_null());
        assert_eq!(value["errors"], json!(0));
    }

    #[test]
    fn not_published_is_the_whole_shape_for_a_missing_wiki() {
        assert_eq!(not_published(), json!({ "published": false }));
    }

    #[test]
    fn doctor_json_derives_ok_and_renders_each_status() {
        let checks = vec![
            Check {
                label: "config".to_owned(),
                status: CheckStatus::Ok,
                detail: "valid".to_owned(),
                remedy: None,
            },
            Check {
                label: "llm api key".to_owned(),
                status: CheckStatus::Fail,
                detail: "unset".to_owned(),
                remedy: Some("export LLM_WIKI_API_KEY=<key>".to_owned()),
            },
            Check {
                label: "endpoint connectivity".to_owned(),
                status: CheckStatus::Skip,
                detail: "pass --live".to_owned(),
                remedy: None,
            },
        ];

        // The count is derived from the list, never passed in: one Fail ⇒ ok=false.
        let value = doctor(&checks);
        assert_eq!(value["ok"], json!(false));
        assert_eq!(value["failures"], json!(1));
        assert_eq!(value["checks"][0]["status"], json!("ok"));
        assert_eq!(value["checks"][1]["status"], json!("fail"));
        assert_eq!(
            value["checks"][1]["remedy"],
            json!("export LLM_WIKI_API_KEY=<key>")
        );
        assert_eq!(value["checks"][2]["status"], json!("skip"));

        // A `skip` is not a failure: no Fail ⇒ ok=true, failures=0.
        let passing = vec![
            Check {
                label: "config".to_owned(),
                status: CheckStatus::Ok,
                detail: "valid".to_owned(),
                remedy: None,
            },
            Check {
                label: "endpoint connectivity".to_owned(),
                status: CheckStatus::Skip,
                detail: "pass --live".to_owned(),
                remedy: None,
            },
        ];
        assert_eq!(doctor(&passing)["ok"], json!(true));
        assert_eq!(doctor(&passing)["failures"], json!(0));
    }

    #[test]
    fn search_json_reports_wiki_hits_and_truncation() {
        let hit = SearchHit {
            page_id: llm_wiki_core::ids::WikiPageId::generate(),
            slug: "auth".to_owned(),
            title: "Auth".to_owned(),
            heading_path: vec!["Overview".to_owned()],
            snippet: "uses SSO".to_owned(),
            rank: -1.5,
        };

        let value = search(&[hit], true);
        assert_eq!(value["published"], json!(true));
        assert_eq!(value["truncated"], json!(true));
        assert_eq!(value["hits"][0]["evidence_kind"], json!("wiki"));
        assert_eq!(value["hits"][0]["slug"], json!("auth"));
        assert_eq!(value["hits"][0]["heading_path"][0], json!("Overview"));
        assert_eq!(value["hits"][0]["rank"], json!(-1.5));

        let empty = search(&[], false);
        assert_eq!(empty["truncated"], json!(false));
        assert_eq!(empty["hits"], json!([]));
    }

    #[test]
    fn status_json_reports_nulls_before_any_scan() {
        let value = status(None, None, None);
        assert!(value["sources"].is_null());
        assert!(value["latest_build"].is_null());
        assert!(value["active_build_id"].is_null());
    }

    #[test]
    fn serve_json_names_the_bind_it_is_about_to_use() {
        let value = serve("127.0.0.1:8080", false);
        assert_eq!(value["status"], json!("starting"));
        assert_eq!(value["bind"], json!("127.0.0.1:8080"));
        assert_eq!(value["remote_enabled"], json!(false));
    }
}
