//! HTTP API integration tests (PRD §30/§31): driven through `build_router`
//! + `tower::ServiceExt::oneshot` — no sockets, no real model (PRD §54).

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use llm_wiki_compiler::{write_current_pointer, PublishPaths};
use llm_wiki_llm::{FakeLlmProvider, LlmProvider};
use llm_wiki_server::{build_router, validate_security, SharedState};
use llm_wiki_storage::{get_job, open};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "llm-wiki-server-{tag}-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A workspace with config + one markdown source; `llm.model` set so the
/// server can construct a provider for build/query endpoints.
fn fixture_workspace(tag: &str) -> PathBuf {
    let workspace = temp_dir(tag);
    let docs = workspace.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("guide.md"),
        "# Guide\n\nThe scheduler retries failed tasks up to three times.\n",
    )
    .unwrap();
    let state = workspace.join(".llm-wiki");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("config.toml"),
        "[project]\nname = \"server-fixture\"\nwiki_dir = \"./wiki\"\n\n[source]\nroot = \"./docs\"\n\n[llm]\nmodel = \"fake-server\"\n",
    )
    .unwrap();
    workspace
}

fn state_for(workspace: PathBuf) -> SharedState {
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    SharedState::new(workspace, config, None)
}

fn state_with_provider(workspace: PathBuf, provider: Arc<dyn LlmProvider>) -> SharedState {
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    SharedState::new(workspace, config, Some(provider))
}

async fn get(app: axum::Router, uri: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&body).to_string())
}

async fn post(
    app: axum::Router,
    uri: &str,
    body: &str,
    idempotency_key: Option<&str>,
) -> (StatusCode, String) {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(key) = idempotency_key {
        builder = builder.header("idempotency-key", key);
    }
    let response = app
        .oneshot(builder.body(Body::from(body.to_owned())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&body).to_string())
}

#[tokio::test]
async fn health_is_open_and_carries_a_request_id() {
    let app = build_router(state_for(fixture_workspace("health")));
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers().get("x-request-id").is_some(),
        "every response carries x-request-id"
    );
}

#[tokio::test]
async fn status_reports_an_empty_workspace() {
    let app = build_router(state_for(fixture_workspace("status")));
    let (status, body) = get(app, "/v1/status").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"sources\":0"), "{body}");
    assert!(body.contains("\"active_build_id\":null"), "{body}");
}

#[tokio::test]
async fn build_rejects_unknown_source_ids_without_scanning() {
    let app = build_router(state_for(fixture_workspace("bad-source")));
    let (status, body) = post(app, "/v1/build", r#"{"source_id":"../etc"}"#, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("unknown source_id"), "{body}");
}

#[tokio::test]
async fn build_without_a_provider_is_a_config_error() {
    let app = build_router(state_for(fixture_workspace("no-provider")));
    let (status, body) = post(app, "/v1/build", "{}", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("llm.model"), "{body}");
}

#[tokio::test]
async fn build_job_lifecycle_runs_to_a_terminal_state() {
    let workspace = fixture_workspace("job-lifecycle");
    // Invalid analysis output → the job runs and terminalizes FAILED with a
    // retryable LLM-stage failure code.
    let provider = Arc::new(FakeLlmProvider::fixed("fake-server", "not json at all"));
    let state = state_with_provider(workspace.clone(), provider);
    let app = build_router(state);

    let (status, body) = post(app.clone(), "/v1/build", "{}", None).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(payload["status"], "queued");
    let job_id = payload["job_id"].as_str().unwrap().to_owned();

    // The job reaches a terminal state and the slot is released afterwards.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut terminal = String::new();
    while tokio::time::Instant::now() < deadline {
        let (get_status, get_body) = get(app.clone(), &format!("/v1/jobs/{job_id}")).await;
        assert_eq!(get_status, StatusCode::OK, "{get_body}");
        let record: serde_json::Value = serde_json::from_str(&get_body).unwrap();
        let job_status = record["status"].as_str().unwrap().to_owned();
        if job_status != "QUEUED" && job_status != "RUNNING" {
            terminal = job_status;
            assert_eq!(record["failure_code"], "llm_error", "{get_body}");
            assert_eq!(record["retryable"], true, "{get_body}");
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(terminal, "FAILED", "job reached a terminal state");

    // The slot was released: a second build is accepted (not 409).
    let (status, _) = post(app, "/v1/build", "{}", None).await;
    assert_eq!(status, StatusCode::ACCEPTED);
}

#[tokio::test]
async fn idempotency_key_replays_the_same_job() {
    let workspace = fixture_workspace("idempotency");
    let provider = Arc::new(FakeLlmProvider::fixed("fake-server", "not json at all"));
    let state = state_with_provider(workspace.clone(), provider);
    let app = build_router(state);

    let (first_status, first_body) =
        post(app.clone(), "/v1/build", "{}", Some("client-key-1")).await;
    assert_eq!(first_status, StatusCode::ACCEPTED, "{first_body}");
    let first: serde_json::Value = serde_json::from_str(&first_body).unwrap();
    let job_id = first["job_id"].as_str().unwrap().to_owned();

    // Wait for the job to terminalize, then replay the same key.
    let conn_path = workspace.join(".llm-wiki").join("state.db");
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let conn = open(&conn_path).unwrap();
        let record = get_job(
            &conn,
            &llm_wiki_core::ids::JobId::from_validated(job_id.clone()),
        )
        .unwrap()
        .unwrap();
        if !matches!(record.status.as_str(), "QUEUED" | "RUNNING") {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "job never finished");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    let (replay_status, replay_body) = post(app, "/v1/build", "{}", Some("client-key-1")).await;
    assert_eq!(replay_status, StatusCode::OK, "{replay_body}");
    let replay: serde_json::Value = serde_json::from_str(&replay_body).unwrap();
    assert_eq!(replay["replayed"], true, "{replay_body}");
    assert_eq!(replay["job"]["job_id"], job_id.as_str(), "{replay_body}");
}

#[tokio::test]
async fn queue_cap_rejects_new_builds() {
    let workspace = fixture_workspace("queue-cap");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let mut config = config;
    config.server.max_queued_jobs = 0;
    let state = SharedState::new(
        workspace,
        config,
        Some(Arc::new(FakeLlmProvider::fixed("fake-server", "x"))),
    );
    let app = build_router(state);
    let (status, body) = post(app, "/v1/build", "{}", None).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    assert!(body.contains("job_queue_full"), "{body}");
}

#[tokio::test]
async fn cancel_a_queued_job_immediately_and_a_running_job_via_the_flag() {
    let workspace = fixture_workspace("cancel");
    let provider = Arc::new(FakeLlmProvider::fixed("fake-server", "not json"));
    let state = state_with_provider(workspace.clone(), provider);
    let app = build_router(state.clone());

    // RUNNING path: plant a RUNNING row + claim the slot ourselves, then
    // cancel — the endpoint must flip OUR flag instance.
    let job_id = llm_wiki_core::ids::JobId::generate();
    let record = llm_wiki_storage::ServerJobRecord {
        job_id: job_id.clone(),
        kind: "build".into(),
        status: "RUNNING".into(),
        phase: Some("ANALYZING".into()),
        build_id: None,
        failure_code: None,
        retryable: false,
        error: None,
        request_id: None,
        idempotency_key: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: Some(chrono::Utc::now().to_rfc3339()),
        finished_at: None,
    };
    let conn = open(&state.db_path()).unwrap();
    llm_wiki_storage::insert_job(&conn, &record).unwrap();
    let cancel = llm_wiki_core::CancelFlag::new();
    assert!(
        state
            .0
            .jobs
            .start(llm_wiki_server::state::ActiveJob {
                job_id: job_id.clone(),
                cancel: cancel.clone(),
            })
            .await
    );

    let (status, body) = post(app.clone(), &format!("/v1/jobs/{job_id}/cancel"), "", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(cancel.is_cancelled(), "the running job's flag was set");
    assert!(body.contains("cancelling"), "{body}");
    state.0.jobs.take(&job_id).await;

    // QUEUED path: a queued row cancels in place.
    let queued_id = llm_wiki_core::ids::JobId::generate();
    let mut queued = record.clone();
    queued.job_id = queued_id.clone();
    queued.status = "QUEUED".into();
    queued.phase = None;
    queued.started_at = None;
    llm_wiki_storage::insert_job(&conn, &queued).unwrap();
    let (status, body) = post(app, &format!("/v1/jobs/{queued_id}/cancel"), "", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("cancelled"), "{body}");
}

#[tokio::test]
async fn search_and_pages_on_a_never_built_workspace() {
    let app = build_router(state_for(fixture_workspace("empty-search")));
    let (status, body) = post(app.clone(), "/v1/search", r#"{"query":"guide"}"#, None).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(body.contains("nothing_published"), "{body}");

    let (status, body) = get(app, "/v1/pages").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(body.contains("nothing_published"), "{body}");
}

#[tokio::test]
async fn list_jobs_is_cursor_paginated() {
    let workspace = fixture_workspace("job-pagination");
    let state = state_for(workspace.clone());
    let conn = open(&state.db_path()).unwrap();
    for i in 0..3 {
        let record = llm_wiki_storage::ServerJobRecord {
            job_id: llm_wiki_core::ids::JobId::generate(),
            kind: "build".into(),
            status: "COMPLETED".into(),
            phase: None,
            build_id: None,
            failure_code: None,
            retryable: false,
            error: None,
            request_id: None,
            idempotency_key: None,
            created_at: format!("2026-01-0{}T00:00:00+00:00", i + 1),
            started_at: None,
            finished_at: None,
        };
        llm_wiki_storage::insert_job(&conn, &record).unwrap();
    }
    let app = build_router(state);
    let (status, body) = get(app.clone(), "/v1/jobs?limit=2&status=COMPLETED").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let page: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(page["truncated"], true);
    assert_eq!(page["jobs"].as_array().unwrap().len(), 2);
    let cursor = page["next_cursor"].as_str().unwrap().to_owned();

    let (status, body) = get(
        app,
        &format!("/v1/jobs?limit=2&status=COMPLETED&cursor={cursor}"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let page_two: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(page_two["truncated"], false);
    assert_eq!(page_two["jobs"].as_array().unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// Security boundary (PRD §30)
// ---------------------------------------------------------------------------

#[test]
fn local_mode_refuses_non_loopback_bind() {
    let mut config = llm_wiki_core::config::Config::default();
    let validate_at = |bind: &str| validate_security(&config, bind).is_err();
    assert!(validate_at("0.0.0.0"));
    assert!(validate_at("192.168.1.10"));
    assert!(!validate_at("127.0.0.1"));
    assert!(!validate_at("::1"));
    assert!(!validate_at("localhost"));

    config.server.remote_enabled = true;
    // Remote mode with no token env set → refuse to start.
    let unique_env = "LLM_WIKI_TEST_TOKEN_LOCAL_MODE";
    std::env::remove_var(unique_env);
    config.server.auth_token_env = unique_env.to_owned();
    assert!(validate_security(&config, "0.0.0.0").is_err());
    std::env::set_var(unique_env, "secret-token");
    assert!(validate_security(&config, "0.0.0.0").is_ok());
}

#[tokio::test]
async fn remote_mode_requires_a_bearer_token() {
    let workspace = fixture_workspace("remote-auth");
    let config = llm_wiki_core::config::Config::load(&workspace).unwrap();
    let unique_env = "LLM_WIKI_TEST_TOKEN_REMOTE";
    std::env::set_var(unique_env, "secret-token");
    let mut config = config;
    config.server.remote_enabled = true;
    config.server.auth_token_env = unique_env.to_owned();
    let state = SharedState::new(workspace, config, None);
    let app = build_router(state);

    let (status, body) = get(app.clone(), "/v1/status").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    assert!(body.contains("unauthorized"), "{body}");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .header("authorization", "Bearer secret-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // /health stays open in remote mode (unauthenticated liveness).
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Cancellation race regressions (#I01/#I02 from the post-delivery review)
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use llm_wiki_core::ids::JobId;
use llm_wiki_llm::LlmError;
use llm_wiki_server::state::ActiveJob;
use llm_wiki_storage::{
    finish_job, insert_job, set_job_running, ServerJobRecord, FAILURE_CANCELLED,
};

fn job_row(job_id: &JobId, status: &str) -> ServerJobRecord {
    ServerJobRecord {
        job_id: job_id.clone(),
        kind: "build".into(),
        status: status.into(),
        phase: None,
        build_id: None,
        failure_code: None,
        retryable: false,
        error: None,
        request_id: None,
        idempotency_key: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        finished_at: None,
    }
}

/// #I01 regression: cancelling a RUNNING job must NOT release the build
/// slot — the cancelled pipeline is still unwinding, and a new build
/// admitted in that window would race it over the registry.
#[tokio::test]
async fn cancelling_a_running_job_keeps_the_build_slot_claimed() {
    let workspace = fixture_workspace("cancel-keeps-slot");
    let state = state_with_provider(
        workspace,
        Arc::new(FakeLlmProvider::fixed("fake-server", "x")),
    );
    let job_id = JobId::generate();

    let conn = open(&state.db_path()).unwrap();
    insert_job(&conn, &job_row(&job_id, "RUNNING")).unwrap();
    let cancel = llm_wiki_core::CancelFlag::new();
    assert!(
        state
            .0
            .jobs
            .start(ActiveJob {
                job_id: job_id.clone(),
                cancel: cancel.clone(),
            })
            .await
    );

    let app = build_router(state.clone());
    let (status, body) = post(app.clone(), &format!("/v1/jobs/{job_id}/cancel"), "", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(cancel.is_cancelled(), "the running job's flag was set");

    // THE CONTRACT: the slot is still claimed while the cancelled pipeline
    // unwinds — a concurrent build request is rejected with 409, not run.
    assert_eq!(
        state
            .0
            .jobs
            .running()
            .await
            .map(|running| running.as_str().to_owned()),
        Some(job_id.as_str().to_owned()),
        "cancelling must not free the build slot"
    );
    let (build_status, build_body) = post(app, "/v1/build", "{}", None).await;
    assert_eq!(build_status, StatusCode::CONFLICT, "{build_body}");
    assert!(build_body.contains("build_already_running"), "{build_body}");

    // Only the job task's own unwind releases the slot.
    assert!(state.0.jobs.take(&job_id).await);
    assert!(state.0.jobs.running().await.is_none());
}

/// #I02 regression: a job cancelled before its task claims the row must
/// never run the pipeline — the task exits immediately, the row stays
/// CANCELLED, and the provider is never invoked.
#[tokio::test]
async fn cancelled_job_task_exits_without_running_the_pipeline() {
    let workspace = fixture_workspace("cancel-before-start");
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let handler = Arc::new(
        move |_request: &llm_wiki_llm::LlmRequest| -> Result<String, LlmError> {
            counter.fetch_add(1, AtomicOrdering::SeqCst);
            Err(LlmError::Api {
                code: 500,
                message: "provider must not be called for a cancelled job".into(),
            })
        },
    );
    let provider: Arc<dyn LlmProvider> = Arc::new(FakeLlmProvider::new("fake-server", handler));
    let state = state_with_provider(workspace, provider.clone());
    let job_id = JobId::generate();

    let conn = open(&state.db_path()).unwrap();
    insert_job(&conn, &job_row(&job_id, "QUEUED")).unwrap();
    assert!(finish_job(
        &conn,
        &job_id,
        "CANCELLED",
        Some(FAILURE_CANCELLED),
        true,
        Some("cancelled before start"),
    )
    .unwrap());

    let cancel = llm_wiki_core::CancelFlag::new();
    assert!(
        state
            .0
            .jobs
            .start(ActiveJob {
                job_id: job_id.clone(),
                cancel: cancel.clone(),
            })
            .await
    );

    llm_wiki_server::jobs::spawn_build_job(state.clone(), job_id.clone(), cancel, provider);

    // The task exits at once (set_job_running returns false): the slot is
    // freed and the provider was never touched.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while state.0.jobs.running().await.is_some() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the cancelled task never released the slot"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        0,
        "no LLM call happened"
    );
    let record = get_job(&conn, &job_id).unwrap().unwrap();
    assert_eq!(record.status, "CANCELLED", "the terminal row is immutable");
}

/// #I02 regression (task side): `set_job_running` on a non-QUEUED row
/// returns false, which is the early-exit signal.
#[tokio::test]
async fn set_job_running_reports_a_cancelled_row_as_not_started() {
    let workspace = fixture_workspace("set-running-guard");
    let state = state_for(workspace);
    let conn = open(&state.db_path()).unwrap();
    let job_id = JobId::generate();
    insert_job(&conn, &job_row(&job_id, "QUEUED")).unwrap();
    finish_job(
        &conn,
        &job_id,
        "CANCELLED",
        Some(FAILURE_CANCELLED),
        true,
        Some("cancelled before start"),
    )
    .unwrap();
    assert!(!set_job_running(&conn, &job_id).unwrap());
}

// ---------------------------------------------------------------------------
// Insights + Embed + auto-resume (V0.4/V1.x functional completion)
// ---------------------------------------------------------------------------

use llm_wiki_core::hash::sha256_hex;
use llm_wiki_core::ids::{BuildId, InsightId, WikiPageId};
use llm_wiki_llm::{EmbeddingProvider, LlmError as EmbedLlmError};
use llm_wiki_storage::{
    insert_insight, persist_generation, requeue_job, start_build, InsightCitation, InsightRecord,
    WikiPageRecord,
};

/// A deterministic fake embedding provider counting its calls.
struct CountingEmbeddings {
    calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl EmbeddingProvider for CountingEmbeddings {
    async fn embed(&self, _model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedLlmError> {
        self.calls.fetch_add(1, AtomicOrdering::SeqCst);
        Ok(texts
            .iter()
            .map(|text| vec![text.len() as f32; 4])
            .collect())
    }
}

/// Seeds a published generation (one page, one citation-backed section row is
/// not needed: context sections come from wiki page sections) plus one
/// insight citing a claim.
fn seed_published_generation(workspace: &std::path::Path) -> (BuildId, String) {
    let db_path = workspace.join(".llm-wiki").join("state.db");
    let mut conn = open(&db_path).unwrap();
    let build_id = start_build(&mut conn, &llm_wiki_storage::BuildDraft::default()).unwrap();
    let claim_id = "kn_fixtureclaim".to_owned();
    let content = "## Overview\n\nThe fixture page cites its stored claims.\n";
    let page = WikiPageRecord {
        page_id: WikiPageId::generate(),
        slug: "fixture".into(),
        title: "Fixture".into(),
        category: "concepts".into(),
        language: "en".into(),
        body_hash: sha256_hex(content.as_bytes()),
        content: content.to_owned(),
        knowledge_refs: Vec::new(),
        citations: Vec::new(),
        links: Vec::new(),
    };
    persist_generation(&mut conn, &build_id, &[page]).unwrap();
    llm_wiki_storage::activate_build(&mut conn, &build_id).unwrap();
    std::fs::create_dir_all(workspace.join("wiki")).unwrap();
    write_current_pointer(&PublishPaths::new(&workspace.join("wiki")), &build_id).unwrap();
    (build_id, claim_id)
}

/// GET /v1/insights: cursor-paginated read surface over the insight layer.
#[tokio::test]
async fn insights_endpoint_lists_paged_records() {
    let workspace = fixture_workspace("insights-endpoint");
    let (build_id, _claim) = seed_published_generation(&workspace);
    let state = state_for(workspace.clone());
    let mut conn = open(&state.db_path()).unwrap();
    for i in 0..3 {
        insert_insight(
            &mut conn,
            &InsightRecord {
                insight_id: InsightId::generate(),
                build_id: build_id.clone(),
                query: format!("question {i}"),
                answer: format!("answer {i}"),
                citations: vec![InsightCitation {
                    claim_node_id: "kn_x".into(),
                    source: "docs/a.md".into(),
                    heading_path: vec![],
                    range: (0, 1),
                    evidence_digest: "d".into(),
                }],
                created_at: format!("2026-01-0{}T00:00:00+00:00", i + 1),
            },
        )
        .unwrap();
    }

    let app = build_router(state);
    let (status, body) = get(app.clone(), "/v1/insights?limit=2").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let page: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(page["protocol_version"], 1);
    assert!(!page["request_id"].as_str().unwrap_or("").is_empty());
    assert_eq!(page["truncated"], true);
    assert_eq!(page["insights"].as_array().unwrap().len(), 2);
    assert_eq!(page["insights"][0]["query"], "question 2", "newest first");
    let cursor = page["next_cursor"].as_str().unwrap().to_owned();

    let (status, body) = get(app, &format!("/v1/insights?limit=2&cursor={cursor}")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let page_two: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(page_two["truncated"], false);
    assert_eq!(page_two["insights"].as_array().unwrap().len(), 1);
    assert_eq!(page_two["insights"][0]["query"], "question 0");
}

/// POST /v1/embed: runs the incremental backfill with the injected embedding
/// provider and reports the counters.
#[tokio::test]
async fn embed_endpoint_runs_the_backfill() {
    let workspace = fixture_workspace("embed-endpoint");
    seed_published_generation(&workspace);
    let calls = Arc::new(AtomicUsize::new(0));
    let state = state_with_provider(
        workspace,
        Arc::new(FakeLlmProvider::fixed("fake-server", "x")),
    )
    .with_embedding_provider(Arc::new(CountingEmbeddings {
        calls: calls.clone(),
    }));
    // The model name comes from the env (same contract as the CLI).
    std::env::set_var("LLM_WIKI_EMBEDDING_MODEL", "fake-embed");

    let app = build_router(state);
    let (status, body) = post(app, "/v1/embed", "{}", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(payload["protocol_version"], 1);
    assert_eq!(payload["model"], "fake-embed");
    assert!(payload["total_sections"].is_u64(), "{body}");

    // Uncovered generation: the endpoint issued at most one embed call.
    assert!(calls.load(AtomicOrdering::SeqCst) <= 1, "{body}");
}

/// Auto-resume (config-gated): recover_interrupted_jobs re-enqueues and
/// re-runs INTERRUPTED-by-restart build jobs one at a time, reusing the SAME
/// job row (idempotency mapping preserved).
#[tokio::test]
async fn interrupted_jobs_are_resumed_through_the_same_row() {
    let workspace = fixture_workspace("job-resume");
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let handler = Arc::new(
        move |_request: &llm_wiki_llm::LlmRequest| -> Result<String, LlmError> {
            counter.fetch_add(1, AtomicOrdering::SeqCst);
            Err(LlmError::Api {
                code: 500,
                message: "deterministic pipeline failure".into(),
            })
        },
    );
    let provider: Arc<dyn LlmProvider> = Arc::new(FakeLlmProvider::new("fake-server", handler));
    let state = state_with_provider(workspace, provider);

    // Seed a job the previous server died with.
    let job_id = JobId::generate();
    let conn = open(&state.db_path()).unwrap();
    let mut record = job_row(&job_id, "INTERRUPTED");
    record.failure_code = Some(llm_wiki_storage::FAILURE_INTERRUPTED.into());
    record.retryable = true;
    record.idempotency_key = Some("client-key-resume".into());
    insert_job(&conn, &record).unwrap();

    let resumed = llm_wiki_server::jobs::recover_interrupted_jobs(&state).await;
    assert_eq!(resumed, 1, "the interrupted job was resumed");

    // Wait for the re-run to terminalize (the fake always fails the pipeline).
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let record = get_job(&conn, &job_id).unwrap().unwrap();
        if !matches!(record.status.as_str(), "QUEUED" | "RUNNING") {
            assert_eq!(
                record.status, "FAILED",
                "the resumed build failed as scripted"
            );
            assert!(calls.load(AtomicOrdering::SeqCst) >= 1, "the pipeline ran");
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "resumed job never finished"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    // The idempotency key still resolves to the SAME row.
    let by_key = llm_wiki_storage::get_job_by_idempotency_key(&conn, "client-key-resume")
        .unwrap()
        .unwrap();
    assert_eq!(by_key.job_id, job_id);
    assert!(state.0.jobs.running().await.is_none(), "slot released");
}

/// requeue_job never resurrects rows interrupted for other reasons.
#[test]
fn requeue_refuses_non_restart_interruptions() {
    let workspace = fixture_workspace("requeue-guard");
    let state = state_for(workspace);
    let conn = open(&state.db_path()).unwrap();
    let job_id = JobId::generate();
    let mut record = job_row(&job_id, "INTERRUPTED");
    record.failure_code = Some("llm_error".into());
    insert_job(&conn, &record).unwrap();
    assert!(!requeue_job(&conn, &job_id).unwrap());
}
