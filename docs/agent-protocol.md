# Agent Protocol — llm-wiki v1

The agent-facing HTTP surface of `llm-wiki-server`. This contract is **frozen
at version 1**: future changes are additive only (new optional request
fields, new response fields, new endpoints) and bump `protocol_version`.
Breaking changes require a new protocol version and a new URL namespace.

The TypeScript SDK in `sdk/typescript` implements exactly this contract.

## Conventions

- Base URL: the `llm-wiki serve` listener (default `127.0.0.1:8080`).
- Every response carries `x-request-id` (header) and, on the agent-facing
  endpoints, `request_id` (body) — use it in bug reports and trace logs.
- Every agent-facing response body carries `protocol_version: 1`.
- Auth: local mode needs no auth (loopback only); remote mode requires
  `Authorization: Bearer <token>` where the token lives in the environment
  variable named by `server.auth_token_env`.
- Errors: non-2xx bodies are `{"error": {"code", "message"}}` with stable
  `code`s (`invalid_request`, `unauthorized`, `nothing_published`,
  `job_queue_full`, `rate_limited`, `build_already_running`, `replan_required`,
  `llm_error`, `storage_error`, `internal_error`, …).
- Pagination: list endpoints take `?limit` (1–100) and `?cursor` (the
  previous page's `next_cursor`); responses carry `next_cursor` (null on the
  last page) and `truncated`.
- Workspace isolation: one server process serves exactly ONE workspace
  (`GET /v1/status` → `workspace` identifies it). Isolation across workspaces
  is deployment-level: run one server per workspace.

## Endpoints (v1)

### GET /v1/status

```json
{
  "protocol_version": 1,
  "workspace": "example",
  "sources": 42,
  "latest_build": {"build_id": "bld_…", "status": "COMPLETED", "started_at": "…"},
  "active_build_id": "bld_…" | null,
  "jobs": {"QUEUED": 0, "RUNNING": 0, "COMPLETED": 3, "...": 0}
}
```

### POST /v1/build

Request: `{"source_id": "default"}` (only the configured root; header
`Idempotency-Key` recommended). Response `202`:
`{"job_id": "job_…", "status": "queued"}`. Same key → the SAME job with
`{"replayed": true}`. One build at a time; queue cap applies.

### GET /v1/jobs, GET /v1/jobs/{id}, POST /v1/jobs/{id}/cancel

Job records mirror the §31 state machine (`QUEUED → RUNNING → COMPLETED |
FAILED | CANCELLED | INTERRUPTED | REPLAN_REQUIRED`) with the live `phase`
(SCANNING…INDEXING), `failure_code`, `retryable`.

### POST /v1/search

Request: `{"query": "…", "limit": 10}`. Response: page/section hits — never
answers.

```json
{
  "protocol_version": 1,
  "request_id": "req_…",
  "generation": "bld_…",
  "hits": [{"page_id", "slug", "title", "heading_path", "snippet", "rank", "citation_count"}],
  "truncated": false
}
```

### POST /v1/context

The Agent-integration surface: the budgeted, diversity-aware retrieval
bundle. Request: `{"query", "hybrid"?, "budget"?: {max_chunks, max_tokens,
max_pages, max_per_source, graph_limit}}`. Response adds `chunks` (each with
`slug`, `title`, `heading_path`, `snippet`, `score`, `sources`), `neighbors`
(graph), `estimated_tokens`, `dropped`, `truncated` — plus
`protocol_version`, `request_id`, `generation`.

### POST /v1/query

Grounded, citation-verified answers (the `llm-wiki ask` service). Request:
`{"query", "write_back"?, "hybrid"?, "embedding_model"?}`. Response: `answer`
(with expanded `<!-- llm-wiki:cite -->` anchors), `citations` (claim id,
source id, section id, range, digests, heading path), `sources`,
`llm_request_count`, `insight_id` (set when `write_back` persisted the
insight), `protocol_version`, `request_id`, `generation`.

### GET /v1/insights

The curated insight layer — verified, query-derived syntheses written back by
`/v1/query` with `write_back: true`. Newest first, cursor-paginated.

```json
{
  "protocol_version": 1,
  "request_id": "req_…",
  "insights": [{"insight_id", "build_id", "query", "answer", "citations", "created_at"}],
  "next_cursor": null,
  "truncated": false
}
```

Consumers: agents that asked before (dedupe / follow-ups), and the wiki's own
quality loop — `llm-wiki lint` verifies every insight against the current
generation (`stale-insight`), and `llm-wiki lint --semantic` re-judges them
(`superseded-insight` / `contradicted-insight`).

### POST /v1/embed

Incremental embedding backfill for the ACTIVE generation (service
counterpart of `llm-wiki embed`). Synchronous; bounded by the LLM semaphore;
zero requests when fully covered. Request: `{"model"?, "batch"?}`. Response:
`{"protocol_version", "request_id", "generation", "model", "total_sections",
"covered_before", "embedded"}`. Requires an embedding model (`model` or
`LLM_WIKI_EMBEDDING_MODEL`).

### GET /v1/pages, GET /v1/pages/{id}

Page metadata / full page content with citations and links of the ACTIVE
generation.
