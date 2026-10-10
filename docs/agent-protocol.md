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

### Retrieval source modes and additive evidence fields (still protocol v1)

Retrieval draws from two corpora: the compiled **wiki** (sections of the
ACTIVE generation) and the raw **sources** (chunked original documents).
`POST /v1/search` and `POST /v1/context` accept an optional request field
`"source_mode": "source" | "wiki" | "fusion"`.

- **Absent field → the configured default.** The server default is the
  `search.source_mode` key in `.llm-wiki/config.toml`, itself defaulting to
  `"wiki"` — a server with no config file behaves exactly like the
  pre-fusion protocol. An unknown `source_mode` value is a `400
  invalid_request`.
- **Semantics.** `"wiki"` is the legacy behavior. `"source"` serves
  raw-source chunks only. `"fusion"` serves both sides merged by reciprocal
  rank fusion; a source chunk containing every query token verbatim keeps a
  guaranteed front slot (exact-match protection), and the merged order is
  never re-sorted by the server.
- **The effective mode is echoed.** `/v1/search` responses carry top-level
  `source_mode` (effective value) and `served`; `/v1/context` responses
  carry the same two under their PR3 names `served_mode` and `degraded`
  (`null` on the legacy wiki-only path, which fails loudly instead of
  degrading).
- **Every hit/chunk carries `evidence_kind: "wiki" | "source"`.**
  - *Wiki hits* keep all their v1 fields (`page_id`, `slug`, `title`,
    `heading_path`, `snippet`, `rank`, and `citation_count` on search hits)
    — one field unchanged, none missing.
  - *Source hits* express identity through
    `source_ref: {source_id, file_path, heading_path, ordinal, range_start,
    range_end}` plus `file_path`, `title`, `heading_path`, `snippet`, `rank`
    — and **never carry `page_id` or `slug`**. Consumers MUST NOT expect
    page identity on `evidence_kind: "source"` entries (on `/v1/context`
    source chunks the `slug` position holds the source-relative file path).
- **Degradation is visible, never silent.** `served` / `degraded` reports
  per corpus side one of `served | not_published | index_not_built |
  no_matches | disabled`. `"disabled"` = the requested mode did not ask for
  that side; structural reasons (`not_published`, `index_not_built`) outrank
  query reasons (`no_matches`).
- **Rerank scope.** The configured reranker is a wiki-side feature
  (source-side reranking is future work): `/v1/search` applies it only in the
  default wiki mode — `source`/`fusion` search results are returned in their
  retrieval order with no rerank pass; `/v1/context` applies it (when
  configured) to the wiki-side candidates in every mode, while source chunks
  keep their fused order.

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

Request: `{"query": "…", "limit": 10, "source_mode"?: "source" | "wiki" | "fusion"}`.
Response: page/section hits — never answers. The example shows the default
wiki mode; with `source_mode: "source" | "fusion"` the array mixes wiki hits
with source hits (`evidence_kind: "source"`, `source_ref`, `file_path`,
`title`, `heading_path`, `snippet`, `rank` — no `page_id`/`slug`).

```json
{
  "protocol_version": 1,
  "request_id": "req_…",
  "generation": "bld_…",
  "source_mode": "wiki",
  "served": {"wiki": "served", "source": "disabled"},
  "hits": [{"evidence_kind": "wiki", "page_id": "wp_…", "slug": "…", "title": "…", "heading_path": ["…"], "snippet": "…", "rank": -3.2, "citation_count": 2}],
  "truncated": false
}
```

The default wiki mode keeps answering `404 nothing_published` on a
never-built workspace; `source_mode: "source" | "fusion"` degrades visibly
instead — `200` with empty `hits` and the per-side `served` reasons.

### POST /v1/context

The Agent-integration surface: the budgeted, diversity-aware retrieval
bundle. Request: `{"query", "hybrid"?, "source_mode"?: "source" | "wiki" |
"fusion", "budget"?: {max_chunks, max_tokens, max_pages, max_per_source,
graph_limit}}` (a fused retrieval uses `max_chunks` as its per-side limit).
Response adds `chunks` (each with `slug`, `title`, `heading_path`, `snippet`,
`score`, `sources`, plus the additive `evidence_kind` / `source_ref`
evidence fields — source chunks carry no page identity), `neighbors`
(graph), `estimated_tokens`, `dropped`, `truncated`, `served_mode` (the
effective mode) and `degraded` (per-side statuses, `null` on the legacy
wiki-only path) — plus `protocol_version`, `request_id`, `generation`.

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
