# llm-wiki MCP server

A **stdio** [Model Context Protocol](https://modelcontextprotocol.io) server that
exposes an `llm-wiki serve` workspace (agent protocol v1) to any MCP client —
Claude Desktop, Cursor, sime-agent, or your own agent runtime.

The server is a **retrieval layer**, not an answer engine: it hands the model
grounded chunks with their provenance (page path, heading path, source file and
byte range) and lets the model do the reasoning. `wiki_ask` — the one tool that
does generate an answer server-side — is opt-in.

- One MCP server per llm-wiki workspace (one `llm-wiki serve` serves exactly one).
- Read-only by default. A build spends LLM tokens; `wiki_build` is opt-in.

## Two builds, one behaviour

| Build | Entry point | Dependencies | Use when |
|---|---|---|---|
| **SDK** (recommended) | `server.sdk.mjs` | `@modelcontextprotocol/sdk` + `zod` | You can `npm install`; wire behaviour is the reference implementation's |
| **Zero-dep** | `server.mjs` | none (Node built-ins) | You cannot install packages (air-gapped, single-file deployment) |

Both share `lib/` (HTTP client, rendering, operation layer), so they return
**byte-identical tool results**. That is enforced, not assumed:
`test/sdk-cross.test.mjs` drives *both* builds with the **official SDK client**
and compares every output string.

The only intentional behavioural difference: bad arguments. The SDK build turns
a validation failure into an `isError` tool result (the model can read it and
self-correct); the zero-dep build raises JSON-RPC `-32602`. Both are legitimate.

Everything else is shared, including error classification:

- a missing page is a **tool** error (`isError`) in both builds — the model can
  retry with another slug;
- `-32002` is reserved for `resources/read` on a URI that does not resolve.

`test/behaviour.test.mjs` pins both rules for both builds.

---

## Requirements

1. `llm-wiki` built and published:

   ```bash
   cargo build --release                 # → target/release/llm-wiki
   cd /srv/knowledge/<workspace>
   llm-wiki init && llm-wiki scan ./docs && llm-wiki build
   llm-wiki serve --port 18086           # keep it running
   ```

2. Node.js ≥ 18 where the MCP client runs.

3. For the SDK build only:

   ```bash
   cd mcp/llm-wiki-mcp
   npm install          # @modelcontextprotocol/sdk + zod (>= 4.6.6: GHSA-mrhx-c7xc-m3w5)
   ```

---

## Run

```bash
# SDK build (recommended)
LLM_WIKI_BASE_URL=http://127.0.0.1:18086 node server.sdk.mjs

# zero-dependency build
LLM_WIKI_BASE_URL=http://127.0.0.1:18086 node server.mjs
```

The process speaks JSON-RPC 2.0 on stdin/stdout, one JSON object per line, and
writes nothing else to stdout. Diagnostics go to stderr and are silent unless
`LLM_WIKI_MCP_DEBUG=1`.

`npm start` runs the SDK build; `npm start:zero-dep` runs the other one. After
`npm install`, `bin.llm-wiki-mcp` points at `server.sdk.mjs`, so `npx .` works too.

---

## Tools

| Tool | Endpoint | Purpose |
|---|---|---|
| `wiki_status` | `GET /v1/status` | Sources, active generation, latest build, version staleness |
| `wiki_context` | `POST /v1/context` | **Primary retrieval**: budgeted chunks with provenance |
| `wiki_search` | `POST /v1/search` | Ranked page/section hits (locate, then read) |
| `wiki_page` | `GET /v1/pages/{id}` | One full page: body, citations, links |
| `wiki_ask` | `POST /v1/query` | Citation-verified answer (opt-in, spends LLM tokens) |
| `wiki_build` | `POST /v1/build` | Queue an incremental build (opt-in) |
| `wiki_job` | `GET /v1/jobs/{id}` | Job status/phase/progress (opt-in with build) |

### `wiki_context` arguments

| Argument | Type | Default | Notes |
|---|---|---|---|
| `query` | string | — | Required. Rewrite first: keep identifiers verbatim, drop chit-chat, split multi-intent questions |
| `source_mode` | `wiki` \| `source` \| `fusion` | `LLM_WIKI_SOURCE_MODE` (`fusion`) | `wiki` = compiled pages, `source` = raw document chunks, `fusion` = both |
| `max_chunks` | 1–50 | `LLM_WIKI_MCP_MAX_CHUNKS` (8) | Per-side limit in fused mode |
| `max_tokens` | 500–50000 | `LLM_WIKI_MCP_MAX_TOKENS` (8000) | Bundle token budget |
| `max_pages` | 1–20 | 4 | Distinct pages |
| `max_per_source` | 1–20 | 2 | Chunks per source file |
| `graph_limit` | 0–50 | 4 | Related-page neighbours |
| `hybrid` | boolean | false | Vector candidates (requires `llm-wiki embed` coverage) |

Output is Markdown: per-chunk title, `location` (page path or `file:start-end`),
`evidence` kind and backing sources, then the snippet, then related pages. The
closing line tells the model to cite `location` and to admit when the chunks do
not answer the question.

### Degradation is visible

`/v1/context` reports per-side status in `degraded` (`served`,
`not_published`, `index_not_built`, `no_matches`, `disabled`). Both builds
surface a non-`served` side as a `> ⚠ retrieval degraded …` line so the model
never mistakes "that corpus side was not searched" for "the wiki has nothing".

---

## Resources

Published pages are exposed as MCP resources, so a client can browse the wiki
without spending tool calls:

- `resources/list` → `wiki://page/<url-encoded slug>`, walking `next_cursor`
  until the wiki is exhausted (capped at 2000 pages, with a stderr warning past
  that; an unpublished workspace yields an empty list, not an error)
- `resources/read` → the page body as `text/markdown`

The SDK build registers them through `ResourceTemplate`; the zero-dep build
implements `resources/list` / `resources/read` directly. Both resolve to the
same URIs and the same bodies.

---

## Configuration

All optional except the base URL. Identical for both builds.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_WIKI_BASE_URL` | `http://127.0.0.1:8080` | Address of `llm-wiki serve` |
| `LLM_WIKI_TOKEN` | — | Bearer token; required when `server.remote_enabled = true` |
| `LLM_WIKI_SOURCE_MODE` | `fusion` | Default retrieval corpus |
| `LLM_WIKI_MCP_MAX_CHUNKS` | `8` | Default `max_chunks` |
| `LLM_WIKI_MCP_MAX_TOKENS` | `8000` | Default `max_tokens` |
| `LLM_WIKI_MCP_RENDER_CHARS` | `16000` | Hard cap on one tool-result text |
| `LLM_WIKI_MCP_TIMEOUT_MS` | `20000` | Per-request HTTP timeout |
| `LLM_WIKI_MCP_ENABLE_ASK` | off | `1` exposes `wiki_ask` |
| `LLM_WIKI_MCP_ENABLE_BUILD` | off | `1` exposes `wiki_build` + `wiki_job` |
| `LLM_WIKI_MCP_DEBUG` | off | `1` logs to stderr |

---

## Client wiring

### Claude Desktop / Cursor (`mcpServers`)

```json
{
  "mcpServers": {
    "llm-wiki": {
      "command": "node",
      "args": ["/abs/path/to/llm-wiki/mcp/llm-wiki-mcp/server.sdk.mjs"],
      "env": {
        "LLM_WIKI_BASE_URL": "http://127.0.0.1:18086",
        "LLM_WIKI_SOURCE_MODE": "fusion"
      }
    }
  }
}
```

Use `server.mjs` instead if that machine has no `node_modules`.

### sime-agent (`McpServerConfig`, stdio)

`sime-agent` accepts `stdio` / `sse` / `http` transports; stdio needs no extra
port and inherits the desktop app's process environment:

```json
{
  "name": "llm-wiki",
  "enabled": true,
  "transport": {
    "type": "stdio",
    "command": "node",
    "args": ["E:\\work\\git\\company_proj\\llm-wiki\\mcp\\llm-wiki-mcp\\server.sdk.mjs"],
    "env": {
      "LLM_WIKI_BASE_URL": "http://127.0.0.1:18086",
      "LLM_WIKI_SOURCE_MODE": "fusion",
      "LLM_WIKI_MCP_MAX_TOKENS": "8000"
    }
  }
}
```

Prefer stdio: it keeps the token off the network and gives each client session
its own server process. If your deployment cannot spawn processes, front 18086
with your gateway and use the `http` transport instead — but note this MCP
server itself is stdio-only; an HTTP MCP endpoint would need a separate bridge.

### Multiple workspaces

One `llm-wiki serve` = one workspace. Register one MCP server entry per
workspace (`llm-wiki-auth`, `llm-wiki-billing`, …), each pointing at its own
`LLM_WIKI_BASE_URL`, and let the client (or the agent) pick. Do not multiplex
workspaces through a single entry: the server has no workspace dimension.

---

## Failure semantics

| Situation | What the client sees |
|---|---|
| `llm-wiki serve` not running / wrong port | `isError: true` with `cannot reach http://… — is \`llm-wiki serve\` running?` |
| Workspace never built | `wiki_context` explains that nothing is published and points at `llm-wiki build`; `resources/list` returns `[]` |
| Upstream 4xx/5xx | `isError: true` with `[code] (HTTP nnn): message · request_id …` — the code is the stable one from the protocol (`nothing_published`, `rate_limited`, `storage_error`, …) |
| Older server without `source_mode` | `wiki_context`/`wiki_search` retry once in `wiki` mode automatically |
| Generation predates the binary | `wiki_status` prints `⚠ generation_stale = true` and says to rebuild |
| Bad tool arguments | SDK build: `isError` result; zero-dep build: JSON-RPC `-32602` |
| `wiki_page` on an unknown slug | `isError: true` in **both** builds — the model can retry with another slug |
| `resources/read` on an unknown URI | JSON-RPC `-32002` (resource not found) |
| Malformed JSON on stdin | `-32700` and the session keeps running (zero-dep build) |

---

## Test

```bash
cd mcp/llm-wiki-mcp
npm install          # once, for the SDK build + client
npm test             # 38 cases, no network, no LLM
```

Three suites:

- `test/handshake.test.mjs` (17) — hand-rolled client against the zero-dep
  build's protocol surface: version negotiation, tool listing, budgets, legacy
  fallback, resources, upstream 500, unreachable server, malformed JSON.
- `test/sdk-cross.test.mjs` (13) — **the official SDK `Client`** connects to
  *both* builds over stdio and asserts the outputs are byte-identical, including
  tool descriptions and resource URIs.
- `test/behaviour.test.mjs` (8) — review regressions: a missing page is an
  `isError` in both builds, `/` in a slug is percent-encoded on the wire,
  `wiki_ask`, `wiki_build` + `wiki_job`, degraded retrieval, request timeout.

Run only the zero-dep suite with nothing installed:

```bash
node --test test/handshake.test.mjs
```

---

## Why stdio (and why `ask` / `build` are off by default)

- **stdio** matches how MCP clients already launch local servers, requires no
  extra port, no auth surface, and no gateway. `llm-wiki serve` stays on loopback.
- **`wiki_ask` off**: the host agent is already an LLM. Calling `/v1/query` makes
  a second model compose an answer, spends tokens, and cannot stream.
  `wiki_context` gives the same evidence without the extra hop. Enable it when
  you specifically want the wiki's audited, citation-checked synthesis (and
  `write_back` for the insight layer).
- **`wiki_build` off**: a build is the only operation that spends LLM budget,
  and it should be triggered deliberately (CI, cron, or a person), not by an
  agent mid-conversation.

---

## Troubleshooting

**Tools return `cannot reach …`** — `llm-wiki serve` is not running, or
`LLM_WIKI_BASE_URL` points at the wrong port. `curl $LLM_WIKI_BASE_URL/health`
should answer `{"status":"ok"}`.

**`401 unauthorized`** — the server runs with `server.remote_enabled = true`;
set `LLM_WIKI_TOKEN` to the value of the env var named by `auth_token_env`.

**`ERR_MODULE_NOT_FOUND: @modelcontextprotocol/sdk`** — you are running
`server.sdk.mjs` without `npm install`, or the client spawns the server from a
different working directory. Either run `npm install` in `mcp/llm-wiki-mcp/`, or
switch the client config to `server.mjs` (zero-dependency).

**Retrieval is empty while `wiki_status` looks healthy** — llm-wiki's default
retrieval is FTS5 (CJK unigram/bigram + Latin words) plus a one-hop graph, not
embeddings. Rewrite the query with the exact identifiers the documents use, or
switch to `source_mode: "fusion"`, or enable the vector layer
(`llm-wiki embed` + `hybrid: true`).

**Nothing shows up in the client's tool list** — the client spawns the server
with its own environment; env vars set in your terminal are not inherited.
Pass them in the client config's `env` block.

**Corrupted stdout / client parse errors** — never add `console.log` to either
entry point. stdout carries protocol frames only; use `log()` (stderr) or
`LLM_WIKI_MCP_DEBUG=1`.
