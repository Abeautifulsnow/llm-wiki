# llm-wiki

> LLM-Wiki is a knowledge compiler that turns evolving documentation into a persistent, grounded and incrementally maintained Wiki for humans and AI agents.

```text
Documents are not merely indexed.
They are compiled into maintainable knowledge.
```

Not another RAG framework: sources stay untouched under `docs/`, and the wiki
is a derived, fully cited artifact — every claim carries a source file,
heading path, byte range and content digest that the compiler (never the
model) writes down.

## How it works

```text
Raw Sources → Parse/Normalize → Knowledge Extraction (LLM, structured + verified)
→ Stable Knowledge Registry → Hierarchical Planning → Page Compilation
→ Atomic Publish (generations/ + current.json) → FTS Search / Graph / lint
```

Key properties (see `docs/llm-wiki-prd.md` for the full spec):

- **Grounded** — claims are verified against source ranges; unverifiable
  output lands in an audited `rejected_claims` table, never in your wiki.
- **Incremental** — rebuilds touch only the pages the diff proves affected;
  anything that cannot be proven safe stops as `REPLAN_REQUIRED` instead of
  silently mixing old and new plans.
- **Reproducible** — identical inputs make zero new LLM requests (validated
  responses are cached per fingerprint).
- **Atomic** — a failed build never damages the currently visible wiki.
- **Chinese-aware search** — FTS5 with a shared CJK unigram/bigram +
  Latin-word analyzer; `search.graph = true` adds one-hop related pages
  from the built-in Wiki Graph.
- **Agent-ready** — a frozen HTTP protocol (v1) with a dependency-free
  TypeScript SDK exposes search, budgeted context retrieval and grounded
  Q&A to agents.

## Quick start

Requires the Rust toolchain and an OpenAI-compatible LLM endpoint
(vLLM / Ollama / OpenRouter / the official API all work).

```bash
cargo build --release

export LLM_WIKI_API_KEY=sk-…        # name configurable via config.api_key_env

llm-wiki init                       # writes .llm-wiki/config.toml (source root ./docs)
llm-wiki scan ./docs                # scan + manifest diff, no LLM calls
llm-wiki build                      # full compile (first build spends real tokens)
llm-wiki search "单点登录"           # CJK full-text search over the compiled wiki
llm-wiki ask "认证流程是怎样的？"     # grounded answer with verified citations
llm-wiki lint                       # broken links, stale citations, hand edits, …
llm-wiki status                     # sources, builds, active generation
llm-wiki doctor                     # config / db / publish / FTS5 availability
llm-wiki doctor --live              # …plus a real request per configured endpoint
```

Output lives under `wiki/generations/{build_id}/*.md` with a `current.json`
pointer — plain Markdown any LLM or agent can consume directly. All working
state (config + SQLite `state.db`) lives under `.llm-wiki/`.

### Changing model, prompt or config

The build fingerprint covers model + prompt + schema + parser + config
versions. A drift stops the incremental build with `REPLAN_REQUIRED`
(exit code 7) instead of rewriting pages behind your back — and so does a
structural change to an existing page's outbound links (the §19 topology
guard: a related-page graph that shifted cannot be extended incrementally
without risking mixed generations). Review what changed, then opt in
explicitly:

```bash
llm-wiki replan --dry-run           # audit the plan diff + cost estimate, no writes
llm-wiki replan                     # execute: stable page IDs survive merges/splits
```

### What to expect on cost

- The **first build** compiles everything: real tokens, real time.
- **Rebuilds** only re-analyze changed sources and recompile affected pages;
  unchanged pages are carried over verbatim.
- **Second build with no changes** makes zero LLM requests (cache).
- `llm-wiki embed` is incremental too: a fully covered generation issues
  zero embedding requests.

Model quality matters: extraction demands strict JSON and precise evidence
ranges. Small local models may trip the rejected-claim safety gate — a
mid-tier or larger model is recommended until measured otherwise. Thinking
models (deepseek-style) burn reasoning tokens from the request output budget:
raise `[llm] max_output_tokens` (e.g. 32768) or analysis fails with
`NO_JSON` before any visible output.

## CLI reference

| Command | Purpose |
|---|---|
| `init` | Write `.llm-wiki/config.toml` with safe, commented defaults (refuses to overwrite). |
| `scan [root]` | Scan the source tree, build the manifest, persist the Source Registry. No LLM calls. |
| `build [root]` | Full pipeline: scan → analyze → plan → compile → publish. Incremental by default. |
| `replan [--dry-run]` | Explicit global re-plan with a stable-ID plan diff. `--dry-run` audits without compiling/publishing; the bare command IS the confirmation. |
| `search <query>` | Full-text search over the published wiki (page sections, never answers); one-hop related pages when `search.graph = true`. |
| `ask <query> [--write-back] [--hybrid] [--embedding-model <m>]` | Retrieval-grounded answer with citation-verified claims. `--write-back` persists the verified insight; `--hybrid` adds vector candidates (needs embedding coverage). |
| `embed [--model <m>] [--batch <n>]` | Incremental embedding backfill of the active generation (vector layer for `--hybrid`). |
| `lint [--semantic]` | Structural lint (citations, links, orphans, hand edits) — Error findings exit 11. `--semantic` adds the LLM-judged review (advisory only, costs model calls). |
| `serve [--host <h>] [--port <p>]` | HTTP API (below). Local loopback-only unless `server.remote_enabled`. |
| `status` | Source counts and the latest build. |
| `doctor [--live]` | Config validation, source root, state db, publish integrity, FTS5 availability, API-key env. Every FAIL prints the command that fixes it. `--live` additionally makes one real request per configured endpoint (chat, embedding, rerank) — the only way to tell a configured endpoint from a live one. Reports, never repairs. |

## Configuration

Config lives at `.llm-wiki/config.toml`. Priority is fixed:
**CLI args > environment (secrets/deployment only) > config file > defaults**.
Missing fields fall back to safe defaults; the file below shows every key
with its default:

```toml
[project]
name = "llm-wiki"
wiki_dir = "./wiki"        # managed generations + current.json; must live OUTSIDE source root

[source]
root = "./docs"
include = ["**/*.md", "**/*.markdown", "**/*.mdx"]
exclude = ["wiki/**", ".llm-wiki/**", ".git/**", "node_modules/**"]

[llm]
provider = "openai-compatible"          # only provider family today
base_url = "http://localhost:8000/v1"
model = ""                              # REQUIRED before build
api_key_env = "LLM_WIKI_API_KEY"        # name of the env var — never the key itself
max_output_tokens = 4096                # raise for thinking models (32768+)
thinking = ""                           # "" = auto (send nothing) | "on" | "off";
                                        # a rejected param downgrades to auto with a warning
thinking_effort = ""                    # "" | "low" | "medium" | "high" (OpenAI-style hint)
max_concurrency = 4
timeout_seconds = 120

# Chat, embedding and rerank may live at THREE different providers: every
# empty/0 field in [embedding] / [rerank] inherits the [llm] value, so fill
# in only what differs (base_url, api_key_env, timeout_seconds).

[embedding]
# provider = "openai-compatible"
# base_url = "https://embeddings.example.com/v1"
# api_key_env = "LLM_WIKI_EMBEDDING_API_KEY"
# model = ""        # or --embedding-model / $LLM_WIKI_EMBEDDING_MODEL
# timeout_seconds = 0

# Used when [search] rerank = "cohere-compatible" (Cohere/Jina-style
# POST {base_url}/rerank — vLLM, Jina, SiliconFlow, Voyage, …).
[rerank]
# base_url = "https://rerank.example.com/v1"
# api_key_env = "LLM_WIKI_RERANK_API_KEY"
# model = ""        # REQUIRED when the rerank strategy is enabled
# timeout_seconds = 0

[analysis]
max_input_tokens = 32000
section_target_tokens = 6000
max_plan_input_tokens = 32000
max_rejected_claim_ratio = 0.10     # build fails above this unverifiable-output ratio

[planning]
hierarchical = true                 # required for large trees; disabling fails loudly
max_cluster_nodes = 24              # subdivide clusters beyond this (never truncate)

[search]
full_text = true
vector = false                      # reserved for the vector layer; hybrid retrieval
                                    # itself is opt-in per query (--hybrid / "hybrid": true)
graph = true                        # one-hop related pages in search output
rerank = "none"                     # "none" | "cohere-compatible"

[build]
incremental = true
keep_generations = 3                # retained past generations; current always kept

[server]
bind = "127.0.0.1"
remote_enabled = false
auth_token_env = "LLM_WIKI_SERVER_TOKEN"
max_queued_jobs = 8                 # queued-build cap; running jobs don't count
max_body_bytes = 1048576            # 1 MiB request cap
rate_limit_per_minute = 120         # per-caller, remote mode only; 0 disables
resume_interrupted_jobs = false     # re-run build jobs interrupted by a restart
                                    # (spends LLM budget on restart — enable deliberately)
```

Validation rules enforced before any build: `source.root` and `wiki_dir`
must not overlap (checked lexically, case-folded); `llm.model` must be set;
`search.rerank = "cohere-compatible"` requires `[rerank] model`; numeric
bounds (`max_concurrency ≥ 1`, `timeout_seconds > 0`, `keep_generations ≥ 1`,
`max_cluster_nodes ≥ 2`, `max_rejected_claim_ratio` within `[0, 1]`).

### Secrets

API keys always come from the environment variable *named* in config —
never from the config file itself. `doctor` reports whether the named env
var is set; the effective-config summary redacts all key material.

### Environment variables

| Variable | Purpose |
|---|---|
| `LLM_WIKI_API_KEY` (or whatever `api_key_env` names) | Chat LLM API key. |
| `LLM_WIKI_EMBEDDING_API_KEY` (or whatever `[embedding] api_key_env` names) | Embedding endpoint key. |
| `LLM_WIKI_RERANK_API_KEY` (or whatever `[rerank] api_key_env` names) | Rerank endpoint key. |
| `LLM_WIKI_EMBEDDING_MODEL` | Embedding model override; resolution order is flag/req body → this env → `[embedding] model`. |
| `LLM_WIKI_SERVER_TOKEN` (or whatever `auth_token_env` names) | Bearer token for remote-mode `serve`. |
| `RUST_LOG` | Tracing filter, defaults to `warn` (e.g. `RUST_LOG=info`). |

### Separate providers for chat / embedding / rerank

The three model families don't have to share a provider. `[embedding]` and
`[rerank]` carry their own `base_url` / `api_key_env` / `model` /
`timeout_seconds`; every field left empty inherits the `[llm]` value, so
fill in only what differs:

```toml
[llm]
base_url = "https://chat.example.com/v1"
model = "gpt-x"
api_key_env = "LLM_WIKI_API_KEY"

[embedding]                          # hybrid retrieval / `llm-wiki embed`
base_url = "https://embeddings.example.com/v1"
api_key_env = "LLM_WIKI_EMBEDDING_API_KEY"
model = "bge-m3"                     # or --embedding-model / $LLM_WIKI_EMBEDDING_MODEL

[search]
rerank = "cohere-compatible"         # Cohere/Jina-style POST {base_url}/rerank

[rerank]                             # vLLM / Jina / SiliconFlow / Voyage / Cohere
base_url = "https://rerank.example.com/v1"
api_key_env = "LLM_WIKI_RERANK_API_KEY"
model = "bge-reranker-v2-m3"
```

## HTTP service

`llm-wiki serve` exposes the workspace over HTTP (default
`127.0.0.1:8080`; `--host`/`--port` override `[server] bind`). One server
process serves exactly ONE workspace — isolation across workspaces is
deployment-level.

Two security modes, chosen by `server.remote_enabled`:

- **Local (default)** — everything under `/v1` accepts loopback peers only;
  a non-loopback bind is refused at startup.
- **Remote** — `remote_enabled = true` requires a non-empty token in the env
  var named by `auth_token_env`, otherwise the server refuses to start
  (fail closed). Requests need `Authorization: Bearer <token>`; the
  per-caller rate limit (`rate_limit_per_minute`) applies. Remote-mode
  guardrails: `max_queued_jobs`, `max_body_bytes`.

`GET /health` is unauthenticated. The full agent-facing contract is frozen
at **protocol version 1** and documented in
[`docs/agent-protocol.md`](docs/agent-protocol.md); all agent responses
carry `protocol_version`, `request_id` and `x-request-id`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/status` | GET | Sources, latest build, active generation, job counts. |
| `/v1/build` | POST | Queue a build job (202). Honors `Idempotency-Key` (same key → same job, `replayed: true`); one build at a time; queue cap applies. |
| `/v1/jobs` · `/v1/jobs/{id}` | GET | List (cursor-paginated, `?status=` filter) / inspect jobs. |
| `/v1/jobs/{id}/cancel` | POST | Cooperative cancel: QUEUED cancels immediately, RUNNING stops at the next pipeline checkpoint, terminal → 409. |
| `/v1/search` | POST | Page/section hits — never answers. Applies the configured rerank strategy. |
| `/v1/context` | POST | **The agent-integration surface**: budgeted, diversity-aware retrieval bundle (`budget: {max_chunks, max_tokens, max_pages, max_per_source, graph_limit}`), with citations, graph neighbors and truncation transparency. `hybrid: true` adds vector candidates. |
| `/v1/query` | POST | Grounded, citation-verified answers (the `ask` service). `write_back: true` persists the insight. |
| `/v1/pages` · `/v1/pages/{id}` | GET | Active generation's page metadata (cursor-paginated) / one full page with citations and links (by id or slug). |
| `/v1/insights` | GET | The curated insight layer written back by `/v1/query` with `write_back: true` — newest first, cursor-paginated. `lint` re-verifies these against the current generation. |
| `/v1/embed` | POST | Incremental embedding backfill of the active generation (synchronous; zero requests when fully covered). |

Errors are `{"error": {"code", "message"}}` with stable codes
(`invalid_request`, `unauthorized`, `nothing_published`, `job_queue_full`,
`rate_limited`, `build_already_running`, `replan_required`, `llm_error`,
`storage_error`, `internal_error`, …). Jobs follow the state machine
`QUEUED → RUNNING → COMPLETED | FAILED | CANCELLED | INTERRUPTED |
REPLAN_REQUIRED` with a live `phase` and `failure_code`. Build jobs
interrupted by a server restart are marked `INTERRUPTED`; with
`server.resume_interrupted_jobs = true` they are re-enqueued on startup
(off by default — resuming spends LLM budget).

### TypeScript SDK

[`sdk/typescript`](sdk/typescript) is a dependency-free fetch client
(Node 18+) implementing exactly protocol v1 — it rejects a mismatching
protocol major version instead of guessing:

```ts
import { LlmWikiClient } from "@llm-wiki/sdk";

const wiki = new LlmWikiClient({ baseUrl: "http://127.0.0.1:8080" });

const hits = await wiki.search("plugin permission", 5);       // retrieval only
const bundle = await wiki.context("plugin permission", { maxChunks: 8 });
const answer = await wiki.ask("How does auth work?", { writeBack: true });
const prior = await wiki.insights(10);                        // curated layer
```

## Exit codes

CLI failures exit with a stable code per category (PRD §34) — scripts and
CI can branch on them:

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Config error / invalid id |
| 3 | Source error / path collision |
| 4 | Parse error |
| 5 | Storage error |
| 6 | LLM / schema / evidence validation error |
| 7 | Planning error / `REPLAN_REQUIRED` |
| 8 | Compilation / index error |
| 9 | Budget exceeded |
| 10 | Publish recovery needed |
| 11 | Lint found errors |
| 12 | Cancelled |

## Development

```bash
cargo test --workspace              # 250+ tests, FakeLlmProvider only
cargo clippy --workspace --all-targets -- -D warnings
python3 evals/check_fixtures.py     # eval corpus integrity (read-only)
cd sdk/typescript && npm install && npm run typecheck
```

Real-LLM E2E is `#[ignore]`-gated and never runs in CI:

```bash
LLM_WIKI_E2E_API_KEY=… LLM_WIKI_E2E_BASE_URL=… \
  cargo test -p llm-wiki-compiler --test golden -- --ignored
```

Eval gates (source coverage, citation correctness, hallucination rate,
cross-document synthesis, rebuild determinism) live in
[`evals/`](evals/README.md) and run in CI against `FakeLlmProvider`.

Workspace layout: `crates/` holds ten Rust crates (`core`, `source`,
`markdown`, `llm`, `storage`, `compiler`, `evals`, `search`, `cli`,
`server`); `prompts/` holds the versioned prompt templates that participate
in the build fingerprint.

Status: incremental build + explicit replan + CJK full-text search +
wiki graph (V0.2), vector retrieval (V0.3), HTTP service (V0.4) and hybrid
retrieval with rerank + the agent protocol/SDK (V0.5) are implemented.
