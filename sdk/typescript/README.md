# @llm-wiki/sdk (TypeScript)

The TypeScript KnowledgeProvider SDK for `llm-wiki-server` — a
dependency-free fetch client over the frozen agent protocol v1
(`docs/agent-protocol.md`).

```ts
import { LlmWikiClient } from "./src/index.ts";

const wiki = new LlmWikiClient({ baseUrl: "http://127.0.0.1:8080" });

// retrieval: page/section hits (never answers)
const hits = await wiki.search("plugin permission", 5);

// the agent-integration surface: budgeted, diversity-aware context
const bundle = await wiki.context("plugin permission", { maxChunks: 8 });
for (const chunk of bundle.chunks) {
  // chunk.snippet + chunk.sources — feed these to your model
}

// grounded answers with verified citations (optionally written back)
const answer = await wiki.ask("How does the permission system work?", {
  writeBack: true,
});
console.log(answer.answer, answer.citations.length, answer.insight_id);

// the curated insight layer (what the wiki has already synthesized)
const prior = await wiki.insights(10);
```

## Embedding backfill (additive, 0014)

`POST /v1/embed` and its async counterpart `POST /v1/embed/jobs` are an
additive protocol surface, so they are exported from `@llm-wiki/sdk/embed`
rather than added to the frozen v1 client. The extended client IS a
`LlmWikiClient`, so nothing else changes:

```ts
import { LlmWikiClientWithEmbeddingJobs } from "./src/embed.ts";

const wiki = new LlmWikiClientWithEmbeddingJobs({ baseUrl: "http://127.0.0.1:8080" });

// async: returns as soon as the job is admitted — poll it, cancel it
const ack = await wiki.startEmbedJob("bge-m3", { idempotencyKey: "nightly-1" });
const job = await wiki.job(ack.job_id); // job.progress: {total, covered_before, embedded}
await wiki.cancelJob(ack.job_id);

// sync: resolves when the whole backfill finishes (the frozen v1 shape)
const report = await wiki.embed("bge-m3", { batch: 32 });
```

Use the async form for a first large backfill: the sync call stays open for
the whole run, so a client timeout loses the outcome even though the server
keeps working.

## Contract guarantees

- Responses carry `protocol_version` — the SDK rejects a mismatching major
  version instead of guessing.
- Errors throw `LlmWikiApiError` with the server's stable `code` and the
  `request_id` for support.
- One client = one server = one workspace (isolation is deployment-level:
  run one `llm-wiki serve` per workspace).

## Development

```bash
npm install
npm run typecheck      # types only
npm test               # compiles src/ to dist/ and runs the contract tests
```

Requires Node 18+ (global fetch). No runtime dependencies. The contract tests
(`test/contract.test.mjs`) pin the exact method, path, headers and body of
every embed-related call, plus the `error.code` and `protocol_version`
guarantees above — they are the client half of the server's
`crates/llm-wiki-server/tests/http.rs`.
