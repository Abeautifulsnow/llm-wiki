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
npm run typecheck
```

Requires Node 18+ (global fetch). No runtime dependencies.
