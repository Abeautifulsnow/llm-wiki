/**
 * Behaviour tests for the gaps found in review:
 *
 *   [#I01] a missing page must be an `isError` tool result in BOTH builds
 *   [wire] page slugs with `/` must be percent-encoded on the request path
 *   [ask]  `wiki_ask` (opt-in) renders the answer, citations and insight id
 *   [build] `wiki_build` sends an Idempotency-Key and reports a replay
 *   [degraded] a non-served corpus side is surfaced, never silent
 *   [timeout] a slow upstream becomes an actionable isError
 *
 * Run: node --test test/behaviour.test.mjs
 */

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import test from "node:test";

import { textOf, withClient } from "./fixtures.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const BUILDS = {
  "zero-dep": join(HERE, "..", "server.mjs"),
  sdk: join(HERE, "..", "server.sdk.mjs"),
};

// ---------------------------------------------------------------------------
// [#I01] missing page — both builds must agree
// ---------------------------------------------------------------------------

for (const [label, file] of Object.entries(BUILDS)) {
  test(`${label}: a missing page is an isError result, not a protocol error`, async () => {
    await withClient(file, async ({ client }) => {
      const outcome = await client
        .callTool({ name: "wiki_page", arguments: { id: "missing/page" } })
        .then((result) => ({ result }))
        .catch((error) => ({ error }));

      assert.equal(outcome.error, undefined, `unexpected protocol error: ${outcome.error?.message}`);
      assert.equal(outcome.result.isError, true);
      assert.match(textOf(outcome.result), /not_found/);
    });
  });

  test(`${label}: a slug containing "/" is percent-encoded on the wire`, async () => {
    await withClient(file, async ({ client, state }) => {
      const result = await client.callTool({ name: "wiki_page", arguments: { id: "auth/sso-flow" } });
      assert.notEqual(result.isError, true);
      assert.ok(
        state.pagePaths.includes("/v1/pages/auth%2Fsso-flow"),
        `expected an encoded path, got ${JSON.stringify(state.pagePaths)}`,
      );
      assert.match(textOf(result), /# SSO flow/);
    });
  });
}

// ---------------------------------------------------------------------------
// [ask] opt-in grounded answer
// ---------------------------------------------------------------------------

test("sdk: wiki_ask renders the answer, citations and insight id", async () => {
  await withClient(
    BUILDS.sdk,
    async ({ client }) => {
      const { tools } = await client.listTools();
      assert.ok(tools.some((tool) => tool.name === "wiki_ask"));

      const result = await client.callTool({ name: "wiki_ask", arguments: { query: "how does sso work", write_back: true } });
      const text = textOf(result);
      assert.match(text, /Grounded answer\./);
      assert.match(text, /docs\/auth\.md:10-80/);
      assert.match(text, /ins_1/);
    },
    { LLM_WIKI_MCP_ENABLE_ASK: "1" },
  );
});

// ---------------------------------------------------------------------------
// [build] opt-in build + job polling
// ---------------------------------------------------------------------------

test("sdk: wiki_build sends an Idempotency-Key and reports a replay", async () => {
  await withClient(
    BUILDS.sdk,
    async ({ client, state }) => {
      const result = await client.callTool({ name: "wiki_build", arguments: {} });
      assert.match(textOf(result), /job_1/);
      assert.match(textOf(result), /replayed/);

      assert.equal(state.buildKeys.length, 1);
      assert.match(state.buildKeys[0], /^llm-wiki-mcp:/);

      const job = await client.callTool({ name: "wiki_job", arguments: { job_id: "job_1" } });
      assert.match(textOf(job), /RUNNING/);
      assert.match(textOf(job), /COMPILING/);
      assert.match(textOf(job), /pages=3/);
    },
    { LLM_WIKI_MCP_ENABLE_BUILD: "1" },
  );
});

// ---------------------------------------------------------------------------
// [degraded] a corpus side that did not serve must be visible
// ---------------------------------------------------------------------------

test("sdk: degraded retrieval is surfaced with the failing side", async () => {
  await withClient(BUILDS.sdk, async ({ client }) => {
    const result = await client.callTool({ name: "wiki_context", arguments: { query: "degraded" } });
    const text = textOf(result);
    assert.match(text, /retrieval degraded/);
    assert.match(text, /wiki=not_published/);
  });
});

// ---------------------------------------------------------------------------
// [timeout] slow upstream → actionable error
// ---------------------------------------------------------------------------

test("sdk: a slow upstream becomes a timeout isError", async () => {
  await withClient(
    BUILDS.sdk,
    async ({ client }) => {
      const result = await client.callTool({ name: "wiki_context", arguments: { query: "slow" } });
      assert.equal(result.isError, true);
      assert.match(textOf(result), /timeout/);
    },
    // The server clamps timeouts to >= 1000ms; the fake delays 2500ms.
    { LLM_WIKI_MCP_TIMEOUT_MS: "1000" },
  );
});
