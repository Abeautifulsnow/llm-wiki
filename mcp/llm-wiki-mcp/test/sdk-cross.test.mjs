/**
 * Cross-validation: drive BOTH builds with the OFFICIAL MCP SDK client.
 *
 * The hand-rolled transport in `server.mjs` is tested by a hand-rolled client
 * in `handshake.test.mjs` — which cannot prove the wire format is right (both
 * sides could share the same misunderstanding). This suite closes that gap:
 * `@modelcontextprotocol/sdk`'s Client connects to each build over stdio, and
 * the two builds must return byte-identical tool results.
 *
 * Run: node --test test/sdk-cross.test.mjs
 */

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import test from "node:test";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const BUILDS = {
  "zero-dep": join(HERE, "..", "server.mjs"),
  sdk: join(HERE, "..", "server.sdk.mjs"),
};

const EXPECTED_TOOLS = ["wiki_context", "wiki_page", "wiki_search", "wiki_status"];

// ---------------------------------------------------------------------------
// Fake llm-wiki API (only what these assertions need)
// ---------------------------------------------------------------------------

const CONTEXT = {
  protocol_version: 1,
  request_id: "req_cross",
  generation: "bld_cross",
  chunks: [
    {
      slug: "auth/sso-flow",
      title: "SSO flow",
      heading_path: ["Auth", "SSO"],
      snippet: "The client redirects to the IdP, then exchanges the code for a token.",
      score: 0.87,
      sources: ["docs/auth.md"],
      evidence_kind: "wiki",
      source_ref: null,
    },
  ],
  neighbors: [{ from_slug: "auth/sso-flow", node_id: "n1", node_type: "concept", label: "OAuth", relation: "uses" }],
  estimated_tokens: 321,
  dropped: 0,
  truncated: false,
  served_mode: "fusion",
  degraded: { wiki: "served", source: "served" },
};

const PAGES = [
  { page_id: "wp_auth", slug: "auth/sso-flow", title: "SSO flow", category: "auth", language: "en", body_hash: "h" },
];

const PAGE = {
  page_id: "wp_auth",
  slug: "auth/sso-flow",
  title: "SSO flow",
  category: "auth",
  language: "en",
  body_hash: "h",
  content: "# SSO flow\n\nThe client redirects to the IdP.",
  knowledge_refs: [],
  citations: [
    {
      claim_node_id: "cl_1",
      source_id: "src_auth",
      section_id: "sec_1",
      range: [10, 80],
      source_hash: "sh",
      evidence_digest: "0123456789abcdef",
      heading_path: ["Auth"],
    },
  ],
  links: [],
};

function startFakeWiki() {
  const server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const raw = Buffer.concat(chunks).toString("utf8");
    let body = null;
    try {
      body = raw === "" ? null : JSON.parse(raw);
    } catch {
      body = null;
    }

    const json = (status, payload) => {
      response.writeHead(status, { "content-type": "application/json", "x-request-id": "req_cross" });
      response.end(JSON.stringify(payload));
    };

    if (request.method === "GET" && url.pathname === "/v1/status") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_cross",
        workspace: "cross-ws",
        sources: 2,
        latest_build: { build_id: "bld_cross", status: "COMPLETED", started_at: "2026-01-01T00:00:00Z" },
        active_build_id: "bld_cross",
        jobs: {},
        versions: { binary: "0.2.0", generation_compiler: "0.2.0", db_schema: 14, generation_stale: false },
      });
    }
    if (request.method === "POST" && url.pathname === "/v1/context") {
      if (body?.query === "fail") {
        return json(500, { error: { code: "storage_error", message: "disk on fire" } });
      }
      return json(200, CONTEXT);
    }
    if (request.method === "POST" && url.pathname === "/v1/search") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_cross",
        generation: "bld_cross",
        source_mode: "wiki",
        served: { wiki: "served", source: "disabled" },
        hits: [
          {
            page_id: "wp_auth",
            slug: "auth/sso-flow",
            title: "SSO flow",
            heading_path: ["Auth"],
            snippet: "redirect to the IdP",
            rank: -1.2,
            citation_count: 2,
            evidence_kind: "wiki",
          },
        ],
        truncated: false,
      });
    }
    if (request.method === "GET" && url.pathname === "/v1/pages") {
      return json(200, {
        protocol_version: 1,
        request_id: "req_cross",
        generation: "bld_cross",
        pages: PAGES,
        next_cursor: null,
        truncated: false,
      });
    }
    if (request.method === "GET" && url.pathname.startsWith("/v1/pages/")) {
      const slug = decodeURIComponent(url.pathname.slice("/v1/pages/".length));
      if (slug === "auth/sso-flow") return json(200, PAGE);
      return json(404, { error: { code: "not_found", message: "no page" } });
    }
    return json(404, { error: { code: "not_found", message: `unhandled ${url.pathname}` } });
  });
  return server;
}

// ---------------------------------------------------------------------------
// Connect the official client to a build
// ---------------------------------------------------------------------------

async function withClient(buildFile, run, extraEnv = {}) {
  const fake = startFakeWiki();
  fake.listen(0, "127.0.0.1");
  await once(fake, "listening");
  const baseUrl = `http://127.0.0.1:${fake.address().port}`;

  const env = Object.fromEntries(
    Object.entries({ ...process.env, LLM_WIKI_BASE_URL: baseUrl, ...extraEnv }).filter(
      ([, value]) => value !== undefined,
    ),
  );

  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [buildFile],
    env,
    stderr: "pipe",
  });
  const client = new Client({ name: "llm-wiki-cross-test", version: "0.0.0" }, { capabilities: {} });
  await client.connect(transport);

  try {
    await run({ client, baseUrl });
  } finally {
    await client.close().catch(() => {});
    fake.close();
    await once(fake, "close").catch(() => {});
  }
}

function textOf(result) {
  return (result?.content ?? [])
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test("the official SDK client performs a handshake with the hand-rolled server", async () => {
  await withClient(BUILDS["zero-dep"], async ({ client }) => {
    const info = client.getServerVersion();
    assert.equal(info.name, "llm-wiki-mcp");
    assert.equal(client.getServerCapabilities().tools !== undefined, true);
  });
});

test("the official SDK client performs a handshake with the SDK server", async () => {
  await withClient(BUILDS.sdk, async ({ client }) => {
    const info = client.getServerVersion();
    assert.equal(info.name, "llm-wiki-mcp");
  });
});

for (const [label, file] of Object.entries(BUILDS)) {
  test(`${label}: listTools exposes the four tools with identical metadata`, async () => {
    await withClient(file, async ({ client }) => {
      const { tools } = await client.listTools();
      assert.deepEqual(tools.map((tool) => tool.name).sort(), EXPECTED_TOOLS);
      for (const tool of tools) {
        assert.ok(tool.description.length > 40, `${tool.name} needs a description`);
        assert.equal(tool.inputSchema.type, "object");
      }
    });
  });

  test(`${label}: callTool(wiki_context) returns grounded chunks`, async () => {
    await withClient(file, async ({ client }) => {
      const result = await client.callTool({ name: "wiki_context", arguments: { query: "sso" } });
      assert.notEqual(result.isError, true);
      const text = textOf(result);
      assert.match(text, /### 1\. SSO flow/);
      assert.match(text, /location: auth\/sso-flow › Auth › SSO/);
      assert.match(text, /Related \(wiki graph\)/);
    });
  });

  test(`${label}: upstream failure is an isError tool result, not a protocol error`, async () => {
    await withClient(file, async ({ client }) => {
      const result = await client.callTool({ name: "wiki_context", arguments: { query: "fail" } });
      assert.equal(result.isError, true);
      assert.match(textOf(result), /storage_error/);
    });
  });

  test(`${label}: resources/list + resources/read round-trip`, async () => {
    await withClient(file, async ({ client }) => {
      const { resources } = await client.listResources();
      assert.equal(resources.length, 1);
      assert.equal(resources[0].uri, "wiki://page/auth%2Fsso-flow");

      const read = await client.readResource({ uri: resources[0].uri });
      assert.match(read.contents[0].text, /# SSO flow/);
      assert.equal(read.contents[0].mimeType, "text/markdown");
    });
  });

  // The two builds reject bad arguments at different layers, and both are
  // legitimate: the hand-rolled server raises JSON-RPC -32602, while the SDK
  // turns validation failures into an `isError` tool result the model can
  // read and self-correct from. Assert "fails loudly", not a specific shape.
  test(`${label}: invalid arguments fail loudly`, async () => {
    await withClient(file, async ({ client }) => {
      const outcome = await client
        .callTool({ name: "wiki_context", arguments: {} })
        .then((result) => ({ result }))
        .catch((error) => ({ error }));

      if (outcome.error) {
        assert.match(`${outcome.error.message} ${outcome.error.code ?? ""}`, /query|invalid/i);
      } else {
        assert.equal(outcome.result.isError, true);
        assert.match(textOf(outcome.result), /query|invalid/i);
      }
    });
  });
}

test("both builds return byte-identical results for the same request", async () => {
  const snapshot = async (file) => {
    let captured;
    await withClient(file, async ({ client }) => {
      const contextResult = await client.callTool({ name: "wiki_context", arguments: { query: "sso" } });
      const searchResult = await client.callTool({ name: "wiki_search", arguments: { query: "sso" } });
      const pageResult = await client.callTool({ name: "wiki_page", arguments: { id: "auth/sso-flow" } });
      const statusResult = await client.callTool({ name: "wiki_status", arguments: {} });
      const resources = await client.listResources();
      const { tools } = await client.listTools();
      captured = {
        context: textOf(contextResult),
        search: textOf(searchResult),
        page: textOf(pageResult),
        status: textOf(statusResult),
        resourceUris: resources.resources.map((resource) => resource.uri),
        toolDescriptions: tools
          .map((tool) => `${tool.name}:${tool.description}`)
          .sort()
          .join("\n"),
      };
    });
    return captured;
  };

  const zeroDep = await snapshot(BUILDS["zero-dep"]);
  const sdk = await snapshot(BUILDS.sdk);

  assert.equal(sdk.context, zeroDep.context);
  assert.equal(sdk.search, zeroDep.search);
  assert.equal(sdk.page, zeroDep.page);
  assert.equal(sdk.status, zeroDep.status);
  assert.deepEqual(sdk.resourceUris, zeroDep.resourceUris);
  assert.equal(sdk.toolDescriptions, zeroDep.toolDescriptions);
});
