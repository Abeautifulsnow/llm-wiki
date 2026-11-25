#!/usr/bin/env node
/**
 * llm-wiki MCP server — official SDK build (recommended).
 *
 * Uses `@modelcontextprotocol/sdk` for framing, capability negotiation,
 * validation and error mapping, so the wire behaviour is the reference
 * implementation's rather than ours. The HTTP calls and rendering are shared
 * with `server.mjs` (the zero-dependency build) via `lib/`, which means both
 * transports return byte-identical tool results — `test/sdk-cross.test.mjs`
 * asserts that with the SDK's own Client.
 *
 * Run:     node server.sdk.mjs
 * Wiring:  same environment variables as server.mjs (see README.md).
 */

import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { McpError } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

import { SERVER_INFO, SOURCE_MODES, config, log } from "./lib/config.mjs";
import { ApiError, apiErrorText } from "./lib/api.mjs";
import { clampText } from "./lib/render.mjs";
import { INSTRUCTIONS, READ_ONLY_ANNOTATIONS, TOOL_SPECS } from "./lib/tools.mjs";
import { PAGE_URI_PREFIX, ResourceNotFoundError, createOps } from "./lib/ops.mjs";

const ops = createOps();

/** Every tool returns Markdown; the SDK wraps it into a text content block. */
function textResult(text) {
  return { content: [{ type: "text", text: clampText(String(text), config.renderChars) }] };
}

/** API failures are tool results (the model must see them), never crashes. */
function errorResult(error) {
  if (error instanceof ApiError) {
    return { content: [{ type: "text", text: apiErrorText(error) }], isError: true };
  }
  throw error;
}

const sourceModeSchema = z
  .enum(SOURCE_MODES)
  .optional()
  .describe(
    "Retrieval corpus: 'wiki' (compiled pages only), 'source' (raw document chunks only), " +
      "'fusion' (both, merged). Defaults to the server setting.",
  );

const server = new McpServer(SERVER_INFO, {
  // Declared explicitly (rather than left at the SDK default of `true`):
  // neither build emits `tools/list_changed`, and the zero-dep build has
  // always advertised `false`. Both transports must look identical on the wire.
  capabilities: {
    tools: { listChanged: false },
    resources: { subscribe: false, listChanged: false },
  },
  instructions: INSTRUCTIONS,
});

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

server.registerTool(
  "wiki_status",
  {
    title: TOOL_SPECS.wiki_status.title,
    description: TOOL_SPECS.wiki_status.description,
    inputSchema: {},
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
  },
  async () => {
    try {
      return textResult(await ops.status());
    } catch (error) {
      return errorResult(error);
    }
  },
);

server.registerTool(
  "wiki_context",
  {
    title: TOOL_SPECS.wiki_context.title,
    description: TOOL_SPECS.wiki_context.description,
    inputSchema: {
      query: z
        .string()
        .min(1)
        .describe("Search query: entities and identifiers kept verbatim, no pleasantries."),
      source_mode: sourceModeSchema,
      max_chunks: z.number().int().min(1).max(50).optional().describe("Maximum chunks to return."),
      max_tokens: z.number().int().min(500).max(50000).optional().describe("Token budget for the bundle."),
      max_pages: z.number().int().min(1).max(20).optional().describe("Maximum distinct pages."),
      max_per_source: z.number().int().min(1).max(20).optional().describe("Maximum chunks per source file."),
      graph_limit: z.number().int().min(0).max(50).optional().describe("Maximum related-page neighbours."),
      hybrid: z.boolean().optional().describe("Add vector candidates (requires an embedded generation)."),
    },
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
  },
  async (args) => {
    try {
      return textResult(await ops.context(args));
    } catch (error) {
      return errorResult(error);
    }
  },
);

server.registerTool(
  "wiki_search",
  {
    title: TOOL_SPECS.wiki_search.title,
    description: TOOL_SPECS.wiki_search.description,
    inputSchema: {
      query: z.string().min(1).describe("Search query."),
      limit: z.number().int().min(1).max(50).optional().describe("Maximum hits (default 10)."),
      source_mode: sourceModeSchema,
    },
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
  },
  async (args) => {
    try {
      return textResult(await ops.search(args));
    } catch (error) {
      return errorResult(error);
    }
  },
);

server.registerTool(
  "wiki_page",
  {
    title: TOOL_SPECS.wiki_page.title,
    description: TOOL_SPECS.wiki_page.description,
    inputSchema: {
      id: z.string().min(1).describe("Page slug (e.g. 'auth/sso-flow') or page_id (wp_…)."),
    },
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
  },
  async ({ id }) => {
    try {
      return textResult(await ops.page(id));
    } catch (error) {
      // A missing page is a TOOL error (`isError`), not a protocol error:
      // the model can see it and try another slug. `-32002` stays reserved for
      // `resources/read`, and the zero-dep build behaves the same way.
      return errorResult(error);
    }
  },
);

if (config.enableAsk) {
  server.registerTool(
    "wiki_ask",
    {
      title: TOOL_SPECS.wiki_ask.title,
      description: TOOL_SPECS.wiki_ask.description,
      inputSchema: {
        query: z.string().min(1).describe("The question to answer from the wiki."),
        write_back: z.boolean().optional().describe("Persist the verified answer to the insight layer (default false)."),
        hybrid: z.boolean().optional().describe("Add vector candidates (requires an embedded generation)."),
      },
      annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: true },
    },
    async (args) => {
      try {
        return textResult(await ops.ask(args));
      } catch (error) {
        return errorResult(error);
      }
    },
  );
}

if (config.enableBuild) {
  server.registerTool(
    "wiki_build",
    {
      title: TOOL_SPECS.wiki_build.title,
      description: TOOL_SPECS.wiki_build.description,
      inputSchema: {
        idempotency_key: z
          .string()
          .optional()
          .describe("Reuse a key to resolve to the same job instead of queueing another build."),
      },
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async (args) => {
      try {
        return textResult(await ops.build(args));
      } catch (error) {
        return errorResult(error);
      }
    },
  );

  server.registerTool(
    "wiki_job",
    {
      title: TOOL_SPECS.wiki_job.title,
      description: TOOL_SPECS.wiki_job.description,
      inputSchema: {
        job_id: z.string().min(1).describe("Job id returned by wiki_build (job_…)."),
      },
      annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
    },
    async ({ job_id }) => {
      try {
        return textResult(await ops.job(job_id));
      } catch (error) {
        return errorResult(error);
      }
    },
  );
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

server.registerResource(
  "wiki-page",
  new ResourceTemplate(`${PAGE_URI_PREFIX}{slug}`, {
    list: async () => ({ resources: await ops.listPageResources() }),
  }),
  {
    title: "Wiki page",
    description: "A page of the active llm-wiki generation (Markdown body with citations).",
    mimeType: "text/markdown",
  },
  async (uri) => {
    try {
      return { contents: [await ops.readPageResource(String(uri))] };
    } catch (error) {
      if (error instanceof ResourceNotFoundError) {
        throw new McpError(-32002, error.message, { uri: String(uri) });
      }
      throw error;
    }
  },
);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

try {
  await server.connect(new StdioServerTransport());
  log(
    `ready (sdk) · base=${config.baseUrl} · mode=${config.sourceMode} · ` +
      `chunks=${config.maxChunks} · tokens=${config.maxTokens} · ` +
      `ask=${config.enableAsk ? "on" : "off"} · build=${config.enableBuild ? "on" : "off"}`,
  );
} catch (error) {
  process.stderr.write(`[llm-wiki-mcp] failed to start: ${error?.stack ?? error}\n`);
  process.exit(1);
}
