#!/usr/bin/env node
/**
 * llm-wiki MCP server — hand-rolled stdio transport (zero runtime dependencies).
 *
 * This file contains ONLY the protocol adapter: JSON-RPC 2.0 framing, method
 * dispatch and parameter validation. The HTTP calls and the rendering live in
 * `lib/`, shared with `server.sdk.mjs` (the official MCP SDK build), so both
 * transports emit identical tool results.
 *
 * Use this build when you cannot install npm dependencies. Prefer
 * `server.sdk.mjs` when you can — it is maintained against the official SDK.
 *
 * Protocol: 2025-06-18 (also accepts 2025-03-26 / 2024-11-05).
 * Transport: stdin/stdout, one JSON object per line, nothing else on stdout.
 */

import { createInterface } from "node:readline";
import {
  KNOWN_PROTOCOL_VERSIONS,
  LATEST_PROTOCOL_VERSION,
  SERVER_INFO,
  SOURCE_MODES,
  config,
  log,
} from "./lib/config.mjs";
import { ApiError, apiErrorText } from "./lib/api.mjs";
import { clampText } from "./lib/render.mjs";
import { INSTRUCTIONS, READ_ONLY_ANNOTATIONS, TOOL_SPECS } from "./lib/tools.mjs";
import { ResourceNotFoundError, createOps } from "./lib/ops.mjs";

const ops = createOps();

// ---------------------------------------------------------------------------
// JSON-RPC error codes + parameter validation
// ---------------------------------------------------------------------------

const RPC = {
  parseError: -32700,
  invalidRequest: -32600,
  methodNotFound: -32601,
  invalidParams: -32602,
  internalError: -32603,
  resourceNotFound: -32002,
};

class RpcError extends Error {
  constructor(code, message, data) {
    super(message);
    this.code = code;
    this.data = data;
  }
}

function requiredString(args, key) {
  const value = args?.[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new RpcError(RPC.invalidParams, `\`${key}\` is required and must be a non-empty string`);
  }
  return value.trim();
}

function optionalString(args, key) {
  const value = args?.[key];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string") throw new RpcError(RPC.invalidParams, `\`${key}\` must be a string`);
  return value;
}

function optionalInt(args, key, { min, max, fallback }) {
  const raw = args?.[key];
  if (raw === undefined || raw === null) return fallback;
  const value = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new RpcError(RPC.invalidParams, `\`${key}\` must be an integer between ${min} and ${max}`);
  }
  return value;
}

function optionalBool(args, key, fallback = false) {
  const raw = args?.[key];
  if (raw === undefined || raw === null) return fallback;
  if (typeof raw !== "boolean") throw new RpcError(RPC.invalidParams, `\`${key}\` must be a boolean`);
  return raw;
}

function optionalMode(args) {
  const raw = args?.source_mode;
  if (raw === undefined || raw === null) return config.sourceMode;
  if (!SOURCE_MODES.includes(raw)) {
    throw new RpcError(RPC.invalidParams, `\`source_mode\` must be one of ${SOURCE_MODES.join(" | ")}`);
  }
  return raw;
}

// ---------------------------------------------------------------------------
// Tool definitions (JSON Schema + validation; handlers come from `ops`)
// ---------------------------------------------------------------------------

const MODE_SCHEMA = {
  type: "string",
  enum: SOURCE_MODES,
  description:
    "Retrieval corpus: 'wiki' (compiled pages only), 'source' (raw document chunks only), " +
    "'fusion' (both, merged). Defaults to the server setting.",
};

const TOOL_DEFS = [
  {
    name: "wiki_status",
    inputSchema: { type: "object", properties: {} },
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
    validate: () => ({}),
    run: () => ops.status(),
  },
  {
    name: "wiki_context",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query: entities and identifiers kept verbatim, no pleasantries.",
        },
        source_mode: MODE_SCHEMA,
        max_chunks: { type: "integer", minimum: 1, maximum: 50, description: "Maximum chunks to return." },
        max_tokens: { type: "integer", minimum: 500, maximum: 50000, description: "Token budget for the bundle." },
        max_pages: { type: "integer", minimum: 1, maximum: 20, description: "Maximum distinct pages." },
        max_per_source: { type: "integer", minimum: 1, maximum: 20, description: "Maximum chunks per source file." },
        graph_limit: { type: "integer", minimum: 0, maximum: 50, description: "Maximum related-page neighbours." },
        hybrid: { type: "boolean", description: "Add vector candidates (requires an embedded generation)." },
      },
      required: ["query"],
    },
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
    validate: (args) => ({
      query: requiredString(args, "query"),
      source_mode: optionalMode(args),
      max_chunks: optionalInt(args, "max_chunks", { min: 1, max: 50, fallback: config.maxChunks }),
      max_tokens: optionalInt(args, "max_tokens", { min: 500, max: 50000, fallback: config.maxTokens }),
      max_pages: optionalInt(args, "max_pages", { min: 1, max: 20, fallback: 4 }),
      max_per_source: optionalInt(args, "max_per_source", { min: 1, max: 20, fallback: 2 }),
      graph_limit: optionalInt(args, "graph_limit", { min: 0, max: 50, fallback: 4 }),
      hybrid: optionalBool(args, "hybrid", false),
    }),
    run: (args) => ops.context(args),
  },
  {
    name: "wiki_search",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query." },
        limit: { type: "integer", minimum: 1, maximum: 50, description: "Maximum hits (default 10)." },
        source_mode: MODE_SCHEMA,
      },
      required: ["query"],
    },
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
    validate: (args) => ({
      query: requiredString(args, "query"),
      limit: optionalInt(args, "limit", { min: 1, max: 50, fallback: 10 }),
      source_mode: optionalMode(args),
    }),
    run: (args) => ops.search(args),
  },
  {
    name: "wiki_page",
    inputSchema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Page slug (e.g. 'auth/sso-flow') or page_id (wp_…)." },
      },
      required: ["id"],
    },
    annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
    validate: (args) => ({ id: requiredString(args, "id") }),
    run: (args) => ops.page(args.id),
  },
];

if (config.enableAsk) {
  TOOL_DEFS.push({
    name: "wiki_ask",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "The question to answer from the wiki." },
        write_back: { type: "boolean", description: "Persist the verified answer to the insight layer (default false)." },
        hybrid: { type: "boolean", description: "Add vector candidates (requires an embedded generation)." },
      },
      required: ["query"],
    },
    annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: true },
    validate: (args) => ({
      query: requiredString(args, "query"),
      write_back: optionalBool(args, "write_back", false),
      hybrid: optionalBool(args, "hybrid", false),
    }),
    run: (args) => ops.ask(args),
  });
}

if (config.enableBuild) {
  TOOL_DEFS.push(
    {
      name: "wiki_build",
      inputSchema: {
        type: "object",
        properties: {
          idempotency_key: {
            type: "string",
            description: "Reuse a key to resolve to the same job instead of queueing another build.",
          },
        },
      },
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
      validate: (args) => ({ idempotency_key: optionalString(args, "idempotency_key") }),
      run: (args) => ops.build(args),
    },
    {
      name: "wiki_job",
      inputSchema: {
        type: "object",
        properties: { job_id: { type: "string", description: "Job id returned by wiki_build (job_…)." } },
        required: ["job_id"],
      },
      annotations: { ...READ_ONLY_ANNOTATIONS, idempotentHint: true },
      validate: (args) => ({ job_id: requiredString(args, "job_id") }),
      run: (args) => ops.job(args.job_id),
    },
  );
}

function toolDescriptor(def) {
  const spec = TOOL_SPECS[def.name];
  return {
    name: def.name,
    title: spec.title,
    description: spec.description,
    inputSchema: def.inputSchema,
    annotations: def.annotations,
  };
}

// ---------------------------------------------------------------------------
// JSON-RPC dispatch
// ---------------------------------------------------------------------------

function negotiateProtocol(requested) {
  if (typeof requested === "string" && KNOWN_PROTOCOL_VERSIONS.includes(requested)) return requested;
  return LATEST_PROTOCOL_VERSION;
}

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function sendResult(id, result) {
  send({ jsonrpc: "2.0", id, result });
}

function sendError(id, code, message, data) {
  const error = { code, message };
  if (data !== undefined) error.data = data;
  send({ jsonrpc: "2.0", id: id ?? null, error });
}

/**
 * Tool invocation. Validation failures stay protocol-level so the client can
 * self-correct; runtime and API failures ride in `isError` so the model reads
 * them. Kept out of the dispatch switch so that stays a pure map.
 */
async function handleToolCall(params) {
  const name = params?.name;
  const def = TOOL_DEFS.find((candidate) => candidate.name === name);
  if (!def) throw new RpcError(RPC.invalidParams, `unknown tool: ${String(name)}`);

  const rawArgs = params?.arguments ?? {};
  if (rawArgs !== null && typeof rawArgs !== "object") {
    throw new RpcError(RPC.invalidParams, "`arguments` must be an object");
  }

  try {
    const args = def.validate(rawArgs);
    const text = await def.run(args);
    return { content: [{ type: "text", text: clampText(String(text), config.renderChars) }], isError: false };
  } catch (error) {
    if (error instanceof RpcError) throw error;
    if (error instanceof ApiError) {
      return { content: [{ type: "text", text: apiErrorText(error) }], isError: true };
    }
    log(`tool ${name} crashed: ${error?.stack ?? error}`);
    return {
      content: [{ type: "text", text: `unexpected error in ${name}: ${error?.message ?? error}` }],
      isError: true,
    };
  }
}

async function handleRequest(method, params) {
  switch (method) {
    case "initialize":
      return {
        protocolVersion: negotiateProtocol(params?.protocolVersion),
        capabilities: {
          tools: { listChanged: false },
          resources: { subscribe: false, listChanged: false },
        },
        serverInfo: SERVER_INFO,
        instructions: INSTRUCTIONS,
      };

    case "ping":
      return {};

    case "tools/list":
      return { tools: TOOL_DEFS.map(toolDescriptor) };

    case "tools/call":
      return await handleToolCall(params);

    case "resources/list":
      return { resources: await ops.listPageResources() };

    case "resources/templates/list":
      return { resourceTemplates: [] };

    case "resources/read": {
      try {
        const content = await ops.readPageResource(params?.uri);
        return { contents: [content] };
      } catch (error) {
        if (error instanceof ResourceNotFoundError) {
          throw new RpcError(RPC.resourceNotFound, error.message, { uri: error.uri });
        }
        throw error;
      }
    }

    case "prompts/list":
      return { prompts: [] };

    case "logging/setLevel":
      return {};

    default:
      throw new RpcError(RPC.methodNotFound, `method not found: ${method}`);
  }
}

async function handleMessage(message) {
  if (message === null || typeof message !== "object" || Array.isArray(message)) {
    sendError(null, RPC.invalidRequest, "expected a single JSON-RPC object");
    return;
  }
  if (message.jsonrpc !== "2.0") {
    sendError(message.id ?? null, RPC.invalidRequest, 'missing "jsonrpc": "2.0"');
    return;
  }
  const { id, method, params } = message;
  if (typeof method !== "string") {
    if (id !== undefined) sendError(id, RPC.invalidRequest, "missing method");
    return;
  }

  // Notifications (no id) never get a response.
  if (id === undefined || id === null) {
    log(`notification: ${method}`);
    return;
  }

  try {
    sendResult(id, await handleRequest(method, params));
  } catch (error) {
    if (error instanceof RpcError) {
      sendError(id, error.code, error.message, error.data);
      return;
    }
    if (error instanceof ApiError) {
      sendError(id, RPC.internalError, apiErrorText(error));
      return;
    }
    log(`request ${method} crashed: ${error?.stack ?? error}`);
    sendError(id, RPC.internalError, error?.message ?? String(error));
  }
}

// ---------------------------------------------------------------------------
// stdio transport
// ---------------------------------------------------------------------------

function main() {
  process.stdout.on("error", (error) => {
    // A closed pipe means the client went away; nothing left to report to.
    log(`stdout closed: ${error?.message ?? error}`);
    process.exit(0);
  });

  process.on("uncaughtException", (error) => {
    process.stderr.write(`[llm-wiki-mcp] uncaught exception: ${error?.stack ?? error}\n`);
    process.exit(1);
  });
  process.on("unhandledRejection", (reason) => {
    process.stderr.write(`[llm-wiki-mcp] unhandled rejection: ${reason?.stack ?? reason}\n`);
  });

  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });

  lines.on("line", (line) => {
    const trimmed = line.trim();
    if (trimmed === "") return;
    let message;
    try {
      message = JSON.parse(trimmed);
    } catch (error) {
      sendError(null, RPC.parseError, `invalid JSON: ${error.message}`);
      return;
    }
    // Handle concurrently: a slow retrieval must not block a ping or a cancel.
    void handleMessage(message);
  });

  lines.on("close", () => {
    log("stdin closed, exiting");
    process.exit(0);
  });

  log(
    `ready (zero-dep) · base=${config.baseUrl} · mode=${config.sourceMode} · ` +
      `chunks=${config.maxChunks} · tokens=${config.maxTokens} · ` +
      `ask=${config.enableAsk ? "on" : "off"} · build=${config.enableBuild ? "on" : "off"}`,
  );
}

main();
