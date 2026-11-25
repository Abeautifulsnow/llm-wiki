/**
 * Shared configuration for both transports (hand-rolled stdio and MCP SDK).
 *
 * Everything is read from the environment once at startup; the MCP client
 * spawns a fresh process per session, so there is no reload path by design.
 */

import { SOURCE_MODES } from "./constants.mjs";

// Re-exported so callers keep importing the enums from the config module
// while the single definition lives in `constants.mjs`.
export { SERVER_INFO, KNOWN_PROTOCOL_VERSIONS, LATEST_PROTOCOL_VERSION, SOURCE_MODES } from "./constants.mjs";

function envInt(name, fallback, { min, max }) {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    log(`ignoring ${name}=${JSON.stringify(raw)} (expected integer ${min}..${max}), using ${fallback}`);
    return fallback;
  }
  return value;
}

function envFlag(name) {
  const raw = (process.env[name] ?? "").trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
}

const configuredMode = (process.env.LLM_WIKI_SOURCE_MODE ?? "").trim();

export const config = {
  baseUrl: (process.env.LLM_WIKI_BASE_URL ?? "http://127.0.0.1:8080").trim().replace(/\/+$/, ""),
  token: (process.env.LLM_WIKI_TOKEN ?? "").trim(),
  sourceMode: SOURCE_MODES.includes(configuredMode) ? configuredMode : "fusion",
  maxChunks: envInt("LLM_WIKI_MCP_MAX_CHUNKS", 8, { min: 1, max: 50 }),
  maxTokens: envInt("LLM_WIKI_MCP_MAX_TOKENS", 8000, { min: 500, max: 50000 }),
  renderChars: envInt("LLM_WIKI_MCP_RENDER_CHARS", 16000, { min: 1000, max: 200000 }),
  timeoutMs: envInt("LLM_WIKI_MCP_TIMEOUT_MS", 20000, { min: 1000, max: 300000 }),
  enableAsk: envFlag("LLM_WIKI_MCP_ENABLE_ASK"),
  enableBuild: envFlag("LLM_WIKI_MCP_ENABLE_BUILD"),
  debug: envFlag("LLM_WIKI_MCP_DEBUG"),
};

// Reads the flag from the environment directly: `log` is used while `config`
// is still being constructed, so it must not touch the config object (TDZ).
export function log(message) {
  if (envFlag("LLM_WIKI_MCP_DEBUG")) process.stderr.write(`[llm-wiki-mcp] ${message}\n`);
}
