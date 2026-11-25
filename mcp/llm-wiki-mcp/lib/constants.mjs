/**
 * Values that must not drift between the config, the API client and the two
 * transports. Kept in their own module so `config.mjs` can own the environment
 * while `api.mjs` imports the enums without pulling in the global config.
 */

export const SERVER_INFO = {
  name: "llm-wiki-mcp",
  title: "llm-wiki",
  version: "0.3.0",
};

/** Protocol versions the hand-rolled transport knows how to speak, newest first. */
export const KNOWN_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"];
export const LATEST_PROTOCOL_VERSION = KNOWN_PROTOCOL_VERSIONS[0];

/** Retrieval corpora accepted by `/v1/search` and `/v1/context`. */
export const SOURCE_MODES = ["wiki", "source", "fusion"];

/** Resource URI namespace for published pages. */
export const PAGE_URI_PREFIX = "wiki://page/";
