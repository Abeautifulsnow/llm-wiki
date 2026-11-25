/**
 * HTTP client for the llm-wiki agent protocol v1.
 *
 * Shared by both transports. Nothing here knows about MCP: it maps HTTP
 * failures onto `ApiError` (stable server code + request id) so the protocol
 * layer can decide whether a failure is a tool result or a protocol error.
 */

import { SOURCE_MODES } from "./constants.mjs";
import { log } from "./config.mjs";

export class ApiError extends Error {
  constructor(status, code, message, requestId) {
    super(message);
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

export function apiErrorText(error) {
  const where = error.status ? `HTTP ${error.status}` : "transport";
  const rid = error.requestId ? ` · request_id ${error.requestId}` : "";
  return `llm-wiki error [${error.code}] (${where}): ${error.message}${rid}`;
}

/**
 * `AbortSignal.timeout` with a Node 18.0-compatible fallback.
 *
 * Returns a `cancel` so the fallback timer is cleared once the request
 * settles; `unref()` alone keeps it alive for the full timeout on every call.
 */
function timeoutSignal(ms) {
  if (typeof AbortSignal.timeout === "function") {
    return { signal: AbortSignal.timeout(ms), cancel() {} };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  timer.unref?.();
  return { signal: controller.signal, cancel: () => clearTimeout(timer) };
}

export function createApi({ baseUrl, token, timeoutMs }) {
  async function callApi(path, { method = "GET", body, headers = {} } = {}) {
    const url = `${baseUrl}${path}`;
    const requestHeaders = { accept: "application/json", ...headers };
    if (body !== undefined) requestHeaders["content-type"] = "application/json";
    if (token) requestHeaders.authorization = `Bearer ${token}`;

    let response;
    const timeout = timeoutSignal(timeoutMs);
    try {
      response = await fetch(url, {
        method,
        headers: requestHeaders,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: timeout.signal,
      });
    } catch (cause) {
      const name = cause?.name ?? "";
      if (name === "TimeoutError" || name === "AbortError") {
        throw new ApiError(0, "timeout", `no response from ${baseUrl} within ${timeoutMs}ms`);
      }
      throw new ApiError(
        0,
        "unreachable",
        `cannot reach ${baseUrl} (${cause?.message ?? cause}) — is \`llm-wiki serve\` running?`,
      );
    } finally {
      timeout.cancel();
    }

    const text = await response.text();
    let payload = null;
    if (text.trim() !== "") {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }

    if (!response.ok) {
      const apiError = payload?.error ?? {};
      throw new ApiError(
        response.status,
        apiError.code ?? "http_error",
        apiError.message ?? text.slice(0, 500) ?? response.statusText,
        response.headers.get("x-request-id") ?? payload?.request_id ?? undefined,
      );
    }
    return payload ?? {};
  }

  /**
   * Runs a request that may need one legacy-server fallback on `source_mode`.
   * Servers predating the fusion fields reject the request field itself.
   */
  async function withSourceModeFallback(mode, run) {
    try {
      return await run(mode);
    } catch (error) {
      const isModeRejection =
        error instanceof ApiError && error.code === "invalid_request" && /source_mode/i.test(error.message);
      if (!isModeRejection || mode === "wiki") throw error;
      // Without this line a silent quality drop is invisible to operators:
      // the retry succeeds, but only the compiled wiki side is searched.
      log(`server rejected source_mode='${mode}'; retrying in wiki mode`);
      return await run("wiki");
    }
  }

  return { callApi, withSourceModeFallback };
}

// ---------------------------------------------------------------------------
// Argument helpers (shared by both transports' validation layers)
// ---------------------------------------------------------------------------

export function clampInt(value, { min, max, fallback }) {
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isInteger(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

export function normalizeMode(value, fallback) {
  return typeof value === "string" && SOURCE_MODES.includes(value) ? value : fallback;
}
