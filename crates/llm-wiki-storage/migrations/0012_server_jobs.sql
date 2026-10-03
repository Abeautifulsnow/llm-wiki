-- 0012_server_jobs: HTTP server job persistence (PRD §30/§31).
-- One row per server-initiated job. Statuses mirror the §31 build state
-- machine (QUEUED + terminal states); live stage phases (SCANNING..INDEXING)
-- are mirrored from the builds table via `build_id`. Rows survive server
-- restarts: startup recovery marks QUEUED/RUNNING jobs INTERRUPTED (§31:
-- no job may stay RUNNING in a fake state across a restart), and write
-- operations accept an idempotency key that maps to the SAME job row within
-- its retention window instead of starting a second LLM run.

CREATE TABLE IF NOT EXISTS server_jobs (
    job_id           TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN (
                         'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED',
                         'CANCELLED', 'INTERRUPTED', 'REPLAN_REQUIRED')),
    phase            TEXT,
    build_id         TEXT REFERENCES builds(build_id),
    failure_code     TEXT,
    retryable        INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    request_id       TEXT,
    idempotency_key  TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_server_jobs_idempotency
    ON server_jobs(idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_server_jobs_status ON server_jobs(status);
CREATE INDEX IF NOT EXISTS idx_server_jobs_created ON server_jobs(created_at, job_id);
