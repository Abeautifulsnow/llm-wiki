-- 0014_server_job_progress: live progress counters for non-build jobs.
-- Embed jobs report {"total","covered_before","embedded"} here while
-- RUNNING; build jobs keep using `phase` (the §31 stage machine).

ALTER TABLE server_jobs ADD COLUMN progress_json TEXT;
