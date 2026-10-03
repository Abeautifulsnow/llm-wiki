-- 0011_section_embeddings: the Vector layer's storage (§19.3 hybrid stack).
-- One row per (context-section text hash, embedding model): the section text
-- is `wiki-ask`-grade semantic content (title + heading path + body) hashed
-- with sha256 — content-addressed, so an unchanged section carries its
-- embedding across generations for free (§28's incremental philosophy, made
-- durable). The model is part of the key: switching embedding models never
-- mixes vector spaces.
--
-- Rows are written ONLY by the explicit `llm-wiki embed` backfill (never by
-- the build pipeline — embeddings cost model calls and the build stays
-- cache/publish-only). Consumers: `ask --hybrid` (brute-force cosine over
-- the active generation's rows) and future ranking surfaces.
-- Embedding BLOB = little-endian f32 sequence, dim entries.

CREATE TABLE IF NOT EXISTS section_embeddings (
    text_hash  TEXT NOT NULL,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    embedding  BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (text_hash, model)
);

CREATE INDEX IF NOT EXISTS idx_section_embeddings_model ON section_embeddings(model);
