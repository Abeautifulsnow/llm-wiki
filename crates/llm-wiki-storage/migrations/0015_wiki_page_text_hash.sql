-- 0015_wiki_page_text_hash: persist the context-section identity hash on the
-- search-index rows themselves (§19.3 Vector layer, LW-001 read path).
--
-- Deriving that hash at query time meant re-reading and RE-HASHING every
-- section body on every hybrid retrieval — measured O(N) materialization:
-- ~1.2 s at 100k sections for the coverage probe alone, and the same text
-- term again inside the top-K scan, even though neither needs the text.
-- With the hash stored, both read a 32-byte key per section instead.
--
-- The formula is `llm_wiki_core::section_text::context_section_hash`
-- (title + heading path + body, sha256) and is written by the index rebuild;
-- it is the same function `active_context_sections` computes, so the stored
-- key and the queried key cannot drift.
--
-- NULL for rows indexed before this migration: readers fall back to deriving
-- the hash from the stored text, and the next publish rewrites every row it
-- touches with the persisted value.

ALTER TABLE wiki_page_text ADD COLUMN text_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_wiki_page_text_build_hash
    ON wiki_page_text(build_id, text_hash);
