//! Section-embedding storage (§19.3 Vector layer): content-addressed vectors
//! for context sections. The table is keyed (text_hash, model) so unchanged
//! sections keep their embeddings across generations and different embedding
//! models never share a vector space. Written only by the explicit
//! `llm-wiki embed` backfill; read by hybrid retrieval.

use std::collections::BTreeMap;

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// Persists one batch of embeddings (little-endian f32 blobs). Rows already
/// present for (hash, model) are replaced — same content re-embeds to the
/// same vector under a deterministic model, so replace is idempotent.
pub fn insert_section_embeddings(
    conn: &mut Connection,
    model: &str,
    embeddings: &[(String, Vec<f32>)],
) -> Result<()> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin embed tx: {e}")))?;
    {
        let mut insert = tx
            .prepare(
                "INSERT INTO section_embeddings (text_hash, model, dim, embedding, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT (text_hash, model) DO UPDATE SET
                     dim = excluded.dim, embedding = excluded.embedding, created_at = excluded.created_at",
            )
            .map_err(|e| WikiError::Storage(format!("prepare embedding insert: {e}")))?;
        for (text_hash, vector) in embeddings {
            let mut blob = Vec::with_capacity(vector.len() * 4);
            for value in vector {
                blob.extend_from_slice(&value.to_le_bytes());
            }
            insert
                .execute(params![
                    text_hash,
                    model,
                    vector.len() as i64,
                    blob,
                    chrono::Utc::now().to_rfc3339()
                ])
                .map_err(db)?;
        }
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit embeddings: {e}")))?;
    Ok(())
}

/// Loads the stored vectors for `model` keyed by text hash. Missing hashes
/// are simply absent — coverage is the caller's question.
///
/// One batched `IN` query per [`HASH_CHUNK`] hashes instead of one query per
/// hash: the retrieval path calls this over every section of the ACTIVE
/// generation on each hybrid query (PRD §19.3 read path).
pub fn section_embeddings_by_hash(
    conn: &Connection,
    model: &str,
    hashes: &[String],
) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut map = BTreeMap::new();
    for chunk in hashes.chunks(HASH_CHUNK) {
        let sql = in_chunk_sql(
            "SELECT text_hash, embedding FROM section_embeddings \
             WHERE model = ?1 AND text_hash IN ({placeholders})",
            chunk.len(),
        );
        // `prepare_cached`: only two SQL shapes ever occur (a full chunk and
        // the shorter tail), so the statement is compiled once and reused —
        // not re-prepared for every chunk.
        let mut stmt = conn
            .prepare_cached(&sql)
            .map_err(|e| WikiError::Storage(format!("prepare embedding lookup: {e}")))?;
        let params = model_then_hashes(model, chunk);
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| WikiError::Storage(format!("embedding lookup: {e}")))?;
        for row in rows {
            let (text_hash, blob) = row.map_err(db)?;
            map.insert(text_hash, decode_embedding(&blob));
        }
    }
    Ok(map)
}

/// Whether ANY of `hashes` has a stored vector for `model` — the coverage
/// pre-check on the hybrid retrieval hot path. One indexed existence probe
/// per chunk (`LIMIT 1`, short-circuits on the first hit), instead of
/// loading every vector of the generation to answer a yes/no question.
pub fn has_any_section_embedding(
    conn: &Connection,
    model: &str,
    hashes: &[String],
) -> Result<bool> {
    for chunk in hashes.chunks(HASH_CHUNK) {
        let sql = in_chunk_sql(
            "SELECT 1 FROM section_embeddings \
             WHERE model = ?1 AND text_hash IN ({placeholders}) LIMIT 1",
            chunk.len(),
        );
        let mut stmt = conn
            .prepare_cached(&sql)
            .map_err(|e| WikiError::Storage(format!("prepare embedding coverage: {e}")))?;
        let params = model_then_hashes(model, chunk);
        if stmt
            .exists(rusqlite::params_from_iter(params))
            .map_err(|e| WikiError::Storage(format!("embedding coverage: {e}")))?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Older SQLite builds cap bound parameters at 999, so hash lists are
/// chunked well under it for the batched `IN` reads.
const HASH_CHUNK: usize = 500;

/// `sql` must contain `{placeholders}`; it is replaced with `count` `?` slots
/// (parameter ?1 is the model, ?2.. are the chunk's hashes).
fn in_chunk_sql(sql: &str, count: usize) -> String {
    let placeholders = vec!["?"; count].join(", ");
    sql.replace("{placeholders}", &placeholders)
}

/// Binds `?1 = model`, `?2.. = chunk hashes` for one batched `IN` query.
/// Owned `Value`s keep the binding free of borrow gymnastics around the
/// unsized `&str → &dyn ToSql` coercion.
fn model_then_hashes(model: &str, chunk: &[String]) -> Vec<rusqlite::types::Value> {
    std::iter::once(rusqlite::types::Value::Text(model.to_owned()))
        .chain(
            chunk
                .iter()
                .map(|hash| rusqlite::types::Value::Text(hash.clone())),
        )
        .collect()
}

/// Decodes a little-endian f32 blob, ignoring any trailing partial bytes.
fn decode_embedding(blob: &[u8]) -> Vec<f32> {
    blob.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;

    fn hashes(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("hash-{i:06}")).collect()
    }

    fn conn_with_vectors(n: usize) -> Connection {
        let mut conn = open_in_memory().unwrap();
        let pairs: Vec<(String, Vec<f32>)> = hashes(n)
            .into_iter()
            .map(|hash| (hash, vec![0.25_f32; 8]))
            .collect();
        insert_section_embeddings(&mut conn, "m", &pairs).unwrap();
        conn
    }

    #[test]
    fn batched_lookup_returns_exactly_the_stored_hashes() {
        let conn = conn_with_vectors(1_200);
        let lookup = hashes(1_300);
        let stored = section_embeddings_by_hash(&conn, "m", &lookup).unwrap();
        assert_eq!(stored.len(), 1_200, "the 1_300th hash was never stored");
        assert!(stored.contains_key("hash-001199"));
        assert_eq!(stored["hash-000000"], vec![0.25_f32; 8]);

        let other_model = section_embeddings_by_hash(&conn, "other", &lookup).unwrap();
        assert!(other_model.is_empty(), "models never share a vector space");
    }

    #[test]
    fn coverage_probe_answers_without_loading_vectors() {
        let conn = conn_with_vectors(600);
        assert!(has_any_section_embedding(&conn, "m", &hashes(2_000)).unwrap());
        assert!(!has_any_section_embedding(&conn, "m", &[]).unwrap());
        assert!(!has_any_section_embedding(&conn, "absent", &hashes(10)).unwrap());
        // A hit in the SECOND chunk is still found (chunking must not skip
        // chunks): HASH_CHUNK absent hashes push the stored one past the
        // first chunk boundary.
        let mut spaced: Vec<String> = (0..HASH_CHUNK).map(|i| format!("absent-{i:06}")).collect();
        spaced.push("hash-000599".into());
        assert!(has_any_section_embedding(&conn, "m", &spaced).unwrap());
    }
}
