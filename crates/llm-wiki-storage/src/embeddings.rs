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
pub fn section_embeddings_by_hash(
    conn: &Connection,
    model: &str,
    hashes: &[String],
) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut map = BTreeMap::new();
    let mut stmt = conn
        .prepare(
            "SELECT text_hash, embedding FROM section_embeddings
             WHERE model = ?1 AND text_hash = ?2",
        )
        .map_err(|e| WikiError::Storage(format!("prepare embedding lookup: {e}")))?;
    for text_hash in hashes {
        let rows = stmt
            .query_map(params![model, text_hash], |row| row.get::<_, Vec<u8>>(1))
            .map_err(|e| WikiError::Storage(format!("embedding lookup: {e}")))?;
        for row in rows {
            let blob = row.map_err(db)?;
            map.insert(text_hash.clone(), decode_embedding(&blob));
        }
    }
    Ok(map)
}

/// Decodes a little-endian f32 blob, ignoring any trailing partial bytes.
fn decode_embedding(blob: &[u8]) -> Vec<f32> {
    blob.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}
