//! Compiled wiki generation persistence (PRD §15/§16): pages, machine
//! citation mapping and resolved WikiLinks for one build, written in a single
//! transaction. Filesystem generations (§35) are the visible artifact; these
//! rows are the machine state backing lint/search.

use rusqlite::{params, Connection};

use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::{
    BuildId, CitationId, KnowledgeNodeId, LinkRowId, SectionId, SourceId, WikiPageId,
};
use llm_wiki_core::model::SourceRange;

fn db(e: rusqlite::Error) -> WikiError {
    WikiError::Storage(e.to_string())
}

/// One expanded citation on a page (PRD §16): the `<!-- llm-wiki:cite ... -->`
/// comment in the Markdown has a matching row here.
#[derive(Debug, Clone)]
pub struct PageCitationRecord {
    pub claim_node_id: KnowledgeNodeId,
    pub source_id: SourceId,
    pub section_id: Option<SectionId>,
    pub range: SourceRange,
    pub source_hash: String,
    pub evidence_digest: String,
    pub heading_path: Vec<String>,
}

/// One resolved WikiLink (`[[Title]]` → target page, PRD §15.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageLinkRecord {
    pub to_page_id: WikiPageId,
    pub target_title: String,
}

/// A compiled page ready to persist.
#[derive(Debug, Clone)]
pub struct WikiPageRecord {
    pub page_id: WikiPageId,
    pub slug: String,
    pub title: String,
    pub category: String,
    pub language: String,
    pub body_hash: String,
    pub content: String,
    pub citations: Vec<PageCitationRecord>,
    pub links: Vec<PageLinkRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationStats {
    pub pages: usize,
    pub citations: usize,
    pub links: usize,
}

/// Persists one generation's pages inside a single transaction. Pages of a
/// failed build are simply never written here (§34: no partial publishes).
pub fn persist_generation(
    conn: &mut Connection,
    build_id: &BuildId,
    pages: &[WikiPageRecord],
) -> Result<GenerationStats> {
    let tx = conn
        .transaction()
        .map_err(|e| WikiError::Storage(format!("begin tx: {e}")))?;
    let mut citations = 0usize;
    let mut links = 0usize;
    // Pages first: links reference sibling page rows created in the same
    // transaction, so every page row must exist before link/citation rows.
    for page in pages {
        tx.execute(
            "INSERT INTO wiki_pages (page_id, build_id, slug, title, category, language, body_hash, content, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                page.page_id.as_str(),
                build_id.as_str(),
                page.slug,
                page.title,
                page.category,
                page.language,
                page.body_hash,
                page.content,
                chrono::Utc::now().to_rfc3339()
            ],
        )
        .map_err(db)?;
    }
    for page in pages {
        for citation in &page.citations {
            tx.execute(
                "INSERT INTO page_citations (citation_id, page_id, claim_node_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    CitationId::generate().as_str(),
                    page.page_id.as_str(),
                    citation.claim_node_id.as_str(),
                    citation.source_id.as_str(),
                    citation.section_id.as_ref().map(|s| s.as_str()),
                    citation.range.start as i64,
                    citation.range.end as i64,
                    citation.source_hash,
                    citation.evidence_digest,
                    serde_json::to_string(&citation.heading_path)
                        .map_err(|e| WikiError::Storage(format!("heading path json: {e}")))?,
                ],
            )
            .map_err(db)?;
            citations += 1;
        }

        for link in &page.links {
            tx.execute(
                "INSERT INTO page_links (link_id, from_page_id, to_page_id, target_title)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    LinkRowId::generate().as_str(),
                    page.page_id.as_str(),
                    link.to_page_id.as_str(),
                    link.target_title,
                ],
            )
            .map_err(db)?;
            links += 1;
        }
    }
    tx.commit()
        .map_err(|e| WikiError::Storage(format!("commit generation: {e}")))?;
    Ok(GenerationStats {
        pages: pages.len(),
        citations,
        links,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::open_in_memory;
    use crate::registry::NodeDraft;
    use crate::sources::upsert_source;
    use crate::{get_or_create_batch, NodeKind};
    use llm_wiki_core::ids::SourceLocatorKey;

    fn seed() -> (Connection, SourceId, KnowledgeNodeId) {
        let mut conn = open_in_memory().unwrap();
        let (source_id, _) = upsert_source(
            &mut conn,
            &SourceLocatorKey::compute("ws", "a.md"),
            "a.md",
            "hash-1",
            10,
            None,
        )
        .unwrap();
        conn.execute(
            "INSERT INTO source_sections (section_id, source_id, heading_path_json, heading_path_key, content_fingerprint, range_start, range_end, status)
             VALUES ('sec_01ARZ3NDEKTSV4RRFFQ69G5FAV', ?1, '[\"Doc\"]', 'Doc', 'fp', 0, 10, 'active')",
            params![source_id.as_str()],
        )
        .unwrap();
        let drafts = vec![NodeDraft {
            kind: NodeKind::Claim,
            canonical_key: "claim-key".to_owned(),
            canonical_name: "claim".into(),
            entity_type: None,
            description: None,
        }];
        let ids = get_or_create_batch(&mut conn, &drafts, None).unwrap();
        let claim_node = ids[0].clone();
        (conn, source_id, claim_node)
    }

    #[test]
    fn generation_persists_pages_citations_links() {
        let (mut conn, source_id, claim_node) = seed();
        let page_a = WikiPageId::generate();
        let page_b = WikiPageId::generate();
        let build_id = BuildId::generate();
        let pages = vec![
            WikiPageRecord {
                page_id: page_a.clone(),
                slug: "plugin-system".into(),
                title: "Plugin System".into(),
                category: "concepts".into(),
                language: "en".into(),
                body_hash: "hash-a".into(),
                content: "# Plugin System\n\ntext <!-- llm-wiki:cite ... -->".into(),
                citations: vec![PageCitationRecord {
                    claim_node_id: claim_node.clone(),
                    source_id: source_id.clone(),
                    section_id: Some(SectionId::parse("sec_01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap()),
                    range: SourceRange::new(4, 20),
                    source_hash: "hash-1".into(),
                    evidence_digest: "digest".into(),
                    heading_path: vec!["Doc".into()],
                }],
                links: vec![PageLinkRecord {
                    to_page_id: page_b.clone(),
                    target_title: "Runtime".into(),
                }],
            },
            WikiPageRecord {
                page_id: page_b.clone(),
                slug: "runtime".into(),
                title: "Runtime".into(),
                category: "architecture".into(),
                language: "en".into(),
                body_hash: "hash-b".into(),
                content: "# Runtime".into(),
                citations: vec![],
                links: vec![],
            },
        ];

        let stats = persist_generation(&mut conn, &build_id, &pages).unwrap();
        assert_eq!(
            stats,
            GenerationStats {
                pages: 2,
                citations: 1,
                links: 1
            }
        );

        let counts: (i64, i64, i64) = conn
            .query_row(
                "SELECT (SELECT COUNT(*) FROM wiki_pages),
                        (SELECT COUNT(*) FROM page_citations),
                        (SELECT COUNT(*) FROM page_links)",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(counts, (2, 1, 1));

        // (build_id, slug) uniqueness rejects a duplicate slug.
        let duplicate = WikiPageRecord {
            page_id: WikiPageId::generate(),
            slug: "runtime".into(),
            title: "Runtime copy".into(),
            category: "architecture".into(),
            language: "en".into(),
            body_hash: "hash-c".into(),
            content: "# Runtime copy".into(),
            citations: vec![],
            links: vec![],
        };
        assert!(persist_generation(&mut conn, &build_id, &[duplicate]).is_err());
    }
}
