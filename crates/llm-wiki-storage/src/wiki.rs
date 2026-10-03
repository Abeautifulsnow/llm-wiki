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
    /// Knowledge nodes the page was compiled from — persisted with the page
    /// so lint (§36 orphan check) and search can use them without replaying
    /// the plan (PRD §45: page identity is persisted).
    pub knowledge_refs: Vec<KnowledgeNodeId>,
    pub citations: Vec<PageCitationRecord>,
    pub links: Vec<PageLinkRecord>,
}

/// One page of the currently published generation with its full machine
/// mapping — the view `lint` (PRD §36) operates on.
#[derive(Debug, Clone)]
pub struct GenerationPageView {
    pub page_id: WikiPageId,
    pub slug: String,
    pub title: String,
    pub category: String,
    pub body_hash: String,
    pub content: String,
    pub knowledge_refs: Vec<KnowledgeNodeId>,
    pub citations: Vec<PageCitationRecord>,
    /// Outbound resolved WikiLinks.
    pub links: Vec<PageLinkRecord>,
    /// Inbound WikiLinks from sibling pages of the same generation.
    pub inbound_links: u64,
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
            "INSERT INTO wiki_pages (page_id, build_id, slug, title, category, language, body_hash, content, knowledge_refs_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                page.page_id.as_str(),
                build_id.as_str(),
                page.slug,
                page.title,
                page.category,
                page.language,
                page.body_hash,
                page.content,
                knowledge_refs_json(&page.knowledge_refs)?,
                chrono::Utc::now().to_rfc3339()
            ],
        )
        .map_err(db)?;
    }
    for page in pages {
        for citation in &page.citations {
            tx.execute(
                "INSERT INTO page_citations (citation_id, build_id, page_id, claim_node_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    CitationId::generate().as_str(),
                    build_id.as_str(),
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
                "INSERT INTO page_links (link_id, build_id, from_page_id, to_page_id, target_title)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    LinkRowId::generate().as_str(),
                    build_id.as_str(),
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

/// Loads one generation's page rows back (publish-journal recovery uses these
/// to re-verify the generation on disk against `body_hash`, PRD §35).
/// Citation/link mappings are not needed for that check and come back empty.
pub fn load_generation_pages(conn: &Connection, build_id: &BuildId) -> Result<Vec<WikiPageRecord>> {
    let mut stmt = conn
        .prepare(
            "SELECT page_id, slug, title, category, language, body_hash, content, knowledge_refs_json
             FROM wiki_pages WHERE build_id = ?1 ORDER BY slug",
        )
        .map_err(|e| WikiError::Storage(format!("prepare load_generation_pages: {e}")))?;
    let rows = stmt
        .query_map(params![build_id.as_str()], |row| {
            Ok((
                WikiPageId::from_validated(row.get::<_, String>("page_id")?),
                row.get::<_, String>("slug")?,
                row.get::<_, String>("title")?,
                row.get::<_, String>("category")?,
                row.get::<_, String>("language")?,
                row.get::<_, String>("body_hash")?,
                row.get::<_, String>("content")?,
                row.get::<_, String>("knowledge_refs_json")?,
            ))
        })
        .map_err(|e| WikiError::Storage(format!("load_generation_pages: {e}")))?;
    let mut out = Vec::new();
    for row in rows {
        let (page_id, slug, title, category, language, body_hash, content, refs_json) =
            row.map_err(db)?;
        out.push(WikiPageRecord {
            page_id,
            slug,
            title,
            category,
            language,
            body_hash,
            content,
            knowledge_refs: parse_knowledge_refs(&refs_json)?,
            citations: Vec::new(),
            links: Vec::new(),
        });
    }
    Ok(out)
}

/// Cheap counts of one persisted generation — the no-change fast path
/// (§37.3) reports the ACTIVE generation without loading its content.
pub fn generation_stats(conn: &Connection, build_id: &BuildId) -> Result<GenerationStats> {
    let count = |sql: &str| -> Result<i64> {
        conn.query_row(sql, params![build_id.as_str()], |r| r.get(0))
            .map_err(|e| WikiError::Storage(format!("generation_stats: {e}")))
    };
    let pages = count("SELECT COUNT(*) FROM wiki_pages WHERE build_id = ?1")?;
    let citations = count("SELECT COUNT(*) FROM page_citations WHERE build_id = ?1")?;
    let links = count("SELECT COUNT(*) FROM page_links WHERE build_id = ?1")?;
    Ok(GenerationStats {
        pages: pages as usize,
        citations: citations as usize,
        links: links as usize,
    })
}

/// Loads the full lint view of one generation: pages with citations, links
/// (outbound and inbound) and persisted knowledge refs (PRD §36).
pub fn load_generation_view(
    conn: &Connection,
    build_id: &BuildId,
) -> Result<Vec<GenerationPageView>> {
    let mut pages: Vec<GenerationPageView> = Vec::new();
    {
        let mut stmt = conn
            .prepare(
                "SELECT page_id, slug, title, category, body_hash, content, knowledge_refs_json
                 FROM wiki_pages WHERE build_id = ?1 ORDER BY slug",
            )
            .map_err(|e| WikiError::Storage(format!("prepare load_generation_view: {e}")))?;
        let rows = stmt
            .query_map(params![build_id.as_str()], |row| {
                Ok((
                    WikiPageId::from_validated(row.get::<_, String>("page_id")?),
                    row.get::<_, String>("slug")?,
                    row.get::<_, String>("title")?,
                    row.get::<_, String>("category")?,
                    row.get::<_, String>("body_hash")?,
                    row.get::<_, String>("content")?,
                    row.get::<_, String>("knowledge_refs_json")?,
                ))
            })
            .map_err(|e| WikiError::Storage(format!("load_generation_view: {e}")))?;
        for row in rows {
            let (page_id, slug, title, category, body_hash, content, refs_json) =
                row.map_err(db)?;
            pages.push(GenerationPageView {
                page_id,
                slug,
                title,
                category,
                body_hash,
                content,
                knowledge_refs: parse_knowledge_refs(&refs_json)?,
                citations: Vec::new(),
                links: Vec::new(),
                inbound_links: 0,
            });
        }
    }

    let mut index_of: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for (idx, page) in pages.iter().enumerate() {
        index_of.insert(page.page_id.as_str().to_owned(), idx);
    }

    {
        // Page ids persist across builds (PRD §45), so every child-row lookup
        // is scoped to THIS generation's build_id.
        let mut stmt = conn
            .prepare(
                "SELECT page_id, claim_node_id, source_id, section_id, range_start, range_end, source_hash, evidence_digest, heading_path_json
                 FROM page_citations WHERE build_id = ?1 AND page_id = ?2 ORDER BY rowid",
            )
            .map_err(|e| WikiError::Storage(format!("prepare view citations: {e}")))?;
        for page in &mut pages {
            let rows = stmt
                .query_map(
                    params![build_id.as_str(), page.page_id.as_str()],
                    map_citation_row,
                )
                .map_err(|e| WikiError::Storage(format!("view citations: {e}")))?;
            for row in rows {
                page.citations.push(row.map_err(db)?);
            }
        }
    }

    {
        let mut stmt = conn
            .prepare(
                "SELECT from_page_id, to_page_id, target_title FROM page_links
                 WHERE build_id = ?1 AND from_page_id = ?2 ORDER BY rowid",
            )
            .map_err(|e| WikiError::Storage(format!("prepare view links: {e}")))?;
        for page in &mut pages {
            let rows = stmt
                .query_map(params![build_id.as_str(), page.page_id.as_str()], |row| {
                    Ok(PageLinkRecord {
                        to_page_id: WikiPageId::from_validated(row.get::<_, String>("to_page_id")?),
                        target_title: row.get("target_title")?,
                    })
                })
                .map_err(|e| WikiError::Storage(format!("view links: {e}")))?;
            for row in rows {
                page.links.push(row.map_err(db)?);
            }
        }
    }

    // Inbound counts: one query over all links of this generation.
    {
        let mut stmt = conn
            .prepare(
                "SELECT to_page_id, COUNT(*) FROM page_links
                 WHERE build_id = ?1 AND to_page_id IN (SELECT page_id FROM wiki_pages WHERE build_id = ?1)
                 GROUP BY to_page_id",
            )
            .map_err(|e| WikiError::Storage(format!("prepare view inbound: {e}")))?;
        let rows = stmt
            .query_map(params![build_id.as_str()], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|e| WikiError::Storage(format!("view inbound: {e}")))?;
        for row in rows {
            let (to_page_id, count) = row.map_err(db)?;
            if let Some(idx) = index_of.get(&to_page_id) {
                pages[*idx].inbound_links = count.max(0) as u64;
            }
        }
    }

    Ok(pages)
}

fn map_citation_row(row: &rusqlite::Row) -> rusqlite::Result<PageCitationRecord> {
    let heading_path_json: String = row.get("heading_path_json")?;
    let heading_path: Vec<String> = serde_json::from_str(&heading_path_json).unwrap_or_default();
    let section_id: Option<String> = row.get("section_id")?;
    Ok(PageCitationRecord {
        claim_node_id: KnowledgeNodeId::from_validated(row.get::<_, String>("claim_node_id")?),
        source_id: SourceId::from_validated(row.get::<_, String>("source_id")?),
        section_id: section_id.map(SectionId::from_validated),
        range: SourceRange::new(
            row.get::<_, i64>("range_start")?.max(0) as usize,
            row.get::<_, i64>("range_end")?.max(0) as usize,
        ),
        source_hash: row.get("source_hash")?,
        evidence_digest: row.get("evidence_digest")?,
        heading_path,
    })
}

fn knowledge_refs_json(refs: &[KnowledgeNodeId]) -> Result<String> {
    serde_json::to_string(
        &refs
            .iter()
            .map(|id| id.as_str().to_owned())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| WikiError::Storage(format!("serialize knowledge refs: {e}")))
}

pub(crate) fn parse_knowledge_refs(json: &str) -> Result<Vec<KnowledgeNodeId>> {
    let raw: Vec<String> = serde_json::from_str(json)
        .map_err(|e| WikiError::Storage(format!("parse knowledge refs: {e}")))?;
    Ok(raw
        .into_iter()
        .map(KnowledgeNodeId::from_validated)
        .collect())
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
                knowledge_refs: vec![claim_node.clone()],
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
                knowledge_refs: Vec::new(),
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
            knowledge_refs: Vec::new(),
            citations: vec![],
            links: vec![],
        };
        assert!(persist_generation(&mut conn, &build_id, &[duplicate]).is_err());
    }

    #[test]
    fn generation_pages_roundtrip_for_recovery_validation() {
        let (mut conn, _source_id, _claim) = seed();
        let build_id = BuildId::generate();
        let pages = vec![WikiPageRecord {
            page_id: WikiPageId::generate(),
            slug: "runtime".into(),
            title: "Runtime".into(),
            category: "architecture".into(),
            language: "en".into(),
            body_hash: "hash-b".into(),
            content: "# Runtime".into(),
            knowledge_refs: Vec::new(),
            citations: vec![],
            links: vec![],
        }];
        persist_generation(&mut conn, &build_id, &pages).unwrap();

        let loaded = load_generation_pages(&conn, &build_id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].slug, "runtime");
        assert_eq!(loaded[0].body_hash, "hash-b");
        assert_eq!(loaded[0].content, "# Runtime");
        assert!(loaded[0].citations.is_empty());

        let other = BuildId::generate();
        assert!(load_generation_pages(&conn, &other).unwrap().is_empty());
    }

    #[test]
    fn generation_view_loads_citations_links_and_inbound_counts() {
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
                content: "# Plugin System".into(),
                knowledge_refs: vec![claim_node.clone()],
                citations: vec![PageCitationRecord {
                    claim_node_id: claim_node.clone(),
                    source_id: source_id.clone(),
                    section_id: None,
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
                knowledge_refs: Vec::new(),
                citations: vec![],
                links: vec![],
            },
        ];
        persist_generation(&mut conn, &build_id, &pages).unwrap();

        let view = load_generation_view(&conn, &build_id).unwrap();
        assert_eq!(view.len(), 2);
        let a = view
            .iter()
            .find(|p| p.page_id == page_a)
            .expect("page a in view");
        assert_eq!(a.knowledge_refs, vec![claim_node.clone()]);
        assert_eq!(a.citations.len(), 1);
        assert_eq!(a.citations[0].range, SourceRange::new(4, 20));
        assert_eq!(a.links.len(), 1);
        assert_eq!(a.inbound_links, 0);
        let b = view
            .iter()
            .find(|p| p.page_id == page_b)
            .expect("page b in view");
        assert_eq!(b.inbound_links, 1, "runtime receives one inbound link");

        let other = BuildId::generate();
        assert!(load_generation_view(&conn, &other).unwrap().is_empty());
    }
}
