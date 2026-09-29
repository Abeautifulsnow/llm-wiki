#![forbid(unsafe_code)]
//! Search layer (PRD §20, §5.5): the [`TextAnalyzer`] shared-normalization
//! abstraction, the [`FullTextSearch`] API and its SQLite FTS5 implementation.
//!
//! Dependency direction: search → {core, storage} only. The tokenization
//! rules live in `llm-wiki-storage::search_index` so the publish-transaction
//! rebuild can share ONE implementation with query time (PRD §20: query and
//! index MUST use the same normalization); this crate owns the query-side
//! surface — MATCH-expression building, hits and the async API.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use llm_wiki_core::error::{Result, WikiError};
use llm_wiki_core::ids::WikiPageId;
use llm_wiki_storage::graph::{graph_expand_from_page, GraphNeighbor};
use llm_wiki_storage::search_index::{self, SearchTokenizer};
use tokio::task::spawn_blocking;

/// Marks crate purpose.
pub const V0_2_SCOPE: &str = "FTS with language-aware analyzers, rank fusion, graph expansion";

// ---------------------------------------------------------------------------
// TextAnalyzer
// ---------------------------------------------------------------------------

/// Query-side façade over the storage-layer tokenizer (PRD §20): one
/// normalization (NFKC, lowercase, full→half width, common punctuation
/// stripped), Latin/digit word tokens and Han unigram+bigram tokens, plus the
/// MATCH-expression builder for pre-tokenized FTS5 fields.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextAnalyzer;

impl TextAnalyzer {
    /// Tokenizes `text` exactly the way the index was built.
    pub fn analyze(&self, text: &str) -> Vec<String> {
        search_index::DefaultSearchTokenizer.analyze(text)
    }

    /// Builds the FTS5 MATCH expression for `text`: deduplicated tokens as
    /// quoted string literals joined by `OR` (a space-joined bag would be an
    /// implicit AND, which no natural-language question survives). Quoting
    /// keeps every token a literal term — FTS5 query syntax (AND/OR/NEAR,
    /// column filters) can never leak in, and the expression travels to SQLite
    /// as a bound parameter. Empty input yields an empty expression.
    pub fn fts_query(&self, text: &str) -> String {
        let mut seen = std::collections::BTreeSet::new();
        let mut terms = Vec::new();
        for token in self.analyze(text) {
            if seen.insert(token.clone()) {
                terms.push(format!("\"{}\"", token.replace('"', "\"\"")));
            }
        }
        terms.join(" OR ")
    }
}

// ---------------------------------------------------------------------------
// FullTextSearch
// ---------------------------------------------------------------------------

/// One section-level hit (PRD §5.5: search returns page/section, not answers).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub page_id: WikiPageId,
    pub slug: String,
    pub title: String,
    /// The section's heading path, outermost first.
    pub heading_path: Vec<String>,
    /// FTS5 `snippet()` over the section body with `[…]` highlight marks.
    pub snippet: String,
    /// FTS5 `bm25()` rank — numerically SMALLER is a better match.
    pub rank: f64,
}

/// Full-text search over the generated wiki (PRD §20 API).
#[async_trait]
pub trait FullTextSearch: Send + Sync {
    /// Returns up to `limit` hits of the ACTIVE generation, best-first.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>>;
}

/// SQLite FTS5 implementation over the state db. The connection is shared
/// behind an `Arc<std::sync::Mutex<…>>` (`Connection` is neither `Sync` nor
/// async) and every query — including the FTS5 probe — runs inside one
/// `spawn_blocking` closure, so the guard never crosses a thread; the index
/// itself is maintained by the publish flow.
pub struct SqliteFullTextSearch {
    conn: Arc<Mutex<llm_wiki_storage::Connection>>,
    analyzer: TextAnalyzer,
    full_text_enabled: bool,
}

impl SqliteFullTextSearch {
    /// `full_text_enabled` mirrors `config.search.full_text` (PRD §32): when
    /// false, [`FullTextSearch::search`] fails with a clear config error.
    pub fn new(conn: llm_wiki_storage::Connection, full_text_enabled: bool) -> Self {
        Self {
            conn: Arc::new(Mutex::new(conn)),
            analyzer: TextAnalyzer,
            full_text_enabled,
        }
    }

    /// Whether anything is published — the CLI renders the friendly
    /// "nothing published" message instead of calling [`Self::search`].
    pub fn has_published(&self) -> Result<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|poisoned| WikiError::Storage(poisoned.to_string()))?;
        Ok(llm_wiki_storage::get_active_build_id(&conn)?.is_some())
    }
}

// ---------------------------------------------------------------------------
// Graph exploration (PRD §17/§22)
// ---------------------------------------------------------------------------

/// One-hop Wiki-Graph expansion façade (PRD §17 graph, §22 limits). The
/// stored graph is maintained by the publish flow regardless of config;
/// `config.search.graph` gates the CONSUMPTION side — the CLI skips calling
/// this façade when it is false, so a disabled flag costs nothing and a
/// enabled one always reads the active generation's graph. Dependency
/// direction unchanged: search → {core, storage}.
pub struct SqliteGraphExploration {
    conn: Arc<Mutex<llm_wiki_storage::Connection>>,
}

impl SqliteGraphExploration {
    pub fn new(conn: llm_wiki_storage::Connection) -> Self {
        Self {
            conn: Arc::new(Mutex::new(conn)),
        }
    }

    /// One-hop neighbors of a wiki page, deterministic and capped (pass
    /// [`llm_wiki_storage::EXPAND_MAX_NODES`] for the §22 default). The page
    /// reaches other pages via `links_to` (both directions) and — once edge
    /// producers exist — entities/concepts via analysis relations, which the
    /// storage expansion already supports for any node type.
    pub async fn neighbors_of_page(
        &self,
        page_id: &WikiPageId,
        limit: usize,
    ) -> Result<Vec<GraphNeighbor>> {
        let shared = Arc::clone(&self.conn);
        let page_id = page_id.clone();
        spawn_blocking(move || {
            let conn = shared
                .lock()
                .map_err(|poisoned| WikiError::Storage(poisoned.to_string()))?;
            graph_expand_from_page(&conn, &page_id, limit)
        })
        .await
        .map_err(|e| WikiError::Storage(format!("graph expansion task failed: {e}")))?
    }
}

#[async_trait]
impl FullTextSearch for SqliteFullTextSearch {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        if !self.full_text_enabled {
            return Err(WikiError::Config(
                "full-text search is disabled in config (search.full_text = false)".into(),
            ));
        }
        let fts_query = self.analyzer.fts_query(query);
        if fts_query.is_empty() {
            return Ok(Vec::new());
        }
        let shared = Arc::clone(&self.conn);
        let rows = spawn_blocking(move || {
            let conn = shared
                .lock()
                .map_err(|poisoned| WikiError::Storage(poisoned.to_string()))?;
            if !search_index::probe_fts5(&conn) {
                return Err(WikiError::Index(search_index::FTS5_UNAVAILABLE.to_owned()));
            }
            search_index::search_index(&conn, &fts_query, limit)
        })
        .await
        .map_err(|e| WikiError::Storage(format!("search task failed: {e}")))??;
        Ok(rows
            .into_iter()
            .map(|row| SearchHit {
                page_id: row.page_id,
                slug: row.slug,
                title: row.title,
                heading_path: row.heading_path,
                snippet: row.snippet,
                rank: row.rank,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_wiki_core::hash::sha256_hex;
    use llm_wiki_storage::WikiPageRecord;
    use llm_wiki_storage::{
        activate_build_with_search_index, default_tokenizer, persist_generation, start_build,
        BuildDraft,
    };

    fn page(slug: &str, title: &str, content: &str) -> WikiPageRecord {
        WikiPageRecord {
            page_id: WikiPageId::generate(),
            slug: slug.to_owned(),
            title: title.to_owned(),
            category: "concepts".into(),
            language: "en".into(),
            body_hash: sha256_hex(content.as_bytes()),
            content: content.to_owned(),
            knowledge_refs: Vec::new(),
            citations: Vec::new(),
            links: Vec::new(),
        }
    }

    fn published_conn() -> llm_wiki_storage::Connection {
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        persist_generation(
            &mut conn,
            &build,
            &[
                page(
                    "streaming",
                    "Streaming Processing",
                    "# Streaming Processing\n\n## Checkpoints\n\n检查点默认每 30 秒持久化一次。\n",
                ),
                page(
                    "sso",
                    "Identity & Access",
                    "# Identity & Access\n\n企业可以通过 SAML 或 OIDC 配置单点登录（SSO）。\n",
                ),
            ],
        )
        .unwrap();
        activate_build_with_search_index(&mut conn, &build, default_tokenizer()).unwrap();
        conn
    }

    // ---- analyzer ----

    #[test]
    fn analyzer_normalizes_case_nfkc_and_width() {
        let analyzer = TextAnalyzer;
        assert_eq!(analyzer.analyze("Access Tokens"), vec!["access", "tokens"]);
        assert_eq!(analyzer.analyze("ＡＰＩ ４２９"), vec!["api", "429"]);
        assert_eq!(
            analyzer.analyze("single sign-on (SSO)"),
            vec!["single", "sign", "on", "sso"]
        );
    }

    #[test]
    fn analyzer_splits_han_into_unigrams_and_bigrams_and_keeps_latin_words() {
        let analyzer = TextAnalyzer;
        assert_eq!(
            analyzer.analyze("检查点"),
            vec!["检", "检查", "查", "查点", "点"]
        );
        assert_eq!(analyzer.analyze("SSO登录"), vec!["sso", "登", "登录", "录"]);
        assert_eq!(
            analyzer.analyze("HMAC webhook 签名"),
            vec!["hmac", "webhook", "签", "签名", "名"]
        );
    }

    #[test]
    fn fts_query_builds_quoted_or_expression_and_is_never_syntax() {
        let analyzer = TextAnalyzer;
        assert_eq!(
            analyzer.fts_query("SSO 登录"),
            "\"sso\" OR \"登\" OR \"登录\" OR \"录\""
        );
        // Deduplication keeps repeated terms from bloating the expression.
        assert_eq!(
            analyzer.fts_query("检查 检查"),
            "\"检\" OR \"检查\" OR \"查\""
        );
        assert_eq!(analyzer.fts_query("  。！ "), "", "punctuation-only input");
        // FTS5 syntax characters in input cannot form operators because every
        // term is quoted (and the analyzer would strip them anyway).
        assert_eq!(
            analyzer.fts_query("NOT OR NEAR"),
            "\"not\" OR \"or\" OR \"near\""
        );
    }

    // ---- FullTextSearch ----

    #[tokio::test]
    async fn search_returns_ranked_section_hits_of_the_active_generation() {
        let search = SqliteFullTextSearch::new(published_conn(), true);
        assert!(search.has_published().unwrap());

        let hits = search.search("检查点", 10).await.unwrap();
        assert_eq!(hits[0].slug, "streaming");
        assert_eq!(
            hits[0].heading_path,
            vec!["Streaming Processing", "Checkpoints"]
        );
        assert!(!hits[0].snippet.is_empty());
        assert!(hits[0].rank <= 0.0);
        assert_eq!(hits[0].title, "Streaming Processing");

        let hits = search.search("SSO 登录", 10).await.unwrap();
        assert_eq!(hits[0].slug, "sso");

        // Limit is passed through to FTS5.
        let hits = search
            .search("检查点 单点登录 streaming sso", 1)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn search_errors_without_a_published_generation() {
        let search = SqliteFullTextSearch::new(llm_wiki_storage::open_in_memory().unwrap(), true);
        assert!(!search.has_published().unwrap());
        let err = search.search("anything", 10).await.unwrap_err();
        assert!(err.to_string().contains("nothing published"), "{err}");
    }

    #[tokio::test]
    async fn disabled_full_text_is_a_clear_config_error() {
        let search = SqliteFullTextSearch::new(published_conn(), false);
        let err = search.search("检查点", 10).await.unwrap_err();
        assert!(
            err.to_string().contains("full-text search is disabled"),
            "{err}"
        );
        assert_eq!(err.exit_code(), 2, "config errors exit 2 (PRD §34)");
    }

    // ---- Graph exploration ----

    #[tokio::test]
    async fn graph_exploration_returns_related_pages_of_the_active_generation() {
        let mut conn = llm_wiki_storage::open_in_memory().unwrap();
        let build = start_build(&mut conn, &BuildDraft::default()).unwrap();
        let target = page("sso", "Identity & Access", "# Identity & Access\n\nbody\n");
        let target_id = target.page_id.clone();
        let mut overview = page("overview", "Overview", "# Overview\n\nbody\n");
        overview.links = vec![llm_wiki_storage::PageLinkRecord {
            to_page_id: target_id.clone(),
            target_title: "Identity & Access".into(),
        }];
        let overview_id = overview.page_id.clone();
        persist_generation(&mut conn, &build, &[target, overview]).unwrap();
        activate_build_with_search_index(&mut conn, &build, default_tokenizer()).unwrap();

        let graph = SqliteGraphExploration::new(conn);
        let neighbors = graph
            .neighbors_of_page(&overview_id, llm_wiki_storage::EXPAND_MAX_NODES)
            .await
            .unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].label, "Identity & Access");
        assert_eq!(neighbors[0].relation, "links_to");
        assert_eq!(
            neighbors[0].direction,
            llm_wiki_storage::NeighborDirection::Outgoing
        );

        // The target page sees the link as a backlink.
        let back = graph
            .neighbors_of_page(&target_id, llm_wiki_storage::EXPAND_MAX_NODES)
            .await
            .unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(
            back[0].direction,
            llm_wiki_storage::NeighborDirection::Incoming
        );

        // An unbuilt page id has no node and therefore no neighbors.
        let empty = graph
            .neighbors_of_page(&WikiPageId::generate(), 10)
            .await
            .unwrap();
        assert!(empty.is_empty());
    }
}
