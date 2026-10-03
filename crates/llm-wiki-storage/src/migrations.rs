//! Embedded, ordered schema migrations. Version N is `MIGRATIONS[N-1]`;
//! applied versions are tracked in `PRAGMA user_version`.

pub const MIGRATIONS: &[&str] = &[
    include_str!("../migrations/0001_init.sql"),
    include_str!("../migrations/0002_analysis.sql"),
    include_str!("../migrations/0003_wiki.sql"),
    include_str!("../migrations/0004_publish.sql"),
    include_str!("../migrations/0005_cache.sql"),
    include_str!("../migrations/0006_plan_decisions.sql"),
    include_str!("../migrations/0007_page_id_map.sql"),
    include_str!("../migrations/0008_search.sql"),
    include_str!("../migrations/0009_graph.sql"),
    include_str!("../migrations/0010_insights.sql"),
];
