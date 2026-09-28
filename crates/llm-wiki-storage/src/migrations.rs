//! Embedded, ordered schema migrations. Version N is `MIGRATIONS[N-1]`;
//! applied versions are tracked in `PRAGMA user_version`.

pub const MIGRATIONS: &[&str] = &[
    include_str!("../migrations/0001_init.sql"),
    include_str!("../migrations/0002_analysis.sql"),
    include_str!("../migrations/0003_wiki.sql"),
    include_str!("../migrations/0004_publish.sql"),
    include_str!("../migrations/0005_cache.sql"),
];
