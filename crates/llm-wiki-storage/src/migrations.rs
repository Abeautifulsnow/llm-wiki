//! Embedded, ordered schema migrations. Version N is `MIGRATIONS[N-1]`;
//! applied versions are tracked in `PRAGMA user_version`.

pub const MIGRATIONS: &[&str] = &[include_str!("../migrations/0001_init.sql")];
