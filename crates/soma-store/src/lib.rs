//! # soma-store — Crash-Safe Persistence
//!
//! WAL (Write-Ahead Log) + zstd-compressed snapshots.
//! No SQLite, no ORM. Append-only journal with periodic snapshots.
//! Recovery: load last snapshot + replay WAL entries.

mod snapshot;
mod store;
mod wal;

pub use snapshot::{SnapshotReader, SnapshotWriter};
pub use store::Store;
pub use wal::{WalEntry, WalReader, WalWriter};
