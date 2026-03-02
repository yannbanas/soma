//! # soma-store — Crash-Safe Persistence
//!
//! WAL (Write-Ahead Log) + zstd-compressed snapshots.
//! No SQLite, no ORM. Append-only journal with periodic snapshots.
//! Recovery: load last snapshot + replay WAL entries.

mod wal;
mod snapshot;
mod store;

pub use wal::{WalEntry, WalWriter, WalReader};
pub use snapshot::{SnapshotWriter, SnapshotReader};
pub use store::Store;
