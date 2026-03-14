use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use soma_core::SomaError;

use crate::snapshot::{SnapshotReader, SnapshotWriter};
use crate::wal::{WalEntry, WalReader, WalWriter};

/// Persistent store managing WAL + snapshots for a workspace.
///
/// Directory layout:
/// ```text
/// <data_dir>/<workspace>/
/// ├── wal.log          # append-only journal (JSON, length-prefixed)
/// ├── snapshot.soma    # last snapshot (zstd-compressed)
/// └── meta.toml        # workspace metadata
/// ```
pub struct Store {
    workspace_dir: PathBuf,
    wal: WalWriter,
    wal_since_snapshot: AtomicU64,
}

impl Store {
    /// Open or create a store for a workspace.
    /// Security: validates path components to prevent directory traversal.
    pub fn open(data_dir: &Path, workspace: &str) -> Result<Self, SomaError> {
        // Security: validate workspace name (no path traversal)
        if workspace.contains("..")
            || workspace.contains('/')
            || workspace.contains('\\')
            || workspace.contains('\0')
        {
            return Err(SomaError::PathTraversal(format!(
                "invalid workspace name: {}",
                workspace
            )));
        }

        let workspace_dir = data_dir.join(workspace);
        // create_dir_all can return AlreadyExists on Windows — ignore it
        if let Err(e) = fs::create_dir_all(&workspace_dir) {
            if e.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(e.into());
            }
        }

        let wal_path = workspace_dir.join("wal.log");
        let wal = WalWriter::open(&wal_path)?;

        Ok(Store {
            workspace_dir,
            wal,
            wal_since_snapshot: AtomicU64::new(0),
        })
    }

    /// Write a WAL entry.
    pub fn write_wal(&mut self, entry: &WalEntry) -> Result<(), SomaError> {
        self.wal.append(entry)?;
        self.wal_since_snapshot.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Read all WAL entries for recovery.
    pub fn read_wal(&self) -> Result<Vec<WalEntry>, SomaError> {
        let wal_path = self.workspace_dir.join("wal.log");
        if !wal_path.exists() {
            return Ok(Vec::new());
        }
        let reader = WalReader::open(&wal_path)?;
        Ok(reader.collect())
    }

    /// Write a snapshot.
    pub fn write_snapshot(
        &mut self,
        graph_data: &str,
        hdc_data: Option<&str>,
        node_count: usize,
        edge_count: usize,
    ) -> Result<(), SomaError> {
        let snap_path = self.workspace_dir.join("snapshot.soma");
        SnapshotWriter::write(&snap_path, graph_data, hdc_data, node_count, edge_count)?;

        // Truncate WAL after successful snapshot
        self.wal.truncate()?;
        self.wal_since_snapshot.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Read the latest snapshot, if it exists.
    pub fn read_snapshot(&self) -> Result<Option<crate::snapshot::GraphSnapshot>, SomaError> {
        let snap_path = self.workspace_dir.join("snapshot.soma");
        if !snap_path.exists() {
            return Ok(None);
        }
        let snapshot = SnapshotReader::read(&snap_path)?;
        Ok(Some(snapshot))
    }

    /// Check if we should create a snapshot (based on WAL entry count).
    pub fn should_snapshot(&self, threshold: u64) -> bool {
        self.wal_since_snapshot.load(Ordering::Relaxed) >= threshold
    }

    /// WAL entries since last snapshot.
    pub fn wal_entries_since_snapshot(&self) -> u64 {
        self.wal_since_snapshot.load(Ordering::Relaxed)
    }

    /// Workspace directory path.
    pub fn workspace_dir(&self) -> &Path {
        &self.workspace_dir
    }

    /// List all workspaces in the data directory.
    pub fn list_workspaces(data_dir: &Path) -> Result<Vec<String>, SomaError> {
        if !data_dir.exists() {
            return Ok(Vec::new());
        }
        let mut workspaces = Vec::new();
        for entry in fs::read_dir(data_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    // Skip hidden directories
                    if !name.starts_with('.') {
                        workspaces.push(name.to_string());
                    }
                }
            }
        }
        workspaces.sort();
        Ok(workspaces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::NodeKind;

    #[test]
    fn store_open_and_write() {
        let dir = std::env::temp_dir().join(format!("soma_store_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut store = Store::open(&dir, "test").unwrap();

        let node = soma_core::SomaNode::new("test", "ChromoQ", NodeKind::Entity);
        store.write_wal(&WalEntry::NodeUpsert(node)).unwrap();

        let entries = store.read_wal().unwrap();
        assert_eq!(entries.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_traversal_blocked() {
        let dir = std::env::temp_dir().join("soma_store_test_sec");
        let result = Store::open(&dir, "../etc/passwd");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, SomaError::PathTraversal(_)));
    }

    #[test]
    fn null_byte_blocked() {
        let dir = std::env::temp_dir().join("soma_store_test_null");
        let result = Store::open(&dir, "test\0evil");
        assert!(result.is_err());
    }

    #[test]
    fn snapshot_roundtrip() {
        let dir = std::env::temp_dir().join(format!("soma_store_snap_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut store = Store::open(&dir, "snap_test").unwrap();

        // Write some WAL entries
        let node = soma_core::SomaNode::new("snap_test", "A", NodeKind::Entity);
        store.write_wal(&WalEntry::NodeUpsert(node)).unwrap();

        // Write snapshot
        store
            .write_snapshot(r#"{"test": true}"#, None, 1, 0)
            .unwrap();

        // WAL should be truncated
        assert_eq!(store.wal_entries_since_snapshot(), 0);

        // Snapshot should be readable
        let snap = store.read_snapshot().unwrap().unwrap();
        assert_eq!(snap.meta.node_count, 1);

        let _ = fs::remove_dir_all(&dir);
    }
}
