use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use soma_core::{EdgeId, NodeId, SomaError, SomaNode, StigreEdge};

/// A single entry in the Write-Ahead Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    NodeUpsert(SomaNode),
    EdgeUpsert(StigreEdge),
    EdgeReinforce {
        id: EdgeId,
        delta: f32,
        ts: DateTime<Utc>,
    },
    EdgePrune(EdgeId),
    NodeArchive(NodeId),
    ConsolidationEvent {
        ts: DateTime<Utc>,
        episodes_merged: u32,
        concepts_created: u32,
    },
    /// Free-form log entry for corrections, merges, etc.
    Custom(String),
}

/// Append-only WAL writer with fsync for crash safety.
pub struct WalWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    entry_count: u64,
}

impl WalWriter {
    /// Open or create a WAL file. Appends to existing data.
    pub fn open(path: &Path) -> Result<Self, SomaError> {
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                if e.kind() != std::io::ErrorKind::AlreadyExists {
                    return Err(e.into());
                }
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        // Count existing entries for stats
        let entry_count = Self::count_entries(path)?;

        Ok(WalWriter {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
            entry_count,
        })
    }

    /// Append an entry to the WAL. Fsyncs for durability.
    pub fn append(&mut self, entry: &WalEntry) -> Result<(), SomaError> {
        let data = serde_json::to_vec(entry)
            .map_err(|e| SomaError::Serialization(e.to_string()))?;

        // Security: bound entry size to prevent DoS via giant entries
        if data.len() > 10 * 1024 * 1024 {
            return Err(SomaError::InputTooLarge {
                max: 10 * 1024 * 1024,
                got: data.len(),
            });
        }

        // Write length-prefixed entry: [len:u32][json_data]
        let len = data.len() as u32;
        self.writer
            .write_all(&len.to_le_bytes())
            .map_err(|e| SomaError::Store(format!("WAL write len: {}", e)))?;
        self.writer
            .write_all(&data)
            .map_err(|e| SomaError::Store(format!("WAL write data: {}", e)))?;
        self.writer
            .flush()
            .map_err(|e| SomaError::Store(format!("WAL flush: {}", e)))?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| SomaError::Store(format!("WAL fsync: {}", e)))?;

        self.entry_count += 1;
        Ok(())
    }

    /// Number of entries written.
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Truncate WAL (after successful snapshot).
    pub fn truncate(&mut self) -> Result<(), SomaError> {
        let file = File::create(&self.path)?;
        file.sync_all()
            .map_err(|e| SomaError::Store(format!("WAL truncate fsync: {}", e)))?;
        drop(std::mem::replace(
            &mut self.writer,
            BufWriter::new(file),
        ));
        self.entry_count = 0;
        Ok(())
    }

    /// Count entries in existing WAL file.
    fn count_entries(path: &Path) -> Result<u64, SomaError> {
        if !path.exists() {
            return Ok(0);
        }
        let reader = WalReader::open(path)?;
        Ok(reader.count() as u64)
    }
}

/// WAL reader — reads entries sequentially.
pub struct WalReader {
    reader: BufReader<File>,
}

impl WalReader {
    pub fn open(path: &Path) -> Result<Self, SomaError> {
        let file = File::open(path)?;
        Ok(WalReader {
            reader: BufReader::new(file),
        })
    }
}

impl Iterator for WalReader {
    type Item = WalEntry;

    fn next(&mut self) -> Option<Self::Item> {
        // Read length prefix
        let mut len_buf = [0u8; 4];
        if self.reader.read_exact(&mut len_buf).is_err() {
            return None;
        }
        let len = u32::from_le_bytes(len_buf) as usize;

        // Security: reject absurdly large entries
        if len > 10 * 1024 * 1024 {
            return None;
        }

        // Read entry data
        let mut data = vec![0u8; len];
        if self.reader.read_exact(&mut data).is_err() {
            return None;
        }

        serde_json::from_slice(&data).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::{Channel, NodeKind};

    fn temp_wal_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("soma_test_wal");
        fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{}_{}.wal", name, std::process::id()))
    }

    #[test]
    fn wal_write_and_read() {
        let path = temp_wal_path("rw");
        let _ = fs::remove_file(&path); // clean up

        // Write entries
        {
            let mut writer = WalWriter::open(&path).unwrap();
            let node = SomaNode::new("test", "ChromoQ", NodeKind::Entity);
            writer.append(&WalEntry::NodeUpsert(node)).unwrap();

            let edge = StigreEdge::new(
                NodeId::from_label("test:ChromoQ"),
                NodeId::from_label("test:EGFP"),
                Channel::DerivesDe,
                0.95,
                "test".to_string(),
            );
            writer.append(&WalEntry::EdgeUpsert(edge)).unwrap();
            assert_eq!(writer.entry_count(), 2);
        }

        // Read entries
        {
            let reader = WalReader::open(&path).unwrap();
            let entries: Vec<WalEntry> = reader.collect();
            assert_eq!(entries.len(), 2);

            match &entries[0] {
                WalEntry::NodeUpsert(n) => assert_eq!(n.label, "ChromoQ"),
                _ => panic!("expected NodeUpsert"),
            }
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn wal_truncate() {
        let path = temp_wal_path("trunc");
        let _ = fs::remove_file(&path);

        let mut writer = WalWriter::open(&path).unwrap();
        let node = SomaNode::new("test", "A", NodeKind::Entity);
        writer.append(&WalEntry::NodeUpsert(node)).unwrap();
        writer.truncate().unwrap();
        assert_eq!(writer.entry_count(), 0);

        let reader = WalReader::open(&path).unwrap();
        let entries: Vec<WalEntry> = reader.collect();
        assert_eq!(entries.len(), 0);

        let _ = fs::remove_file(&path);
    }
}
