use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use soma_core::SomaError;

/// Snapshot metadata.
#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotMeta {
    pub version: u32,
    pub timestamp: String,
    pub node_count: usize,
    pub edge_count: usize,
}

/// Serializable graph data for snapshot.
#[derive(Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub meta: SnapshotMeta,
    pub graph_json: String,
    pub hdc_json: Option<String>,
}

/// Writes compressed snapshots.
pub struct SnapshotWriter;

impl SnapshotWriter {
    /// Write a zstd-compressed snapshot.
    pub fn write(
        path: &Path,
        graph_data: &str,
        hdc_data: Option<&str>,
        node_count: usize,
        edge_count: usize,
    ) -> Result<(), SomaError> {
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                if e.kind() != std::io::ErrorKind::AlreadyExists {
                    return Err(e.into());
                }
            }
        }

        let snapshot = GraphSnapshot {
            meta: SnapshotMeta {
                version: 1,
                timestamp: chrono::Utc::now().to_rfc3339(),
                node_count,
                edge_count,
            },
            graph_json: graph_data.to_string(),
            hdc_json: hdc_data.map(|s| s.to_string()),
        };

        let json =
            serde_json::to_vec(&snapshot).map_err(|e| SomaError::Serialization(e.to_string()))?;

        // Compress with zstd (level 3 — good balance speed/compression)
        let compressed = zstd::encode_all(json.as_slice(), 3)
            .map_err(|e| SomaError::Snapshot(format!("zstd compress: {}", e)))?;

        // Atomic write: write to temp file, then rename
        let temp_path = path.with_extension("soma.tmp");
        {
            let mut file = File::create(&temp_path)?;
            // Write magic bytes for format detection
            file.write_all(b"SOMA")?;
            file.write_all(&1u32.to_le_bytes())?; // version
            file.write_all(&(compressed.len() as u64).to_le_bytes())?;
            file.write_all(&compressed)?;
            file.flush()?;
            file.sync_all()
                .map_err(|e| SomaError::Snapshot(format!("snapshot fsync: {}", e)))?;
        }
        fs::rename(&temp_path, path)?;

        Ok(())
    }
}

/// Reads compressed snapshots.
pub struct SnapshotReader;

impl SnapshotReader {
    /// Read a zstd-compressed snapshot.
    pub fn read(path: &Path) -> Result<GraphSnapshot, SomaError> {
        let mut file = File::open(path)?;

        // Read magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"SOMA" {
            return Err(SomaError::Snapshot("invalid magic bytes".to_string()));
        }

        // Read version
        let mut version_buf = [0u8; 4];
        file.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version != 1 {
            return Err(SomaError::Snapshot(format!(
                "unsupported snapshot version: {}",
                version
            )));
        }

        // Read compressed data length
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let compressed_len = u64::from_le_bytes(len_buf) as usize;

        // Security: reject absurdly large snapshots (1GB max)
        if compressed_len > 1_073_741_824 {
            return Err(SomaError::InputTooLarge {
                max: 1_073_741_824,
                got: compressed_len,
            });
        }

        // Read compressed data
        let mut compressed = vec![0u8; compressed_len];
        file.read_exact(&mut compressed)?;

        // Decompress
        let decompressed = zstd::decode_all(compressed.as_slice())
            .map_err(|e| SomaError::Snapshot(format!("zstd decompress: {}", e)))?;

        // Parse JSON
        let snapshot: GraphSnapshot = serde_json::from_slice(&decompressed)
            .map_err(|e| SomaError::Serialization(e.to_string()))?;

        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip() {
        let path = std::env::temp_dir().join(format!("soma_test_snap_{}.soma", std::process::id()));
        let _ = fs::remove_file(&path);

        let graph_data = r#"{"nodes":[],"edges":[]}"#;
        SnapshotWriter::write(&path, graph_data, None, 0, 0).unwrap();

        let snapshot = SnapshotReader::read(&path).unwrap();
        assert_eq!(snapshot.meta.version, 1);
        assert_eq!(snapshot.meta.node_count, 0);
        assert_eq!(snapshot.graph_json, graph_data);
        assert!(snapshot.hdc_json.is_none());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn invalid_magic_rejected() {
        let path =
            std::env::temp_dir().join(format!("soma_test_bad_snap_{}.soma", std::process::id()));
        fs::write(
            &path,
            b"BADM\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();

        let result = SnapshotReader::read(&path);
        assert!(result.is_err());

        let _ = fs::remove_file(&path);
    }
}
