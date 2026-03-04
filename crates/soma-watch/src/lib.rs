//! # soma-watch — File watcher for auto-ingest
//!
//! Monitors a directory (optionally recursive) and auto-ingests new/modified files
//! into the SOMA graph via `IngestPipeline`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::{mpsc, RwLock};
use tracing::info;

use soma_graph::StigreGraph;
use soma_ingest::{IngestPipeline, IngestSource};

/// File watcher that auto-ingests on changes.
pub struct FileWatcher {
    path: PathBuf,
    recursive: bool,
    debounce: Duration,
}

impl FileWatcher {
    pub fn new(path: PathBuf, recursive: bool, debounce_secs: u64) -> Self {
        FileWatcher {
            path,
            recursive,
            debounce: Duration::from_secs(debounce_secs),
        }
    }

    /// Run the watcher loop. Blocks until Ctrl+C.
    pub async fn run(
        &self,
        graph: Arc<RwLock<StigreGraph>>,
        pipeline: IngestPipeline,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (tx, mut rx) = mpsc::channel::<PathBuf>(100);

        let mode = if self.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        let tx_clone = tx.clone();
        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    use notify::EventKind::*;
                    match event.kind {
                        Create(_) | Modify(_) => {
                            for path in event.paths {
                                if is_ingestable(&path) {
                                    let _ = tx_clone.blocking_send(path);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            },
            Config::default(),
        )?;

        watcher.watch(&self.path, mode)?;
        info!(
            "[watch] Watching {} (recursive={})",
            self.path.display(),
            self.recursive
        );

        let debounce = self.debounce;
        loop {
            tokio::select! {
                Some(path) = rx.recv() => {
                    // Debounce: wait a bit for more events
                    tokio::time::sleep(debounce).await;
                    // Drain any pending events for the same or other files
                    let mut paths = vec![path];
                    while let Ok(p) = rx.try_recv() {
                        if !paths.contains(&p) {
                            paths.push(p);
                        }
                    }

                    for p in &paths {
                        let source = IngestSource::File(p.clone());
                        let source_name = p.display().to_string();
                        let mut g = graph.write().await;
                        match pipeline.ingest(&source, &mut g, &source_name) {
                            Ok(result) => {
                                info!(
                                    "[watch] Ingested {} → {} nodes, {} edges",
                                    p.display(),
                                    result.nodes_created,
                                    result.edges_created,
                                );
                            }
                            Err(e) => {
                                tracing::warn!("[watch] Failed to ingest {}: {}", p.display(), e);
                            }
                        }
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    info!("[watch] Shutting down");
                    break;
                }
            }
        }

        Ok(())
    }
}

/// Check if a path has an ingestable extension.
fn has_ingestable_extension(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("txt" | "md" | "json" | "csv" | "log" | "pdf" | "rs" | "py" | "toml" | "yaml" | "yml")
    )
}

/// Check if a file is something we can ingest (exists + right extension).
fn is_ingestable(path: &Path) -> bool {
    path.is_file() && has_ingestable_extension(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingestable_extensions() {
        assert!(has_ingestable_extension(Path::new("test.md")));
        assert!(has_ingestable_extension(Path::new("test.txt")));
        assert!(has_ingestable_extension(Path::new("test.json")));
        assert!(!has_ingestable_extension(Path::new("test.exe")));
        assert!(!has_ingestable_extension(Path::new("test.png")));
    }
}
