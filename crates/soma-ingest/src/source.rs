use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Input sources for SOMA ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IngestSource {
    /// Free text or Markdown
    RawText(String),
    /// File path (auto-detect format)
    File(PathBuf),
    /// Claude conversation JSON export
    ClaudeConversation(PathBuf),
    /// Ollama session log
    OllamaSession { log: String, model: String },
    /// Web URL (fetch + extract)
    Url(String),
    /// Structured data (JSON/TOML/CSV)
    Structured(serde_json::Value),
}

impl IngestSource {
    /// Extract raw text from the source.
    /// Security: validates file paths and bounds input size.
    pub fn to_text(&self) -> Result<String, soma_core::SomaError> {
        match self {
            IngestSource::RawText(text) => {
                // Security: limit input size (10MB max)
                if text.len() > 10 * 1024 * 1024 {
                    return Err(soma_core::SomaError::InputTooLarge {
                        max: 10 * 1024 * 1024,
                        got: text.len(),
                    });
                }
                Ok(text.clone())
            }
            IngestSource::File(path) => {
                // Security: validate path (no traversal)
                let path_str = path.to_string_lossy();
                if path_str.contains("..") {
                    return Err(soma_core::SomaError::PathTraversal(
                        path_str.to_string(),
                    ));
                }
                let content = std::fs::read_to_string(path)?;
                if content.len() > 10 * 1024 * 1024 {
                    return Err(soma_core::SomaError::InputTooLarge {
                        max: 10 * 1024 * 1024,
                        got: content.len(),
                    });
                }
                Ok(content)
            }
            IngestSource::ClaudeConversation(path) => {
                let path_str = path.to_string_lossy();
                if path_str.contains("..") {
                    return Err(soma_core::SomaError::PathTraversal(
                        path_str.to_string(),
                    ));
                }
                let content = std::fs::read_to_string(path)?;
                // Parse Claude JSON and extract messages
                let json: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| soma_core::SomaError::Ingest(e.to_string()))?;

                let mut texts = Vec::new();
                if let Some(messages) = json.as_array() {
                    for msg in messages {
                        if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                            texts.push(text.to_string());
                        }
                    }
                }
                Ok(texts.join("\n\n"))
            }
            IngestSource::OllamaSession { log, .. } => {
                if log.len() > 10 * 1024 * 1024 {
                    return Err(soma_core::SomaError::InputTooLarge {
                        max: 10 * 1024 * 1024,
                        got: log.len(),
                    });
                }
                Ok(log.clone())
            }
            IngestSource::Url(_url) => {
                // URL fetching will be implemented in Phase 3
                Err(soma_core::SomaError::Ingest(
                    "URL ingestion not yet implemented".to_string(),
                ))
            }
            IngestSource::Structured(value) => {
                Ok(serde_json::to_string_pretty(value)
                    .map_err(|e| soma_core::SomaError::Serialization(e.to_string()))?)
            }
        }
    }
}
