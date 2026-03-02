use thiserror::Error;

use crate::ids::{EdgeId, NodeId};

/// Unified error type for all SOMA operations.
#[derive(Debug, Error)]
pub enum SomaError {
    #[error("node not found: {0}")]
    NodeNotFound(NodeId),

    #[error("edge not found: {0}")]
    EdgeNotFound(EdgeId),

    #[error("node label not found: {0}")]
    LabelNotFound(String),

    #[error("invalid channel: {0}")]
    InvalidChannel(String),

    #[error("workspace not found: {0}")]
    WorkspaceNotFound(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("WAL corrupted: {0}")]
    WalCorrupted(String),

    #[error("snapshot error: {0}")]
    Snapshot(String),

    #[error("ingest error: {0}")]
    Ingest(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("MCP protocol error: {0}")]
    Mcp(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("path traversal blocked: {0}")]
    PathTraversal(String),

    #[error("input too large: max {max} bytes, got {got}")]
    InputTooLarge { max: usize, got: usize },
}

/// Result type alias for SOMA operations.
#[allow(dead_code)]
pub type SomaResult<T> = Result<T, SomaError>;
