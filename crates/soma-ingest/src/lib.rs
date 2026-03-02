//! # soma-ingest — Universal Ingestion Pipeline
//!
//! Three-level extraction: L0 (heuristics), L1 (regex patterns), L2 (optional LLM).
//! Supports: RawText, File, ClaudeConversation, OllamaSession, Structured data.
//! Intelligent chunking with overlap for context preservation.

mod chunker;
mod patterns;
mod pipeline;
mod source;

pub use chunker::Chunker;
pub use patterns::PatternExtractor;
pub use pipeline::{IngestPipeline, IngestResult, Triplet};
pub use source::IngestSource;
