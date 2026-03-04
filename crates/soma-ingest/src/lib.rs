//! # soma-ingest — Universal Ingestion Pipeline
//!
//! Four-level extraction:
//! - L0: Heuristics (structured sources)
//! - L1: Regex patterns (domain-specific)
//! - L1.5: Automatic NER + co-occurrence (domain-agnostic)
//! - L2: Optional LLM (Ollama)
//!
//! Supports: RawText, File, ClaudeConversation, OllamaSession, Structured data.
//! Intelligent chunking with overlap for context preservation.

mod chunker;
pub mod ner;
mod patterns;
mod pipeline;
mod source;

pub use chunker::Chunker;
pub use patterns::PatternExtractor;
pub use pipeline::{IngestPipeline, IngestResult, Triplet};
pub use source::IngestSource;
