//! # soma-llm — Ollama LLM client for SOMA
//!
//! Provides two capabilities:
//! 1. **L2 Extraction**: extract knowledge triplets via Ollama generate API
//! 2. **Neural Embeddings**: get dense vector embeddings via Ollama embedding API
//!
//! Both features are opt-in (disabled by default) with graceful degradation.

mod client;
mod prompt;
mod types;

pub use client::{LlmError, OllamaClient};
pub use prompt::build_extraction_prompt;
pub use types::LlmTriplet;
