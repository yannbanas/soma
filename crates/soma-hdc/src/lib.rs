//! # soma-hdc — Hyperdimensional Computing Semantic Index
//!
//! Random Indexing with D=10000 dimensions.
//! Distributional semantics: tokens represented by context co-occurrence.
//! TF-IDF weighted, sparse base vectors, mmap-ready binary storage.

mod engine;
mod storage;

pub use engine::{HdcEngine, HdcSnapshot};
