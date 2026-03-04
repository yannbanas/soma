//! # soma-bench — Academic Benchmark Suite
//!
//! Evaluation infrastructure for SOMA using standard QA benchmarks
//! (MuSiQue, HotpotQA) and a temporal knowledge benchmark.
//!
//! Metrics: Entity Recall@K, Path Recall, Token F1, Exact Match.

pub mod loader;
pub mod metrics;
pub mod runner;
pub mod temporal;
pub mod ablation;
