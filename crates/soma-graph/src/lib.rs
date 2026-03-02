//! # soma-graph — Living Knowledge Graph
//!
//! Built on petgraph. Supports idempotent upserts, lazy evaporation,
//! weighted traversal (Dijkstra by effective intensity), and graph statistics.

mod graph;
mod traversal;
mod stats;

pub use graph::StigreGraph;
pub use stats::GraphStats;
