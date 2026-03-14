//! # soma-graph — Living Knowledge Graph
//!
//! Built on petgraph. Supports idempotent upserts, lazy evaporation,
//! weighted traversal (Dijkstra by effective intensity), and graph statistics.

pub mod community;
pub mod csr;
mod graph;
mod stats;
mod traversal;

pub use community::{detect_communities, CommunityResult};
pub use graph::StigreGraph;
pub use stats::GraphStats;
pub use traversal::extract_query_entities;
