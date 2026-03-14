//! # soma-graph — Living Knowledge Graph
//!
//! Built on petgraph. Supports idempotent upserts, lazy evaporation,
//! weighted traversal (Dijkstra by effective intensity), and graph statistics.

mod graph;
mod traversal;
mod stats;
pub mod csr;
pub mod community;

pub use graph::StigreGraph;
pub use stats::GraphStats;
pub use traversal::extract_query_entities;
pub use community::{detect_communities, CommunityResult};
