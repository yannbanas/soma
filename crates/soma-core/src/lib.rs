//! # soma-core — Fundamental types for SOMA
//!
//! Zero internal dependencies. Defines NodeId, EdgeId, Channel, SomaNode, StigreEdge, SomaQuery.
//! All types are type-safe, deterministic, and serializable.

mod ids;
mod channel;
mod node;
mod edge;
mod query;
mod error;
mod config;
mod hybrid;

pub use ids::{NodeId, EdgeId};
pub use channel::Channel;
pub use node::{SomaNode, NodeKind};
pub use edge::{StigreEdge, Provenance};
pub use query::{SomaQuery, QueryResult};
pub use error::SomaError;
pub use config::{SomaConfig, LlmSection};
pub use hybrid::{rrf_merge, rrf_merge_with_sources, rrf_merge_with_specificity, fuzzy_label_search, rerank_temporal, mmr_diversify, HybridResult};
