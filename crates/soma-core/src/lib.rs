//! # soma-core — Fundamental types for SOMA
//!
//! Zero internal dependencies. Defines NodeId, EdgeId, Channel, SomaNode, StigreEdge, SomaQuery.
//! All types are type-safe, deterministic, and serializable.

mod channel;
mod config;
mod edge;
mod error;
mod hybrid;
mod ids;
mod node;
mod query;

pub use channel::Channel;
pub use config::{LlmSection, SomaConfig};
pub use edge::{Provenance, StigreEdge};
pub use error::SomaError;
pub use hybrid::{
    fuzzy_label_search, mmr_diversify, rerank_temporal, rrf_merge, rrf_merge_with_sources,
    rrf_merge_with_specificity, HybridResult,
};
pub use ids::{EdgeId, NodeId};
pub use node::{NodeKind, SomaNode};
pub use query::{QueryResult, SomaQuery};
