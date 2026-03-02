use serde::{Deserialize, Serialize};

/// Statistics about the current state of the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub nodes: usize,
    pub edges: usize,
    pub dead_edges: usize,
    pub workspace: String,
    pub avg_intensity: f32,
}
