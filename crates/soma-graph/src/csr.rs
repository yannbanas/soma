//! CSR-style adjacency cache for fast filtered traversal.
//!
//! Precomputes outgoing edges with effective intensities at a given time,
//! avoiding repeated `effective_intensity(now)` calls during BFS/PPR.
//! Invalidated on any mutation (upsert_edge, prune, remove_node).

use chrono::{DateTime, Utc};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

use soma_core::{Channel, SomaNode, StigreEdge};

/// A precomputed outgoing edge with its effective intensity already calculated.
#[derive(Debug, Clone)]
pub struct CachedEdge {
    pub target: NodeIndex,
    pub channel: Channel,
    pub intensity: f32,
    pub confidence: f32,
    pub uses: u32,
    pub last_touch: DateTime<Utc>,
}

/// CSR-like adjacency cache. Maps NodeIndex → Vec<CachedEdge>.
/// Built lazily from the petgraph DiGraph, invalidated on mutation.
pub struct AdjacencyCache {
    /// Precomputed adjacency lists indexed by NodeIndex.index()
    adj: Vec<Vec<CachedEdge>>,
    /// Timestamp when the cache was built
    built_at: DateTime<Utc>,
    /// Whether the cache is still valid
    valid: bool,
}

impl AdjacencyCache {
    /// Create an empty (invalid) cache.
    pub fn new() -> Self {
        AdjacencyCache {
            adj: Vec::new(),
            built_at: Utc::now(),
            valid: false,
        }
    }

    /// Build/rebuild the cache from the graph.
    pub fn rebuild(&mut self, graph: &DiGraph<SomaNode, StigreEdge>) {
        let now = Utc::now();
        let n = graph.node_count();
        self.adj.clear();
        self.adj.resize_with(n, Vec::new);

        for edge_ref in graph.edge_references() {
            let src = edge_ref.source().index();
            let w = edge_ref.weight();
            let intensity = w.effective_intensity(now);

            if src < n {
                self.adj[src].push(CachedEdge {
                    target: edge_ref.target(),
                    channel: w.channel,
                    intensity,
                    confidence: w.confidence,
                    uses: w.uses,
                    last_touch: w.last_touch,
                });
            }
        }

        // Sort each adjacency list by intensity descending for best-first traversal
        for list in &mut self.adj {
            list.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap_or(std::cmp::Ordering::Equal));
        }

        self.built_at = now;
        self.valid = true;
    }

    /// Invalidate the cache (called on mutation).
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Check if cache is valid.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Get outgoing edges for a node (returns empty if index out of range).
    pub fn outgoing(&self, idx: NodeIndex) -> &[CachedEdge] {
        let i = idx.index();
        if i < self.adj.len() {
            &self.adj[i]
        } else {
            &[]
        }
    }

    /// Get filtered outgoing edges for a node.
    pub fn outgoing_filtered<'a>(
        &'a self,
        idx: NodeIndex,
        channels: &'a [Channel],
        min_intensity: f32,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
    ) -> impl Iterator<Item = &'a CachedEdge> + 'a {
        self.outgoing(idx).iter().filter(move |e| {
            // Channel filter
            let ch_ok = if channels.is_empty() {
                !matches!(e.channel, Channel::Alarm)
            } else {
                channels.contains(&e.channel)
            };
            if !ch_ok {
                return false;
            }

            // Intensity filter (edges are sorted desc, so we could early-break in caller)
            if e.intensity < min_intensity {
                return false;
            }

            // Temporal filters
            if let Some(s) = since {
                if e.last_touch < s {
                    return false;
                }
            }
            if let Some(u) = until {
                if e.last_touch > u {
                    return false;
                }
            }

            true
        })
    }

    /// Total cached edges (for stats).
    pub fn total_edges(&self) -> usize {
        self.adj.iter().map(|v| v.len()).sum()
    }

    /// How many nodes are cached.
    pub fn node_count(&self) -> usize {
        self.adj.len()
    }
}

impl Default for AdjacencyCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::DiGraph;
    use soma_core::{NodeKind, SomaNode, StigreEdge, NodeId};

    fn make_test_graph() -> DiGraph<SomaNode, StigreEdge> {
        let mut g = DiGraph::new();
        let a = g.add_node(SomaNode::new("test", "A", NodeKind::Entity));
        let b = g.add_node(SomaNode::new("test", "B", NodeKind::Entity));
        let c = g.add_node(SomaNode::new("test", "C", NodeKind::Entity));

        let id_a = NodeId::from_label("test:A");
        let id_b = NodeId::from_label("test:B");
        let id_c = NodeId::from_label("test:C");

        g.add_edge(a, b, StigreEdge::new(id_a, id_b, Channel::Trail, 0.9, "test".into()));
        g.add_edge(a, c, StigreEdge::new(id_a, id_c, Channel::Causal, 0.7, "test".into()));
        g.add_edge(b, c, StigreEdge::new(id_b, id_c, Channel::Trail, 0.5, "test".into()));
        g
    }

    #[test]
    fn cache_build_and_query() {
        let g = make_test_graph();
        let mut cache = AdjacencyCache::new();
        assert!(!cache.is_valid());

        cache.rebuild(&g);
        assert!(cache.is_valid());
        assert_eq!(cache.node_count(), 3);
        assert_eq!(cache.total_edges(), 3);

        // Node A (index 0) has 2 outgoing edges
        let a_out = cache.outgoing(NodeIndex::new(0));
        assert_eq!(a_out.len(), 2);
        // Sorted by intensity desc: Trail(0.9) then Causal(0.7)
        assert!(a_out[0].intensity >= a_out[1].intensity);
    }

    #[test]
    fn cache_filtered_query() {
        let g = make_test_graph();
        let mut cache = AdjacencyCache::new();
        cache.rebuild(&g);

        // Filter by Trail channel only
        let trail_edges: Vec<_> = cache.outgoing_filtered(
            NodeIndex::new(0), &[Channel::Trail], 0.0, None, None,
        ).collect();
        assert_eq!(trail_edges.len(), 1);
        assert_eq!(trail_edges[0].channel, Channel::Trail);

        // Filter by min intensity 0.8 — only Trail(0.9) passes
        let high_edges: Vec<_> = cache.outgoing_filtered(
            NodeIndex::new(0), &[], 0.8, None, None,
        ).collect();
        assert_eq!(high_edges.len(), 1);
    }

    #[test]
    fn cache_invalidation() {
        let g = make_test_graph();
        let mut cache = AdjacencyCache::new();
        cache.rebuild(&g);
        assert!(cache.is_valid());

        cache.invalidate();
        assert!(!cache.is_valid());
    }

    #[test]
    fn cache_empty_node() {
        let g = make_test_graph();
        let mut cache = AdjacencyCache::new();
        cache.rebuild(&g);

        // Node C (index 2) has no outgoing edges
        let c_out = cache.outgoing(NodeIndex::new(2));
        assert!(c_out.is_empty());
    }
}
