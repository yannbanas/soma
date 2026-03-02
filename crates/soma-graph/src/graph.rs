use std::collections::HashMap;

use chrono::Utc;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;

use soma_core::{
    Channel, EdgeId, NodeId, NodeKind, QueryResult, SomaNode, SomaQuery, StigreEdge,
};

use crate::stats::GraphStats;

/// The living knowledge graph.
/// Uses petgraph DiGraph as backend with O(1) lookups by label and NodeId.
pub struct StigreGraph {
    /// Backend directed graph
    inner: DiGraph<SomaNode, StigreEdge>,
    /// O(1) lookup: normalized label → NodeIndex (for deduplication)
    label_idx: HashMap<String, NodeIndex>,
    /// O(1) lookup: NodeId → NodeIndex
    id_idx: HashMap<NodeId, NodeIndex>,
    /// Workspace identifier
    workspace: String,
    /// Global pruning threshold
    prune_threshold: f32,
}

impl StigreGraph {
    /// Create a new empty graph for a workspace.
    pub fn new(workspace: &str, prune_threshold: f32) -> Self {
        StigreGraph {
            inner: DiGraph::new(),
            label_idx: HashMap::new(),
            id_idx: HashMap::new(),
            workspace: workspace.to_string(),
            prune_threshold,
        }
    }

    /// Get workspace name.
    pub fn workspace(&self) -> &str {
        &self.workspace
    }

    /// Get prune threshold.
    pub fn prune_threshold(&self) -> f32 {
        self.prune_threshold
    }

    /// Idempotent node upsert — if label already exists, returns existing ID and updates last_seen.
    /// No duplicates ever.
    pub fn upsert_node(&mut self, label: &str, kind: NodeKind) -> NodeId {
        let node_id = NodeId::from_label(&format!("{}:{}", self.workspace, label));

        if let Some(&idx) = self.id_idx.get(&node_id) {
            // Update last_seen
            self.inner[idx].touch();
            return node_id;
        }

        let node = SomaNode::new(&self.workspace, label, kind);
        let idx = self.inner.add_node(node);
        self.label_idx.insert(label.to_string(), idx);
        self.id_idx.insert(node_id, idx);
        node_id
    }

    /// Upsert node with tags.
    pub fn upsert_node_with_tags(
        &mut self,
        label: &str,
        kind: NodeKind,
        tags: Vec<String>,
    ) -> NodeId {
        let node_id = self.upsert_node(label, kind);
        if let Some(&idx) = self.id_idx.get(&node_id) {
            let node = &mut self.inner[idx];
            // Merge tags (deduplicate)
            for tag in tags {
                if !node.tags.contains(&tag) {
                    node.tags.push(tag);
                }
            }
        }
        node_id
    }

    /// Idempotent edge upsert — if (from, to, channel) already exists, reinforce instead of duplicate.
    pub fn upsert_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        channel: Channel,
        confidence: f32,
        source: &str,
    ) -> Option<EdgeId> {
        let from_idx = self.id_idx.get(&from).copied()?;
        let to_idx = self.id_idx.get(&to).copied()?;
        let now = Utc::now();

        // Search for existing edge with same (from, to, channel)
        let existing = self.find_edge_by_channel(from_idx, to_idx, channel);

        if let Some(edge_idx) = existing {
            let edge = &mut self.inner[edge_idx];
            edge.reinforce(now);
            return Some(edge.id);
        }

        // New edge
        let edge = StigreEdge::new(from, to, channel, confidence, source.to_string());
        let eid = edge.id;
        self.inner.add_edge(from_idx, to_idx, edge);
        Some(eid)
    }

    /// Find an edge between two nodes with a specific channel.
    fn find_edge_by_channel(
        &self,
        from: NodeIndex,
        to: NodeIndex,
        channel: Channel,
    ) -> Option<EdgeIndex> {
        self.inner
            .edges_connecting(from, to)
            .find(|e| e.weight().channel == channel)
            .map(|e| e.id())
    }

    /// Get a node by its ID.
    pub fn get_node(&self, id: NodeId) -> Option<&SomaNode> {
        self.id_idx.get(&id).map(|&idx| &self.inner[idx])
    }

    /// Get a node by label.
    pub fn get_node_by_label(&self, label: &str) -> Option<&SomaNode> {
        self.label_idx.get(label).map(|&idx| &self.inner[idx])
    }

    /// Get node ID by label.
    pub fn node_id_by_label(&self, label: &str) -> Option<NodeId> {
        self.get_node_by_label(label).map(|n| n.id)
    }

    /// Get all nodes (iterator).
    pub fn all_nodes(&self) -> impl Iterator<Item = &SomaNode> {
        self.inner.node_weights()
    }

    /// Get all edges (iterator).
    pub fn all_edges(&self) -> impl Iterator<Item = &StigreEdge> {
        self.inner.edge_weights()
    }

    /// Get mutable reference to an edge by EdgeId.
    pub fn get_edge_mut(&mut self, id: EdgeId) -> Option<&mut StigreEdge> {
        self.inner
            .edge_weights_mut()
            .find(|e| e.id == id)
    }

    /// Get outgoing edges from a node.
    pub fn outgoing_edges(&self, node_id: NodeId) -> Vec<&StigreEdge> {
        let Some(&idx) = self.id_idx.get(&node_id) else {
            return Vec::new();
        };
        self.inner
            .edges(idx)
            .map(|e| e.weight())
            .collect()
    }

    /// Get incoming edges to a node.
    pub fn incoming_edges(&self, node_id: NodeId) -> Vec<&StigreEdge> {
        let Some(&idx) = self.id_idx.get(&node_id) else {
            return Vec::new();
        };
        self.inner
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|e| e.weight())
            .collect()
    }

    /// Direct neighbors filtered by channel and minimum intensity.
    pub fn neighbors(
        &self,
        node_id: NodeId,
        channels: &[Channel],
        min_intensity: f32,
    ) -> Vec<(NodeId, &StigreEdge)> {
        let Some(&idx) = self.id_idx.get(&node_id) else {
            return Vec::new();
        };
        let now = Utc::now();

        self.inner
            .edges(idx)
            .filter(|e| {
                let w = e.weight();
                let ch_ok = channels.is_empty() || channels.contains(&w.channel);
                let int_ok = w.effective_intensity(now) >= min_intensity;
                ch_ok && int_ok
            })
            .map(|e| {
                let target = e.target();
                (self.inner[target].id, e.weight())
            })
            .collect()
    }

    /// BFS/Dijkstra traversal weighted by effective intensity.
    /// Score of a path = product of effective intensities on edges.
    pub fn traverse(&self, query: &SomaQuery) -> Vec<QueryResult> {
        let now = Utc::now();

        // Find start node
        let start_idx = match self.label_idx.get(&query.start) {
            Some(&idx) => idx,
            None => {
                // Try fuzzy match (prefix)
                match self.label_idx.iter().find(|(k, _)| {
                    k.to_lowercase().contains(&query.start.to_lowercase())
                }) {
                    Some((_, &idx)) => idx,
                    None => return Vec::new(),
                }
            }
        };

        let mut results = Vec::new();
        let mut visited = HashMap::new();

        // BFS with intensity-weighted scoring
        let mut queue: Vec<(NodeIndex, Vec<StigreEdge>, f32, u8)> = vec![];

        // Start node is always a result
        let start_node = self.inner[start_idx].clone();
        results.push(QueryResult::new(start_node, Vec::new(), 1.0, 0));
        visited.insert(start_idx, 1.0_f32);

        // Seed the queue with outgoing edges from start
        for edge_ref in self.inner.edges(start_idx) {
            let w = edge_ref.weight();
            if !query.allows_channel(&w.channel) {
                continue;
            }
            let eff = w.effective_intensity(now);
            if eff < query.min_intensity {
                continue;
            }
            // Temporal filter
            if let Some(since) = query.since {
                if w.last_touch < since {
                    continue;
                }
            }
            if let Some(until) = query.until {
                if w.last_touch > until {
                    continue;
                }
            }
            queue.push((edge_ref.target(), vec![w.clone()], eff, 1));
        }

        while let Some((node_idx, path, score, hops)) = queue.pop() {
            if hops > query.max_hops {
                continue;
            }

            // Skip if we already visited with a better score
            if let Some(&prev_score) = visited.get(&node_idx) {
                if prev_score >= score {
                    continue;
                }
            }
            visited.insert(node_idx, score);

            let node = self.inner[node_idx].clone();
            results.push(QueryResult::new(node, path.clone(), score, hops));

            // Expand neighbors
            if hops < query.max_hops {
                for edge_ref in self.inner.edges(node_idx) {
                    let w = edge_ref.weight();
                    if !query.allows_channel(&w.channel) {
                        continue;
                    }
                    let eff = w.effective_intensity(now);
                    if eff < query.min_intensity {
                        continue;
                    }
                    if let Some(since) = query.since {
                        if w.last_touch < since {
                            continue;
                        }
                    }
                    if let Some(until) = query.until {
                        if w.last_touch > until {
                            continue;
                        }
                    }

                    let new_score = score * eff;
                    if new_score < query.min_intensity {
                        continue;
                    }

                    let target = edge_ref.target();
                    if let Some(&prev) = visited.get(&target) {
                        if prev >= new_score {
                            continue;
                        }
                    }

                    let mut new_path = path.clone();
                    new_path.push(w.clone());
                    queue.push((target, new_path, new_score, hops + 1));
                }
            }
        }

        // Sort by score descending, limit results
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(query.limit);
        results
    }

    /// Reinforce an edge by its ID.
    pub fn reinforce_edge(&mut self, edge_id: EdgeId) -> bool {
        let now = Utc::now();
        if let Some(edge) = self.get_edge_mut(edge_id) {
            edge.reinforce(now);
            true
        } else {
            false
        }
    }

    /// Prune dead edges. Returns the number of edges removed.
    pub fn prune_dead_edges(&mut self) -> usize {
        let now = Utc::now();
        let threshold = self.prune_threshold;

        // Collect dead edge indices
        let dead: Vec<EdgeIndex> = self
            .inner
            .edge_indices()
            .filter(|&idx| self.inner[idx].is_dead(threshold, now))
            .collect();

        let count = dead.len();
        // Remove in reverse order to maintain index validity
        for idx in dead.into_iter().rev() {
            self.inner.remove_edge(idx);
        }
        count
    }

    /// Archive orphan nodes (no edges). Returns archived node IDs.
    pub fn archive_orphans(&mut self) -> Vec<NodeId> {
        let orphans: Vec<NodeIndex> = self
            .inner
            .node_indices()
            .filter(|&idx| {
                self.inner.edges(idx).next().is_none()
                    && self
                        .inner
                        .edges_directed(idx, petgraph::Direction::Incoming)
                        .next()
                        .is_none()
            })
            .collect();

        let mut archived = Vec::new();
        for idx in orphans.into_iter().rev() {
            if let Some(node) = self.inner.remove_node(idx) {
                self.label_idx.remove(&node.label);
                self.id_idx.remove(&node.id);
                archived.push(node.id);
            }
        }

        // Rebuild indices after node removal (petgraph swaps indices)
        self.rebuild_indices();
        archived
    }

    /// Remove a node by label (and all its edges). Returns the NodeId if found.
    pub fn remove_node_by_label(&mut self, label: &str) -> Option<NodeId> {
        let idx = self.label_idx.get(label).copied()?;
        let node = self.inner.remove_node(idx)?;
        let node_id = node.id;
        self.label_idx.remove(&node.label);
        self.id_idx.remove(&node.id);
        self.rebuild_indices();
        Some(node_id)
    }

    /// Rebuild lookup indices after structural changes.
    fn rebuild_indices(&mut self) {
        self.label_idx.clear();
        self.id_idx.clear();
        for idx in self.inner.node_indices() {
            let node = &self.inner[idx];
            self.label_idx.insert(node.label.clone(), idx);
            self.id_idx.insert(node.id, idx);
        }
    }

    /// Graph statistics.
    pub fn stats(&self) -> GraphStats {
        let now = Utc::now();
        let total_edges = self.inner.edge_count();
        let dead_edges = self
            .inner
            .edge_weights()
            .filter(|e| e.is_dead(self.prune_threshold, now))
            .count();

        let avg_intensity = if total_edges > 0 {
            self.inner
                .edge_weights()
                .map(|e| e.effective_intensity(now))
                .sum::<f32>()
                / total_edges as f32
        } else {
            0.0
        };

        GraphStats {
            nodes: self.inner.node_count(),
            edges: total_edges,
            dead_edges,
            workspace: self.workspace.clone(),
            avg_intensity,
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Get all labels (for HDC training).
    pub fn all_labels(&self) -> Vec<String> {
        self.inner.node_weights().map(|n| n.label.clone()).collect()
    }

    /// Get internal petgraph (for serialization).
    pub fn inner(&self) -> &DiGraph<SomaNode, StigreEdge> {
        &self.inner
    }

    /// Set internal graph from deserialized data (for store recovery).
    pub fn set_inner(&mut self, graph: DiGraph<SomaNode, StigreEdge>) {
        self.inner = graph;
        self.rebuild_indices();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_deduplicates() {
        let mut g = StigreGraph::new("test", 0.05);
        let id1 = g.upsert_node("ChromoQ", NodeKind::Entity);
        let id2 = g.upsert_node("ChromoQ", NodeKind::Entity);
        assert_eq!(id1, id2);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn different_labels_different_nodes() {
        let mut g = StigreGraph::new("test", 0.05);
        let id1 = g.upsert_node("ChromoQ", NodeKind::Entity);
        let id2 = g.upsert_node("EGFP", NodeKind::Entity);
        assert_ne!(id1, id2);
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn edge_upsert_and_reinforce() {
        let mut g = StigreGraph::new("test", 0.05);
        let a = g.upsert_node("A", NodeKind::Entity);
        let b = g.upsert_node("B", NodeKind::Entity);

        let e1 = g.upsert_edge(a, b, Channel::Trail, 0.5, "test");
        assert!(e1.is_some());
        assert_eq!(g.edge_count(), 1);

        // Same (from, to, channel) → reinforce, not duplicate
        let e2 = g.upsert_edge(a, b, Channel::Trail, 0.5, "test");
        assert_eq!(e1, e2);
        assert_eq!(g.edge_count(), 1);

        // Different channel → new edge
        let e3 = g.upsert_edge(a, b, Channel::Causal, 0.8, "test");
        assert_ne!(e1, e3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn traverse_basic() {
        let mut g = StigreGraph::new("test", 0.05);
        let a = g.upsert_node("ChromoQ", NodeKind::Entity);
        let b = g.upsert_node("EGFP", NodeKind::Entity);
        let c = g.upsert_node("GFP", NodeKind::Entity);

        g.upsert_edge(a, b, Channel::DerivesDe, 0.95, "test");
        g.upsert_edge(b, c, Channel::DerivesDe, 0.90, "test");

        let query = SomaQuery::new("ChromoQ").with_max_hops(3);
        let results = g.traverse(&query);

        assert!(results.len() >= 2); // ChromoQ + at least EGFP
        assert_eq!(results[0].node.label, "ChromoQ");
        assert_eq!(results[0].score, 1.0); // start node
    }

    #[test]
    fn neighbors_filtered() {
        let mut g = StigreGraph::new("test", 0.05);
        let a = g.upsert_node("A", NodeKind::Entity);
        let b = g.upsert_node("B", NodeKind::Entity);
        let c = g.upsert_node("C", NodeKind::Entity);

        g.upsert_edge(a, b, Channel::Trail, 0.8, "test");
        g.upsert_edge(a, c, Channel::Alarm, 0.9, "test");

        let trail_neighbors = g.neighbors(a, &[Channel::Trail], 0.0);
        assert_eq!(trail_neighbors.len(), 1);

        let all_neighbors = g.neighbors(a, &[], 0.0);
        assert_eq!(all_neighbors.len(), 2);
    }

    #[test]
    fn stats_correct() {
        let mut g = StigreGraph::new("test", 0.05);
        let a = g.upsert_node("A", NodeKind::Entity);
        let b = g.upsert_node("B", NodeKind::Entity);
        g.upsert_edge(a, b, Channel::Trail, 0.8, "test");

        let stats = g.stats();
        assert_eq!(stats.nodes, 2);
        assert_eq!(stats.edges, 1);
        assert_eq!(stats.dead_edges, 0);
    }

    #[test]
    fn prune_dead_edges() {
        let mut g = StigreGraph::new("test", 0.5); // high threshold
        let a = g.upsert_node("A", NodeKind::Entity);
        let b = g.upsert_node("B", NodeKind::Entity);
        g.upsert_edge(a, b, Channel::Episodic, 0.1, "test"); // below threshold

        let pruned = g.prune_dead_edges();
        assert_eq!(pruned, 1);
        assert_eq!(g.edge_count(), 0);
    }
}
