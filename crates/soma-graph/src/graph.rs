use std::collections::HashMap;

use chrono::{DateTime, Utc};
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

        // Find start node — cascading resolution:
        // 1. Exact case-sensitive match
        // 2. Case-insensitive exact match
        // 3. Label contains query (prefer shortest label)
        // 4. Query contains label (prefer longest label, min 3 chars)
        let start_idx = match self.label_idx.get(&query.start) {
            Some(&idx) => idx,
            None => {
                let query_lower = query.start.to_lowercase();
                if let Some((_, &idx)) = self
                    .label_idx
                    .iter()
                    .find(|(k, _)| k.to_lowercase() == query_lower)
                {
                    idx
                } else if let Some((_, &idx)) = self
                    .label_idx
                    .iter()
                    .filter(|(k, _)| k.to_lowercase().contains(&query_lower))
                    .min_by_key(|(k, _)| k.len())
                {
                    idx
                } else if let Some((_, &idx)) = self
                    .label_idx
                    .iter()
                    .filter(|(k, _)| {
                        k.len() >= 3 && query_lower.contains(&k.to_lowercase())
                    })
                    .max_by_key(|(k, _)| k.len())
                {
                    idx
                } else {
                    return Vec::new();
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

    // ── Personalized PageRank ──────────────────────────────────

    /// Personalized PageRank from seed nodes.
    ///
    /// Uses power iteration with edge weights = effective_intensity(now),
    /// giving temporal awareness that static KGs lack.
    ///
    /// - `alpha`: teleport probability (typically 0.15)
    /// - `max_iterations`: convergence limit (typically 50)
    /// - `tolerance`: convergence threshold (typically 1e-6)
    /// - `node_specificity`: optional IDF-based node weights for seed initialization
    pub fn ppr(
        &self,
        seed_labels: &[String],
        alpha: f32,
        max_iterations: usize,
        tolerance: f32,
        node_specificity: Option<&HashMap<NodeId, f32>>,
    ) -> Vec<(NodeId, String, f32)> {
        let now = Utc::now();
        let n = self.inner.node_count();
        if n == 0 {
            return Vec::new();
        }

        // Find seed node indices
        let seed_indices: Vec<NodeIndex> = seed_labels
            .iter()
            .filter_map(|label| self.label_idx.get(label).copied())
            .collect();

        if seed_indices.is_empty() {
            return Vec::new();
        }

        // Personalization vector
        let mut personalization = vec![0.0f32; n];
        for &idx in &seed_indices {
            let weight = if let Some(spec) = node_specificity {
                let node = &self.inner[idx];
                spec.get(&node.id).copied().unwrap_or(1.0)
            } else {
                1.0
            };
            personalization[idx.index()] = weight;
        }
        // Normalize
        let p_sum: f32 = personalization.iter().sum();
        if p_sum > 0.0 {
            for v in personalization.iter_mut() {
                *v /= p_sum;
            }
        }

        // Build sparse transition: for each node, outgoing (target_index, weight)
        let mut out_weights: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut out_sums: Vec<f32> = vec![0.0; n];

        for edge_ref in self.inner.edge_references() {
            let u = edge_ref.source().index();
            let v = edge_ref.target().index();
            let w = edge_ref.weight().effective_intensity(now);
            if w > 0.001 {
                out_weights[u].push((v, w));
                out_sums[u] += w;
            }
        }

        // Power iteration
        let mut rank = personalization.clone();
        let mut new_rank = vec![0.0f32; n];

        for _ in 0..max_iterations {
            new_rank.fill(0.0);

            // Propagation
            for u in 0..n {
                if rank[u] < 1e-10 || out_sums[u] < 1e-10 {
                    continue;
                }
                let factor = (1.0 - alpha) * rank[u] / out_sums[u];
                for &(v, w) in &out_weights[u] {
                    new_rank[v] += factor * w;
                }
            }

            // Teleport
            for v in 0..n {
                new_rank[v] += alpha * personalization[v];
            }

            // Convergence check
            let diff: f32 = rank
                .iter()
                .zip(new_rank.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            std::mem::swap(&mut rank, &mut new_rank);

            if diff < tolerance {
                break;
            }
        }

        // Collect non-zero results sorted by rank
        let mut results: Vec<(NodeId, String, f32)> = self
            .inner
            .node_indices()
            .filter(|&idx| rank[idx.index()] > 1e-8)
            .map(|idx| {
                let node = &self.inner[idx];
                (node.id, node.label.clone(), rank[idx.index()])
            })
            .collect();

        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // ── Node Specificity (IDF weighting) ───────────────────────

    /// Compute IDF-based node specificity.
    ///
    /// `s_i = log(N / (1 + degree_i))` normalized to [0, 1].
    /// Hub nodes (high degree) → low specificity.
    /// Rare nodes (low degree) → high specificity.
    pub fn node_specificity_idf(&self) -> HashMap<NodeId, f32> {
        let n = self.inner.node_count() as f32;
        if n == 0.0 {
            return HashMap::new();
        }
        let now = Utc::now();
        let mut specificity = HashMap::new();

        for idx in self.inner.node_indices() {
            let node = &self.inner[idx];
            let out_deg = self
                .inner
                .edges(idx)
                .filter(|e| e.weight().effective_intensity(now) > self.prune_threshold)
                .count();
            let in_deg = self
                .inner
                .edges_directed(idx, petgraph::Direction::Incoming)
                .filter(|e| e.weight().effective_intensity(now) > self.prune_threshold)
                .count();
            let degree = out_deg + in_deg;
            let idf = (n / (1.0 + degree as f32)).ln() + 1.0;
            specificity.insert(node.id, idf);
        }

        // Normalize to [0, 1]
        let max_spec = specificity.values().cloned().fold(0.0f32, f32::max);
        if max_spec > 0.0 {
            for v in specificity.values_mut() {
                *v /= max_spec;
            }
        }

        specificity
    }

    /// Node specificity keyed by label (for use with RRF merge).
    pub fn node_specificity_by_label(&self) -> HashMap<String, f32> {
        let id_map = self.node_specificity_idf();
        let mut label_map = HashMap::new();
        for node in self.all_nodes() {
            if let Some(&spec) = id_map.get(&node.id) {
                label_map.insert(node.label.clone(), spec);
            }
        }
        label_map
    }

    // ── Temporal helpers ────────────────────────────────────────

    /// Set the last_touch timestamp on all edges from a given source.
    /// Used by temporal benchmarks to simulate time evolution.
    pub fn set_edge_timestamps_for_source(&mut self, source: &str, timestamp: DateTime<Utc>) {
        for edge in self.inner.edge_weights_mut() {
            if edge.source == source {
                edge.last_touch = timestamp;
            }
        }
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

    // ── PPR tests ──────────────────────────────────────────────

    #[test]
    fn ppr_basic_propagation() {
        let mut g = StigreGraph::new("test", 0.05);
        let a = g.upsert_node("Alice", NodeKind::Entity);
        let b = g.upsert_node("Company", NodeKind::Entity);
        let c = g.upsert_node("Industry", NodeKind::Entity);
        g.upsert_edge(a, b, Channel::Trail, 0.9, "test");
        g.upsert_edge(b, c, Channel::Trail, 0.9, "test");

        let results = g.ppr(&["Alice".to_string()], 0.15, 50, 1e-6, None);
        assert!(results.len() >= 2);
        // Alice (seed) should have highest score
        assert_eq!(results[0].1, "Alice");
        // Company (1 hop) should rank above Industry (2 hops)
        let company_pos = results.iter().position(|r| r.1 == "Company");
        let industry_pos = results.iter().position(|r| r.1 == "Industry");
        assert!(company_pos.is_some());
        assert!(industry_pos.is_some());
        assert!(company_pos.unwrap() < industry_pos.unwrap());
    }

    #[test]
    fn ppr_multi_seed() {
        let mut g = StigreGraph::new("test", 0.05);
        let a = g.upsert_node("A", NodeKind::Entity);
        let b = g.upsert_node("B", NodeKind::Entity);
        let c = g.upsert_node("C", NodeKind::Entity);
        g.upsert_edge(a, c, Channel::Trail, 0.9, "test");
        g.upsert_edge(b, c, Channel::Trail, 0.9, "test");

        let results = g.ppr(
            &["A".to_string(), "B".to_string()],
            0.15, 50, 1e-6, None,
        );
        // C should be found (reachable from both seeds)
        assert!(results.iter().any(|r| r.1 == "C"));
    }

    #[test]
    fn ppr_empty_seeds() {
        let mut g = StigreGraph::new("test", 0.05);
        g.upsert_node("X", NodeKind::Entity);
        let results = g.ppr(&["nonexistent".to_string()], 0.15, 50, 1e-6, None);
        assert!(results.is_empty());
    }

    #[test]
    fn ppr_empty_graph() {
        let g = StigreGraph::new("test", 0.05);
        let results = g.ppr(&["A".to_string()], 0.15, 50, 1e-6, None);
        assert!(results.is_empty());
    }

    // ── Node specificity tests ─────────────────────────────────

    #[test]
    fn node_specificity_hub_vs_leaf() {
        let mut g = StigreGraph::new("test", 0.01);
        let hub = g.upsert_node("hub", NodeKind::Entity);
        let leaf1 = g.upsert_node("leaf1", NodeKind::Entity);
        let leaf2 = g.upsert_node("leaf2", NodeKind::Entity);
        let leaf3 = g.upsert_node("leaf3", NodeKind::Entity);
        let isolated = g.upsert_node("isolated", NodeKind::Entity);

        g.upsert_edge(hub, leaf1, Channel::Trail, 0.8, "test");
        g.upsert_edge(hub, leaf2, Channel::Trail, 0.8, "test");
        g.upsert_edge(hub, leaf3, Channel::Trail, 0.8, "test");

        let spec = g.node_specificity_idf();
        let hub_spec = spec[&hub];
        let leaf_spec = spec[&leaf1];
        let iso_spec = spec[&isolated];

        // Isolated (degree 0) > leaf (degree 1) > hub (degree 3)
        assert!(iso_spec > hub_spec, "isolated={} should > hub={}", iso_spec, hub_spec);
        assert!(leaf_spec > hub_spec, "leaf={} should > hub={}", leaf_spec, hub_spec);
    }

    #[test]
    fn node_specificity_empty_graph() {
        let g = StigreGraph::new("test", 0.05);
        let spec = g.node_specificity_idf();
        assert!(spec.is_empty());
    }
}
