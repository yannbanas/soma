//! Louvain community detection for the SOMA knowledge graph.
//!
//! Bio-aware: uses effective_intensity(now) as edge weight,
//! so communities reflect current (evaporated) state.

use std::collections::HashMap;

use chrono::Utc;
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;

use soma_core::{NodeId, SomaNode, StigreEdge};

/// Result of community detection.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Node label → community ID
    pub assignments: HashMap<String, usize>,
    /// Community ID → list of node labels
    pub communities: HashMap<usize, Vec<String>>,
    /// Modularity score
    pub modularity: f64,
}

/// Louvain community detection on the knowledge graph.
///
/// Treats the directed graph as undirected for modularity computation.
/// Edge weight = effective_intensity(now).
pub fn detect_communities(
    graph: &DiGraph<SomaNode, StigreEdge>,
    min_community_size: usize,
) -> CommunityResult {
    let now = Utc::now();
    let n = graph.node_count();
    if n == 0 {
        return CommunityResult {
            assignments: HashMap::new(),
            communities: HashMap::new(),
            modularity: 0.0,
        };
    }

    // Build adjacency with weights (undirected view)
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut total_weight = 0.0f64;

    for edge in graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        let w = edge.weight().effective_intensity(now) as f64;
        if w < 0.001 { continue; }
        adj[u].push((v, w));
        adj[v].push((u, w)); // undirected
        total_weight += w;
    }

    if total_weight < 1e-10 {
        return CommunityResult {
            assignments: HashMap::new(),
            communities: HashMap::new(),
            modularity: 0.0,
        };
    }

    let m2 = total_weight; // total weight (each edge counted once from each direction = 2m, but we already doubled)

    // Node strengths (sum of edge weights)
    let strength: Vec<f64> = adj.iter().map(|edges| edges.iter().map(|(_, w)| w).sum()).collect();

    // Phase 1: each node starts in its own community
    let mut community: Vec<usize> = (0..n).collect();
    let mut improved = true;

    while improved {
        improved = false;

        for i in 0..n {
            let current_comm = community[i];

            // Compute weight to each neighboring community
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &adj[i] {
                *comm_weights.entry(community[j]).or_insert(0.0) += w;
            }

            // Sum of strengths per community
            let mut comm_strength: HashMap<usize, f64> = HashMap::new();
            for (node, &comm) in community.iter().enumerate() {
                *comm_strength.entry(comm).or_insert(0.0) += strength[node];
            }

            let ki = strength[i];
            let w_current = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
            let sigma_current = comm_strength.get(&current_comm).copied().unwrap_or(0.0) - ki;

            let mut best_comm = current_comm;
            let mut best_delta = 0.0f64;

            for (&comm, &w_comm) in &comm_weights {
                if comm == current_comm { continue; }
                let sigma_comm = comm_strength.get(&comm).copied().unwrap_or(0.0);

                // Modularity gain formula
                let delta = (w_comm - w_current) / m2
                    - ki * (sigma_comm - sigma_current) / (m2 * m2);

                if delta > best_delta {
                    best_delta = delta;
                    best_comm = comm;
                }
            }

            if best_comm != current_comm {
                community[i] = best_comm;
                improved = true;
            }
        }
    }

    // Renumber communities to be contiguous
    let mut comm_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    for c in &community {
        if !comm_map.contains_key(c) {
            comm_map.insert(*c, next_id);
            next_id += 1;
        }
    }
    for c in community.iter_mut() {
        *c = comm_map[c];
    }

    // Build result
    let mut assignments = HashMap::new();
    let mut communities: HashMap<usize, Vec<String>> = HashMap::new();

    for (idx, &comm) in community.iter().enumerate() {
        if let Some(node) = graph.node_weights().nth(idx) {
            assignments.insert(node.label.clone(), comm);
            communities.entry(comm).or_default().push(node.label.clone());
        }
    }

    // Filter by min size
    communities.retain(|_, members| members.len() >= min_community_size);

    // Compute modularity
    let modularity = compute_modularity(&community, &adj, m2);

    CommunityResult {
        assignments,
        communities,
        modularity,
    }
}

/// Compute modularity Q = (1/2m) Σ [A_ij - k_i*k_j/2m] δ(c_i, c_j)
fn compute_modularity(community: &[usize], adj: &[Vec<(usize, f64)>], m2: f64) -> f64 {
    if m2 < 1e-10 { return 0.0; }

    let strength: Vec<f64> = adj.iter().map(|edges| edges.iter().map(|(_, w)| w).sum()).collect();
    let mut q = 0.0f64;

    for (i, edges) in adj.iter().enumerate() {
        for &(j, w) in edges {
            if community[i] == community[j] {
                q += w - strength[i] * strength[j] / m2;
            }
        }
    }

    q / m2
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::DiGraph;
    use soma_core::{Channel, NodeKind, NodeId};

    #[test]
    fn empty_graph() {
        let g: DiGraph<SomaNode, StigreEdge> = DiGraph::new();
        let result = detect_communities(&g, 1);
        assert!(result.communities.is_empty());
        assert_eq!(result.modularity, 0.0);
    }

    #[test]
    fn two_clusters() {
        let mut g = DiGraph::new();
        // Cluster 1: A-B-C (fully connected)
        let a = g.add_node(SomaNode::new("test", "A", NodeKind::Entity));
        let b = g.add_node(SomaNode::new("test", "B", NodeKind::Entity));
        let c = g.add_node(SomaNode::new("test", "C", NodeKind::Entity));
        // Cluster 2: D-E-F (fully connected)
        let d = g.add_node(SomaNode::new("test", "D", NodeKind::Entity));
        let e = g.add_node(SomaNode::new("test", "E", NodeKind::Entity));
        let f = g.add_node(SomaNode::new("test", "F", NodeKind::Entity));

        let id = |s: &str| NodeId::from_label(&format!("test:{}", s));
        let edge = |from: &str, to: &str| StigreEdge::new(id(from), id(to), Channel::Trail, 0.9, "test".into());

        // Cluster 1 edges
        g.add_edge(a, b, edge("A", "B"));
        g.add_edge(b, c, edge("B", "C"));
        g.add_edge(a, c, edge("A", "C"));

        // Cluster 2 edges
        g.add_edge(d, e, edge("D", "E"));
        g.add_edge(e, f, edge("E", "F"));
        g.add_edge(d, f, edge("D", "F"));

        // Weak bridge between clusters
        let mut bridge = edge("C", "D");
        bridge.set_intensity(0.1);
        g.add_edge(c, d, bridge);

        let result = detect_communities(&g, 2);
        assert!(result.communities.len() >= 2, "should find at least 2 communities");
        assert!(result.modularity > 0.0, "modularity should be positive");
    }

    #[test]
    fn single_node() {
        let mut g = DiGraph::new();
        g.add_node(SomaNode::new("test", "lonely", NodeKind::Entity));
        let result = detect_communities(&g, 1);
        // No edges → no communities of interest
        assert_eq!(result.modularity, 0.0);
    }
}
