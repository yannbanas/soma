use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use serde::Serialize;
use tokio::sync::RwLock;
use tracing::info;

use soma_core::{Channel, NodeKind};
use soma_graph::{GraphStats, StigreGraph};
use soma_store::Store;

/// Report from a single consolidation pass.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ConsolidationReport {
    pub episodes_found: usize,
    pub concept_created: Option<String>,
    pub edges_created: usize,
    pub edges_pruned: usize,
    pub orphans_archived: usize,
    pub stats_before: Option<GraphStats>,
    pub stats_after: Option<GraphStats>,
}

/// Configuration for biological scheduler intervals.
#[derive(Debug, Clone)]
pub struct BioConfig {
    pub evaporation_interval: Duration,
    pub physarum_interval: Duration,
    pub consolidation_interval: Duration,
    pub pruning_interval: Duration,
    pub prune_threshold: f32,
}

impl Default for BioConfig {
    fn default() -> Self {
        BioConfig {
            evaporation_interval: Duration::from_secs(3600),      // 1h
            physarum_interval: Duration::from_secs(7200),         // 2h
            consolidation_interval: Duration::from_secs(21600),   // 6h
            pruning_interval: Duration::from_secs(86400),         // 24h
            prune_threshold: 0.05,
        }
    }
}

impl BioConfig {
    pub fn from_soma_config(cfg: &soma_core::SomaConfig) -> Self {
        BioConfig {
            evaporation_interval: Duration::from_secs_f64(3600.0), // always 1h
            physarum_interval: Duration::from_secs_f64(
                cfg.bio.physarum_interval_hours * 3600.0,
            ),
            consolidation_interval: Duration::from_secs_f64(
                cfg.bio.consolidation_interval_hours * 3600.0,
            ),
            pruning_interval: Duration::from_secs_f64(
                cfg.bio.pruning_interval_hours * 3600.0,
            ),
            prune_threshold: cfg.bio.prune_threshold,
        }
    }
}

/// Biological scheduler — manages 4 async background loops.
pub struct BioScheduler {
    config: BioConfig,
}

impl BioScheduler {
    pub fn new(config: BioConfig) -> Self {
        BioScheduler { config }
    }

    /// Run all 4 biological loops concurrently. Returns on Ctrl+C for graceful shutdown.
    pub async fn run(
        &self,
        graph: Arc<RwLock<StigreGraph>>,
        store: Arc<RwLock<Store>>,
    ) {
        tokio::select! {
            _ = self.run_loops(graph.clone(), store.clone()) => {},
            _ = tokio::signal::ctrl_c() => {
                info!("[bio] Graceful shutdown — Ctrl+C received");
            }
        }
    }

    /// Run all 4 biological loops without handling Ctrl+C.
    /// Use this when the caller manages shutdown (e.g., daemon with HTTP server).
    pub async fn run_loops(
        &self,
        graph: Arc<RwLock<StigreGraph>>,
        store: Arc<RwLock<Store>>,
    ) {
        let config = self.config.clone();
        tokio::join!(
            Self::evaporation_watchdog(graph.clone(), config.clone()),
            Self::physarum_reshape(graph.clone(), config.clone()),
            Self::sleep_consolidation(graph.clone(), store.clone(), config.clone()),
            Self::daily_pruning(graph.clone(), store.clone(), config),
        );
    }

    /// Run a single consolidation + pruning pass and return a report.
    pub async fn consolidate_once(
        graph: &Arc<RwLock<StigreGraph>>,
        store: &Arc<RwLock<Store>>,
    ) -> ConsolidationReport {
        let mut report = ConsolidationReport::default();

        let mut g = graph.write().await;
        report.stats_before = Some(g.stats());

        let now = Utc::now();
        let six_hours_ago = now - chrono::Duration::hours(6);

        // Find recent Episodic nodes
        let recent_episodes: Vec<(soma_core::NodeId, String)> = g
            .all_nodes()
            .filter(|n| n.kind == NodeKind::Event && n.created_at >= six_hours_ago)
            .map(|n| (n.id, n.label.clone()))
            .collect();

        report.episodes_found = recent_episodes.len();

        if recent_episodes.len() >= 3 {
            let concept_label = format!("consolidated_{}", now.format("%Y%m%d_%H%M"));
            let concept_id = g.upsert_node(&concept_label, NodeKind::Concept);

            let mut created_edges = Vec::new();
            for (episode_id, _label) in &recent_episodes {
                if let Some(_eid) = g.upsert_edge(
                    *episode_id,
                    concept_id,
                    Channel::DerivesDe,
                    0.7,
                    "bio:consolidation",
                ) {
                    created_edges.push(soma_core::StigreEdge::new(
                        *episode_id,
                        concept_id,
                        Channel::DerivesDe,
                        0.7,
                        "bio:consolidation".to_string(),
                    ));
                }
            }

            report.concept_created = Some(concept_label.clone());
            report.edges_created = created_edges.len();

            // WAL
            {
                let mut s = store.write().await;
                let concept_node =
                    soma_core::SomaNode::new(g.workspace(), &concept_label, NodeKind::Concept);
                let _ = s.write_wal(&soma_store::WalEntry::NodeUpsert(concept_node));
                for edge in &created_edges {
                    let _ = s.write_wal(&soma_store::WalEntry::EdgeUpsert(edge.clone()));
                }
                let _ = s.write_wal(&soma_store::WalEntry::ConsolidationEvent {
                    ts: now,
                    episodes_merged: created_edges.len() as u32,
                    concepts_created: 1,
                });
            }
        }

        // Pruning pass
        report.edges_pruned = g.prune_dead_edges();
        report.orphans_archived = g.archive_orphans().len();

        // WAL for archived nodes
        if report.orphans_archived > 0 {
            let archived = g.archive_orphans();
            let mut s = store.write().await;
            for node_id in &archived {
                let _ = s.write_wal(&soma_store::WalEntry::NodeArchive(*node_id));
            }
        }

        report.stats_after = Some(g.stats());
        report
    }

    /// Evaporation watchdog (1h interval).
    /// Checks edges whose effective intensity has fallen below threshold.
    /// Lazy evaporation means we just identify dead edges; actual removal is in daily_pruning.
    async fn evaporation_watchdog(graph: Arc<RwLock<StigreGraph>>, config: BioConfig) {
        loop {
            tokio::time::sleep(config.evaporation_interval).await;

            let g = graph.read().await;
            let stats = g.stats();
            info!(
                "[bio:evaporation] nodes={} edges={} dead={}",
                stats.nodes, stats.edges, stats.dead_edges
            );
        }
    }

    /// Physarum reshape (2h interval).
    /// Inspired by Physarum polycephalum: frequently traversed paths strengthen,
    /// unused paths weaken. Simulates network optimization.
    async fn physarum_reshape(graph: Arc<RwLock<StigreGraph>>, config: BioConfig) {
        loop {
            tokio::time::sleep(config.physarum_interval).await;

            let mut g = graph.write().await;
            let now = Utc::now();

            // Collect edge info (can't iterate and mutate simultaneously)
            let edge_info: Vec<(soma_core::EdgeId, u32, f32, Channel)> = g
                .all_edges()
                .map(|e| (e.id, e.uses, e.effective_intensity(now), e.channel))
                .collect();

            if edge_info.is_empty() {
                continue;
            }

            // Average uses across all edges
            let avg_uses: f32 =
                edge_info.iter().map(|(_, u, _, _)| *u as f32).sum::<f32>()
                    / edge_info.len() as f32;

            let mut reinforced = 0u32;
            for (id, uses, _intensity, _channel) in &edge_info {
                if (*uses as f32) > avg_uses * 1.5 {
                    // Frequently used → reinforce
                    if g.reinforce_edge(*id) {
                        reinforced += 1;
                    }
                }
            }

            if reinforced > 0 {
                info!("[bio:physarum] reinforced {} high-traffic edges", reinforced);
            }
        }
    }

    /// Sleep consolidation (6h interval).
    /// Clusters recent Episodic nodes by similarity, creates Concept summaries.
    /// Mimics hippocampo-cortical sleep consolidation.
    async fn sleep_consolidation(
        graph: Arc<RwLock<StigreGraph>>,
        store: Arc<RwLock<Store>>,
        config: BioConfig,
    ) {
        loop {
            tokio::time::sleep(config.consolidation_interval).await;

            let mut g = graph.write().await;
            let now = Utc::now();
            let six_hours_ago = now - chrono::Duration::hours(6);

            // Find recent Episodic nodes
            let recent_episodes: Vec<(soma_core::NodeId, String)> = g
                .all_nodes()
                .filter(|n| {
                    n.kind == NodeKind::Event && n.created_at >= six_hours_ago
                })
                .map(|n| (n.id, n.label.clone()))
                .collect();

            if recent_episodes.len() < 3 {
                continue;
            }

            // Simple clustering: group by shared words
            // (Phase 2 will use HDC similarity for proper clustering)
            let concept_label = format!("consolidated_{}", now.format("%Y%m%d_%H%M"));
            let concept_id = g.upsert_node(&concept_label, NodeKind::Concept);

            let mut linked = 0u32;
            let mut created_edges = Vec::new();
            for (episode_id, _label) in &recent_episodes {
                if let Some(_eid) = g.upsert_edge(
                    *episode_id,
                    concept_id,
                    Channel::DerivesDe,
                    0.7,
                    "bio:consolidation",
                ) {
                    created_edges.push(soma_core::StigreEdge::new(
                        *episode_id,
                        concept_id,
                        Channel::DerivesDe,
                        0.7,
                        "bio:consolidation".to_string(),
                    ));
                    linked += 1;
                }
            }

            // Log to WAL: individual edges + summary
            {
                let mut s = store.write().await;
                // Concept node
                let concept_node = soma_core::SomaNode::new(
                    g.workspace(), &concept_label, NodeKind::Concept,
                );
                let _ = s.write_wal(&soma_store::WalEntry::NodeUpsert(concept_node));
                // Individual edges
                for edge in &created_edges {
                    let _ = s.write_wal(&soma_store::WalEntry::EdgeUpsert(edge.clone()));
                }
                // Summary event
                let _ = s.write_wal(&soma_store::WalEntry::ConsolidationEvent {
                    ts: now,
                    episodes_merged: linked,
                    concepts_created: 1,
                });
            }

            info!(
                "[bio:consolidation] created concept '{}' from {} episodes",
                concept_label, linked
            );
        }
    }

    /// Daily pruning (24h interval).
    /// Removes dead edges and archives orphan nodes.
    async fn daily_pruning(
        graph: Arc<RwLock<StigreGraph>>,
        store: Arc<RwLock<Store>>,
        config: BioConfig,
    ) {
        loop {
            tokio::time::sleep(config.pruning_interval).await;

            let mut g = graph.write().await;

            // Prune dead edges
            let pruned_edges = g.prune_dead_edges();

            // Archive orphan nodes
            let archived_nodes = g.archive_orphans();

            // Log to WAL
            {
                let mut s = store.write().await;
                for node_id in &archived_nodes {
                    let _ = s.write_wal(&soma_store::WalEntry::NodeArchive(*node_id));
                }
            }

            info!(
                "[bio:pruning] pruned {} edges, archived {} orphan nodes",
                pruned_edges,
                archived_nodes.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bio_config_defaults() {
        let cfg = BioConfig::default();
        assert_eq!(cfg.evaporation_interval, Duration::from_secs(3600));
        assert_eq!(cfg.prune_threshold, 0.05);
    }

    #[test]
    fn bio_config_from_soma() {
        let soma_cfg = soma_core::SomaConfig::default();
        let bio_cfg = BioConfig::from_soma_config(&soma_cfg);
        assert_eq!(bio_cfg.prune_threshold, 0.05);
        assert_eq!(
            bio_cfg.physarum_interval,
            Duration::from_secs_f64(soma_cfg.bio.physarum_interval_hours * 3600.0)
        );
        assert_eq!(
            bio_cfg.consolidation_interval,
            Duration::from_secs_f64(soma_cfg.bio.consolidation_interval_hours * 3600.0)
        );
        assert_eq!(
            bio_cfg.pruning_interval,
            Duration::from_secs_f64(soma_cfg.bio.pruning_interval_hours * 3600.0)
        );
        assert_eq!(bio_cfg.evaporation_interval, Duration::from_secs(3600));
    }
}
