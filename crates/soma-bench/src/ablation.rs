//! Ablation framework: test each retrieval path independently and combined.

use crate::loader::BenchQuestion;
use crate::metrics;
use serde::Serialize;
use soma_core::SomaQuery;
use soma_core::{fuzzy_label_search, rrf_merge_with_sources};
use soma_graph::StigreGraph;
use soma_hdc::HdcEngine;
use std::collections::HashMap;

/// Which retrieval paths are active in this ablation config.
#[derive(Debug, Clone)]
pub struct AblationConfig {
    pub name: String,
    pub use_graph_bfs: bool,
    pub use_hdc: bool,
    pub use_fuzzy: bool,
    pub use_ppr: bool,
}

impl AblationConfig {
    /// All 8 standard ablation configurations.
    pub fn all_configs() -> Vec<Self> {
        vec![
            Self {
                name: "graph_only".into(),
                use_graph_bfs: true,
                use_hdc: false,
                use_fuzzy: false,
                use_ppr: false,
            },
            Self {
                name: "hdc_only".into(),
                use_graph_bfs: false,
                use_hdc: true,
                use_fuzzy: false,
                use_ppr: false,
            },
            Self {
                name: "fuzzy_only".into(),
                use_graph_bfs: false,
                use_hdc: false,
                use_fuzzy: true,
                use_ppr: false,
            },
            Self {
                name: "ppr_only".into(),
                use_graph_bfs: false,
                use_hdc: false,
                use_fuzzy: false,
                use_ppr: true,
            },
            Self {
                name: "graph+hdc".into(),
                use_graph_bfs: true,
                use_hdc: true,
                use_fuzzy: false,
                use_ppr: false,
            },
            Self {
                name: "graph+hdc+fuzzy".into(),
                use_graph_bfs: true,
                use_hdc: true,
                use_fuzzy: true,
                use_ppr: false,
            },
            Self {
                name: "graph+ppr".into(),
                use_graph_bfs: true,
                use_hdc: false,
                use_fuzzy: false,
                use_ppr: true,
            },
            Self {
                name: "all_four".into(),
                use_graph_bfs: true,
                use_hdc: true,
                use_fuzzy: true,
                use_ppr: true,
            },
        ]
    }
}

/// Results of a single ablation configuration.
#[derive(Debug, Serialize)]
pub struct AblationReport {
    pub config_name: String,
    pub entity_recall_at_5: f32,
    pub entity_recall_at_10: f32,
    pub path_recall_avg: f32,
    pub num_questions: usize,
}

/// Run hybrid search with a specific ablation configuration.
///
/// Returns ranked list of (label, score).
pub fn search_with_ablation(
    graph: &StigreGraph,
    hdc: &HdcEngine,
    query: &str,
    config: &AblationConfig,
    limit: usize,
) -> Vec<(String, f32)> {
    let all_labels = graph.all_labels();
    let mut ranked_lists: Vec<(&str, Vec<(String, f32)>)> = Vec::new();

    if config.use_graph_bfs {
        let query_entities = soma_graph::extract_query_entities(query);
        let mut graph_list: Vec<(String, f32)> = Vec::new();

        // Try each extracted entity as BFS seed
        for entity in &query_entities {
            let q = SomaQuery::new(entity).with_max_hops(4).with_limit(limit);
            let results = graph.traverse(&q);
            for r in &results {
                if !graph_list.iter().any(|(l, _)| l == &r.node.label) {
                    graph_list.push((r.node.label.clone(), r.score));
                }
            }
        }
        // Also try fuzzy-matched labels as BFS seeds
        for (seed_label, seed_score) in fuzzy_label_search(query, &all_labels, 3) {
            if seed_score >= 0.5 {
                let q = SomaQuery::new(&seed_label)
                    .with_max_hops(4)
                    .with_limit(limit);
                let results = graph.traverse(&q);
                for r in &results {
                    if !graph_list.iter().any(|(l, _)| l == &r.node.label) {
                        graph_list.push((r.node.label.clone(), r.score));
                    }
                }
            }
        }
        ranked_lists.push(("graph", graph_list));
    }

    if config.use_hdc {
        let query_ents = soma_graph::extract_query_entities(query);
        let mut hdc_list: Vec<(String, f32)> = Vec::new();
        // Search full query
        for (label, score) in hdc.search_labels(query, &all_labels, limit) {
            hdc_list.push((label, score));
        }
        // Also search per extracted entity
        for entity in &query_ents {
            for (label, score) in hdc.search_labels(entity, &all_labels, limit) {
                if !hdc_list.iter().any(|(l, _)| l == &label) {
                    hdc_list.push((label, score));
                }
            }
        }
        hdc_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hdc_list.truncate(limit);
        ranked_lists.push(("hdc", hdc_list));
    }

    if config.use_fuzzy {
        let query_ents = soma_graph::extract_query_entities(query);
        let mut fuzzy_list: Vec<(String, f32)> = Vec::new();
        // Search per extracted entity
        for entity in &query_ents {
            for (label, score) in fuzzy_label_search(entity, &all_labels, limit) {
                if !fuzzy_list.iter().any(|(l, _)| l == &label) {
                    fuzzy_list.push((label, score));
                }
            }
        }
        fuzzy_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fuzzy_list.truncate(limit);
        ranked_lists.push(("fuzzy", fuzzy_list));
    }

    if config.use_ppr {
        // Extract seed entities from query
        let query_entities = soma_graph::extract_query_entities(query);
        let mut seeds = query_entities;
        for (label, score) in fuzzy_label_search(query, &all_labels, 5) {
            if score >= 0.7 && !seeds.contains(&label) {
                seeds.push(label);
            }
        }
        let ppr_results = graph.ppr(&seeds, 0.15, 50, 1e-6, None);
        let list: Vec<(String, f32)> = ppr_results
            .iter()
            .take(limit)
            .map(|(_, label, score)| (label.clone(), *score))
            .collect();
        ranked_lists.push(("ppr", list));
    }

    if ranked_lists.is_empty() {
        return Vec::new();
    }

    let hybrid = rrf_merge_with_sources(&ranked_lists, 60.0);
    hybrid
        .into_iter()
        .take(limit)
        .map(|hr| (hr.label, hr.score))
        .collect()
}

/// Run ablation across all configurations on a pre-built graph.
pub fn run_ablation(
    questions: &[BenchQuestion],
    graph: &StigreGraph,
    hdc: &HdcEngine,
) -> HashMap<String, AblationReport> {
    let mut reports = HashMap::new();

    for config in AblationConfig::all_configs() {
        let mut total_recall_5 = 0.0f32;
        let mut total_recall_10 = 0.0f32;
        let mut total_path_recall = 0.0f32;
        let mut count = 0;

        for question in questions {
            let results = search_with_ablation(graph, hdc, &question.question, &config, 20);
            let labels: Vec<String> = results.iter().map(|r| r.0.clone()).collect();

            let gold: Vec<String> = question
                .supporting_paragraphs
                .iter()
                .flat_map(|p| p.entities.clone())
                .collect();

            let (r5, _, _) = metrics::entity_recall_at_k(&gold, &labels, 5);
            let (r10, _, _) = metrics::entity_recall_at_k(&gold, &labels, 10);
            total_recall_5 += r5;
            total_recall_10 += r10;

            let chain: Vec<String> = question
                .supporting_paragraphs
                .iter()
                .map(|p| p.title.clone())
                .collect();
            total_path_recall += metrics::path_recall(&chain, &labels);

            count += 1;
        }

        let n = count.max(1) as f32;
        reports.insert(
            config.name.clone(),
            AblationReport {
                config_name: config.name,
                entity_recall_at_5: total_recall_5 / n,
                entity_recall_at_10: total_recall_10 / n,
                path_recall_avg: total_path_recall / n,
                num_questions: count,
            },
        );
    }

    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_configs_count() {
        assert_eq!(AblationConfig::all_configs().len(), 8);
    }

    #[test]
    fn search_empty_graph() {
        let graph = StigreGraph::new("test", 0.05);
        let hdc = HdcEngine::new(100, 3, false);
        let config = AblationConfig {
            name: "test".into(),
            use_graph_bfs: true,
            use_hdc: true,
            use_fuzzy: true,
            use_ppr: true,
        };
        let results = search_with_ablation(&graph, &hdc, "test query", &config, 10);
        assert!(results.is_empty());
    }
}
