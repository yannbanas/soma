use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::node::SomaNode;

/// Result of a hybrid search with source attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    pub label: String,
    pub score: f32,
    /// Which search paths contributed: "graph", "hdc", "fuzzy"
    pub sources: Vec<String>,
    pub node: Option<SomaNode>,
    pub hops: Option<u8>,
}

/// Reciprocal Rank Fusion — merges N ranked lists into one.
///
/// Formula: `score(d) = Σ 1/(k + rank_i(d))` with k typically 60.
/// Each input list must be sorted by score descending.
pub fn rrf_merge(ranked_lists: &[Vec<(String, f32)>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for list in ranked_lists {
        for (rank, (label, _score)) in list.iter().enumerate() {
            *scores.entry(label.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }

    let mut results: Vec<(String, f32)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Track which sources contributed to each label across ranked lists.
pub fn rrf_merge_with_sources(
    ranked_lists: &[(&str, Vec<(String, f32)>)],
    k: f32,
) -> Vec<HybridResult> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut source_map: HashMap<String, Vec<String>> = HashMap::new();

    for (source_name, list) in ranked_lists {
        for (rank, (label, _score)) in list.iter().enumerate() {
            *scores.entry(label.clone()).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
            source_map
                .entry(label.clone())
                .or_default()
                .push(source_name.to_string());
        }
    }

    let mut results: Vec<HybridResult> = scores
        .into_iter()
        .map(|(label, score)| HybridResult {
            sources: source_map.remove(&label).unwrap_or_default(),
            label,
            score,
            node: None,
            hops: None,
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Fuzzy label search — exact, prefix, and substring matching with scoring.
///
/// Scoring:
/// - Exact match (case-insensitive) → 1.0
/// - Prefix match → 0.9
/// - Contains match → 0.7
pub fn fuzzy_label_search(query: &str, labels: &[String], limit: usize) -> Vec<(String, f32)> {
    if query.trim().is_empty() {
        return Vec::new();
    }
    let query_lower = query.to_lowercase();
    let mut results: Vec<(String, f32)> = Vec::new();

    for label in labels {
        let label_lower = label.to_lowercase();

        let score = if label_lower == query_lower {
            1.0
        } else if label_lower.starts_with(&query_lower) {
            0.9
        } else if label_lower.contains(&query_lower) {
            0.7
        } else if query_lower.len() >= 3 {
            // Check if any word in the label starts with the query
            let has_word_prefix = label_lower
                .split_whitespace()
                .any(|w| w.starts_with(&query_lower));
            if has_word_prefix {
                0.6
            } else {
                continue;
            }
        } else {
            continue;
        };

        results.push((label.clone(), score));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_single_list() {
        let list = vec![
            ("A".to_string(), 0.9),
            ("B".to_string(), 0.5),
            ("C".to_string(), 0.1),
        ];
        let merged = rrf_merge(&[list], 60.0);
        assert_eq!(merged[0].0, "A");
        assert_eq!(merged[1].0, "B");
        assert_eq!(merged[2].0, "C");
    }

    #[test]
    fn rrf_two_lists_interleave() {
        let list1 = vec![
            ("A".to_string(), 0.9),
            ("B".to_string(), 0.5),
        ];
        let list2 = vec![
            ("B".to_string(), 0.9),
            ("C".to_string(), 0.5),
        ];
        let merged = rrf_merge(&[list1, list2], 60.0);

        // B appears in both lists → highest RRF score
        assert_eq!(merged[0].0, "B");
        // A and C appear in one list each
        assert!(merged.len() == 3);
    }

    #[test]
    fn rrf_empty_lists() {
        let merged = rrf_merge(&[], 60.0);
        assert!(merged.is_empty());

        let merged2 = rrf_merge(&[vec![]], 60.0);
        assert!(merged2.is_empty());
    }

    #[test]
    fn rrf_k_parameter_affects_scoring() {
        let list1 = vec![("A".to_string(), 1.0)];
        let list2 = vec![("A".to_string(), 1.0)];

        let merged_low_k = rrf_merge(&[list1.clone(), list2.clone()], 1.0);
        let merged_high_k = rrf_merge(&[list1, list2], 100.0);

        // Lower k → higher score for same ranks
        assert!(merged_low_k[0].1 > merged_high_k[0].1);
    }

    #[test]
    fn fuzzy_exact_match_first() {
        let labels = vec![
            "ChromoQ".to_string(),
            "ChromoQ-variant".to_string(),
            "something ChromoQ".to_string(),
        ];
        let results = fuzzy_label_search("ChromoQ", &labels, 10);
        assert_eq!(results[0].0, "ChromoQ");
        assert_eq!(results[0].1, 1.0);
    }

    #[test]
    fn fuzzy_prefix_match() {
        let labels = vec![
            "ChromoQ-variant".to_string(),
            "xChromoQ".to_string(),
        ];
        let results = fuzzy_label_search("ChromoQ", &labels, 10);
        // "ChromoQ-variant" is a prefix match (0.9), "xChromoQ" is a contains match (0.7)
        assert_eq!(results[0].0, "ChromoQ-variant");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn fuzzy_case_insensitive() {
        let labels = vec!["ChromoQ".to_string()];
        let results = fuzzy_label_search("chromoq", &labels, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "ChromoQ");
        assert_eq!(results[0].1, 1.0);
    }

    #[test]
    fn rrf_with_sources_tracks_origin() {
        let lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("graph", vec![("A".to_string(), 0.9), ("B".to_string(), 0.5)]),
            ("hdc", vec![("B".to_string(), 0.8), ("C".to_string(), 0.3)]),
        ];
        let results = rrf_merge_with_sources(&lists, 60.0);

        let b = results.iter().find(|r| r.label == "B").unwrap();
        assert!(b.sources.contains(&"graph".to_string()));
        assert!(b.sources.contains(&"hdc".to_string()));

        let a = results.iter().find(|r| r.label == "A").unwrap();
        assert_eq!(a.sources, vec!["graph".to_string()]);
    }
}
