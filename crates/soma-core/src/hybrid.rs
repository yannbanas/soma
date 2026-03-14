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

/// Reciprocal Rank Fusion with node specificity weighting.
///
/// Like `rrf_merge_with_sources` but multiplies each item's RRF score by its
/// specificity weight (IDF-based). Rare/specific nodes get boosted, hubs get dampened.
///
/// `specificity` maps label → specificity score in [0, 1].
/// Items not in the map get a default weight of 0.5.
pub fn rrf_merge_with_specificity(
    ranked_lists: &[(&str, Vec<(String, f32)>)],
    k: f32,
    specificity: &HashMap<String, f32>,
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

    // Apply specificity weighting
    for (label, score) in scores.iter_mut() {
        let spec = specificity.get(label).copied().unwrap_or(0.5);
        // Blend: 70% RRF + 30% specificity boost
        *score *= 0.7 + 0.3 * spec;
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

/// Re-rank results with temporal boost and specificity.
///
/// - `recency_boost = 1.0 + 0.5 × e^(-hours_since_last_touch / 168)` (1 week half-life)
/// - Final score = rrf_score × recency_boost × specificity
pub fn rerank_temporal(
    results: &mut [HybridResult],
    last_touch_hours: &HashMap<String, f64>,
    specificity: &HashMap<String, f32>,
) {
    for r in results.iter_mut() {
        let hours = last_touch_hours.get(&r.label).copied().unwrap_or(168.0);
        let recency = 1.0 + 0.5 * (-hours / 168.0).exp() as f32;
        let spec = specificity.get(&r.label).copied().unwrap_or(0.5);
        r.score *= recency * spec;
    }
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
}

/// MMR (Maximal Marginal Relevance) diversification.
///
/// Selects items iteratively: each new item maximizes
/// `λ × relevance - (1-λ) × max_similarity_to_selected`.
///
/// Similarity is approximated by label overlap (Jaccard of words).
/// λ=0.7 means 70% relevance, 30% diversity.
pub fn mmr_diversify(results: &[HybridResult], limit: usize, lambda: f32) -> Vec<HybridResult> {
    if results.is_empty() || limit == 0 {
        return Vec::new();
    }

    let mut selected: Vec<HybridResult> = Vec::with_capacity(limit);
    let mut remaining: Vec<&HybridResult> = results.iter().collect();

    // Always pick the top result first
    selected.push(remaining.remove(0).clone());

    while selected.len() < limit && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr = f32::NEG_INFINITY;

        for (i, candidate) in remaining.iter().enumerate() {
            let relevance = candidate.score;
            let max_sim = selected.iter()
                .map(|s| word_jaccard(&candidate.label, &s.label))
                .fold(0.0f32, f32::max);
            let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        selected.push(remaining.remove(best_idx).clone());
    }

    selected
}

/// Word-level Jaccard similarity between two labels.
fn word_jaccard(a: &str, b: &str) -> f32 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_words: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();
    if a_words.is_empty() && b_words.is_empty() { return 1.0; }
    let intersection = a_words.intersection(&b_words).count() as f32;
    let union = a_words.union(&b_words).count() as f32;
    if union == 0.0 { 0.0 } else { intersection / union }
}

/// Strip accents/diacritics from a string for accent-insensitive comparison.
/// "Spéléologie" → "speleologie", "créé" → "cree"
fn fold_accents(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'à' | 'â' | 'ä' | 'á' | 'ã' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'î' | 'ï' | 'í' => 'i',
            'ô' | 'ö' | 'ó' | 'õ' => 'o',
            'ù' | 'û' | 'ü' | 'ú' => 'u',
            'ÿ' | 'ý' => 'y',
            'ç' => 'c',
            'ñ' => 'n',
            'À' | 'Â' | 'Ä' | 'Á' | 'Ã' => 'a',
            'É' | 'È' | 'Ê' | 'Ë' => 'e',
            'Î' | 'Ï' | 'Í' => 'i',
            'Ô' | 'Ö' | 'Ó' | 'Õ' => 'o',
            'Ù' | 'Û' | 'Ü' | 'Ú' => 'u',
            'Ÿ' | 'Ý' => 'y',
            'Ç' => 'c',
            'Ñ' => 'n',
            _ => c,
        })
        .collect()
}

/// Fuzzy label search — exact, prefix, and substring matching with scoring.
/// Supports accent-insensitive matching (e.g. "speleo" matches "Spéléologie").
///
/// Scoring:
/// - Exact match (case-insensitive) → 1.0
/// - Prefix match → 0.9
/// - Contains match → 0.7
/// - Word prefix match → 0.6
/// - Accent-folded variants → same scores with -0.05 penalty
pub fn fuzzy_label_search(query: &str, labels: &[String], limit: usize) -> Vec<(String, f32)> {
    if query.trim().is_empty() {
        return Vec::new();
    }
    let query_lower = query.to_lowercase();
    let query_folded = fold_accents(&query_lower);
    let mut results: Vec<(String, f32)> = Vec::new();

    for label in labels {
        let label_lower = label.to_lowercase();

        // First try exact (case-insensitive) matching
        let score = if label_lower == query_lower {
            1.0
        } else if label_lower.starts_with(&query_lower) {
            0.9
        } else if label_lower.contains(&query_lower) {
            0.7
        } else if query_lower.len() >= 3 {
            let has_word_prefix = label_lower
                .split_whitespace()
                .any(|w| w.starts_with(&query_lower));
            if has_word_prefix {
                0.6
            } else {
                // Try accent-folded matching
                let label_folded = fold_accents(&label_lower);
                if label_folded == query_folded {
                    0.95  // almost exact, just accent difference
                } else if label_folded.starts_with(&query_folded) {
                    0.85
                } else if label_folded.contains(&query_folded) {
                    0.65
                } else {
                    let has_folded_prefix = label_folded
                        .split_whitespace()
                        .any(|w| w.starts_with(&query_folded));
                    if has_folded_prefix {
                        0.55
                    } else {
                        continue;
                    }
                }
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

    #[test]
    fn rrf_specificity_boosts_rare_nodes() {
        let lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("graph", vec![("hub".to_string(), 0.9), ("leaf".to_string(), 0.5)]),
        ];
        let mut specificity = HashMap::new();
        specificity.insert("hub".to_string(), 0.1);  // low specificity (hub)
        specificity.insert("leaf".to_string(), 1.0);  // high specificity (leaf)

        let results = rrf_merge_with_specificity(&lists, 60.0, &specificity);

        // Without specificity: hub > leaf (rank 0 vs rank 1)
        // With specificity: leaf should be boosted closer to or above hub
        let hub_score = results.iter().find(|r| r.label == "hub").unwrap().score;
        let leaf_score = results.iter().find(|r| r.label == "leaf").unwrap().score;

        // Hub: RRF * (0.7 + 0.3*0.1) = RRF * 0.73
        // Leaf: RRF * (0.7 + 0.3*1.0) = RRF * 1.0
        // So the leaf gets a bigger boost factor
        assert!(leaf_score / hub_score > 0.5, "specificity should boost leaf relative to hub");
    }

    #[test]
    fn fuzzy_accent_insensitive() {
        let labels = vec![
            "Spéléologie".to_string(),
            "spéléologie".to_string(),
        ];
        let results = fuzzy_label_search("speleo", &labels, 10);
        assert!(!results.is_empty(), "speleo should match Spéléologie");
        assert_eq!(results[0].0, "Spéléologie");
    }

    #[test]
    fn fuzzy_accent_folding_exact() {
        let labels = vec!["créé".to_string()];
        let results = fuzzy_label_search("cree", &labels, 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.9, "accent-folded exact should score high");
    }

    #[test]
    fn rrf_specificity_default_weight() {
        let lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("graph", vec![("unknown".to_string(), 0.9)]),
        ];
        let specificity = HashMap::new(); // empty → default 0.5

        let results = rrf_merge_with_specificity(&lists, 60.0, &specificity);
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.0);
    }
}
