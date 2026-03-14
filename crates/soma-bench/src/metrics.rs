//! Evaluation metrics for knowledge graph retrieval.
//!
//! - Entity Recall@K: fraction of gold entities in top-K results.
//! - Path Recall: multi-hop reasoning chain coverage.
//! - Token F1: standard QA token-level F1.
//! - Exact Match: normalized string comparison.

/// Entity Recall@K: fraction of gold entities found in top-K retrieval results.
///
/// Uses case-insensitive substring matching (a gold entity "Einstein" matches
/// a retrieved label "Albert Einstein" and vice versa).
///
/// Returns: (recall, hit_count, total_gold)
pub fn entity_recall_at_k(
    gold_entities: &[String],
    retrieved_labels: &[String],
    k: usize,
) -> (f32, usize, usize) {
    let top_k: Vec<String> = retrieved_labels
        .iter()
        .take(k)
        .map(|s| s.to_lowercase())
        .collect();

    let hits = gold_entities
        .iter()
        .filter(|g| {
            let g_lower = g.to_lowercase();
            top_k
                .iter()
                .any(|r| r.contains(&g_lower) || g_lower.contains(r))
        })
        .count();

    let total = gold_entities.len();
    let recall = if total > 0 {
        hits as f32 / total as f32
    } else {
        0.0
    };
    (recall, hits, total)
}

/// Path Recall: for multi-hop questions, checks whether the retrieved entities
/// cover the full reasoning chain.
///
/// `chain_entities`: ordered entities in the gold reasoning path.
/// `retrieved_labels`: labels from SOMA search results.
///
/// Returns: fraction of chain entities found in retrieved labels.
pub fn path_recall(chain_entities: &[String], retrieved_labels: &[String]) -> f32 {
    if chain_entities.is_empty() {
        return 0.0;
    }

    let retrieved_lower: Vec<String> = retrieved_labels.iter().map(|s| s.to_lowercase()).collect();

    let hits = chain_entities
        .iter()
        .filter(|c| {
            let c_lower = c.to_lowercase();
            retrieved_lower
                .iter()
                .any(|r| r.contains(&c_lower) || c_lower.contains(r))
        })
        .count();

    hits as f32 / chain_entities.len() as f32
}

/// Token-level F1 score (standard for QA evaluation).
///
/// Computes precision and recall over whitespace-separated tokens,
/// then returns their harmonic mean.
pub fn token_f1(prediction: &str, gold: &str) -> f32 {
    let pred_tokens: std::collections::HashSet<String> = normalize_answer(prediction)
        .split_whitespace()
        .map(String::from)
        .collect();
    let gold_tokens: std::collections::HashSet<String> = normalize_answer(gold)
        .split_whitespace()
        .map(String::from)
        .collect();

    if pred_tokens.is_empty() && gold_tokens.is_empty() {
        return 1.0;
    }
    if pred_tokens.is_empty() || gold_tokens.is_empty() {
        return 0.0;
    }

    let common = pred_tokens.intersection(&gold_tokens).count() as f32;
    let precision = common / pred_tokens.len() as f32;
    let recall = common / gold_tokens.len() as f32;

    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Exact Match: returns true if normalized strings are identical.
pub fn exact_match(prediction: &str, gold: &str) -> bool {
    normalize_answer(prediction) == normalize_answer(gold)
}

/// Normalize an answer string: lowercase, remove punctuation, collapse whitespace.
fn normalize_answer(s: &str) -> String {
    s.to_lowercase()
        .replace(
            &['(', ')', ',', '.', '!', '?', ';', ':', '\'', '"', '-'][..],
            " ",
        )
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Mean Reciprocal Rank: average of 1/rank for first correct result.
pub fn mean_reciprocal_rank(ranks: &[Option<usize>]) -> f32 {
    if ranks.is_empty() {
        return 0.0;
    }
    let sum: f32 = ranks
        .iter()
        .map(|r| match r {
            Some(rank) => 1.0 / (*rank as f32 + 1.0),
            None => 0.0,
        })
        .sum();
    sum / ranks.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── entity_recall_at_k ─────────────────────────────────────

    #[test]
    fn recall_perfect() {
        let gold = vec!["Alice".to_string(), "Bob".to_string()];
        let retrieved = vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()];
        let (recall, hits, total) = entity_recall_at_k(&gold, &retrieved, 5);
        assert_eq!(recall, 1.0);
        assert_eq!(hits, 2);
        assert_eq!(total, 2);
    }

    #[test]
    fn recall_partial() {
        let gold = vec!["Alice".to_string(), "Bob".to_string()];
        let retrieved = vec!["Alice".to_string(), "Carol".to_string()];
        let (recall, hits, _) = entity_recall_at_k(&gold, &retrieved, 5);
        assert_eq!(hits, 1);
        assert!((recall - 0.5).abs() < 0.001);
    }

    #[test]
    fn recall_substring_match() {
        let gold = vec!["Einstein".to_string()];
        let retrieved = vec!["Albert Einstein".to_string()];
        let (recall, hits, _) = entity_recall_at_k(&gold, &retrieved, 5);
        assert_eq!(hits, 1);
        assert_eq!(recall, 1.0);
    }

    #[test]
    fn recall_case_insensitive() {
        let gold = vec!["alice".to_string()];
        let retrieved = vec!["ALICE".to_string()];
        let (recall, _, _) = entity_recall_at_k(&gold, &retrieved, 5);
        assert_eq!(recall, 1.0);
    }

    #[test]
    fn recall_respects_k() {
        let gold = vec!["Bob".to_string()];
        let retrieved = vec!["Alice".to_string(), "Carol".to_string(), "Bob".to_string()];
        let (recall_k2, _, _) = entity_recall_at_k(&gold, &retrieved, 2);
        let (recall_k3, _, _) = entity_recall_at_k(&gold, &retrieved, 3);
        assert_eq!(recall_k2, 0.0); // Bob is at position 3, beyond k=2
        assert_eq!(recall_k3, 1.0);
    }

    #[test]
    fn recall_empty_gold() {
        let gold: Vec<String> = vec![];
        let retrieved = vec!["Alice".to_string()];
        let (recall, _, _) = entity_recall_at_k(&gold, &retrieved, 5);
        assert_eq!(recall, 0.0);
    }

    // ── path_recall ────────────────────────────────────────────

    #[test]
    fn path_recall_full_chain() {
        let chain = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let retrieved = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        assert_eq!(path_recall(&chain, &retrieved), 1.0);
    }

    #[test]
    fn path_recall_partial_chain() {
        let chain = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let retrieved = vec!["A".to_string(), "D".to_string()];
        assert!((path_recall(&chain, &retrieved) - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn path_recall_empty() {
        let chain: Vec<String> = vec![];
        let retrieved = vec!["A".to_string()];
        assert_eq!(path_recall(&chain, &retrieved), 0.0);
    }

    // ── token_f1 ───────────────────────────────────────────────

    #[test]
    fn f1_perfect() {
        assert_eq!(token_f1("the cat", "the cat"), 1.0);
    }

    #[test]
    fn f1_partial() {
        let f1 = token_f1("the cat sat", "the cat");
        // precision = 2/3, recall = 2/2 = 1.0, F1 = 2*(2/3)*1/(2/3+1) = 0.8
        assert!((f1 - 0.8).abs() < 0.01);
    }

    #[test]
    fn f1_no_overlap() {
        assert_eq!(token_f1("dog", "cat"), 0.0);
    }

    #[test]
    fn f1_both_empty() {
        assert_eq!(token_f1("", ""), 1.0);
    }

    // ── exact_match ────────────────────────────────────────────

    #[test]
    fn em_identical() {
        assert!(exact_match("Paris", "paris"));
    }

    #[test]
    fn em_with_punctuation() {
        assert!(exact_match("New York, USA", "new york usa"));
    }

    #[test]
    fn em_different() {
        assert!(!exact_match("Paris", "London"));
    }

    // ── mrr ────────────────────────────────────────────────────

    #[test]
    fn mrr_basic() {
        let ranks = vec![Some(0), Some(2), None];
        let mrr = mean_reciprocal_rank(&ranks);
        // (1/1 + 1/3 + 0) / 3 = 1.333/3 = 0.444
        assert!((mrr - 0.444).abs() < 0.01);
    }

    #[test]
    fn mrr_empty() {
        assert_eq!(mean_reciprocal_rank(&[]), 0.0);
    }
}
