//! Advanced traversal and query entity extraction.
//!
//! Basic traversal (BFS by effective intensity) lives in `StigreGraph::traverse()`.
//! This module provides helpers for PPR seed extraction and future algorithms.

/// Extract potential entity names from a natural-language query.
///
/// Heuristics:
/// 1. Multi-word capitalized sequences (e.g. "Acme Corp", "Alice Smith")
/// 2. Single capitalized words that are not sentence-initial stop words
/// 3. Quoted substrings
///
/// Used to seed Personalized PageRank.
pub fn extract_query_entities(query: &str) -> Vec<String> {
    let mut entities = Vec::new();

    // 1. Extract quoted strings: "Foo Bar" or 'Foo Bar'
    let mut in_quote = false;
    let mut quote_char = '"';
    let mut current_quoted = String::new();
    for ch in query.chars() {
        if !in_quote && (ch == '"' || ch == '\'') {
            in_quote = true;
            quote_char = ch;
            current_quoted.clear();
        } else if in_quote && ch == quote_char {
            in_quote = false;
            let trimmed = current_quoted.trim().to_string();
            if !trimmed.is_empty() && !entities.contains(&trimmed) {
                entities.push(trimmed);
            }
        } else if in_quote {
            current_quoted.push(ch);
        }
    }

    // 2. Extract capitalized sequences from unquoted text
    let stop_words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "what", "which", "who",
        "whom", "this", "that", "these", "those", "i", "me", "my", "myself",
        "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
        "its", "they", "them", "their",
    ];

    let words: Vec<&str> = query.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
        if word.is_empty() {
            i += 1;
            continue;
        }

        let first_char = word.chars().next().unwrap();
        if first_char.is_uppercase() && !stop_words.contains(&word.to_lowercase().as_str()) {
            // Start a capitalized sequence
            let mut seq = vec![word.to_string()];
            let mut j = i + 1;
            while j < words.len() {
                let next = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                if next.is_empty() {
                    break;
                }
                let nc = next.chars().next().unwrap();
                // Continue if capitalized, or if it's a short connective (of, the, de, von, etc.)
                if nc.is_uppercase() {
                    seq.push(next.to_string());
                    j += 1;
                } else if ["of", "the", "de", "von", "van", "del", "la", "le", "di"]
                    .contains(&next.to_lowercase().as_str())
                    && j + 1 < words.len()
                {
                    let after = words[j + 1].trim_matches(|c: char| !c.is_alphanumeric());
                    if !after.is_empty() && after.chars().next().unwrap().is_uppercase() {
                        seq.push(next.to_string());
                        j += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            let entity = seq.join(" ");
            if !entities.contains(&entity) {
                entities.push(entity);
            }
            i = j;
        } else {
            i += 1;
        }
    }

    // 3. Extract non-stopword tokens (>= 4 chars) as potential entities
    for word in &words {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
        if clean.len() >= 4
            && !stop_words.contains(&clean.to_lowercase().as_str())
            && !entities.iter().any(|e| e.to_lowercase().contains(&clean.to_lowercase()))
        {
            entities.push(clean);
        }
    }

    // 4. Fallback: if no capitalized entities found, extract consecutive non-stop bigrams/trigrams
    //    from lowercase text (handles "who founded acme corp?" → "acme corp")
    let has_capitalized = entities.iter().any(|e| e.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));
    if !has_capitalized {
        let content_words: Vec<&str> = words
            .iter()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty() && !stop_words.contains(&w.to_lowercase().as_str()))
            .collect();
        // Build bigrams and trigrams from consecutive content words
        for window in 2..=3usize {
            if content_words.len() >= window {
                for chunk in content_words.windows(window) {
                    let phrase = chunk.join(" ");
                    if phrase.len() >= 4
                        && !entities.iter().any(|e| e.to_lowercase() == phrase.to_lowercase())
                    {
                        entities.push(phrase);
                    }
                }
            }
        }
    }

    entities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_capitalized_names() {
        let entities = extract_query_entities("Who is the CEO of Acme Corp?");
        assert!(entities.iter().any(|e| e.contains("Acme Corp")));
    }

    #[test]
    fn extract_quoted_entities() {
        let entities = extract_query_entities("What is \"ChromoQ\" used for?");
        assert!(entities.contains(&"ChromoQ".to_string()));
    }

    #[test]
    fn extract_multi_word_names() {
        let entities = extract_query_entities("Tell me about Alice Smith and Bob Jones");
        assert!(entities.iter().any(|e| e.contains("Alice Smith")));
        assert!(entities.iter().any(|e| e.contains("Bob Jones")));
    }

    #[test]
    fn extract_empty_query() {
        let entities = extract_query_entities("");
        assert!(entities.is_empty());
    }

    #[test]
    fn extract_no_entities() {
        let entities = extract_query_entities("what is the best way to do this?");
        // Should not extract stopwords — may extract content word bigrams
        assert!(entities.iter().all(|e| {
            let words: Vec<&str> = e.split_whitespace().collect();
            words.iter().all(|w| {
                let lower = w.to_lowercase();
                !["the", "a", "an", "is", "are", "was", "to", "of", "in", "for", "do", "this"]
                    .contains(&lower.as_str())
                    || words.len() > 1
            })
        }));
    }

    #[test]
    fn extract_lowercase_entities() {
        let entities = extract_query_entities("who founded acme corp in new york?");
        // Should extract bigrams like "acme corp", "founded acme", etc.
        assert!(entities.iter().any(|e| e.to_lowercase().contains("acme")));
    }
}
