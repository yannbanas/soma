//! Automatic Named Entity Recognition (NER) via capitalization heuristics.
//!
//! Extracts proper nouns (capitalized multi-word sequences) from text and
//! creates co-occurrence links between entities in the same context window.
//! This is domain-agnostic and works for any language with capitalization.

/// An extracted named entity with its position in the original text.
#[derive(Debug, Clone)]
pub struct NamedEntity {
    pub name: String,
    /// Character offset in the original text.
    pub offset: usize,
}

/// Extract named entities from text using capitalization heuristics.
///
/// Detects:
/// - Multi-word capitalized sequences: "Grant Green", "Blue Note", "New York City"
/// - Single capitalized words that are long enough (>= 4 chars) and not sentence-initial
///
/// Filters out:
/// - Common English/French words (The, This, After, Pour, Dans...)
/// - Very short words (< 4 chars for single-word entities)
/// - Duplicates
pub fn extract_entities(text: &str) -> Vec<NamedEntity> {
    let mut entities: Vec<NamedEntity> = Vec::new();
    let mut seen: Vec<String> = Vec::new();

    let words: Vec<(usize, &str)> = text
        .split_whitespace()
        .scan(0usize, |offset, word| {
            let start = text[*offset..].find(word).unwrap_or(0) + *offset;
            *offset = start + word.len();
            Some((start, word))
        })
        .collect();

    let mut i = 0;
    while i < words.len() {
        let (offset, word) = words[i];
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-');

        if clean.is_empty() || !starts_upper(clean) || is_stopword(clean) {
            i += 1;
            continue;
        }

        // Try to build a multi-word entity
        let mut parts = vec![clean.to_string()];
        let mut j = i + 1;

        while j < words.len() {
            let (_, next_word) = words[j];
            let next_clean =
                next_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-');

            if next_clean.is_empty() {
                break;
            }

            // Allow connectives inside names: "University of Cambridge", "Tour de France"
            if is_name_connective(next_clean) && j + 1 < words.len() {
                let (_, after) = words[j + 1];
                let after_clean =
                    after.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-');
                if !after_clean.is_empty() && starts_upper(after_clean) {
                    parts.push(next_clean.to_string());
                    parts.push(after_clean.to_string());
                    j += 2;
                    continue;
                }
            }

            if starts_upper(next_clean) && !is_stopword(next_clean) {
                parts.push(next_clean.to_string());
                j += 1;
            } else {
                break;
            }
        }

        let entity_name = parts.join(" ");

        // Skip single short words (likely sentence-initial common words)
        let is_single = parts.len() == 1;
        if is_single && clean.len() < 4 {
            i += 1;
            continue;
        }

        // Skip if it's a sentence-initial single word (after ".", "!", "?")
        if is_single && i > 0 {
            let prev = words[i - 1].1;
            if !prev.ends_with('.') && !prev.ends_with('!') && !prev.ends_with('?') {
                // Not sentence-initial, keep it
            } else {
                // Sentence-initial single word — skip (likely just capitalization)
                i = j;
                continue;
            }
        }
        // First word of text and single word — also skip
        if is_single && i == 0 {
            i = j;
            continue;
        }

        // Deduplicate
        let name_lower = entity_name.to_lowercase();
        if !seen.iter().any(|s| s == &name_lower) {
            seen.push(name_lower);
            entities.push(NamedEntity {
                name: entity_name,
                offset,
            });
        }

        i = j;
    }

    entities
}

/// Generate co-occurrence pairs from entities extracted from the same chunk.
/// Returns (entity_a, entity_b) pairs — each pair appears once.
pub fn cooccurrence_pairs(entities: &[NamedEntity]) -> Vec<(&str, &str)> {
    let mut pairs = Vec::new();
    for i in 0..entities.len() {
        for j in (i + 1)..entities.len() {
            pairs.push((entities[i].name.as_str(), entities[j].name.as_str()));
        }
    }
    pairs
}

fn starts_upper(s: &str) -> bool {
    s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
}

fn is_name_connective(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "of" | "de"
            | "du"
            | "des"
            | "von"
            | "van"
            | "di"
            | "da"
            | "del"
            | "la"
            | "le"
            | "les"
            | "el"
            | "al"
            | "ibn"
            | "bin"
            | "the"
            | "and"
            | "und"
            | "et"
            | "y"
            | "e"
    )
}

fn is_stopword(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "the" | "a" | "an" | "and" | "or" | "but" | "in" | "on" | "at"
            | "to" | "for" | "is" | "it" | "by" | "as" | "he" | "she"
            | "they" | "this" | "that" | "was" | "were" | "has" | "had"
            | "have" | "been" | "are" | "his" | "her" | "its" | "from"
            | "with" | "not" | "also" | "who" | "which" | "when" | "where"
            | "what" | "how" | "after" | "before" | "during" | "between"
            | "into" | "over" | "under" | "their" | "these" | "those"
            | "then" | "than" | "some" | "such" | "only" | "other"
            | "more" | "most" | "many" | "much" | "each" | "every"
            | "both" | "few" | "all" | "any" | "own" | "same" | "so"
            | "no" | "nor" | "if" | "just" | "about" | "up" | "out"
            | "one" | "two" | "three" | "four" | "five" | "six"
            // French
            | "le" | "la" | "les" | "un" | "une" | "des" | "du" | "de"
            | "ce" | "cette" | "ces" | "il" | "elle" | "ils" | "elles"
            | "est" | "sont" | "dans" | "pour" | "avec" | "sur" | "par"
            | "plus" | "mais" | "donc" | "car" | "qui" | "que" | "dont"
            // Common sentence starters
            | "however" | "although" | "because" | "since" | "while"
            | "meanwhile" | "therefore" | "furthermore" | "moreover"
            | "according" | "following" | "several" | "various"
            | "another" | "first" | "second" | "third" | "later"
            | "early" | "earlier" | "recent" | "recently"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_multi_word_names() {
        let entities = extract_entities(
            "the album by American jazz guitarist Grant Green released on the Blue Note label",
        );
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Grant Green"), "got: {:?}", names);
        assert!(names.contains(&"Blue Note"), "got: {:?}", names);
    }

    #[test]
    fn extract_names_with_connectives() {
        let entities =
            extract_entities("he studied at the University of Cambridge and toured in New York");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"University of Cambridge"),
            "got: {:?}",
            names
        );
        assert!(names.contains(&"New York"), "got: {:?}", names);
    }

    #[test]
    fn skip_stopwords() {
        let entities = extract_entities("The quick brown fox jumps over the lazy dog");
        // No real proper nouns here
        assert!(entities.is_empty(), "got: {:?}", entities);
    }

    #[test]
    fn dedup_entities() {
        let entities = extract_entities(
            "Albert Einstein studied physics. Later Albert Einstein won the Nobel Prize",
        );
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        let einstein_count = names.iter().filter(|n| **n == "Albert Einstein").count();
        assert_eq!(einstein_count, 1);
    }

    #[test]
    fn cooccurrence_basic() {
        let entities =
            extract_entities("Scott Derrickson directed Doctor Strange for Marvel Studios");
        let pairs = cooccurrence_pairs(&entities);
        assert!(!pairs.is_empty());
    }

    #[test]
    fn empty_text() {
        assert!(extract_entities("").is_empty());
    }
}
