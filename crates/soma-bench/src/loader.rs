//! Dataset loaders for MuSiQue and HotpotQA.

use serde::{Deserialize, Serialize};
use soma_core::SomaError;
use std::path::Path;

/// Dataset origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetKind {
    MuSiQue,
    HotpotQA,
}

/// A supporting paragraph with ground-truth entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportingParagraph {
    pub title: String,
    pub text: String,
    /// Key entities extracted from title + text.
    pub entities: Vec<String>,
}

/// A single benchmark question with ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchQuestion {
    pub id: String,
    pub question: String,
    pub answer: String,
    /// Supporting paragraphs (gold evidence).
    pub supporting_paragraphs: Vec<SupportingParagraph>,
    /// All context paragraphs (gold + distractors).
    pub all_paragraphs: Vec<(String, String)>,
    /// Number of reasoning hops required.
    pub num_hops: u8,
    pub dataset: DatasetKind,
}

// ── MuSiQue raw JSON structures ────────────────────────────────

#[derive(Deserialize)]
struct MuSiQueParagraph {
    #[serde(default)]
    title: String,
    paragraph_text: String,
    is_supporting: bool,
}

#[derive(Deserialize)]
struct MuSiQueEntry {
    id: String,
    question: String,
    answer: String,
    paragraphs: Vec<MuSiQueParagraph>,
    #[serde(default, rename = "answer_aliases")]
    _answer_aliases: Vec<String>,
}

/// Load MuSiQue-Ans dataset from a JSONL file.
///
/// Format: one JSON object per line.
/// Each object has: id, question, answer, paragraphs[{title, paragraph_text, is_supporting}].
pub fn load_musique(path: &Path, limit: usize) -> Result<Vec<BenchQuestion>, SomaError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| SomaError::Store(format!("Failed to read MuSiQue file: {}", e)))?;

    let mut questions = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: MuSiQueEntry = serde_json::from_str(line)
            .map_err(|e| SomaError::Store(format!("MuSiQue parse error: {}", e)))?;

        let supporting: Vec<SupportingParagraph> = entry
            .paragraphs
            .iter()
            .filter(|p| p.is_supporting)
            .map(|p| {
                let entities =
                    extract_entities_from_text(&p.title, &p.paragraph_text, &entry.answer);
                SupportingParagraph {
                    title: p.title.clone(),
                    text: p.paragraph_text.clone(),
                    entities,
                }
            })
            .collect();

        let all_paragraphs: Vec<(String, String)> = entry
            .paragraphs
            .iter()
            .map(|p| (p.title.clone(), p.paragraph_text.clone()))
            .collect();

        let num_hops = supporting.len().max(1) as u8;

        questions.push(BenchQuestion {
            id: entry.id,
            question: entry.question,
            answer: entry.answer,
            supporting_paragraphs: supporting,
            all_paragraphs,
            num_hops,
            dataset: DatasetKind::MuSiQue,
        });

        if questions.len() >= limit {
            break;
        }
    }

    Ok(questions)
}

// ── HotpotQA raw JSON structures ───────────────────────────────

#[derive(Deserialize)]
struct HotpotQAEntry {
    #[serde(rename = "_id")]
    id: String,
    question: String,
    answer: String,
    /// Format: [["title", sent_id], ["title", sent_id], ...]
    supporting_facts: Vec<(String, usize)>,
    context: HotpotQAContext,
    #[serde(default, rename = "level")]
    _level: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HotpotQAContext {
    Nested(Vec<(String, Vec<String>)>),
    Flat(Vec<Vec<serde_json::Value>>),
}

/// Load HotpotQA distractor-setting dataset from a JSON file.
///
/// Format: JSON array of objects.
/// Each object has: _id, question, answer, supporting_facts, context.
pub fn load_hotpotqa(path: &Path, limit: usize) -> Result<Vec<BenchQuestion>, SomaError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| SomaError::Store(format!("Failed to read HotpotQA file: {}", e)))?;

    let entries: Vec<HotpotQAEntry> = serde_json::from_str(&content)
        .map_err(|e| SomaError::Store(format!("HotpotQA parse error: {}", e)))?;

    let mut questions = Vec::new();
    let support_titles_set = |entry: &HotpotQAEntry| -> std::collections::HashSet<String> {
        entry
            .supporting_facts
            .iter()
            .map(|(title, _)| title.clone())
            .collect()
    };

    for entry in entries.iter().take(limit) {
        let gold_titles = support_titles_set(entry);

        let all_paragraphs: Vec<(String, String)> = match &entry.context {
            HotpotQAContext::Nested(ctx) => ctx
                .iter()
                .map(|(title, sents)| (title.clone(), sents.join(" ")))
                .collect(),
            HotpotQAContext::Flat(ctx) => ctx
                .iter()
                .filter_map(|item| {
                    if item.len() >= 2 {
                        let title = item[0].as_str().unwrap_or("").to_string();
                        let text = if let Some(arr) = item[1].as_array() {
                            arr.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(" ")
                        } else {
                            item[1].as_str().unwrap_or("").to_string()
                        };
                        Some((title, text))
                    } else {
                        None
                    }
                })
                .collect(),
        };

        let supporting: Vec<SupportingParagraph> = all_paragraphs
            .iter()
            .filter(|(title, _)| gold_titles.contains(title))
            .map(|(title, text)| {
                let entities = extract_entities_from_text(title, text, &entry.answer);
                SupportingParagraph {
                    title: title.clone(),
                    text: text.clone(),
                    entities,
                }
            })
            .collect();

        questions.push(BenchQuestion {
            id: entry.id.clone(),
            question: entry.question.clone(),
            answer: entry.answer.clone(),
            supporting_paragraphs: supporting,
            all_paragraphs,
            num_hops: 2,
            dataset: DatasetKind::HotpotQA,
        });
    }

    Ok(questions)
}

// ── Entity extraction from text ────────────────────────────────

/// Extract probable entity names from paragraph title, text, and answer.
///
/// Heuristic: title words, capitalized phrases in text, answer tokens.
fn extract_entities_from_text(title: &str, text: &str, answer: &str) -> Vec<String> {
    let mut entities = Vec::new();

    // Title is almost always an entity name
    let title_trimmed = title.trim();
    if !title_trimmed.is_empty() {
        entities.push(title_trimmed.to_string());
    }

    // Answer tokens are gold entities
    let answer_trimmed = answer.trim();
    if !answer_trimmed.is_empty() && answer_trimmed.len() < 100 {
        entities.push(answer_trimmed.to_string());
    }

    // Capitalized multi-word phrases from text
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
        if !word.is_empty()
            && word.len() >= 2
            && word
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
            && !is_common_word(word)
        {
            let mut parts = vec![word.to_string()];
            let mut j = i + 1;
            while j < words.len() {
                let next = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                if !next.is_empty()
                    && next
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                {
                    parts.push(next.to_string());
                    j += 1;
                } else {
                    break;
                }
            }
            let entity = parts.join(" ");
            if entity.len() >= 2 && !entities.contains(&entity) {
                entities.push(entity);
            }
            i = j;
        } else {
            i += 1;
        }
    }

    entities
}

fn is_common_word(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "the"
            | "a"
            | "an"
            | "and"
            | "or"
            | "but"
            | "in"
            | "on"
            | "at"
            | "to"
            | "for"
            | "of"
            | "is"
            | "it"
            | "by"
            | "as"
            | "he"
            | "she"
            | "they"
            | "this"
            | "that"
            | "was"
            | "were"
            | "has"
            | "had"
            | "have"
            | "been"
            | "are"
            | "his"
            | "her"
            | "its"
            | "from"
            | "with"
            | "not"
            | "also"
            | "who"
            | "which"
            | "when"
            | "where"
            | "what"
            | "how"
            | "after"
            | "before"
            | "during"
            | "between"
            | "into"
            | "over"
            | "under"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_entities_basic() {
        let entities = extract_entities_from_text(
            "Albert Einstein",
            "Albert Einstein was born in Ulm, Germany. He developed the Theory of Relativity.",
            "physicist",
        );
        assert!(entities.contains(&"Albert Einstein".to_string()));
        assert!(entities.contains(&"physicist".to_string()));
    }

    #[test]
    fn extract_entities_empty() {
        let entities = extract_entities_from_text("", "", "");
        assert!(entities.is_empty());
    }

    #[test]
    fn extract_entities_skips_common() {
        let entities =
            extract_entities_from_text("Paris", "The city is known for its art.", "France");
        assert!(entities.contains(&"Paris".to_string()));
        assert!(entities.contains(&"France".to_string()));
        // "The" should not be extracted
        assert!(!entities.iter().any(|e| e == "The"));
    }
}
