/// Build the L2 extraction prompt for a text chunk.
/// Instructs the LLM to return a JSON array of triplets.
pub fn build_extraction_prompt(chunk: &str) -> String {
    format!(
        r#"Extract knowledge triplets from the following text.
Return ONLY a JSON array of objects with keys: "subject", "relation", "object", "confidence".
Rules:
- subject and object are named entities or concepts (1-4 words)
- relation is a verb phrase (e.g. "is a", "derives from", "causes", "uses", "produces")
- confidence is a float between 0.0 and 1.0
- Extract 1-10 triplets, focusing on the most important facts
- If no clear triplets can be extracted, return an empty array []

Text:
{chunk}

JSON:"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_contains_chunk_text() {
        let prompt = build_extraction_prompt("ChromoQ is a fluorescent protein");
        assert!(prompt.contains("ChromoQ is a fluorescent protein"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn prompt_instructs_json_format() {
        let prompt = build_extraction_prompt("test");
        assert!(prompt.contains("subject"));
        assert!(prompt.contains("relation"));
        assert!(prompt.contains("object"));
        assert!(prompt.contains("confidence"));
    }
}
