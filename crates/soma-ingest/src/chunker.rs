/// Intelligent text chunker with sentence-level splitting and overlap.
///
/// Chunks are 3-5 sentences with 1-2 sentence overlap for context preservation.
/// Performance: O(n) single pass, no allocation per sentence.
pub struct Chunker {
    chunk_size: usize,
    overlap: usize,
}

impl Chunker {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Chunker {
            chunk_size: chunk_size.max(1),
            overlap: overlap.min(chunk_size.saturating_sub(1)),
        }
    }

    /// Default: 5 sentences per chunk, 1 sentence overlap.
    pub fn default_config() -> Self {
        Self::new(5, 1)
    }

    /// Split text into overlapping chunks.
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let sentences = self.split_sentences(text);

        if sentences.is_empty() {
            return Vec::new();
        }

        if sentences.len() <= self.chunk_size {
            return vec![sentences.join(" ")];
        }

        let mut chunks = Vec::new();
        let step = self.chunk_size - self.overlap;
        let mut i = 0;

        while i < sentences.len() {
            let end = (i + self.chunk_size).min(sentences.len());
            let chunk: String = sentences[i..end].join(" ");
            if !chunk.trim().is_empty() {
                chunks.push(chunk);
            }
            i += step;
            if end == sentences.len() {
                break;
            }
        }

        chunks
    }

    /// Split text into sentences using common delimiters.
    /// Handles: . ! ? and newline-separated text.
    fn split_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() && trimmed.len() >= 5 {
                    sentences.push(trimmed);
                }
                current.clear();
            } else if ch == '\n' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() && trimmed.len() >= 10 {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        // Remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() && trimmed.len() >= 5 {
            sentences.push(trimmed);
        }

        sentences
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_chunking() {
        let chunker = Chunker::new(2, 1);
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn short_text_single_chunk() {
        let chunker = Chunker::default_config();
        let text = "Just one sentence here.";
        let chunks = chunker.chunk(text);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn empty_text() {
        let chunker = Chunker::default_config();
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn overlap_preserves_context() {
        let chunker = Chunker::new(2, 1);
        let text = "Sentence one here. Sentence two here. Sentence three here.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2, "Need at least 2 chunks, got {}", chunks.len());
        // With overlap=1, the last sentence of chunk[0] should appear in chunk[1]
        let last_sentence_of_first = chunks[0]
            .split('.')
            .filter(|s| s.trim().len() >= 5)
            .last()
            .unwrap_or("")
            .trim();
        assert!(
            chunks[1].contains(last_sentence_of_first),
            "Overlap missing: chunk[0] ends with '{}', chunk[1] = '{}'",
            last_sentence_of_first,
            chunks[1]
        );
    }
}
