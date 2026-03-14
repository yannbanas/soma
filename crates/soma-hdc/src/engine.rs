use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// HDC Engine — Hyperdimensional Computing semantic index.
///
/// Uses Random Indexing (distributional semantics) with D=10000 dimensions.
/// Each token gets a sparse random base vector (±1, ~1% active).
/// Context vectors are built by accumulating TF-IDF weighted neighbor base vectors
/// within a sliding window.
///
/// Performance: all vector ops are O(D) with no allocation after init.
/// Security: no external calls, deterministic from seed.
#[derive(Clone)]
pub struct HdcEngine {
    /// Vector dimensionality
    dim: usize,
    /// Sparse base vectors per token (±1 at ~1% of positions)
    base_vectors: HashMap<String, Vec<f32>>,
    /// Distributional context vectors (enriched by co-occurrence)
    context_vectors: HashMap<String, Vec<f32>>,
    /// Co-occurrence window size
    window: usize,
    /// IDF weights per token
    idf: HashMap<String, f32>,
    /// Use TF-IDF weighting
    use_tfidf: bool,
    /// Total number of documents trained on
    doc_count: usize,
    /// Document frequency per token
    df: HashMap<String, usize>,
    /// Optional neural embeddings from Ollama (keyed by lowercased token/sentence).
    /// When present, these are preferred over Random Indexing vectors for similarity.
    neural_embeddings: HashMap<String, Vec<f32>>,
    /// Dimensionality of neural embeddings (may differ from HDC dim).
    neural_dim: Option<usize>,
}

/// Serializable snapshot of HDC engine state.
#[derive(Serialize, Deserialize)]
pub struct HdcSnapshot {
    pub dim: usize,
    pub window: usize,
    pub use_tfidf: bool,
    pub doc_count: usize,
    pub vocab: Vec<String>,
    pub base_data: Vec<f32>,
    pub context_data: Vec<f32>,
    pub df: Vec<(String, usize)>,
    #[serde(default)]
    pub neural_vocab: Vec<String>,
    #[serde(default)]
    pub neural_data: Vec<f32>,
    #[serde(default)]
    pub neural_dim: Option<usize>,
}

impl HdcEngine {
    /// Create a new HDC engine with given dimensionality and window size.
    pub fn new(dim: usize, window: usize, use_tfidf: bool) -> Self {
        HdcEngine {
            dim,
            base_vectors: HashMap::new(),
            context_vectors: HashMap::new(),
            window,
            idf: HashMap::new(),
            use_tfidf,
            doc_count: 0,
            df: HashMap::new(),
            neural_embeddings: HashMap::new(),
            neural_dim: None,
        }
    }

    /// Default configuration: D=10000, window=5, TF-IDF on.
    pub fn default_config() -> Self {
        Self::new(10_000, 5, true)
    }

    /// Get or create a sparse base vector for a token.
    /// Deterministic from token (seeded RNG from token hash).
    fn get_or_create_base(&mut self, token: &str) -> Vec<f32> {
        if let Some(v) = self.base_vectors.get(token) {
            return v.clone();
        }

        // Deterministic seed from token for reproducibility
        let seed = {
            let mut h = 0u64;
            for (i, b) in token.bytes().enumerate() {
                h = h
                    .wrapping_mul(31)
                    .wrapping_add(b as u64)
                    .wrapping_add(i as u64);
            }
            h
        };
        let mut rng = StdRng::seed_from_u64(seed);

        // Sparse random vector: ~1% of positions are ±1
        let active_count = (self.dim as f32 * 0.01).ceil() as usize;
        let mut vec = vec![0.0f32; self.dim];

        for _ in 0..active_count {
            let pos = rng.gen_range(0..self.dim);
            vec[pos] = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        }

        self.base_vectors.insert(token.to_string(), vec.clone());
        vec
    }

    /// Tokenize text into lowercase tokens, filtering noise.
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|s| s.len() >= 2)
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Train on a corpus of sentences/documents.
    /// Builds context vectors from co-occurrence within sliding window.
    pub fn train(&mut self, sentences: &[String]) {
        // Phase 1: compute document frequencies for IDF
        for sentence in sentences {
            let tokens = Self::tokenize(sentence);
            let unique: std::collections::HashSet<&str> =
                tokens.iter().map(|s| s.as_str()).collect();
            for token in unique {
                *self.df.entry(token.to_string()).or_insert(0) += 1;
            }
            self.doc_count += 1;
        }

        // Compute IDF: log(N / df(t))
        if self.use_tfidf && self.doc_count > 0 {
            let n = self.doc_count as f32;
            for (token, &df) in &self.df {
                let idf_val = (n / (df as f32 + 1.0)).ln() + 1.0;
                self.idf.insert(token.clone(), idf_val);
            }
        }

        // Phase 2: build context vectors
        for sentence in sentences {
            let tokens = Self::tokenize(sentence);
            let len = tokens.len();

            // Ensure all base vectors exist
            for token in &tokens {
                self.get_or_create_base(token);
            }

            // For each token, accumulate weighted base vectors of neighbors
            for (i, token) in tokens.iter().enumerate() {
                let start = i.saturating_sub(self.window);
                let end = (i + self.window + 1).min(len);

                // Single-token sentence: use base vector as context vector
                if len == 1 {
                    let base = self
                        .base_vectors
                        .get(token)
                        .expect("base vector must exist")
                        .clone();
                    self.context_vectors.entry(token.clone()).or_insert(base);
                    continue;
                }

                for (j, neighbor) in tokens[start..end].iter().enumerate() {
                    let j = j + start;
                    if i == j {
                        continue;
                    }
                    let base = self
                        .base_vectors
                        .get(neighbor)
                        .expect("base vector must exist (created in loop above)")
                        .clone();

                    let weight = if self.use_tfidf {
                        *self.idf.get(neighbor).unwrap_or(&1.0)
                    } else {
                        1.0
                    };

                    let ctx = self
                        .context_vectors
                        .entry(token.clone())
                        .or_insert_with(|| vec![0.0f32; self.dim]);

                    for (k, &v) in base.iter().enumerate() {
                        ctx[k] += v * weight;
                    }
                }
            }
        }

        // Phase 3: normalize context vectors
        for vec in self.context_vectors.values_mut() {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for v in vec.iter_mut() {
                    *v /= norm;
                }
            }
        }
    }

    /// Cosine similarity between two token vectors.
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        let va = match self.context_vectors.get(&a_lower) {
            Some(v) => v,
            None => return 0.0,
        };
        let vb = match self.context_vectors.get(&b_lower) {
            Some(v) => v,
            None => return 0.0,
        };

        cosine_similarity(va, vb)
    }

    /// Top-K most similar tokens to a given token.
    pub fn most_similar(&self, token: &str, k: usize) -> Vec<(String, f32)> {
        let token_lower = token.to_lowercase();
        let target = match self.context_vectors.get(&token_lower) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut scores: Vec<(String, f32)> = self
            .context_vectors
            .iter()
            .filter(|(t, _)| *t != &token_lower)
            .map(|(t, v)| (t.clone(), cosine_similarity(target, v)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Encode a sentence as a centroid vector (TF-IDF weighted average).
    /// Prefers neural embedding if available for the exact sentence.
    pub fn encode_sentence(&self, sentence: &str) -> Vec<f32> {
        // Check if we have a pre-computed neural embedding for this exact sentence
        let key = sentence.to_lowercase();
        if let Some(neural) = self.neural_embeddings.get(&key) {
            return neural.clone();
        }

        let tokens = Self::tokenize(sentence);
        let mut centroid = vec![0.0f32; self.dim];
        let mut count = 0.0f32;

        for token in &tokens {
            if let Some(vec) = self.context_vectors.get(token) {
                let weight = if self.use_tfidf {
                    *self.idf.get(token).unwrap_or(&1.0)
                } else {
                    1.0
                };
                for (i, &v) in vec.iter().enumerate() {
                    centroid[i] += v * weight;
                }
                count += weight;
            }
        }

        if count > 1e-8 {
            for v in centroid.iter_mut() {
                *v /= count;
            }
        }

        centroid
    }

    /// Search: top-K candidates by sentence similarity.
    pub fn search(&self, query: &str, candidates: &[String], k: usize) -> Vec<(String, f32)> {
        let query_vec = self.encode_sentence(query);

        let mut scores: Vec<(String, f32)> = candidates
            .iter()
            .map(|c| {
                let c_vec = self.encode_sentence(c);
                (c.clone(), cosine_similarity(&query_vec, &c_vec))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Modulate a token's vector based on graph topology.
    /// Hub nodes (high Trail strength) get amplified.
    /// Alarmed nodes get dampened.
    pub fn modulate(&mut self, label: &str, trail_strength: f32, has_alarm: bool) {
        let label_lower = label.to_lowercase();
        if let Some(vec) = self.context_vectors.get_mut(&label_lower) {
            let factor = if has_alarm {
                0.5 // dampen alarmed nodes
            } else {
                1.0 + trail_strength.min(2.0) * 0.1 // amplify hubs (max 1.2x)
            };
            for v in vec.iter_mut() {
                *v *= factor;
            }
            // Re-normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for v in vec.iter_mut() {
                    *v /= norm;
                }
            }
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.context_vectors.len()
    }

    /// Get dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Check if a token has a context vector.
    pub fn has_token(&self, token: &str) -> bool {
        self.context_vectors.contains_key(&token.to_lowercase())
    }

    // --- Neural embedding methods ---

    /// Store a neural embedding for a token or sentence.
    /// The embedding will be preferred over HDC vectors for similarity.
    pub fn set_neural_embedding(&mut self, key: &str, embedding: Vec<f32>) {
        let dim = embedding.len();
        if self.neural_dim.is_none() {
            self.neural_dim = Some(dim);
        }
        self.neural_embeddings.insert(key.to_lowercase(), embedding);
    }

    /// Check if a neural embedding exists for a key.
    pub fn has_neural_embedding(&self, key: &str) -> bool {
        self.neural_embeddings.contains_key(&key.to_lowercase())
    }

    /// Number of neural embeddings stored.
    pub fn neural_count(&self) -> usize {
        self.neural_embeddings.len()
    }

    /// Get neural embedding dimension.
    pub fn neural_dim(&self) -> Option<usize> {
        self.neural_dim
    }

    /// Compute similarity using neural embeddings if both keys have them,
    /// otherwise fall back to HDC similarity.
    pub fn similarity_hybrid(&self, a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();

        if let (Some(va), Some(vb)) = (
            self.neural_embeddings.get(&a_lower),
            self.neural_embeddings.get(&b_lower),
        ) {
            return cosine_similarity(va, vb);
        }

        // Fall back to HDC
        self.similarity(a, b)
    }

    /// Dimension-safe semantic search using similarity_hybrid().
    /// Unlike search(), this handles mixed neural/HDC dimensions correctly
    /// by never comparing vectors of different dimensions directly.
    pub fn search_labels(&self, query: &str, labels: &[String], k: usize) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = labels
            .iter()
            .map(|label| {
                let sim = self.similarity_hybrid(query, label);
                (label.clone(), sim)
            })
            .filter(|(_, sim)| *sim > 0.01) // filter near-zero scores
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }

    /// Check if a token has a context vector (for auto-training on ingest).
    pub fn has_context_vector(&self, token: &str) -> bool {
        self.context_vectors.contains_key(&token.to_lowercase())
    }

    /// Get all labels that have been trained (for embedding commands).
    pub fn all_labels(&self) -> Vec<String> {
        self.context_vectors.keys().cloned().collect()
    }

    /// Export to serializable snapshot.
    pub fn to_snapshot(&self) -> HdcSnapshot {
        let vocab: Vec<String> = self.context_vectors.keys().cloned().collect();
        let mut base_data = Vec::with_capacity(vocab.len() * self.dim);
        let mut context_data = Vec::with_capacity(vocab.len() * self.dim);

        for token in &vocab {
            if let Some(bv) = self.base_vectors.get(token) {
                base_data.extend_from_slice(bv);
            } else {
                base_data.extend(std::iter::repeat_n(0.0f32, self.dim));
            }
            if let Some(cv) = self.context_vectors.get(token) {
                context_data.extend_from_slice(cv);
            } else {
                context_data.extend(std::iter::repeat_n(0.0f32, self.dim));
            }
        }

        let df: Vec<(String, usize)> = self.df.iter().map(|(k, &v)| (k.clone(), v)).collect();

        // Neural embeddings
        let neural_vocab: Vec<String> = self.neural_embeddings.keys().cloned().collect();
        let ndim = self.neural_dim.unwrap_or(0);
        let mut neural_data = Vec::with_capacity(neural_vocab.len() * ndim);
        for key in &neural_vocab {
            if let Some(emb) = self.neural_embeddings.get(key) {
                neural_data.extend_from_slice(emb);
            }
        }

        HdcSnapshot {
            dim: self.dim,
            window: self.window,
            use_tfidf: self.use_tfidf,
            doc_count: self.doc_count,
            vocab,
            base_data,
            context_data,
            df,
            neural_vocab,
            neural_data,
            neural_dim: self.neural_dim,
        }
    }

    /// Restore from snapshot.
    pub fn from_snapshot(snap: HdcSnapshot) -> Self {
        let mut engine = HdcEngine::new(snap.dim, snap.window, snap.use_tfidf);
        engine.doc_count = snap.doc_count;

        for (i, token) in snap.vocab.iter().enumerate() {
            let offset = i * snap.dim;
            let end = offset + snap.dim;

            if end <= snap.base_data.len() {
                engine
                    .base_vectors
                    .insert(token.clone(), snap.base_data[offset..end].to_vec());
            }
            if end <= snap.context_data.len() {
                engine
                    .context_vectors
                    .insert(token.clone(), snap.context_data[offset..end].to_vec());
            }
        }

        for (token, count) in snap.df {
            engine.df.insert(token, count);
        }

        // Rebuild IDF
        if engine.use_tfidf && engine.doc_count > 0 {
            let n = engine.doc_count as f32;
            for (token, &df) in &engine.df {
                let idf_val = (n / (df as f32 + 1.0)).ln() + 1.0;
                engine.idf.insert(token.clone(), idf_val);
            }
        }

        // Restore neural embeddings
        if let Some(ndim) = snap.neural_dim {
            engine.neural_dim = Some(ndim);
            if ndim > 0 {
                for (i, key) in snap.neural_vocab.iter().enumerate() {
                    let offset = i * ndim;
                    let end = offset + ndim;
                    if end <= snap.neural_data.len() {
                        engine
                            .neural_embeddings
                            .insert(key.clone(), snap.neural_data[offset..end].to_vec());
                    }
                }
            }
        }

        engine
    }
}

/// Cosine similarity between two f32 slices. Returns 0 for zero vectors.
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    // Manual loop for performance (avoids iterator overhead on hot path)
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trained_engine() -> HdcEngine {
        let mut engine = HdcEngine::new(1000, 3, true); // smaller dim for tests
        engine.train(&[
            "ChromoQ derives from EGFP protein".to_string(),
            "EGFP is a green fluorescent protein".to_string(),
            "ChromoQ emission at 523nm wavelength".to_string(),
            "AlphaFold2 predicts protein structure".to_string(),
            "Rust is a systems programming language".to_string(),
            "WebAssembly runs in the browser".to_string(),
        ]);
        engine
    }

    #[test]
    fn training_creates_vectors() {
        let engine = make_trained_engine();
        assert!(engine.vocab_size() > 0);
        assert!(engine.has_token("chromoq"));
        assert!(engine.has_token("egfp"));
    }

    #[test]
    fn similar_tokens_have_higher_similarity() {
        let engine = make_trained_engine();
        // Protein-related terms should be more similar to each other than to unrelated terms
        let chromoq_egfp = engine.similarity("chromoq", "egfp");
        let chromoq_rust = engine.similarity("chromoq", "rust");
        assert!(chromoq_egfp.is_finite());
        assert!(chromoq_rust.is_finite());
        // chromoq and egfp co-occur in protein sentences; chromoq and rust don't
        assert!(
            chromoq_egfp > chromoq_rust,
            "chromoq-egfp ({:.4}) should be > chromoq-rust ({:.4})",
            chromoq_egfp,
            chromoq_rust
        );
    }

    #[test]
    fn most_similar_returns_correct_count() {
        let engine = make_trained_engine();
        let results = engine.most_similar("protein", 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn encode_sentence_returns_correct_dim() {
        let engine = make_trained_engine();
        let vec = engine.encode_sentence("ChromoQ protein structure");
        assert_eq!(vec.len(), 1000);
    }

    #[test]
    fn search_returns_ranked_results() {
        let engine = make_trained_engine();
        let candidates = vec![
            "ChromoQ protein".to_string(),
            "Rust language".to_string(),
            "EGFP fluorescent".to_string(),
        ];
        let results = engine.search("protein structure", &candidates, 3);
        assert_eq!(results.len(), 3);
        // Results should be sorted by score descending
        for i in 0..results.len() - 1 {
            assert!(results[i].1 >= results[i + 1].1);
        }
    }

    #[test]
    fn snapshot_roundtrip() {
        let engine = make_trained_engine();
        let snap = engine.to_snapshot();
        let restored = HdcEngine::from_snapshot(snap);

        assert_eq!(engine.dim(), restored.dim());
        assert_eq!(engine.vocab_size(), restored.vocab_size());

        // Similarity should be preserved
        let orig_sim = engine.similarity("chromoq", "egfp");
        let rest_sim = restored.similarity("chromoq", "egfp");
        assert!((orig_sim - rest_sim).abs() < 1e-6);
    }

    #[test]
    fn deterministic_base_vectors() {
        let mut e1 = HdcEngine::new(100, 3, false);
        let mut e2 = HdcEngine::new(100, 3, false);
        let v1 = e1.get_or_create_base("test_token");
        let v2 = e2.get_or_create_base("test_token");
        assert_eq!(v1, v2);
    }

    // --- Neural embedding tests ---

    #[test]
    fn neural_embedding_stored_and_retrieved() {
        let mut engine = HdcEngine::new(100, 3, false);
        let emb = vec![0.1f32; 768];
        engine.set_neural_embedding("test_token", emb);
        assert!(engine.has_neural_embedding("test_token"));
        assert!(engine.has_neural_embedding("TEST_TOKEN")); // case-insensitive
        assert_eq!(engine.neural_count(), 1);
        assert_eq!(engine.neural_dim(), Some(768));
    }

    #[test]
    fn neural_embedding_preferred_in_encode() {
        let mut engine = HdcEngine::new(100, 3, false);
        engine.train(&["test sentence here".to_string()]);

        let hdc_vec = engine.encode_sentence("test sentence here");

        // Set a neural embedding (different dimension is fine — it's a separate space)
        let neural = vec![1.0f32; 100];
        engine.set_neural_embedding("test sentence here", neural.clone());

        let result = engine.encode_sentence("test sentence here");
        assert_eq!(
            result, neural,
            "Should return neural embedding when available"
        );
        assert_ne!(result, hdc_vec, "Should differ from HDC vector");
    }

    #[test]
    fn hybrid_similarity_with_neural() {
        let mut engine = HdcEngine::new(100, 3, false);

        // Two similar neural vectors
        let mut va = vec![0.0f32; 10];
        va[0] = 1.0;
        let mut vb = vec![0.0f32; 10];
        vb[0] = 0.9;
        vb[1] = 0.1;

        engine.set_neural_embedding("alpha", va);
        engine.set_neural_embedding("beta", vb);

        let sim = engine.similarity_hybrid("alpha", "beta");
        assert!(sim > 0.9, "Neural cosine should be high: {}", sim);

        // Without neural → falls back to HDC (0.0 since not trained)
        let sim_hdc = engine.similarity_hybrid("unknown_a", "unknown_b");
        assert_eq!(sim_hdc, 0.0);
    }

    #[test]
    fn snapshot_roundtrip_with_neural() {
        let mut engine = HdcEngine::new(100, 3, false);
        engine.train(&["hello world".to_string()]);
        engine.set_neural_embedding("hello", vec![0.5f32; 768]);

        let snap = engine.to_snapshot();
        let restored = HdcEngine::from_snapshot(snap);

        assert!(restored.has_neural_embedding("hello"));
        assert_eq!(restored.neural_count(), 1);
        assert_eq!(restored.neural_dim(), Some(768));
        assert_eq!(engine.vocab_size(), restored.vocab_size());
    }
}
