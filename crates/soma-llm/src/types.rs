use serde::{Deserialize, Serialize};

/// A triplet extracted by the LLM (L2 extraction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmTriplet {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f32,
}

/// Request body for Ollama /api/generate endpoint.
#[derive(Debug, Serialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// Response from Ollama /api/generate endpoint.
#[derive(Debug, Deserialize)]
pub struct OllamaGenerateResponse {
    #[allow(dead_code)]
    pub model: String,
    pub response: String,
    #[allow(dead_code)]
    pub done: bool,
    #[serde(default)]
    pub total_duration: u64,
    #[serde(default)]
    pub eval_count: u64,
}

/// Request body for Ollama /api/embeddings endpoint.
#[derive(Debug, Serialize)]
pub struct OllamaEmbeddingRequest {
    pub model: String,
    pub prompt: String,
}

/// Response from Ollama /api/embeddings endpoint.
#[derive(Debug, Deserialize)]
pub struct OllamaEmbeddingResponse {
    pub embedding: Vec<f64>,
}
