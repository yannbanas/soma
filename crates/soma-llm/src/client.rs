use std::time::Duration;

use reqwest::blocking::Client;
use tracing::{debug, warn};

use soma_core::LlmSection;

use crate::prompt;
use crate::types::*;

/// Send a blocking HTTP request, handling tokio runtime context.
/// reqwest::blocking creates its own internal tokio runtime, which conflicts
/// with an outer tokio runtime. When inside tokio, we spawn a dedicated OS thread.
fn blocking_send(
    request: reqwest::blocking::RequestBuilder,
) -> Result<reqwest::blocking::Response, reqwest::Error> {
    if tokio::runtime::Handle::try_current().is_ok() {
        // Inside a tokio runtime — spawn a dedicated thread to avoid nested runtime conflict
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(request.send());
        });
        rx.recv().expect("blocking_send: thread panicked")
    } else {
        request.send()
    }
}

/// Ollama client for generation and embedding API calls.
/// Uses reqwest::blocking — safe to call from both sync and async contexts
/// (automatically uses block_in_place when inside a tokio runtime).
#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    endpoint: String,
    model: String,
    embedding_model: String,
    enabled: bool,
}

impl OllamaClient {
    /// Create from LlmSection config.
    pub fn from_config(config: &LlmSection) -> Self {
        let timeout = Duration::from_millis(config.timeout_ms.max(1000));
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| Client::new());

        let embedding_model = config
            .embedding_model
            .clone()
            .unwrap_or_else(|| config.model.clone());

        OllamaClient {
            client,
            endpoint: config.endpoint.clone(),
            model: config.model.clone(),
            embedding_model,
            enabled: config.enabled,
        }
    }

    /// Check if the client is enabled AND Ollama is reachable.
    pub fn is_available(&self) -> bool {
        if !self.enabled {
            return false;
        }
        blocking_send(
            self.client
                .get(format!("{}/api/tags", self.endpoint))
                .timeout(Duration::from_secs(2)),
        )
        .map(|r| r.status().is_success())
        .unwrap_or(false)
    }

    /// L2 extraction: extract triplets from a text chunk.
    /// Returns Ok(vec![]) if LLM is disabled or unreachable (graceful degradation).
    pub fn extract_triplets(&self, chunk: &str) -> Result<Vec<LlmTriplet>, LlmError> {
        if !self.enabled {
            return Ok(Vec::new());
        }

        let prompt_text = prompt::build_extraction_prompt(chunk);

        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: prompt_text,
            stream: false,
            format: Some("json".to_string()),
        };

        let url = format!("{}/api/generate", self.endpoint);

        let response =
            blocking_send(self.client.post(&url).json(&request)).map_err(|e| {
                warn!("[llm] Ollama unreachable: {}", e);
                LlmError::ConnectionFailed(e.to_string())
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            warn!("[llm] Ollama error {}: {}", status, body);
            return Err(LlmError::ApiError {
                status: status.as_u16(),
                body,
            });
        }

        let gen_response: OllamaGenerateResponse = response
            .json()
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        debug!(
            "[llm] L2 extraction: {} tokens in {}ns",
            gen_response.eval_count, gen_response.total_duration
        );

        parse_triplets_response(&gen_response.response)
    }

    /// Batch embed multiple texts at once via /api/embed.
    /// Returns vectors in the same order as input. Returns Ok(None) if disabled.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Option<Vec<Vec<f64>>>, LlmError> {
        if !self.enabled || texts.is_empty() {
            return Ok(None);
        }

        let request = OllamaBatchEmbedRequest {
            model: self.embedding_model.clone(),
            input: texts.to_vec(),
        };

        let url = format!("{}/api/embed", self.endpoint);

        let response =
            blocking_send(self.client.post(&url).json(&request)).map_err(|e| {
                warn!("[llm] Ollama batch embed unreachable: {}", e);
                LlmError::ConnectionFailed(e.to_string())
            })?;

        if !response.status().is_success() {
            return Ok(None);
        }

        let emb_response: OllamaBatchEmbedResponse = response
            .json()
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        Ok(Some(emb_response.embeddings))
    }

    /// Get embedding vector for a text.
    /// Returns Ok(None) if disabled or unreachable.
    pub fn embed(&self, text: &str) -> Result<Option<Vec<f64>>, LlmError> {
        if !self.enabled {
            return Ok(None);
        }

        let request = OllamaEmbeddingRequest {
            model: self.embedding_model.clone(),
            prompt: text.to_string(),
        };

        let url = format!("{}/api/embeddings", self.endpoint);

        let response =
            blocking_send(self.client.post(&url).json(&request)).map_err(|e| {
                warn!("[llm] Ollama embed unreachable: {}", e);
                LlmError::ConnectionFailed(e.to_string())
            })?;

        if !response.status().is_success() {
            return Ok(None); // graceful degradation
        }

        let emb_response: OllamaEmbeddingResponse = response
            .json()
            .map_err(|e| LlmError::ParseError(e.to_string()))?;

        Ok(Some(emb_response.embedding))
    }
}

/// Strip Cogito-style `<think>...</think>` tags and other non-JSON preamble.
fn clean_llm_response(response: &str) -> String {
    let mut s = response.to_string();

    // Remove <think>...</think> blocks (Cogito chain-of-thought)
    while let Some(start) = s.find("<think>") {
        if let Some(end) = s.find("</think>") {
            s = format!("{}{}", &s[..start], &s[end + 8..]);
        } else {
            // Unclosed <think> — remove everything from <think> onwards
            s = s[..start].to_string();
            break;
        }
    }

    s.trim().to_string()
}

/// Parse LLM response into triplets. Handles various JSON formats robustly.
pub(crate) fn parse_triplets_response(response: &str) -> Result<Vec<LlmTriplet>, LlmError> {
    let cleaned = clean_llm_response(response);

    // Try direct array parse
    if let Ok(triplets) = serde_json::from_str::<Vec<LlmTriplet>>(&cleaned) {
        return Ok(triplets);
    }

    // Try wrapped in object with "triplets" key
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cleaned) {
        // Try "triplets" key
        if let Some(arr) = val.get("triplets").and_then(|v| v.as_array()) {
            if let Ok(triplets) =
                serde_json::from_value::<Vec<LlmTriplet>>(serde_json::Value::Array(arr.clone()))
            {
                return Ok(triplets);
            }
        }
        // Try any key that contains an array of objects
        if let Some(obj) = val.as_object() {
            for (_key, v) in obj {
                if let Some(arr) = v.as_array() {
                    if let Ok(triplets) = serde_json::from_value::<Vec<LlmTriplet>>(
                        serde_json::Value::Array(arr.clone()),
                    ) {
                        if !triplets.is_empty() {
                            return Ok(triplets);
                        }
                    }
                }
            }
        }
    }

    // Extract JSON from markdown code block
    let trimmed = cleaned.trim();
    if trimmed.starts_with("```") {
        let inner = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        if let Ok(triplets) = serde_json::from_str::<Vec<LlmTriplet>>(inner) {
            return Ok(triplets);
        }
    }

    // Last resort: find first JSON array or object in the text
    if let Some(start) = cleaned.find('[') {
        if let Some(end) = cleaned.rfind(']') {
            if let Ok(triplets) =
                serde_json::from_str::<Vec<LlmTriplet>>(&cleaned[start..=end])
            {
                return Ok(triplets);
            }
        }
    }

    warn!(
        "[llm] could not parse triplets from response: {}...",
        &cleaned[..cleaned.len().min(200)]
    );
    Ok(Vec::new()) // graceful: return empty rather than error
}

/// LLM-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Ollama connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Ollama API error (HTTP {status}): {body}")]
    ApiError { status: u16, body: String },
    #[error("Response parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_triplets_json_array() {
        let json = r#"[{"subject":"ChromoQ","relation":"derives from","object":"EGFP","confidence":0.95}]"#;
        let triplets = parse_triplets_response(json).unwrap();
        assert_eq!(triplets.len(), 1);
        assert_eq!(triplets[0].subject, "ChromoQ");
        assert_eq!(triplets[0].relation, "derives from");
        assert_eq!(triplets[0].object, "EGFP");
        assert!((triplets[0].confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn parse_triplets_wrapped_object() {
        let json =
            r#"{"triplets":[{"subject":"A","relation":"uses","object":"B","confidence":0.8}]}"#;
        let triplets = parse_triplets_response(json).unwrap();
        assert_eq!(triplets.len(), 1);
        assert_eq!(triplets[0].subject, "A");
    }

    #[test]
    fn parse_triplets_markdown_block() {
        let json = "```json\n[{\"subject\":\"X\",\"relation\":\"is\",\"object\":\"Y\",\"confidence\":0.7}]\n```";
        let triplets = parse_triplets_response(json).unwrap();
        assert_eq!(triplets.len(), 1);
        assert_eq!(triplets[0].subject, "X");
    }

    #[test]
    fn parse_triplets_garbage_returns_empty() {
        let triplets = parse_triplets_response("not json at all").unwrap();
        assert!(triplets.is_empty());
    }

    #[test]
    fn parse_triplets_empty_array() {
        let triplets = parse_triplets_response("[]").unwrap();
        assert!(triplets.is_empty());
    }

    #[test]
    fn disabled_client_returns_empty() {
        let config = LlmSection {
            enabled: false,
            ..Default::default()
        };
        let client = OllamaClient::from_config(&config);
        let result = client.extract_triplets("some text").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn disabled_client_embed_returns_none() {
        let config = LlmSection {
            enabled: false,
            ..Default::default()
        };
        let client = OllamaClient::from_config(&config);
        let result = client.embed("some text").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn client_from_default_config() {
        let config = LlmSection::default();
        let client = OllamaClient::from_config(&config);
        assert!(!client.enabled);
        assert_eq!(client.model, "cogito:8b");
        assert_eq!(client.embedding_model, "cogito:8b"); // None → falls back to model
    }

    #[test]
    fn client_with_separate_embedding_model() {
        let config = LlmSection {
            embedding_model: Some("nomic-embed-text".to_string()),
            ..Default::default()
        };
        let client = OllamaClient::from_config(&config);
        assert_eq!(client.model, "cogito:8b");
        assert_eq!(client.embedding_model, "nomic-embed-text");
    }
}
