use serde::{Deserialize, Serialize};

/// MCP Tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Incoming MCP request (JSON-RPC 2.0 style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// Outgoing MCP response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
}

impl McpResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        McpResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<serde_json::Value>, code: i32, message: &str) -> Self {
        McpResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(McpError {
                code,
                message: message.to_string(),
            }),
        }
    }
}

/// Build the list of all SOMA MCP tools.
pub fn soma_tools() -> Vec<McpTool> {
    vec![
        McpTool {
            name: "soma_add".into(),
            description: "Add text or a note to SOMA memory".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Text content to add"},
                    "source": {"type": "string", "description": "Source identifier"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "workspace": {"type": "string"},
                    "channel": {"type": "string"}
                },
                "required": ["content"]
            }),
        },
        McpTool {
            name: "soma_ingest".into(),
            description: "Ingest a file (PDF, Markdown, JSON, logs)".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to ingest"},
                    "source_type": {"type": "string"},
                    "workspace": {"type": "string"}
                },
                "required": ["path"]
            }),
        },
        McpTool {
            name: "soma_search".into(),
            description: "Hybrid search (semantic + graph traversal)".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "channels": {"type": "array", "items": {"type": "string"}},
                    "max_hops": {"type": "integer"},
                    "min_intensity": {"type": "number"},
                    "workspace": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }),
        },
        McpTool {
            name: "soma_relate".into(),
            description: "Create a typed relation between two entities".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "channel": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source": {"type": "string"},
                    "workspace": {"type": "string"}
                },
                "required": ["from", "to", "channel"]
            }),
        },
        McpTool {
            name: "soma_reinforce".into(),
            description: "Reinforce a relation after external validation".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "channel": {"type": "string"},
                    "workspace": {"type": "string"}
                },
                "required": ["from", "to"]
            }),
        },
        McpTool {
            name: "soma_alarm".into(),
            description: "Mark an entity as dangerous/erroneous".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "reason": {"type": "string"},
                    "source": {"type": "string"},
                    "workspace": {"type": "string"}
                },
                "required": ["label", "reason"]
            }),
        },
        McpTool {
            name: "soma_forget".into(),
            description: "Archive (without deleting) an entity or relation".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "workspace": {"type": "string"}
                },
                "required": ["label"]
            }),
        },
        McpTool {
            name: "soma_stats".into(),
            description: "Full graph and index state".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "workspace": {"type": "string"}
                }
            }),
        },
        McpTool {
            name: "soma_workspace".into(),
            description: "Manage isolated workspaces".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "switch", "list", "delete"]},
                    "name": {"type": "string"}
                },
                "required": ["action"]
            }),
        },
        McpTool {
            name: "soma_context".into(),
            description: "Return formatted LLM-ready context block for a query".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What context to retrieve"},
                    "max_tokens": {"type": "integer"},
                    "workspace": {"type": "string"}
                },
                "required": ["query"]
            }),
        },
    ]
}
