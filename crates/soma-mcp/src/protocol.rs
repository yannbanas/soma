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
            description: "Return formatted LLM-ready context block for a query, with token budget".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What context to retrieve"},
                    "max_tokens": {"type": "integer", "description": "Token budget (default 2000, ~4 chars/token)"},
                    "workspace": {"type": "string"}
                },
                "required": ["query"]
            }),
        },
        McpTool {
            name: "soma_cypher".into(),
            description: "Execute a Cypher query against the SOMA knowledge graph".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Cypher query (MATCH, CREATE, SET, DELETE)"}
                },
                "required": ["query"]
            }),
        },
        McpTool {
            name: "soma_correct".into(),
            description: "Correct an edge's confidence when AI detects an error (feedback loop)".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "Source entity label"},
                    "to": {"type": "string", "description": "Target entity label"},
                    "new_confidence": {"type": "number", "description": "New confidence value [0.0, 1.0]"},
                    "reason": {"type": "string", "description": "Why the correction is needed"}
                },
                "required": ["from", "to", "new_confidence"]
            }),
        },
        McpTool {
            name: "soma_validate".into(),
            description: "Validate an edge after external verification (positive feedback)".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "Source entity label"},
                    "to": {"type": "string", "description": "Target entity label"},
                    "source": {"type": "string", "description": "Verification source (e.g. 'UniProt P42212')"}
                },
                "required": ["from", "to"]
            }),
        },
        McpTool {
            name: "soma_compact".into(),
            description: "Save a session summary before context compaction to preserve memory".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Session summary text"},
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "entities": {"type": "array", "items": {"type": "string"}, "description": "Key entities discussed"},
                    "decisions": {"type": "array", "items": {"type": "string"}, "description": "Decisions made"}
                },
                "required": ["summary"]
            }),
        },
        McpTool {
            name: "soma_session_restore".into(),
            description: "Restore context from previous sessions via semantic search on session summaries".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for session context"},
                    "limit": {"type": "integer", "description": "Max sessions to return (default 5)"}
                },
                "required": ["query"]
            }),
        },
        McpTool {
            name: "soma_explain".into(),
            description: "Find and explain paths between two entities in the knowledge graph".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "Source entity"},
                    "to": {"type": "string", "description": "Target entity"},
                    "max_paths": {"type": "integer", "description": "Max paths to return (default 3)"}
                },
                "required": ["from", "to"]
            }),
        },
        McpTool {
            name: "soma_merge".into(),
            description: "Merge duplicate nodes, transferring all edges from absorbed to kept node".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "keep": {"type": "string", "description": "Label of node to keep"},
                    "absorb": {"type": "string", "description": "Label of node to absorb/remove"},
                    "reason": {"type": "string", "description": "Why merging (e.g. 'same entity, different case')"}
                },
                "required": ["keep", "absorb"]
            }),
        },
        McpTool {
            name: "soma_communities".into(),
            description: "Detect communities in the knowledge graph using Louvain algorithm".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "min_size": {"type": "integer", "description": "Minimum community size (default 3)"}
                }
            }),
        },
        McpTool {
            name: "soma_think".into(),
            description: "Record a reasoning step (Graph of Thoughts) with dependencies".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "The reasoning step or conclusion"},
                    "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Premises this thought depends on"},
                    "conclusion": {"type": "boolean", "description": "If true, stored with durable Causal channel"}
                },
                "required": ["thought"]
            }),
        },
    ]
}
