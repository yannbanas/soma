use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use tracing::{error, info};

use soma_graph::StigreGraph;
use soma_hdc::HdcEngine;
use soma_llm::OllamaClient;
use soma_store::Store;

use crate::protocol::{self, McpRequest, McpResponse};
use crate::tools::ToolHandler;

/// MCP Server — handles JSON-RPC 2.0 requests over stdio or TCP.
pub struct McpServer {
    handler: ToolHandler,
}

impl McpServer {
    pub fn new(
        graph: Arc<RwLock<StigreGraph>>,
        hdc: Arc<RwLock<HdcEngine>>,
        store: Arc<RwLock<Store>>,
    ) -> Self {
        McpServer {
            handler: ToolHandler::new(graph, hdc, store),
        }
    }

    pub fn with_llm(mut self, client: OllamaClient) -> Self {
        self.handler = self.handler.with_llm(client);
        self
    }

    /// Run MCP server on stdio (for Claude Desktop, KOLOSS).
    pub async fn run_stdio(&self) -> Result<(), soma_core::SomaError> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let reader = BufReader::new(stdin);
        let mut lines = reader.lines();

        info!("[mcp] SOMA MCP server started (stdio)");

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            let response = self.process_message(&line).await;

            let json = serde_json::to_string(&response).unwrap_or_else(|e| {
                serde_json::to_string(&McpResponse::error(
                    None,
                    -32603,
                    &format!("serialization error: {}", e),
                ))
                .unwrap()
            });

            if let Err(e) = stdout.write_all(json.as_bytes()).await {
                error!("[mcp] stdout write error: {}", e);
                break;
            }
            if let Err(e) = stdout.write_all(b"\n").await {
                error!("[mcp] stdout newline error: {}", e);
                break;
            }
            if let Err(e) = stdout.flush().await {
                error!("[mcp] stdout flush error: {}", e);
                break;
            }
        }

        Ok(())
    }

    /// Run MCP server on TCP (for Ollama agents, Django apps).
    pub async fn run_tcp(&self, port: u16) -> Result<(), soma_core::SomaError> {
        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
            .await
            .map_err(|e| soma_core::SomaError::Mcp(format!("TCP bind: {}", e)))?;

        info!("[mcp] SOMA MCP server started (TCP 127.0.0.1:{})", port);

        loop {
            let (stream, addr) = listener
                .accept()
                .await
                .map_err(|e| soma_core::SomaError::Mcp(format!("TCP accept: {}", e)))?;

            info!("[mcp] connection from {}", addr);

            let (reader, mut writer) = stream.into_split();
            let buf_reader = BufReader::new(reader);
            let mut lines = buf_reader.lines();

            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                let response = self.process_message(&line).await;
                let json = serde_json::to_string(&response).unwrap_or_default();

                if writer.write_all(json.as_bytes()).await.is_err()
                    || writer.write_all(b"\n").await.is_err()
                    || writer.flush().await.is_err()
                {
                    break;
                }
            }
        }
    }

    /// Process a single JSON-RPC message.
    async fn process_message(&self, message: &str) -> McpResponse {
        // Security: limit message size (1MB)
        if message.len() > 1_048_576 {
            return McpResponse::error(None, -32600, "message too large (max 1MB)");
        }

        let request: McpRequest = match serde_json::from_str(message) {
            Ok(r) => r,
            Err(e) => {
                return McpResponse::error(None, -32700, &format!("parse error: {}", e));
            }
        };

        match request.method.as_str() {
            "initialize" => McpResponse::success(
                request.id,
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "soma",
                        "version": "0.1.0"
                    }
                }),
            ),
            "tools/list" => {
                let tools = protocol::soma_tools();
                McpResponse::success(request.id, serde_json::json!({"tools": tools}))
            }
            "tools/call" => {
                let tool_name = request
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let arguments = request
                    .params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                self.handler.handle(tool_name, &arguments, request.id).await
            }
            "notifications/initialized" => {
                // Client notification — no response needed
                McpResponse::success(request.id, serde_json::json!({}))
            }
            _ => McpResponse::error(
                request.id,
                -32601,
                &format!("method not found: {}", request.method),
            ),
        }
    }
}
