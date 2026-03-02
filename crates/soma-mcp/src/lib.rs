//! # soma-mcp — Native MCP Server
//!
//! Provides 10 tools via MCP protocol (stdio or TCP transport).
//! Designed for Claude, KOLOSS, Ollama agents, and any MCP-compatible client.

mod server;
mod tools;
mod protocol;

pub use server::McpServer;
pub use tools::ToolHandler;
pub use protocol::{McpRequest, McpResponse, McpTool};
