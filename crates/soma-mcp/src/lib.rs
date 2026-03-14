//! # soma-mcp — Native MCP Server
//!
//! Provides 10 tools via MCP protocol (stdio or TCP transport).
//! Designed for Claude, KOLOSS, Ollama agents, and any MCP-compatible client.

mod protocol;
mod server;
mod tools;

pub use protocol::{McpRequest, McpResponse, McpTool};
pub use server::McpServer;
pub use tools::ToolHandler;
