//! # soma-http — REST API server
//!
//! Provides HTTP endpoints for SOMA, delegating to the same `ToolHandler`
//! used by the MCP server. Zero logic duplication.

mod routes;
pub mod webhooks;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::{broadcast, RwLock};
use tracing::info;

use soma_graph::StigreGraph;
use soma_mcp::ToolHandler;
use soma_store::Store;

/// A graph event emitted on mutations (for webhooks + SSE).
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphEvent {
    pub kind: String, // "node_added", "edge_added", "node_removed", etc.
    pub label: String,
    pub detail: serde_json::Value,
    pub timestamp: String,
}

/// Registered webhook target.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebhookRegistration {
    pub id: String,
    pub url: String,
    pub events: Vec<String>, // filter: ["node_added", "edge_added", "*"]
    pub secret: Option<String>,
}

/// API key → tenant config.
#[derive(Debug, Clone)]
pub struct TenantConfig {
    pub workspace: String,
    pub name: String,
}

/// Shared application state accessible from all handlers.
#[derive(Clone)]
pub struct AppState {
    pub tool_handler: Arc<ToolHandler>,
    pub graph: Arc<RwLock<StigreGraph>>,
    pub store: Arc<RwLock<Store>>,
    pub started_at: Instant,
    pub event_tx: broadcast::Sender<GraphEvent>,
    pub webhooks: Arc<RwLock<Vec<WebhookRegistration>>>,
    pub tenants: Arc<RwLock<HashMap<String, TenantConfig>>>,
}

/// HTTP server wrapping the SOMA REST API.
pub struct HttpServer {
    state: AppState,
}

impl HttpServer {
    pub fn new(
        tool_handler: Arc<ToolHandler>,
        graph: Arc<RwLock<StigreGraph>>,
        store: Arc<RwLock<Store>>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        HttpServer {
            state: AppState {
                tool_handler,
                graph,
                store,
                started_at: Instant::now(),
                event_tx,
                webhooks: Arc::new(RwLock::new(Vec::new())),
                tenants: Arc::new(RwLock::new(HashMap::new())),
            },
        }
    }

    /// Build the axum Router with all endpoints.
    pub fn router(&self) -> Router {
        routes::build_router(self.state.clone())
    }

    /// Get event sender for emitting graph events from outside.
    pub fn event_sender(&self) -> broadcast::Sender<GraphEvent> {
        self.state.event_tx.clone()
    }

    /// Run the HTTP server on the given port. Blocks until shutdown.
    pub async fn run(&self, port: u16) -> std::io::Result<()> {
        // Spawn webhook dispatcher
        webhooks::spawn_dispatcher(self.state.event_tx.subscribe(), self.state.webhooks.clone());

        let app = self.router();
        let addr = format!("0.0.0.0:{}", port);
        info!("[http] Listening on http://{}", addr);
        let listener = TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}
