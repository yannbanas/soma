//! # soma-http — REST API server
//!
//! Provides HTTP endpoints for SOMA, delegating to the same `ToolHandler`
//! used by the MCP server. Zero logic duplication.

mod routes;

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing::info;

use soma_graph::StigreGraph;
use soma_mcp::ToolHandler;
use soma_store::Store;

/// Shared application state accessible from all handlers.
#[derive(Clone)]
pub struct AppState {
    pub tool_handler: Arc<ToolHandler>,
    pub graph: Arc<RwLock<StigreGraph>>,
    pub store: Arc<RwLock<Store>>,
    pub started_at: Instant,
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
        HttpServer {
            state: AppState {
                tool_handler,
                graph,
                store,
                started_at: Instant::now(),
            },
        }
    }

    /// Build the axum Router with all endpoints.
    pub fn router(&self) -> Router {
        routes::build_router(self.state.clone())
    }

    /// Run the HTTP server on the given port. Blocks until shutdown.
    pub async fn run(&self, port: u16) -> std::io::Result<()> {
        let app = self.router();
        let addr = format!("0.0.0.0:{}", port);
        info!("[http] Listening on http://{}", addr);
        let listener = TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}
