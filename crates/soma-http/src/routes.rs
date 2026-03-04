use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::json;

use crate::AppState;

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/search", get(search))
        .route("/context", get(context))
        .route("/add", post(add))
        .route("/ingest", post(ingest))
        .route("/relate", post(relate))
        .route("/reinforce", post(reinforce))
        .route("/alarm", post(alarm))
        .route("/forget", post(forget))
        .route("/sleep", post(sleep))
        .with_state(state)
}

// ── Health ──────────────────────────────────────────────────────────

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let g = state.graph.read().await;
    let stats = g.stats();
    let uptime = state.started_at.elapsed().as_secs();

    Json(json!({
        "status": "ok",
        "uptime_secs": uptime,
        "nodes": stats.nodes,
        "edges": stats.edges,
        "workspace": stats.workspace,
    }))
}

// ── Stats ───────────────────────────────────────────────────────────

async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let resp = state.tool_handler.handle("soma_stats", &json!({}), None).await;
    mcp_to_http(resp)
}

// ── Search ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize { 20 }

async fn search(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_search",
        &json!({"query": params.q, "limit": params.limit}),
        None,
    ).await;
    mcp_to_http(resp)
}

// ── Context ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ContextParams {
    q: String,
}

async fn context(
    State(state): State<AppState>,
    Query(params): Query<ContextParams>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_context",
        &json!({"query": params.q}),
        None,
    ).await;
    mcp_to_http(resp)
}

// ── Add ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AddRequest {
    content: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

async fn add(
    State(state): State<AppState>,
    Json(req): Json<AddRequest>,
) -> impl IntoResponse {
    let mut params = json!({"content": req.content});
    if let Some(src) = &req.source {
        params["source"] = json!(src);
    }
    if !req.tags.is_empty() {
        params["tags"] = json!(req.tags);
    }
    let resp = state.tool_handler.handle("soma_add", &params, None).await;
    mcp_to_http(resp)
}

// ── Ingest ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct IngestRequest {
    path: String,
}

async fn ingest(
    State(state): State<AppState>,
    Json(req): Json<IngestRequest>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_ingest",
        &json!({"path": req.path}),
        None,
    ).await;
    mcp_to_http(resp)
}

// ── Relate ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RelateRequest {
    from: String,
    to: String,
    channel: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

fn default_confidence() -> f32 { 0.8 }

async fn relate(
    State(state): State<AppState>,
    Json(req): Json<RelateRequest>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_relate",
        &json!({
            "from": req.from,
            "to": req.to,
            "channel": req.channel,
            "confidence": req.confidence,
        }),
        None,
    ).await;
    mcp_to_http(resp)
}

// ── Reinforce ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ReinforceRequest {
    from: String,
    to: String,
}

async fn reinforce(
    State(state): State<AppState>,
    Json(req): Json<ReinforceRequest>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_reinforce",
        &json!({"from": req.from, "to": req.to}),
        None,
    ).await;
    mcp_to_http(resp)
}

// ── Alarm ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AlarmRequest {
    label: String,
    reason: String,
}

async fn alarm(
    State(state): State<AppState>,
    Json(req): Json<AlarmRequest>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_alarm",
        &json!({"label": req.label, "reason": req.reason}),
        None,
    ).await;
    mcp_to_http(resp)
}

// ── Forget ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ForgetRequest {
    label: String,
}

async fn forget(
    State(state): State<AppState>,
    Json(req): Json<ForgetRequest>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_forget",
        &json!({"label": req.label}),
        None,
    ).await;
    mcp_to_http(resp)
}

// ── Sleep (manual consolidation) ────────────────────────────────────

async fn sleep(State(state): State<AppState>) -> impl IntoResponse {
    let report = soma_bio::BioScheduler::consolidate_once(
        &state.graph,
        &state.store,
    ).await;
    Json(json!(report))
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Convert McpResponse to HTTP JSON response.
fn mcp_to_http(resp: soma_mcp::McpResponse) -> impl IntoResponse {
    if let Some(result) = resp.result {
        (StatusCode::OK, Json(result))
    } else if let Some(err) = resp.error {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": err.message})),
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "unknown"})),
        )
    }
}
