use std::collections::HashMap;
use std::convert::Infallible;
use std::time::Duration;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Sse},
    response::sse::{Event, KeepAlive},
    routing::{get, post, delete},
};
use chrono::Utc;
use futures::stream::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::BroadcastStream;

use crate::{AppState, WebhookRegistration};

/// Dashboard HTML embedded at compile time.
const DASHBOARD_HTML: &str = include_str!("../dashboard.html");

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(dashboard))
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/search", get(search))
        .route("/context", get(context))
        .route("/api/graph", get(graph_data))
        .route("/add", post(add))
        .route("/ingest", post(ingest))
        .route("/relate", post(relate))
        .route("/reinforce", post(reinforce))
        .route("/alarm", post(alarm))
        .route("/forget", post(forget))
        .route("/sleep", post(sleep))
        .route("/snapshot", post(snapshot))
        .route("/cypher", post(cypher))
        .route("/ingest-code", post(ingest_code))
        .route("/correct", post(correct))
        .route("/validate", post(validate))
        .route("/compact", post(compact))
        .route("/session-restore", get(session_restore))
        .route("/explain", get(explain))
        .route("/merge", post(merge))
        .route("/communities", get(communities))
        .route("/think", post(think))
        // SSE streaming
        .route("/search/stream", get(search_stream))
        .route("/events", get(event_stream))
        // Webhooks CRUD
        .route("/webhooks", get(list_webhooks).post(register_webhook))
        .route("/webhooks/:id", delete(delete_webhook))
        // Multi-tenancy
        .route("/tenants", get(list_tenants).post(create_tenant))
        .with_state(state)
}

// ── Dashboard ──────────────────────────────────────────────────────

async fn dashboard() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        DASHBOARD_HTML,
    )
}

// ── Graph Data (for D3 visualization) ──────────────────────────────

#[derive(Deserialize)]
struct GraphParams {
    /// Max nodes to return (0 = all)
    #[serde(default = "default_graph_limit")]
    limit: usize,
    /// Minimum degree to include a node
    #[serde(default)]
    min_degree: usize,
    /// Filter by tag
    #[serde(default)]
    tag: Option<String>,
    /// Center node — load its neighborhood
    #[serde(default)]
    center: Option<String>,
    /// Depth of neighborhood exploration (default 1)
    #[serde(default = "default_depth")]
    depth: usize,
}

fn default_graph_limit() -> usize { 200 }
fn default_depth() -> usize { 1 }

async fn graph_data(
    State(state): State<AppState>,
    Query(params): Query<GraphParams>,
) -> impl IntoResponse {
    let g = state.graph.read().await;
    let now = Utc::now();

    // Count degrees for all nodes
    let mut degrees: HashMap<String, usize> = HashMap::new();
    for edge in g.all_edges() {
        if let (Some(from), Some(to)) = (g.get_node(edge.from), g.get_node(edge.to)) {
            *degrees.entry(from.label.clone()).or_default() += 1;
            *degrees.entry(to.label.clone()).or_default() += 1;
        }
    }

    let total_nodes = g.all_nodes().count();
    let total_edges = g.all_edges().count();

    // Determine which nodes to include
    let mut visible_labels: std::collections::HashSet<String> = std::collections::HashSet::new();

    if let Some(center_label) = &params.center {
        // Neighborhood mode: BFS from center node
        let mut frontier: Vec<String> = vec![center_label.clone()];
        visible_labels.insert(center_label.clone());

        for _ in 0..params.depth {
            let mut next_frontier = Vec::new();
            for label in &frontier {
                // Find all neighbors
                for edge in g.all_edges() {
                    if let (Some(from), Some(to)) = (g.get_node(edge.from), g.get_node(edge.to)) {
                        if from.label == *label && !visible_labels.contains(&to.label) {
                            visible_labels.insert(to.label.clone());
                            next_frontier.push(to.label.clone());
                        }
                        if to.label == *label && !visible_labels.contains(&from.label) {
                            visible_labels.insert(from.label.clone());
                            next_frontier.push(from.label.clone());
                        }
                    }
                }
            }
            frontier = next_frontier;
        }
    } else {
        // Top-N mode: sort by degree descending, take top limit
        let mut all_labels: Vec<(String, usize)> = g.all_nodes()
            .map(|n| (n.label.clone(), degrees.get(&n.label).copied().unwrap_or(0)))
            .collect();

        // Apply min_degree filter
        if params.min_degree > 0 {
            all_labels.retain(|(_, deg)| *deg >= params.min_degree);
        }

        // Apply tag filter
        if let Some(ref tag) = params.tag {
            let tag_lower = tag.to_lowercase();
            let tag_labels: std::collections::HashSet<String> = g.all_nodes()
                .filter(|n| n.tags.iter().any(|t| t.to_lowercase().contains(&tag_lower)))
                .map(|n| n.label.clone())
                .collect();
            all_labels.retain(|(label, _)| tag_labels.contains(label));
        }

        // Sort by degree descending
        all_labels.sort_by(|a, b| b.1.cmp(&a.1));

        // Apply limit
        let limit = if params.limit == 0 { all_labels.len() } else { params.limit };
        for (label, _) in all_labels.into_iter().take(limit) {
            visible_labels.insert(label);
        }
    }

    let nodes: Vec<_> = g.all_nodes()
        .filter(|n| visible_labels.contains(&n.label))
        .map(|n| {
            let degree = degrees.get(&n.label).copied().unwrap_or(0);
            json!({
                "id": n.label,
                "kind": n.kind.as_str(),
                "degree": degree,
                "tags": n.tags,
                "created": n.created_at.format("%Y-%m-%d %H:%M").to_string(),
            })
        }).collect();

    let links: Vec<_> = g.all_edges().filter_map(|e| {
        let from = g.get_node(e.from)?;
        let to = g.get_node(e.to)?;
        // Only include edges where both endpoints are visible
        if !visible_labels.contains(&from.label) || !visible_labels.contains(&to.label) {
            return None;
        }
        Some(json!({
            "source": from.label,
            "target": to.label,
            "channel": e.channel.to_string(),
            "intensity": e.effective_intensity(now),
            "uses": e.uses,
        }))
    }).collect();

    Json(json!({
        "nodes": nodes,
        "links": links,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "showing": nodes.len(),
    }))
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
    /// Semantic label for the relationship (e.g. "passion", "valeur", "ami")
    #[serde(default)]
    label: Option<String>,
}

fn default_confidence() -> f32 { 0.8 }

async fn relate(
    State(state): State<AppState>,
    Json(req): Json<RelateRequest>,
) -> impl IntoResponse {
    let mut params = json!({
        "from": req.from,
        "to": req.to,
        "channel": req.channel,
        "confidence": req.confidence,
    });
    if let Some(label) = &req.label {
        params["label"] = json!(label);
    }
    let resp = state.tool_handler.handle("soma_relate", &params, None).await;
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

// ── Snapshot (manual persistence) ────────────────────────────────────

async fn snapshot(State(state): State<AppState>) -> impl IntoResponse {
    let g = state.graph.read().await;
    let stats = g.stats();

    match serde_json::to_string(g.inner()) {
        Ok(graph_json) => {
            drop(g);
            let mut s = state.store.write().await;
            match s.write_snapshot(&graph_json, None, stats.nodes, stats.edges) {
                Ok(_) => {
                    Json(json!({
                        "saved": true,
                        "nodes": stats.nodes,
                        "edges": stats.edges,
                    }))
                }
                Err(e) => {
                    Json(json!({"saved": false, "error": format!("{}", e)}))
                }
            }
        }
        Err(e) => {
            Json(json!({"saved": false, "error": format!("{}", e)}))
        }
    }
}

// ── Cypher Query ────────────────────────────────────────────────

#[derive(Deserialize)]
struct CypherRequest {
    query: String,
}

async fn cypher(
    State(state): State<AppState>,
    Json(req): Json<CypherRequest>,
) -> impl IntoResponse {
    let mut g = state.graph.write().await;
    match soma_cypher::CypherExecutor::execute(&mut g, &req.query) {
        Ok(result) => (StatusCode::OK, Json(json!(result))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e}))),
    }
}

// ── Code Ingestion ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct IngestCodeRequest {
    path: String,
}

async fn ingest_code(
    State(state): State<AppState>,
    Json(req): Json<IngestCodeRequest>,
) -> impl IntoResponse {
    let path = std::path::Path::new(&req.path);
    if !path.exists() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "Path does not exist"})));
    }

    let mut g = state.graph.write().await;
    let result = soma_ingest::code::ingest_rust_directory(&mut g, path, "http/ingest-code");

    (StatusCode::OK, Json(json!({
        "files_processed": result.files_processed,
        "functions_found": result.functions_found,
        "structs_found": result.structs_found,
        "traits_found": result.traits_found,
        "impls_found": result.impls_found,
        "edges_created": result.edges_created,
    })))
}

// ── New SOMA v3 Tools (HTTP routes) ─────────────────────────────────

async fn correct(
    State(state): State<AppState>,
    Json(params): Json<serde_json::Value>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle("soma_correct", &params, None).await;
    mcp_to_http(resp)
}

async fn validate(
    State(state): State<AppState>,
    Json(params): Json<serde_json::Value>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle("soma_validate", &params, None).await;
    mcp_to_http(resp)
}

async fn compact(
    State(state): State<AppState>,
    Json(params): Json<serde_json::Value>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle("soma_compact", &params, None).await;
    mcp_to_http(resp)
}

#[derive(Deserialize)]
struct SessionRestoreParams {
    q: String,
    #[serde(default = "default_sr_limit")]
    limit: usize,
}
fn default_sr_limit() -> usize { 5 }

async fn session_restore(
    State(state): State<AppState>,
    Query(params): Query<SessionRestoreParams>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_session_restore",
        &json!({"query": params.q, "limit": params.limit}),
        None,
    ).await;
    mcp_to_http(resp)
}

#[derive(Deserialize)]
struct ExplainParams {
    from: String,
    to: String,
    #[serde(default = "default_max_paths")]
    max_paths: usize,
}
fn default_max_paths() -> usize { 3 }

async fn explain(
    State(state): State<AppState>,
    Query(params): Query<ExplainParams>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle(
        "soma_explain",
        &json!({"from": params.from, "to": params.to, "max_paths": params.max_paths}),
        None,
    ).await;
    mcp_to_http(resp)
}

async fn merge(
    State(state): State<AppState>,
    Json(params): Json<serde_json::Value>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle("soma_merge", &params, None).await;
    mcp_to_http(resp)
}

async fn communities(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let min_size = params.get("min_size")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let resp = state.tool_handler.handle(
        "soma_communities",
        &json!({"min_size": min_size}),
        None,
    ).await;
    mcp_to_http(resp)
}

async fn think(
    State(state): State<AppState>,
    Json(params): Json<serde_json::Value>,
) -> impl IntoResponse {
    let resp = state.tool_handler.handle("soma_think", &params, None).await;
    mcp_to_http(resp)
}

// ── SSE Streaming Search ──────────────────────────────────────────

async fn search_stream(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let query = params.q.clone();
    let limit = params.limit;
    let tool_handler = state.tool_handler.clone();

    let stream = async_stream::stream! {
        // Emit start event
        yield Ok(Event::default()
            .event("start")
            .data(json!({"query": &query, "limit": limit}).to_string()));

        // Execute search
        let resp = tool_handler.handle(
            "soma_search",
            &json!({"query": &query, "limit": limit}),
            None,
        ).await;

        if let Some(result) = resp.result {
            // Stream each result individually
            if let Some(results) = result.get("results").and_then(|r| r.as_array()) {
                for (i, item) in results.iter().enumerate() {
                    yield Ok(Event::default()
                        .event("result")
                        .data(json!({"index": i, "item": item}).to_string()));
                }
            } else {
                // Single result object
                yield Ok(Event::default()
                    .event("result")
                    .data(result.to_string()));
            }
        } else if let Some(err) = resp.error {
            yield Ok(Event::default()
                .event("error")
                .data(json!({"error": err.message}).to_string()));
        }

        // Emit done
        yield Ok(Event::default()
            .event("done")
            .data(json!({"status": "complete"}).to_string()));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ── SSE Event Stream (live graph events) ────────────────────────

async fn event_stream(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        match result {
            Ok(event) => Some(Ok(Event::default()
                .event(&event.kind)
                .data(serde_json::to_string(&event).unwrap_or_default()))),
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    )
}

// ── Webhooks CRUD ────────────────────────────────────────────────

async fn list_webhooks(State(state): State<AppState>) -> impl IntoResponse {
    let hooks = state.webhooks.read().await;
    Json(json!({"webhooks": *hooks}))
}

#[derive(Deserialize)]
struct WebhookRequest {
    url: String,
    events: Vec<String>,
    secret: Option<String>,
}

async fn register_webhook(
    State(state): State<AppState>,
    Json(req): Json<WebhookRequest>,
) -> impl IntoResponse {
    let id = format!("wh_{:x}", rand_id());
    let hook = WebhookRegistration {
        id: id.clone(),
        url: req.url,
        events: req.events,
        secret: req.secret,
    };
    state.webhooks.write().await.push(hook.clone());
    (StatusCode::CREATED, Json(json!(hook)))
}

async fn delete_webhook(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut hooks = state.webhooks.write().await;
    let before = hooks.len();
    hooks.retain(|h| h.id != id);
    if hooks.len() < before {
        Json(json!({"deleted": true, "id": id}))
    } else {
        Json(json!({"deleted": false, "error": "not found"}))
    }
}

fn rand_id() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ── Multi-tenancy ────────────────────────────────────────────────

async fn list_tenants(State(state): State<AppState>) -> impl IntoResponse {
    let tenants = state.tenants.read().await;
    let list: Vec<_> = tenants
        .iter()
        .map(|(key, cfg)| json!({"api_key": key, "workspace": cfg.workspace, "name": cfg.name}))
        .collect();
    Json(json!({"tenants": list}))
}

#[derive(Deserialize)]
struct CreateTenantRequest {
    name: String,
    workspace: String,
}

async fn create_tenant(
    State(state): State<AppState>,
    Json(req): Json<CreateTenantRequest>,
) -> impl IntoResponse {
    let api_key = format!("sk_soma_{:x}", rand_id());
    let cfg = crate::TenantConfig {
        workspace: req.workspace.clone(),
        name: req.name.clone(),
    };
    state.tenants.write().await.insert(api_key.clone(), cfg);
    (
        StatusCode::CREATED,
        Json(json!({
            "api_key": api_key,
            "workspace": req.workspace,
            "name": req.name,
        })),
    )
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
