use std::sync::Arc;

use chrono::Utc;
use tokio::sync::RwLock;

use soma_core::{Channel, NodeKind, SomaError, SomaQuery, fuzzy_label_search, rrf_merge_with_sources};
use soma_graph::StigreGraph;
use soma_hdc::HdcEngine;
use soma_ingest::{IngestPipeline, IngestSource};
use soma_llm::OllamaClient;
use soma_store::Store;

use crate::protocol::McpResponse;

/// Handles MCP tool invocations against SOMA state.
pub struct ToolHandler {
    graph: Arc<RwLock<StigreGraph>>,
    hdc: Arc<RwLock<HdcEngine>>,
    store: Arc<RwLock<Store>>,
    pipeline: IngestPipeline,
}

impl ToolHandler {
    pub fn new(
        graph: Arc<RwLock<StigreGraph>>,
        hdc: Arc<RwLock<HdcEngine>>,
        store: Arc<RwLock<Store>>,
    ) -> Self {
        ToolHandler {
            graph,
            hdc,
            store,
            pipeline: IngestPipeline::default_config(),
        }
    }

    pub fn with_llm(mut self, client: OllamaClient) -> Self {
        self.pipeline = self.pipeline.with_llm(client);
        self
    }

    /// Dispatch a tool call by name.
    pub async fn handle(
        &self,
        tool: &str,
        params: &serde_json::Value,
        id: Option<serde_json::Value>,
    ) -> McpResponse {
        let result = match tool {
            "soma_add" => self.soma_add(params).await,
            "soma_ingest" => self.soma_ingest(params).await,
            "soma_search" => self.soma_search(params).await,
            "soma_relate" => self.soma_relate(params).await,
            "soma_reinforce" => self.soma_reinforce(params).await,
            "soma_alarm" => self.soma_alarm(params).await,
            "soma_forget" => self.soma_forget(params).await,
            "soma_stats" => self.soma_stats(params).await,
            "soma_workspace" => self.soma_workspace(params).await,
            "soma_context" => self.soma_context(params).await,
            _ => Err(SomaError::Mcp(format!("unknown tool: {}", tool))),
        };

        match result {
            Ok(value) => McpResponse::success(id, value),
            Err(e) => McpResponse::error(id, -32000, &e.to_string()),
        }
    }

    async fn soma_add(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'content' parameter".into()))?;
        let source = params
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("mcp");
        let tags: Vec<String> = params
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let mut graph = self.graph.write().await;
        let result = self.pipeline.ingest_text(content, &mut graph, source)?;

        // Also create a direct node for the content
        if result.triplets_extracted == 0 {
            let label = if content.len() > 80 {
                format!("{}...", &content[..77])
            } else {
                content.to_string()
            };
            graph.upsert_node_with_tags(&label, NodeKind::Event, tags);
        }

        Ok(serde_json::json!({
            "nodes_created": result.nodes_created,
            "edges_created": result.edges_created,
            "duration_ms": 0
        }))
    }

    async fn soma_ingest(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'path' parameter".into()))?;

        let source = IngestSource::File(std::path::PathBuf::from(path));
        let mut graph = self.graph.write().await;
        let result = self.pipeline.ingest(&source, &mut graph, path)?;

        Ok(serde_json::json!({
            "chunks": result.chunks_processed,
            "nodes": result.nodes_created,
            "edges": result.edges_created,
            "triplets": result.triplets_extracted,
        }))
    }

    async fn soma_search(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let query_str = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'query' parameter".into()))?;

        let max_hops = params
            .get("max_hops")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as u8;
        let min_intensity = params
            .get("min_intensity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.15) as f32;
        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let query = SomaQuery::new(query_str)
            .with_max_hops(max_hops)
            .with_min_intensity(min_intensity)
            .with_limit(limit);

        // Path 1: Graph BFS traverse
        let graph = self.graph.read().await;
        let graph_results = graph.traverse(&query);
        let graph_list: Vec<(String, f32)> = graph_results
            .iter()
            .map(|r| (r.node.label.clone(), r.score))
            .collect();

        // Path 2: HDC/Neural semantic search
        let all_labels = graph.all_labels();
        let h = self.hdc.read().await;
        let hdc_list = h.search_labels(query_str, &all_labels, limit);

        // Path 3: Fuzzy label search
        let fuzzy_list = fuzzy_label_search(query_str, &all_labels, limit);

        drop(h);

        // RRF Merge with source tracking
        let ranked_lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("graph", graph_list),
            ("hdc", hdc_list),
            ("fuzzy", fuzzy_list),
        ];
        let hybrid_results = rrf_merge_with_sources(&ranked_lists, 60.0);

        let json_results: Vec<serde_json::Value> = hybrid_results
            .iter()
            .take(limit)
            .filter_map(|hr| {
                graph.get_node_by_label(&hr.label).map(|node| {
                    let graph_match = graph_results.iter().find(|r| r.node.label == hr.label);
                    let hops = graph_match.map(|r| r.hops).unwrap_or(0);
                    serde_json::json!({
                        "label": node.label,
                        "kind": node.kind.as_str(),
                        "score": hr.score,
                        "hops": hops,
                        "tags": node.tags,
                        "sources": hr.sources,
                    })
                })
            })
            .collect();

        Ok(serde_json::json!(json_results))
    }

    async fn soma_relate(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let from = params.get("from").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'from'".into()))?;
        let to = params.get("to").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'to'".into()))?;
        let channel_str = params.get("channel").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'channel'".into()))?;
        let confidence = params.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;
        let source = params.get("source").and_then(|v| v.as_str()).unwrap_or("mcp:relate");

        let channel = Channel::from_str_name(channel_str)
            .ok_or_else(|| SomaError::InvalidChannel(channel_str.into()))?;

        let mut graph = self.graph.write().await;
        let from_id = graph.upsert_node(from, NodeKind::Entity);
        let to_id = graph.upsert_node(to, NodeKind::Entity);
        let edge_id = graph.upsert_edge(from_id, to_id, channel, confidence, source);

        Ok(serde_json::json!({
            "edge_id": edge_id.map(|e| e.to_string())
        }))
    }

    async fn soma_reinforce(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let from = params.get("from").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'from'".into()))?;
        let to = params.get("to").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'to'".into()))?;

        let graph = self.graph.read().await;
        let from_id = graph.node_id_by_label(from)
            .ok_or_else(|| SomaError::LabelNotFound(from.into()))?;

        // Find and reinforce all edges from → to
        let edges = graph.outgoing_edges(from_id);
        let matching: Vec<_> = edges.iter()
            .filter(|e| {
                if let Some(to_node) = graph.get_node(e.to) {
                    to_node.label == to
                } else {
                    false
                }
            })
            .map(|e| e.id)
            .collect();
        drop(graph);

        let mut graph = self.graph.write().await;
        for eid in &matching {
            graph.reinforce_edge(*eid);
        }

        Ok(serde_json::json!({
            "reinforced": matching.len()
        }))
    }

    async fn soma_alarm(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let label = params.get("label").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'label'".into()))?;
        let reason = params.get("reason").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'reason'".into()))?;
        let source = params.get("source").and_then(|v| v.as_str()).unwrap_or("mcp:alarm");

        let mut graph = self.graph.write().await;
        let entity_id = graph.upsert_node(label, NodeKind::Entity);
        let warning_id = graph.upsert_node(reason, NodeKind::Warning);
        let edge_id = graph.upsert_edge(entity_id, warning_id, Channel::Alarm, 0.9, source);

        Ok(serde_json::json!({
            "alarm_id": edge_id.map(|e| e.to_string())
        }))
    }

    async fn soma_forget(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let label = params.get("label").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'label'".into()))?;

        let mut graph = self.graph.write().await;
        if let Some(node_id) = graph.remove_node_by_label(label) {
            // Log archive to WAL
            let mut s = self.store.write().await;
            let _ = s.write_wal(&soma_store::WalEntry::NodeArchive(node_id));
            Ok(serde_json::json!({
                "archived": true,
                "node_id": node_id.to_string(),
                "label": label
            }))
        } else {
            Ok(serde_json::json!({
                "archived": false,
                "error": format!("node '{}' not found", label)
            }))
        }
    }

    async fn soma_stats(
        &self,
        _params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let graph = self.graph.read().await;
        let stats = graph.stats();
        let hdc = self.hdc.read().await;

        Ok(serde_json::json!({
            "nodes": stats.nodes,
            "edges": stats.edges,
            "dead_edges": stats.dead_edges,
            "avg_intensity": stats.avg_intensity,
            "workspace": stats.workspace,
            "hdc_vocab_size": hdc.vocab_size(),
        }))
    }

    async fn soma_workspace(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let action = params.get("action").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'action'".into()))?;

        match action {
            "list" => {
                let graph = self.graph.read().await;
                Ok(serde_json::json!({
                    "workspaces": [graph.workspace()],
                    "current": graph.workspace()
                }))
            }
            "create" => {
                let name = params.get("name").and_then(|v| v.as_str())
                    .ok_or_else(|| SomaError::Mcp("missing 'name' for create".into()))?;
                // Creating a workspace just requires opening a store (creates the directory)
                let data_dir = self.store.read().await.workspace_dir().parent()
                    .map(|p| p.to_path_buf())
                    .ok_or_else(|| SomaError::Mcp("cannot determine data directory".into()))?;
                let _ = Store::open(&data_dir, name)?;
                Ok(serde_json::json!({
                    "created": name,
                    "status": "ok"
                }))
            }
            "switch" | "delete" => {
                // Switch/delete require reloading graph state — not supported in MCP session
                Ok(serde_json::json!({
                    "status": format!("'{}' requires CLI (soma workspace {})", action, action),
                    "hint": "Use 'soma workspace list' to see available workspaces"
                }))
            }
            _ => Err(SomaError::Mcp(format!("unknown workspace action: {}", action))),
        }
    }

    async fn soma_context(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, SomaError> {
        let query_str = params.get("query").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'query'".into()))?;

        let graph = self.graph.read().await;
        let query = SomaQuery::new(query_str)
            .with_max_hops(3)
            .with_min_intensity(0.1)
            .with_limit(20);

        // Path 1: Graph BFS traverse
        let graph_results = graph.traverse(&query);
        let graph_list: Vec<(String, f32)> = graph_results
            .iter()
            .map(|r| (r.node.label.clone(), r.score))
            .collect();

        // Path 2: HDC/Neural semantic search
        let all_labels = graph.all_labels();
        let h = self.hdc.read().await;
        let hdc_list = h.search_labels(query_str, &all_labels, 20);
        drop(h);

        // Path 3: Fuzzy label search
        let fuzzy_list = fuzzy_label_search(query_str, &all_labels, 20);

        // RRF Merge
        let ranked_lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("graph", graph_list),
            ("hdc", hdc_list),
            ("fuzzy", fuzzy_list),
        ];
        let hybrid_results = rrf_merge_with_sources(&ranked_lists, 60.0);

        // Format as LLM-ready context block
        let now = Utc::now().format("%Y-%m-%d");
        let workspace = graph.workspace();

        let mut lines = vec![
            format!("[SOMA MEMORY — {} — workspace: {}]", now, workspace),
            String::new(),
        ];

        // Collect facts from hybrid results (skip the start node)
        let facts: Vec<_> = hybrid_results.iter()
            .take(20)
            .filter_map(|hr| {
                graph.get_node_by_label(&hr.label).map(|node| {
                    (node, hr)
                })
            })
            .collect();

        if !facts.is_empty() {
            lines.push("Relevant facts (hybrid: graph+hdc+fuzzy):".to_string());
            for (node, hr) in &facts {
                let source_str = format!("[{}]", hr.sources.join("+"));
                lines.push(format!(
                    "  - {} ({}, score={:.3}) {}",
                    node.label,
                    node.kind,
                    hr.score,
                    source_str,
                ));
            }
        }

        // Alarms
        let alarms: Vec<_> = graph.all_nodes()
            .filter(|n| n.kind == NodeKind::Warning)
            .collect();

        if !alarms.is_empty() {
            lines.push(String::new());
            lines.push("Active alarms:".to_string());
            for a in &alarms {
                lines.push(format!("  ! {}", a.label));
            }
        }

        let context = lines.join("\n");
        let fact_count = facts.len();

        Ok(serde_json::json!({
            "context": context,
            "facts": fact_count,
            "sources": hybrid_results.iter().take(20).flat_map(|r| r.sources.clone()).collect::<Vec<_>>(),
        }))
    }
}
