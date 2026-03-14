use std::sync::Arc;

use chrono::Utc;
use tokio::sync::RwLock;

use soma_core::{Channel, NodeKind, Provenance, SomaError, SomaQuery, fuzzy_label_search, rrf_merge_with_sources};
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
            "soma_cypher" => self.soma_cypher(params).await,
            "soma_correct" => self.soma_correct(params).await,
            "soma_validate" => self.soma_validate(params).await,
            "soma_compact" => self.soma_compact(params).await,
            "soma_session_restore" => self.soma_session_restore(params).await,
            "soma_explain" => self.soma_explain(params).await,
            "soma_merge" => self.soma_merge(params).await,
            "soma_communities" => self.soma_communities(params).await,
            "soma_think" => self.soma_think(params).await,
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
        let label = params.get("label").and_then(|v| v.as_str());

        let channel = Channel::from_str_name(channel_str)
            .ok_or_else(|| SomaError::InvalidChannel(channel_str.into()))?;

        let mut graph = self.graph.write().await;
        let from_id = graph.upsert_node(from, NodeKind::Entity);
        let to_id = graph.upsert_node(to, NodeKind::Entity);
        let edge_id = graph.upsert_edge_labeled(from_id, to_id, channel, confidence, source, label);

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
        // Token budget: ~4 chars per token
        let max_tokens = params.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(2000) as usize;
        let max_chars = max_tokens * 4;

        let graph = self.graph.read().await;
        let now_dt = Utc::now();
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

        // Path 3: Fuzzy label search (full query + individual keywords)
        let mut fuzzy_list = fuzzy_label_search(query_str, &all_labels, 20);
        // Also search by individual significant words (>3 chars, skip stopwords)
        let stopwords = ["les","des","une","est","sont","que","qui","pour","par","dans",
            "avec","pas","sur","mais","the","and","for","you","this","that","not",
            "fais","fait","quoi","comment","quel","quelle","ton","tes","mon","mes",
            "dire","dit","dis","peut","veux","elle","nous","vous","ils","toi","moi"];
        for word in query_str.split(|c: char| !c.is_alphanumeric()) {
            if word.len() > 3 && !stopwords.contains(&word.to_lowercase().as_str()) {
                let word_results = fuzzy_label_search(word, &all_labels, 10);
                for (label, score) in word_results {
                    // Reduce score slightly for per-word matches
                    if !fuzzy_list.iter().any(|(l, _)| l == &label) {
                        fuzzy_list.push((label, score * 0.8));
                    }
                }
            }
        }

        // RRF Merge
        let ranked_lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("graph", graph_list),
            ("hdc", hdc_list),
            ("fuzzy", fuzzy_list),
        ];
        let hybrid_results = rrf_merge_with_sources(&ranked_lists, 60.0);

        // Format as LLM-ready context block
        let now = now_dt.format("%Y-%m-%d");
        let workspace = graph.workspace();

        let mut lines = vec![
            format!("[SOMA MEMORY — {} — workspace: {}]", now, workspace),
            String::new(),
        ];

        // Collect top facts from hybrid results
        let facts: Vec<_> = hybrid_results.iter()
            .take(15)
            .filter_map(|hr| {
                graph.get_node_by_label(&hr.label).map(|node| {
                    (node, hr)
                })
            })
            .collect();

        // --- Section 1: Relationships (edges with labels) ---
        // For each found node, show its labeled relationships
        let mut seen_relations = std::collections::HashSet::new();
        let mut relation_lines = Vec::new();

        for (node, _hr) in &facts {
            // Outgoing edges
            let out_edges = graph.outgoing_edges(node.id);
            for edge in &out_edges {
                if edge.effective_intensity(now_dt) < 0.1 { continue; }
                if let Some(target) = graph.get_node(edge.to) {
                    let rel_label = edge.label.as_deref()
                        .unwrap_or(edge.channel.as_str());
                    let key = format!("{}->{}->{}", node.label, rel_label, target.label);
                    if seen_relations.insert(key) {
                        relation_lines.push(format!(
                            "  {} --[{}]--> {}",
                            node.label, rel_label, target.label,
                        ));
                    }
                }
            }

            // Incoming edges
            let in_edges = graph.incoming_edges(node.id);
            for edge in &in_edges {
                if edge.effective_intensity(now_dt) < 0.1 { continue; }
                if let Some(source_node) = graph.get_node(edge.from) {
                    let rel_label = edge.label.as_deref()
                        .unwrap_or(edge.channel.as_str());
                    let key = format!("{}->{}->{}", source_node.label, rel_label, node.label);
                    if seen_relations.insert(key) {
                        relation_lines.push(format!(
                            "  {} --[{}]--> {}",
                            source_node.label, rel_label, node.label,
                        ));
                    }
                }
            }
        }

        if !relation_lines.is_empty() {
            lines.push("Knowledge graph relationships:".to_string());
            // Cap at 30 relations to avoid overwhelming the LLM
            for line in relation_lines.iter().take(30) {
                lines.push(line.clone());
            }
            lines.push(String::new());
        }

        // --- Section 2: Relevant entities ---
        if !facts.is_empty() {
            lines.push("Relevant entities:".to_string());
            for (node, _hr) in &facts {
                lines.push(format!("  - {} ({})", node.label, node.kind));
            }
        }

        // --- Section 3: Alarms ---
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

        // Budget-aware truncation: prioritize Alarms > Relations > Entities
        let mut context = lines.join("\n");
        let estimated_tokens = context.len() / 4;
        if context.len() > max_chars {
            context.truncate(max_chars);
            // Find last newline to avoid cutting mid-line
            if let Some(pos) = context.rfind('\n') {
                context.truncate(pos);
            }
            context.push_str("\n[... truncated to fit token budget]");
        }

        let fact_count = facts.len();

        Ok(serde_json::json!({
            "context": context,
            "facts": fact_count,
            "estimated_tokens": estimated_tokens.min(max_tokens),
            "budget": max_tokens,
            "sources": hybrid_results.iter().take(20).flat_map(|r| r.sources.clone()).collect::<Vec<_>>(),
        }))
    }

    async fn soma_cypher(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'query' parameter".into()))?;

        let mut g = self.graph.write().await;
        let result = soma_cypher::CypherExecutor::execute(&mut g, query)
            .map_err(|e| SomaError::Mcp(format!("Cypher error: {}", e)))?;

        Ok(serde_json::to_value(result).unwrap_or_default())
    }

    /// B.1 — Correct an edge's confidence (AI feedback loop).
    async fn soma_correct(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let from = params.get("from").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'from'".into()))?;
        let to = params.get("to").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'to'".into()))?;
        let new_confidence = params.get("new_confidence").and_then(|v| v.as_f64())
            .ok_or_else(|| SomaError::Mcp("missing 'new_confidence'".into()))? as f32;
        let reason = params.get("reason").and_then(|v| v.as_str()).unwrap_or("correction");

        let mut graph = self.graph.write().await;
        let edges = graph.find_edges_by_labels(from, to);
        if edges.is_empty() {
            return Err(SomaError::Mcp(format!("no edge from '{}' to '{}'", from, to)));
        }

        let mut corrected = 0;
        for (edge_id, _channel, _old_confidence) in &edges {
            if graph.weaken_edge(*edge_id, new_confidence).is_some() {
                graph.set_edge_provenance(*edge_id, Provenance::AiValidated);
                corrected += 1;
            }
        }

        // Create a Warning node with the reason
        let warning_label = format!("correction: {}", reason);
        let warning_id = graph.upsert_node(&warning_label, NodeKind::Warning);
        let from_id = graph.node_id_by_label(from);
        if let Some(fid) = from_id {
            graph.upsert_edge(fid, warning_id, Channel::Alarm, 0.8, "mcp:correct");
        }

        // Log to WAL
        let mut s = self.store.write().await;
        let _ = s.write_wal(&soma_store::WalEntry::Custom(format!(
            "EdgeCorrection: {}→{} confidence {} reason: {}",
            from, to, new_confidence, reason
        )));

        Ok(serde_json::json!({
            "corrected": corrected,
            "old_confidences": edges.iter().map(|(_, _, c)| *c).collect::<Vec<_>>(),
            "new_confidence": new_confidence,
            "reason": reason,
        }))
    }

    /// B.2 — Validate an edge (positive AI feedback).
    async fn soma_validate(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let from = params.get("from").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'from'".into()))?;
        let to = params.get("to").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'to'".into()))?;
        let source = params.get("source").and_then(|v| v.as_str()).unwrap_or("mcp:validate");

        let mut graph = self.graph.write().await;
        let edges = graph.find_edges_by_labels(from, to);
        if edges.is_empty() {
            return Err(SomaError::Mcp(format!("no edge from '{}' to '{}'", from, to)));
        }

        let mut validated = 0;
        for (edge_id, _, _) in &edges {
            graph.reinforce_edge(*edge_id);
            graph.set_edge_provenance(*edge_id, Provenance::AiValidated);
            validated += 1;
        }

        Ok(serde_json::json!({
            "validated": validated,
            "source": source,
        }))
    }

    /// A.1 — Compact a session summary into the graph before context compaction.
    async fn soma_compact(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let summary = params.get("summary").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'summary'".into()))?;
        let session_id = params.get("session_id").and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let entities: Vec<String> = params.get("entities")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let decisions: Vec<String> = params.get("decisions")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let mut graph = self.graph.write().await;

        // Create session summary node
        let summary_label = format!("session_{}_summary", session_id);
        let summary_id = graph.upsert_node_with_tags(
            &summary_label,
            NodeKind::Event,
            vec!["session_summary".to_string(), session_id.to_string()],
        );

        // Link entities via Trail
        let mut edges_created = 0;
        for entity in &entities {
            let entity_id = graph.upsert_node(entity, NodeKind::Entity);
            if graph.upsert_edge(summary_id, entity_id, Channel::Trail, 0.7, "mcp:compact").is_some() {
                edges_created += 1;
            }
        }

        // Decisions become Concept nodes linked via Causal
        for decision in &decisions {
            let decision_id = graph.upsert_node(decision, NodeKind::Concept);
            if graph.upsert_edge(summary_id, decision_id, Channel::Causal, 0.8, "mcp:compact").is_some() {
                edges_created += 1;
            }
        }

        // Train HDC on summary text for semantic retrieval
        let mut h = self.hdc.write().await;
        h.train(&[summary.to_string()]);
        drop(h);

        // WAL
        let mut s = self.store.write().await;
        let _ = s.write_wal(&soma_store::WalEntry::Custom(format!(
            "SessionCompact: {} entities={} decisions={}",
            session_id, entities.len(), decisions.len()
        )));

        Ok(serde_json::json!({
            "session_id": session_id,
            "summary_node": summary_label,
            "entities_linked": entities.len(),
            "decisions_stored": decisions.len(),
            "edges_created": edges_created,
        }))
    }

    /// A.3 — Restore previous session context.
    async fn soma_session_restore(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let query_str = params.get("query").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'query'".into()))?;
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let graph = self.graph.read().await;

        // Find session summary nodes
        let session_nodes: Vec<_> = graph.all_nodes()
            .filter(|n| n.tags.contains(&"session_summary".to_string()))
            .collect();

        if session_nodes.is_empty() {
            return Ok(serde_json::json!({ "sessions": [], "count": 0 }));
        }

        // Use fuzzy search against session labels
        let session_labels: Vec<String> = session_nodes.iter().map(|n| n.label.clone()).collect();
        let matches = fuzzy_label_search(query_str, &session_labels, limit);

        // Also try HDC search
        let h = self.hdc.read().await;
        let hdc_matches = h.search_labels(query_str, &session_labels, limit);
        drop(h);

        // RRF merge
        let ranked_lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("fuzzy", matches),
            ("hdc", hdc_matches),
        ];
        let hybrid = rrf_merge_with_sources(&ranked_lists, 60.0);

        let sessions: Vec<serde_json::Value> = hybrid.iter()
            .take(limit)
            .filter_map(|hr| {
                graph.get_node_by_label(&hr.label).map(|node| {
                    // Get linked entities and decisions
                    let edges = graph.outgoing_edges(node.id);
                    let entities: Vec<String> = edges.iter()
                        .filter(|e| e.channel == Channel::Trail)
                        .filter_map(|e| graph.get_node(e.to).map(|n| n.label.clone()))
                        .collect();
                    let decisions: Vec<String> = edges.iter()
                        .filter(|e| e.channel == Channel::Causal)
                        .filter_map(|e| graph.get_node(e.to).map(|n| n.label.clone()))
                        .collect();
                    serde_json::json!({
                        "session": node.label,
                        "entities": entities,
                        "decisions": decisions,
                        "score": hr.score,
                    })
                })
            })
            .collect();

        let count = sessions.len();
        Ok(serde_json::json!({
            "sessions": sessions,
            "count": count,
        }))
    }

    /// E.2 — Explain paths between two entities.
    async fn soma_explain(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let from = params.get("from").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'from'".into()))?;
        let to = params.get("to").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'to'".into()))?;
        let max_paths = params.get("max_paths").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

        let graph = self.graph.read().await;

        // BFS from 'from' to find paths to 'to'
        let query = SomaQuery::new(from)
            .with_max_hops(5)
            .with_min_intensity(0.05)
            .with_limit(100);
        let results = graph.traverse(&query);

        let paths: Vec<serde_json::Value> = results.iter()
            .filter(|r| r.node.label.to_lowercase() == to.to_lowercase())
            .take(max_paths)
            .map(|r| {
                let path_str: Vec<String> = r.path.iter().map(|e| {
                    let from_node = graph.get_node(e.from).map(|n| n.label.as_str()).unwrap_or("?");
                    let to_node = graph.get_node(e.to).map(|n| n.label.as_str()).unwrap_or("?");
                    let label = e.label.as_deref().unwrap_or(e.channel.as_str());
                    format!("{} --[{}]--> {}", from_node, label, to_node)
                }).collect();
                serde_json::json!({
                    "path": path_str,
                    "score": r.score,
                    "hops": r.hops,
                })
            })
            .collect();

        if paths.is_empty() {
            return Ok(serde_json::json!({
                "paths": [],
                "message": format!("no path found from '{}' to '{}'", from, to),
            }));
        }

        Ok(serde_json::json!({
            "paths": paths,
            "from": from,
            "to": to,
        }))
    }

    /// E.3 — Merge duplicate nodes.
    async fn soma_merge(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let keep = params.get("keep").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'keep'".into()))?;
        let absorb = params.get("absorb").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'absorb'".into()))?;
        let reason = params.get("reason").and_then(|v| v.as_str()).unwrap_or("merge");

        let mut graph = self.graph.write().await;
        match graph.merge_nodes(keep, absorb) {
            Some(transferred) => {
                let mut s = self.store.write().await;
                let _ = s.write_wal(&soma_store::WalEntry::Custom(format!(
                    "NodeMerge: keep={} absorb={} reason={} transferred={}",
                    keep, absorb, reason, transferred
                )));
                Ok(serde_json::json!({
                    "merged": true,
                    "keep": keep,
                    "absorbed": absorb,
                    "edges_transferred": transferred,
                    "reason": reason,
                }))
            }
            None => Err(SomaError::Mcp(format!("node '{}' or '{}' not found", keep, absorb))),
        }
    }

    /// C.2 — Community detection via Louvain.
    async fn soma_communities(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let min_size = params.get("min_size").and_then(|v| v.as_u64()).unwrap_or(3) as usize;

        let graph = self.graph.read().await;
        let result = soma_graph::detect_communities(graph.inner(), min_size);

        let communities: Vec<serde_json::Value> = result.communities.iter()
            .map(|(id, members)| {
                serde_json::json!({
                    "id": id,
                    "size": members.len(),
                    "members": members,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "communities": communities,
            "total": result.communities.len(),
            "modularity": result.modularity,
        }))
    }

    /// F.2 — Record a reasoning step (Graph of Thoughts).
    async fn soma_think(&self, params: &serde_json::Value) -> Result<serde_json::Value, SomaError> {
        let thought = params.get("thought").and_then(|v| v.as_str())
            .ok_or_else(|| SomaError::Mcp("missing 'thought'".into()))?;
        let depends_on: Vec<String> = params.get("depends_on")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let is_conclusion = params.get("conclusion").and_then(|v| v.as_bool()).unwrap_or(false);

        let mut graph = self.graph.write().await;

        // Create thought node
        let thought_id = graph.upsert_node_with_tags(
            thought,
            NodeKind::Concept,
            vec!["thought".to_string()],
        );

        let mut edges_created = 0;

        // Link to dependencies via Reasoning channel (fast decay)
        // If conclusion, use Causal (more durable)
        let channel = if is_conclusion { Channel::Causal } else { Channel::Reasoning };

        for dep in &depends_on {
            let dep_id = graph.upsert_node(dep, NodeKind::Concept);
            if graph.upsert_edge(dep_id, thought_id, channel, 0.8, "mcp:think").is_some() {
                edges_created += 1;
            }
        }

        Ok(serde_json::json!({
            "thought": thought,
            "is_conclusion": is_conclusion,
            "channel": channel.as_str(),
            "dependencies": depends_on.len(),
            "edges_created": edges_created,
        }))
    }
}
