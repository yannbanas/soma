//! Cypher executor — runs parsed AST against a StigreGraph.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use soma_core::{Channel, NodeKind, SomaNode, SomaQuery};
use soma_graph::StigreGraph;

use crate::ast::*;
use crate::lexer::Lexer;
use crate::parser::Parser;

/// Result value from a Cypher query.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CypherValue {
    String(String),
    Float(f64),
    Int(i64),
    Bool(bool),
    Null,
    Node(CypherNode),
}

/// Simplified node representation for query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypherNode {
    pub label: String,
    pub kind: String,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

impl From<&SomaNode> for CypherNode {
    fn from(node: &SomaNode) -> Self {
        CypherNode {
            label: node.label.clone(),
            kind: node.kind.as_str().to_string(),
            tags: node.tags.clone(),
            meta: node.meta.clone(),
        }
    }
}

/// Result of a Cypher query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypherResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<CypherValue>>,
    pub message: Option<String>,
}

impl CypherResult {
    fn empty_with_message(msg: &str) -> Self {
        CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            message: Some(msg.to_string()),
        }
    }
}

/// Executor that runs Cypher statements against a StigreGraph.
pub struct CypherExecutor;

impl CypherExecutor {
    /// Parse and execute a Cypher query string.
    pub fn execute(graph: &mut StigreGraph, query: &str) -> Result<CypherResult, String> {
        let mut lexer = Lexer::new(query);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(tokens);
        let stmt = parser.parse()?;
        Self::execute_statement(graph, &stmt)
    }

    /// Execute a parsed statement.
    pub fn execute_statement(
        graph: &mut StigreGraph,
        stmt: &CypherStatement,
    ) -> Result<CypherResult, String> {
        match stmt {
            CypherStatement::Query(q) => Self::execute_query(graph, q),
            CypherStatement::Create(c) => Self::execute_create(graph, c),
            CypherStatement::Delete(d) => Self::execute_delete(graph, d),
            CypherStatement::Set(s) => Self::execute_set(graph, s),
        }
    }

    // --- Query execution ---

    fn execute_query(graph: &StigreGraph, query: &CypherQuery) -> Result<CypherResult, String> {
        // Step 1: Find all matching bindings from MATCH clause
        let bindings = Self::match_patterns(graph, &query.match_clause)?;

        // Step 2: Filter by WHERE clause
        let now = Utc::now();
        let filtered: Vec<_> = bindings
            .into_iter()
            .filter(|binding| {
                query
                    .where_clause
                    .as_ref()
                    .map_or(true, |w| Self::eval_where(w, binding, graph, now))
            })
            .collect();

        // Step 3: Build RETURN columns
        let columns: Vec<String> = query
            .return_clause
            .items
            .iter()
            .map(|item| match item {
                ReturnItem::Property(pa) => format!("{}.{}", pa.variable, pa.property),
                ReturnItem::Variable(v) => v.clone(),
                ReturnItem::Function(f, arg) => format!("{}({})", f, arg),
            })
            .collect();

        let mut rows: Vec<Vec<CypherValue>> = filtered
            .iter()
            .map(|binding| {
                query
                    .return_clause
                    .items
                    .iter()
                    .map(|item| Self::eval_return_item(item, binding, graph))
                    .collect()
            })
            .collect();

        // Step 4: ORDER BY
        if let Some(ref order) = query.order_by {
            let var = &order.property.variable;
            let prop = &order.property.property;
            rows.sort_by(|a, b| {
                // Find the column index for this property
                let idx = columns
                    .iter()
                    .position(|c| c == &format!("{}.{}", var, prop))
                    .unwrap_or(0);
                let cmp = compare_cypher_values(
                    a.get(idx).unwrap_or(&CypherValue::Null),
                    b.get(idx).unwrap_or(&CypherValue::Null),
                );
                if order.descending {
                    cmp.reverse()
                } else {
                    cmp
                }
            });
        }

        // Step 5: LIMIT
        if let Some(limit) = query.limit {
            rows.truncate(limit);
        }

        Ok(CypherResult {
            columns,
            rows,
            message: None,
        })
    }

    /// Match patterns against the graph and return variable bindings.
    /// Each binding maps variable names to node labels.
    fn match_patterns(
        graph: &StigreGraph,
        match_clause: &MatchClause,
    ) -> Result<Vec<HashMap<String, String>>, String> {
        let mut all_bindings: Vec<HashMap<String, String>> = vec![HashMap::new()];

        for pattern in &match_clause.patterns {
            let mut new_bindings = Vec::new();

            for existing in &all_bindings {
                let pattern_bindings = Self::match_single_pattern(graph, pattern, existing)?;
                new_bindings.extend(pattern_bindings);
            }

            all_bindings = new_bindings;
        }

        Ok(all_bindings)
    }

    fn match_single_pattern(
        graph: &StigreGraph,
        pattern: &Pattern,
        existing: &HashMap<String, String>,
    ) -> Result<Vec<HashMap<String, String>>, String> {
        // Extract the node patterns from elements
        let mut node_patterns: Vec<&NodePattern> = Vec::new();
        let mut rel_patterns: Vec<&RelPattern> = Vec::new();

        for elem in &pattern.elements {
            match elem {
                PatternElement::Node(n) => node_patterns.push(n),
                PatternElement::Relationship(r) => rel_patterns.push(r),
            }
        }

        if node_patterns.is_empty() {
            return Ok(vec![existing.clone()]);
        }

        // Single node pattern without relationships — scan all matching nodes
        if rel_patterns.is_empty() {
            let first = node_patterns[0];
            let candidates = Self::find_matching_nodes(graph, first);
            let mut results = Vec::new();

            for node in candidates {
                let mut binding = existing.clone();
                if let Some(ref var) = first.variable {
                    if let Some(existing_label) = binding.get(var) {
                        if existing_label != &node.label {
                            continue;
                        }
                    }
                    binding.insert(var.clone(), node.label.clone());
                }
                results.push(binding);
            }

            return Ok(results);
        }

        // Pattern with relationships — use graph traversal
        let first_node = node_patterns[0];
        let start_nodes = Self::find_matching_nodes(graph, first_node);
        let now = Utc::now();

        let mut results = Vec::new();

        for start in &start_nodes {
            let mut binding = existing.clone();
            if let Some(ref var) = first_node.variable {
                binding.insert(var.clone(), start.label.clone());
            }

            // Walk the pattern chain
            let chain_results =
                Self::walk_pattern_chain(graph, start, &node_patterns[1..], &rel_patterns, &binding, now);
            results.extend(chain_results);
        }

        Ok(results)
    }

    fn walk_pattern_chain(
        graph: &StigreGraph,
        current_node: &SomaNode,
        remaining_nodes: &[&NodePattern],
        remaining_rels: &[&RelPattern],
        binding: &HashMap<String, String>,
        now: chrono::DateTime<Utc>,
    ) -> Vec<HashMap<String, String>> {
        if remaining_rels.is_empty() || remaining_nodes.is_empty() {
            return vec![binding.clone()];
        }

        let rel_pattern = remaining_rels[0];
        let next_node_pattern = remaining_nodes[0];

        let max_hops = rel_pattern.max_hops.unwrap_or(1) as usize;
        let min_hops = rel_pattern.min_hops.unwrap_or(1) as usize;

        // Get channel filter from relationship type
        let channel_filter: Vec<Channel> = rel_pattern
            .rel_type
            .as_deref()
            .and_then(Channel::from_str_name)
            .map(|c| vec![c])
            .unwrap_or_default();

        // Use SOMA's existing traverse for variable-length paths
        let query = SomaQuery {
            start: current_node.label.clone(),
            channels: channel_filter,
            max_hops: max_hops as u8,
            min_intensity: 0.0, // WHERE clause handles filtering
            semantic: false,
            workspace: graph.workspace().to_string(),
            since: None,
            until: None,
            limit: 10_000, // gather all, filter later
        };

        let traverse_results = graph.traverse(&query);

        let mut results = Vec::new();

        for qr in &traverse_results {
            let hops = qr.hops as usize;
            if hops < min_hops || hops > max_hops {
                continue;
            }
            // Skip the start node itself (hops=0)
            if hops == 0 {
                continue;
            }

            // Check if target matches the next node pattern
            if !Self::node_matches_pattern(&qr.node, next_node_pattern) {
                continue;
            }

            let mut new_binding = binding.clone();
            if let Some(ref var) = next_node_pattern.variable {
                new_binding.insert(var.clone(), qr.node.label.clone());
            }
            if let Some(ref var) = rel_pattern.variable {
                // Store edge info as "channel:from->to"
                if let Some(edge) = qr.path.last() {
                    new_binding.insert(
                        var.clone(),
                        format!(
                            "{}:{}->{}",
                            edge.channel,
                            current_node.label,
                            qr.node.label
                        ),
                    );
                }
            }

            // Continue walking if more patterns remain
            let more = Self::walk_pattern_chain(
                graph,
                &qr.node,
                &remaining_nodes[1..],
                &remaining_rels[1..],
                &new_binding,
                now,
            );
            results.extend(more);
        }

        results
    }

    fn find_matching_nodes<'a>(graph: &'a StigreGraph, pattern: &NodePattern) -> Vec<&'a SomaNode> {
        graph
            .all_nodes()
            .filter(|node| Self::node_matches_pattern(node, pattern))
            .collect()
    }

    fn node_matches_pattern(node: &SomaNode, pattern: &NodePattern) -> bool {
        // Check label filter (NodeKind)
        if let Some(ref label) = pattern.label {
            if let Some(kind) = NodeKind::from_str_name(label) {
                if node.kind != kind {
                    return false;
                }
            }
            // If not a valid NodeKind, treat as a tag filter
            else {
                let label_lower = label.to_lowercase();
                if !node.tags.iter().any(|t| t.to_lowercase() == label_lower) {
                    return false;
                }
            }
        }

        // Check property filters
        for (key, val) in &pattern.properties {
            match key.as_str() {
                "label" => {
                    if let PropValue::String(s) = val {
                        if node.label != *s {
                            return false;
                        }
                    }
                }
                "kind" => {
                    if let PropValue::String(s) = val {
                        if node.kind.as_str() != s.to_lowercase() {
                            return false;
                        }
                    }
                }
                _ => {
                    // Check in tags
                    if let PropValue::String(s) = val {
                        if key == "tag" && !node.tags.contains(s) {
                            return false;
                        }
                    }
                    // Check in meta
                    if let Some(ref meta) = node.meta {
                        if let Some(meta_val) = meta.get(key) {
                            if !prop_matches_json(val, meta_val) {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        true
    }

    // --- WHERE evaluation ---

    fn eval_where(
        expr: &WhereExpr,
        binding: &HashMap<String, String>,
        graph: &StigreGraph,
        now: chrono::DateTime<Utc>,
    ) -> bool {
        match expr {
            WhereExpr::And(a, b) => {
                Self::eval_where(a, binding, graph, now)
                    && Self::eval_where(b, binding, graph, now)
            }
            WhereExpr::Or(a, b) => {
                Self::eval_where(a, binding, graph, now)
                    || Self::eval_where(b, binding, graph, now)
            }
            WhereExpr::Not(e) => !Self::eval_where(e, binding, graph, now),
            WhereExpr::Contains(prop, s) => {
                let val = Self::resolve_property(prop, binding, graph, now);
                match val {
                    CypherValue::String(v) => v.to_lowercase().contains(&s.to_lowercase()),
                    _ => false,
                }
            }
            WhereExpr::Comparison(cmp) => {
                let left_val = Self::resolve_property(&cmp.left, binding, graph, now);
                Self::compare_value(&left_val, &cmp.op, &cmp.right)
            }
        }
    }

    fn resolve_property(
        prop: &PropertyAccess,
        binding: &HashMap<String, String>,
        graph: &StigreGraph,
        now: chrono::DateTime<Utc>,
    ) -> CypherValue {
        let Some(node_label) = binding.get(&prop.variable) else {
            return CypherValue::Null;
        };

        let Some(node) = graph.get_node_by_label(node_label) else {
            return CypherValue::Null;
        };

        match prop.property.as_str() {
            "label" => CypherValue::String(node.label.clone()),
            "kind" => CypherValue::String(node.kind.as_str().to_string()),
            "tags" => CypherValue::String(node.tags.join(", ")),
            "created_at" => CypherValue::String(node.created_at.to_rfc3339()),
            "last_seen" => CypherValue::String(node.last_seen.to_rfc3339()),
            // Edge properties — check outgoing edges
            "intensity" => {
                let edges = graph.outgoing_edges(node.id);
                let max_intensity = edges
                    .iter()
                    .map(|e| e.effective_intensity(now))
                    .fold(0.0_f32, f32::max);
                CypherValue::Float(max_intensity as f64)
            }
            "confidence" => {
                let edges = graph.outgoing_edges(node.id);
                let max_conf = edges
                    .iter()
                    .map(|e| e.confidence)
                    .fold(0.0_f32, f32::max);
                CypherValue::Float(max_conf as f64)
            }
            "uses" => {
                let edges = graph.outgoing_edges(node.id);
                let total: u32 = edges.iter().map(|e| e.uses).sum();
                CypherValue::Int(total as i64)
            }
            // Check meta fields
            other => {
                if let Some(ref meta) = node.meta {
                    if let Some(val) = meta.get(other) {
                        return json_to_cypher(val);
                    }
                }
                // Check if it's a tag query
                if other == "tag" {
                    return CypherValue::String(node.tags.join(", "));
                }
                CypherValue::Null
            }
        }
    }

    fn compare_value(left: &CypherValue, op: &CompOp, right: &PropValue) -> bool {
        match (left, right) {
            (CypherValue::String(l), PropValue::String(r)) => match op {
                CompOp::Eq => l == r,
                CompOp::Neq => l != r,
                CompOp::Lt => l < r,
                CompOp::Gt => l > r,
                CompOp::Lte => l <= r,
                CompOp::Gte => l >= r,
            },
            (CypherValue::Float(l), PropValue::Float(r)) => match op {
                CompOp::Eq => (l - r).abs() < f64::EPSILON,
                CompOp::Neq => (l - r).abs() >= f64::EPSILON,
                CompOp::Lt => l < r,
                CompOp::Gt => l > r,
                CompOp::Lte => l <= r,
                CompOp::Gte => l >= r,
            },
            (CypherValue::Float(l), PropValue::Int(r)) => {
                let r = *r as f64;
                match op {
                    CompOp::Eq => (l - r).abs() < f64::EPSILON,
                    CompOp::Neq => (l - r).abs() >= f64::EPSILON,
                    CompOp::Lt => *l < r,
                    CompOp::Gt => *l > r,
                    CompOp::Lte => *l <= r,
                    CompOp::Gte => *l >= r,
                }
            }
            (CypherValue::Int(l), PropValue::Int(r)) => match op {
                CompOp::Eq => l == r,
                CompOp::Neq => l != r,
                CompOp::Lt => l < r,
                CompOp::Gt => l > r,
                CompOp::Lte => l <= r,
                CompOp::Gte => l >= r,
            },
            (CypherValue::Bool(l), PropValue::Bool(r)) => match op {
                CompOp::Eq => l == r,
                CompOp::Neq => l != r,
                _ => false,
            },
            _ => false,
        }
    }

    // --- RETURN evaluation ---

    fn eval_return_item(
        item: &ReturnItem,
        binding: &HashMap<String, String>,
        graph: &StigreGraph,
    ) -> CypherValue {
        let now = Utc::now();
        match item {
            ReturnItem::Property(prop) => Self::resolve_property(prop, binding, graph, now),
            ReturnItem::Variable(var) => {
                if let Some(label) = binding.get(var) {
                    if let Some(node) = graph.get_node_by_label(label) {
                        CypherValue::Node(CypherNode::from(node))
                    } else {
                        CypherValue::String(label.clone())
                    }
                } else {
                    CypherValue::Null
                }
            }
            ReturnItem::Function(func, _arg) => {
                // Simple aggregation functions
                match func.to_uppercase().as_str() {
                    "COUNT" => {
                        // Count would need full result set — for now return 1 per binding
                        CypherValue::Int(1)
                    }
                    _ => CypherValue::Null,
                }
            }
        }
    }

    // --- CREATE ---

    fn execute_create(graph: &mut StigreGraph, create: &CreateClause) -> Result<CypherResult, String> {
        // Resolve source nodes from MATCH clause
        let bindings = if let Some(ref match_clause) = create.match_clause {
            let b = Self::match_patterns(graph, match_clause)?;
            if b.is_empty() {
                return Ok(CypherResult::empty_with_message("No matching nodes found for CREATE"));
            }
            b
        } else {
            vec![HashMap::new()]
        };

        let mut created = 0;

        for binding in &bindings {
            let from_label = binding.get(&create.from_var).ok_or_else(|| {
                format!("Variable '{}' not bound in MATCH clause", create.from_var)
            })?;
            let to_label = binding.get(&create.to_var).ok_or_else(|| {
                format!("Variable '{}' not bound in MATCH clause", create.to_var)
            })?;

            let channel = Channel::from_str_name(&create.rel_type)
                .ok_or_else(|| format!("Unknown channel/relationship type: '{}'", create.rel_type))?;

            let confidence = create
                .properties
                .get("confidence")
                .and_then(|v| match v {
                    PropValue::Float(f) => Some(*f as f32),
                    PropValue::Int(i) => Some(*i as f32),
                    _ => None,
                })
                .unwrap_or(0.7);

            let source = create
                .properties
                .get("source")
                .and_then(|v| match v {
                    PropValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or("cypher");

            let edge_label = create
                .properties
                .get("label")
                .and_then(|v| match v {
                    PropValue::String(s) => Some(s.as_str()),
                    _ => None,
                });

            let from_id = graph
                .node_id_by_label(from_label)
                .ok_or_else(|| format!("Node '{}' not found", from_label))?;
            let to_id = graph
                .node_id_by_label(to_label)
                .ok_or_else(|| format!("Node '{}' not found", to_label))?;

            graph.upsert_edge_labeled(from_id, to_id, channel, confidence, source, edge_label);
            created += 1;
        }

        Ok(CypherResult::empty_with_message(&format!(
            "Created {} relationship(s)",
            created
        )))
    }

    // --- DELETE ---

    fn execute_delete(graph: &mut StigreGraph, delete: &DeleteClause) -> Result<CypherResult, String> {
        let bindings = Self::match_patterns(graph, &delete.match_clause)?;

        let now = Utc::now();
        let filtered: Vec<_> = bindings
            .into_iter()
            .filter(|b| {
                delete
                    .where_clause
                    .as_ref()
                    .map_or(true, |w| Self::eval_where(w, b, graph, now))
            })
            .collect();

        let mut deleted = 0;

        // Collect labels to delete first (avoid borrow issues)
        let labels_to_delete: Vec<String> = filtered
            .iter()
            .flat_map(|binding| {
                delete
                    .variables
                    .iter()
                    .filter_map(|var| binding.get(var).cloned())
            })
            .collect();

        for label in labels_to_delete {
            if graph.remove_node_by_label(&label).is_some() {
                deleted += 1;
            }
        }

        Ok(CypherResult::empty_with_message(&format!(
            "Deleted {} node(s)",
            deleted
        )))
    }

    // --- SET ---

    fn execute_set(_graph: &mut StigreGraph, _set: &SetClause) -> Result<CypherResult, String> {
        // SET on nodes is limited — SOMA nodes have fixed fields
        // For now, support setting tags via meta
        Ok(CypherResult::empty_with_message(
            "SET operations on nodes are not yet supported (SOMA nodes have fixed schema)",
        ))
    }
}

// --- Helpers ---

fn prop_matches_json(prop: &PropValue, json: &serde_json::Value) -> bool {
    match (prop, json) {
        (PropValue::String(s), serde_json::Value::String(j)) => s == j,
        (PropValue::Int(i), serde_json::Value::Number(n)) => n.as_i64() == Some(*i),
        (PropValue::Float(f), serde_json::Value::Number(n)) => {
            n.as_f64().map_or(false, |nf| (nf - f).abs() < f64::EPSILON)
        }
        (PropValue::Bool(b), serde_json::Value::Bool(j)) => b == j,
        _ => false,
    }
}

fn json_to_cypher(val: &serde_json::Value) -> CypherValue {
    match val {
        serde_json::Value::String(s) => CypherValue::String(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                CypherValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                CypherValue::Float(f)
            } else {
                CypherValue::Null
            }
        }
        serde_json::Value::Bool(b) => CypherValue::Bool(*b),
        serde_json::Value::Null => CypherValue::Null,
        _ => CypherValue::String(val.to_string()),
    }
}

fn compare_cypher_values(a: &CypherValue, b: &CypherValue) -> std::cmp::Ordering {
    match (a, b) {
        (CypherValue::String(a), CypherValue::String(b)) => a.cmp(b),
        (CypherValue::Int(a), CypherValue::Int(b)) => a.cmp(b),
        (CypherValue::Float(a), CypherValue::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (CypherValue::Null, _) => std::cmp::Ordering::Greater,
        (_, CypherValue::Null) => std::cmp::Ordering::Less,
        _ => std::cmp::Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::NodeKind;

    fn setup_graph() -> StigreGraph {
        let mut g = StigreGraph::new("test", 0.05);
        let chromoq = g.upsert_node("ChromoQ", NodeKind::Entity);
        let fluor = g.upsert_node("fluorescence", NodeKind::Concept);
        let rust = g.upsert_node_with_tags("Rust", NodeKind::Entity, vec!["language".into()]);
        let soma = g.upsert_node("SOMA", NodeKind::Entity);

        g.upsert_edge(chromoq, fluor, Channel::Trail, 0.9, "test");
        g.upsert_edge(chromoq, rust, Channel::Causal, 0.7, "test");
        g.upsert_edge(rust, soma, Channel::Trail, 0.8, "test");
        g
    }

    #[test]
    fn test_simple_match() {
        let mut g = setup_graph();
        let result = CypherExecutor::execute(&mut g, "MATCH (n:Entity) RETURN n.label").unwrap();
        assert!(!result.rows.is_empty());
        assert_eq!(result.columns, vec!["n.label"]);
    }

    #[test]
    fn test_match_with_property() {
        let mut g = setup_graph();
        let result = CypherExecutor::execute(
            &mut g,
            r#"MATCH (n {label: "ChromoQ"}) RETURN n.label, n.kind"#,
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
        if let CypherValue::String(ref s) = result.rows[0][0] {
            assert_eq!(s, "ChromoQ");
        }
    }

    #[test]
    fn test_match_with_relationship() {
        let mut g = setup_graph();
        let result = CypherExecutor::execute(
            &mut g,
            "MATCH (a:Entity)-[:Trail]->(b) RETURN a.label, b.label",
        )
        .unwrap();
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_where_contains() {
        let mut g = setup_graph();
        let result = CypherExecutor::execute(
            &mut g,
            r#"MATCH (n) WHERE n.label CONTAINS "Chromo" RETURN n.label"#,
        )
        .unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_where_comparison() {
        let mut g = setup_graph();
        let result = CypherExecutor::execute(
            &mut g,
            "MATCH (n:Entity) WHERE n.intensity > 0.5 RETURN n.label",
        )
        .unwrap();
        // Nodes with outgoing edges having intensity > 0.5
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_limit() {
        let mut g = setup_graph();
        let result = CypherExecutor::execute(
            &mut g,
            "MATCH (n) RETURN n.label LIMIT 2",
        )
        .unwrap();
        assert!(result.rows.len() <= 2);
    }

    #[test]
    fn test_create_relationship() {
        let mut g = setup_graph();
        let result = CypherExecutor::execute(
            &mut g,
            r#"MATCH (a {label: "SOMA"}), (b {label: "Rust"}) CREATE (a)-[:Trail {confidence: 0.9}]->(b)"#,
        )
        .unwrap();
        assert!(result.message.unwrap().contains("Created 1"));
    }

    #[test]
    fn test_delete_node() {
        let mut g = setup_graph();
        assert!(g.get_node_by_label("SOMA").is_some());
        let result = CypherExecutor::execute(
            &mut g,
            r#"MATCH (n {label: "SOMA"}) DELETE n"#,
        )
        .unwrap();
        assert!(result.message.unwrap().contains("Deleted 1"));
        assert!(g.get_node_by_label("SOMA").is_none());
    }

    #[test]
    fn test_variable_length_path() {
        let mut g = setup_graph();
        // ChromoQ -[:Trail]-> fluorescence (1 hop)
        // ChromoQ -[:Causal]-> Rust -[:Trail]-> SOMA (2 hops via mixed channels)
        let result = CypherExecutor::execute(
            &mut g,
            r#"MATCH (a {label: "ChromoQ"})-[*1..3]->(b) RETURN b.label"#,
        )
        .unwrap();
        // Should find fluorescence, Rust, and SOMA
        assert!(result.rows.len() >= 2);
    }
}
