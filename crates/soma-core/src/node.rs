use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::NodeId;

/// Kind of knowledge entity stored in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Concrete object: tool, file, protein, character
    Entity,
    /// Abstract idea: performance, reliability, technique
    Concept,
    /// Dated event: session, experiment, incident
    Event,
    /// Measured value: pLDDT=94.2, λ=523nm, latency=16ms
    Measurement,
    /// Know-how: procedure, recipe, pattern
    Procedure,
    /// Known problem: bug, incompatibility, constraint
    Warning,
}

impl NodeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeKind::Entity => "entity",
            NodeKind::Concept => "concept",
            NodeKind::Event => "event",
            NodeKind::Measurement => "measurement",
            NodeKind::Procedure => "procedure",
            NodeKind::Warning => "warning",
        }
    }

    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "entity" => Some(NodeKind::Entity),
            "concept" => Some(NodeKind::Concept),
            "event" => Some(NodeKind::Event),
            "measurement" => Some(NodeKind::Measurement),
            "procedure" => Some(NodeKind::Procedure),
            "warning" => Some(NodeKind::Warning),
            _ => None,
        }
    }
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A node in the SOMA knowledge graph.
/// Label is unique within a workspace — deterministic NodeId from label ensures deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaNode {
    pub id: NodeId,
    /// Primary text — unique within a workspace
    pub label: String,
    pub kind: NodeKind,
    pub created_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    /// Keywords for fast filtering
    pub tags: Vec<String>,
    /// Free-form metadata
    pub meta: Option<serde_json::Value>,
}

impl SomaNode {
    /// Create a new node with deterministic ID from workspace-scoped label.
    pub fn new(workspace: &str, label: &str, kind: NodeKind) -> Self {
        let now = Utc::now();
        let scoped = format!("{}:{}", workspace, label);
        SomaNode {
            id: NodeId::from_label(&scoped),
            label: label.to_string(),
            kind,
            created_at: now,
            last_seen: now,
            tags: Vec::new(),
            meta: None,
        }
    }

    /// Create with tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Create with metadata.
    pub fn with_meta(mut self, meta: serde_json::Value) -> Self {
        self.meta = Some(meta);
        self
    }

    /// Touch — update last_seen timestamp.
    pub fn touch(&mut self) {
        self.last_seen = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_creation() {
        let node = SomaNode::new("default", "ChromoQ", NodeKind::Entity);
        assert_eq!(node.label, "ChromoQ");
        assert_eq!(node.kind, NodeKind::Entity);
        assert!(node.tags.is_empty());
    }

    #[test]
    fn node_with_tags() {
        let node = SomaNode::new("default", "ChromoQ", NodeKind::Entity)
            .with_tags(vec!["protein".into(), "fluorescent".into()]);
        assert_eq!(node.tags.len(), 2);
    }

    #[test]
    fn deterministic_id_from_workspace_label() {
        let a = SomaNode::new("research", "ChromoQ", NodeKind::Entity);
        let b = SomaNode::new("research", "ChromoQ", NodeKind::Entity);
        assert_eq!(a.id, b.id);
    }

    #[test]
    fn different_workspace_different_id() {
        let a = SomaNode::new("research", "ChromoQ", NodeKind::Entity);
        let b = SomaNode::new("panlunadra", "ChromoQ", NodeKind::Entity);
        assert_ne!(a.id, b.id);
    }
}
