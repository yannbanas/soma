use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::channel::Channel;
use crate::edge::StigreEdge;
use crate::node::SomaNode;

/// Query object for graph traversal.
/// Supports channel filtering, hop depth, intensity thresholds,
/// semantic enrichment, temporal filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaQuery {
    /// Start point — free text (matched by label or HDC)
    pub start: String,
    /// Allowed channels for traversal (empty = all except Alarm)
    pub channels: Vec<Channel>,
    /// Maximum traversal depth
    pub max_hops: u8,
    /// Minimum effective intensity — edges below are ignored
    pub min_intensity: f32,
    /// If true, enrich results with HDC semantic similarity
    pub semantic: bool,
    /// Target workspace
    pub workspace: String,
    /// Optional temporal filter — only edges touched after this
    pub since: Option<DateTime<Utc>>,
    /// Optional temporal filter — only edges touched before this
    pub until: Option<DateTime<Utc>>,
    /// Maximum number of results
    pub limit: usize,
}

impl Default for SomaQuery {
    fn default() -> Self {
        SomaQuery {
            start: String::new(),
            channels: Vec::new(), // empty = all except Alarm
            max_hops: 3,
            min_intensity: 0.15,
            semantic: true,
            workspace: "default".to_string(),
            since: None,
            until: None,
            limit: 10,
        }
    }
}

impl SomaQuery {
    pub fn new(start: &str) -> Self {
        SomaQuery {
            start: start.to_string(),
            ..Default::default()
        }
    }

    pub fn with_workspace(mut self, workspace: &str) -> Self {
        self.workspace = workspace.to_string();
        self
    }

    pub fn with_channels(mut self, channels: Vec<Channel>) -> Self {
        self.channels = channels;
        self
    }

    pub fn with_max_hops(mut self, hops: u8) -> Self {
        self.max_hops = hops;
        self
    }

    pub fn with_min_intensity(mut self, min: f32) -> Self {
        self.min_intensity = min.clamp(0.0, 1.0);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        // Security: cap at reasonable maximum to prevent memory exhaustion
        self.limit = limit.min(10_000);
        self
    }

    /// Check if a channel is allowed by this query.
    pub fn allows_channel(&self, ch: &Channel) -> bool {
        if self.channels.is_empty() {
            // Default: all except Alarm
            !matches!(ch, Channel::Alarm)
        } else {
            self.channels.contains(ch)
        }
    }
}

/// Result of a graph traversal query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub node: SomaNode,
    pub path: Vec<StigreEdge>,
    /// Product of effective intensities along the path
    pub score: f32,
    pub hops: u8,
    /// Which search paths contributed: "graph", "hdc", "fuzzy"
    #[serde(default)]
    pub sources: Vec<String>,
}

impl QueryResult {
    pub fn new(node: SomaNode, path: Vec<StigreEdge>, score: f32, hops: u8) -> Self {
        QueryResult {
            node,
            path,
            score,
            hops,
            sources: Vec::new(),
        }
    }

    pub fn with_sources(mut self, sources: Vec<String>) -> Self {
        self.sources = sources;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_query() {
        let q = SomaQuery::default();
        assert_eq!(q.max_hops, 3);
        assert_eq!(q.min_intensity, 0.15);
        assert!(q.semantic);
        assert_eq!(q.workspace, "default");
        assert_eq!(q.limit, 10);
    }

    #[test]
    fn channel_filter_default_excludes_alarm() {
        let q = SomaQuery::default();
        assert!(q.allows_channel(&Channel::Trail));
        assert!(q.allows_channel(&Channel::Causal));
        assert!(!q.allows_channel(&Channel::Alarm));
    }

    #[test]
    fn channel_filter_explicit() {
        let q = SomaQuery::new("test").with_channels(vec![Channel::Trail, Channel::Alarm]);
        assert!(q.allows_channel(&Channel::Trail));
        assert!(q.allows_channel(&Channel::Alarm));
        assert!(!q.allows_channel(&Channel::Episodic));
    }

    #[test]
    fn limit_capped() {
        let q = SomaQuery::new("test").with_limit(999_999);
        assert_eq!(q.limit, 10_000);
    }
}
