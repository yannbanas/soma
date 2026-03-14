use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::channel::Channel;
use crate::ids::{EdgeId, NodeId};

/// Provenance tracks how an edge was created/validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provenance {
    /// Created by human input
    Human,
    /// Inferred by AI (not yet validated)
    AiInferred,
    /// Validated by AI against external source
    AiValidated,
    /// Automated extraction (ingestion pipeline, code analysis)
    Automated,
}

impl Default for Provenance {
    fn default() -> Self {
        Provenance::Automated
    }
}

impl std::fmt::Display for Provenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provenance::Human => write!(f, "human"),
            Provenance::AiInferred => write!(f, "ai_inferred"),
            Provenance::AiValidated => write!(f, "ai_validated"),
            Provenance::Automated => write!(f, "automated"),
        }
    }
}

impl Provenance {
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "human" => Some(Provenance::Human),
            "ai_inferred" | "aiinferred" => Some(Provenance::AiInferred),
            "ai_validated" | "aivalidated" => Some(Provenance::AiValidated),
            "automated" => Some(Provenance::Automated),
            _ => None,
        }
    }
}

/// A living edge in the SOMA knowledge graph.
///
/// Edges have biological properties: intensity decays over time (evaporation),
/// gets reinforced on traversal, and can die when below threshold.
/// Evaporation is lazy — computed on demand, not by timer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StigreEdge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub channel: Channel,

    /// Intensity at last touch — [0.0, 1.0]
    /// Effective intensity (after evaporation) is computed on demand.
    intensity: f32,

    /// Timestamp of last reinforcement or creation
    pub last_touch: DateTime<Utc>,

    /// Number of traversals or confirmations
    pub uses: u32,

    /// Confidence in the source — [0.0, 1.0]
    pub confidence: f32,

    /// Origin: "claude/conv-abc123", "ollama/session-42", "file/paper.pdf"
    pub source: String,

    /// Optional human-readable label
    pub label: Option<String>,

    /// How this edge was created/validated
    #[serde(default)]
    pub provenance: Provenance,
}

impl StigreEdge {
    /// Create a new edge with validated intensity and confidence bounds.
    pub fn new(
        from: NodeId,
        to: NodeId,
        channel: Channel,
        confidence: f32,
        source: String,
    ) -> Self {
        StigreEdge {
            id: EdgeId::random(),
            from,
            to,
            channel,
            // Security: clamp values to valid range
            intensity: confidence.clamp(0.0, 1.0),
            last_touch: Utc::now(),
            uses: 1,
            confidence: confidence.clamp(0.0, 1.0),
            source,
            label: None,
            provenance: Provenance::default(),
        }
    }

    /// Get raw intensity (at last touch, before evaporation).
    #[inline]
    pub fn raw_intensity(&self) -> f32 {
        self.intensity
    }

    /// Set intensity directly (used by store recovery).
    /// Clamped to [0.0, 1.0] for safety.
    #[inline]
    pub fn set_intensity(&mut self, val: f32) {
        self.intensity = val.clamp(0.0, 1.0);
    }

    /// Effective intensity after evaporation — lazy computation, no active timer.
    ///
    /// Formula: intensity × exp(-τ × Δt_hours)
    /// For permanent channels (τ=0), returns raw intensity.
    #[inline]
    pub fn effective_intensity(&self, now: DateTime<Utc>) -> f32 {
        let tau = self.channel.tau_decay();
        if tau == 0.0 {
            return self.intensity;
        }
        let dt_hours = (now - self.last_touch)
            .num_seconds()
            .max(0) as f32
            / 3600.0;
        self.intensity * (-tau * dt_hours).exp()
    }

    /// Reinforce the edge — reset clock, combine current effective intensity + delta.
    /// Capped at 1.0 for safety.
    pub fn reinforce(&mut self, now: DateTime<Utc>) {
        let current = self.effective_intensity(now);
        let delta = self.channel.reinforce_delta();
        self.intensity = (current + delta).min(1.0);
        self.last_touch = now;
        self.uses = self.uses.saturating_add(1);
    }

    /// Check if this edge is dead (below threshold and prunable).
    #[inline]
    pub fn is_dead(&self, threshold: f32, now: DateTime<Utc>) -> bool {
        self.channel.is_prunable() && self.effective_intensity(now) < threshold
    }

    /// Add a human-readable label.
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edge(channel: Channel, intensity: f32) -> StigreEdge {
        let mut edge = StigreEdge::new(
            NodeId::from_label("a"),
            NodeId::from_label("b"),
            channel,
            intensity,
            "test".to_string(),
        );
        edge.intensity = intensity;
        edge
    }

    #[test]
    fn permanent_channel_no_decay() {
        let edge = make_edge(Channel::DerivesDe, 0.8);
        let future = Utc::now() + chrono::Duration::days(365);
        assert_eq!(edge.effective_intensity(future), 0.8);
    }

    #[test]
    fn episodic_decays_fast() {
        let mut edge = make_edge(Channel::Episodic, 1.0);
        edge.last_touch = Utc::now() - chrono::Duration::hours(24);
        let now = Utc::now();
        let eff = edge.effective_intensity(now);
        // After 24h with tau=0.05: exp(-0.05 * 24) ≈ 0.30
        assert!(eff < 0.35, "effective={}", eff);
        assert!(eff > 0.25, "effective={}", eff);
    }

    #[test]
    fn trail_decays_slowly() {
        let mut edge = make_edge(Channel::Trail, 1.0);
        edge.last_touch = Utc::now() - chrono::Duration::hours(24);
        let now = Utc::now();
        let eff = edge.effective_intensity(now);
        // After 24h with tau=0.001: exp(-0.001 * 24) ≈ 0.976
        assert!(eff > 0.95, "effective={}", eff);
    }

    #[test]
    fn reinforce_increases_intensity() {
        let mut edge = make_edge(Channel::Trail, 0.5);
        let now = Utc::now();
        let before = edge.effective_intensity(now);
        edge.reinforce(now);
        assert!(edge.raw_intensity() > before);
        assert_eq!(edge.uses, 2);
    }

    #[test]
    fn reinforce_caps_at_one() {
        let mut edge = make_edge(Channel::Causal, 0.9);
        let now = Utc::now();
        edge.reinforce(now); // 0.9 + 0.5 should cap at 1.0
        assert_eq!(edge.raw_intensity(), 1.0);
    }

    #[test]
    fn dead_edge_detection() {
        let mut edge = make_edge(Channel::Episodic, 0.01);
        edge.last_touch = Utc::now() - chrono::Duration::hours(100);
        assert!(edge.is_dead(0.05, Utc::now()));
    }

    #[test]
    fn derives_de_never_dead() {
        let mut edge = make_edge(Channel::DerivesDe, 0.01);
        edge.last_touch = Utc::now() - chrono::Duration::hours(10000);
        assert!(!edge.is_dead(0.05, Utc::now()));
    }

    #[test]
    fn intensity_clamped_on_creation() {
        let edge = StigreEdge::new(
            NodeId::from_label("a"),
            NodeId::from_label("b"),
            Channel::Trail,
            5.0, // exceeds 1.0
            "test".into(),
        );
        assert_eq!(edge.raw_intensity(), 1.0);
        assert_eq!(edge.confidence, 1.0);
    }

    #[test]
    fn uses_saturating_add() {
        let mut edge = make_edge(Channel::Trail, 0.5);
        edge.uses = u32::MAX;
        edge.reinforce(Utc::now());
        assert_eq!(edge.uses, u32::MAX); // no overflow
    }
}
