use serde::{Deserialize, Serialize};

/// Channel determines the biological behavior of an edge:
/// evaporation rate, reinforcement amplitude, consolidation treatment.
///
/// Each channel has a specific decay constant (tau) modeling natural forgetting.
/// tau = 0.0 means permanent; tau = 0.05 means ~20h half-life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Channel {
    /// Validated knowledge, established trust path
    Trail,
    /// Direct causal relation: A causes B
    Causal,
    /// Filiation: B derives from A (version, mutation, inspiration)
    DerivesDe,
    /// Time-bound event — naturally fades
    Episodic,
    /// Confirmed error, incompatibility, dead-end — inhibits paths
    Alarm,
    /// Semantic similarity computed by soma-hdc (auto-generated)
    SemanticSim,
    /// Reasoning step — fast decay (~12h), for Graph of Thoughts
    Reasoning,
    /// Custom channel declared in soma.toml
    Custom(u16),
}

impl Channel {
    /// Decay constant (intensity lost per hour, exponential base).
    /// tau = 0.0 → permanent | tau = 0.05 → fades in ~20h
    #[inline]
    pub fn tau_decay(self) -> f32 {
        match self {
            Channel::DerivesDe => 0.0,    // permanent — filiation never fades
            Channel::Causal => 0.0001,    // quasi-permanent
            Channel::Trail => 0.001,      // very slow (~40 days)
            Channel::Alarm => 0.005,      // slow (~8 days — errors persist)
            Channel::SemanticSim => 0.02, // moderate (~2 days)
            Channel::Episodic => 0.05,    // fast (~20h)
            Channel::Reasoning => 0.08,   // very fast (~12h)
            Channel::Custom(_) => 0.01,
        }
    }

    /// Delta added to intensity on traversal (positive feedback).
    #[inline]
    pub fn reinforce_delta(self) -> f32 {
        match self {
            Channel::Causal => 0.50,
            Channel::DerivesDe => 0.40,
            Channel::Alarm => 0.35, // errors strengthen if repeated
            Channel::Trail => 0.30,
            Channel::SemanticSim => 0.20,
            Channel::Reasoning => 0.15,
            Channel::Episodic => 0.10,
            Channel::Custom(_) => 0.20,
        }
    }

    /// Whether this edge type can be pruned by biological scheduler.
    #[inline]
    pub fn is_prunable(self) -> bool {
        !matches!(self, Channel::DerivesDe)
    }

    /// Parse channel name from string (case-insensitive).
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "trail" => Some(Channel::Trail),
            "causal" => Some(Channel::Causal),
            "derives_de" | "derivesde" => Some(Channel::DerivesDe),
            "episodic" => Some(Channel::Episodic),
            "alarm" => Some(Channel::Alarm),
            "semantic_sim" | "semanticsim" => Some(Channel::SemanticSim),
            "reasoning" => Some(Channel::Reasoning),
            _ => {
                // Try parsing "custom:NNN"
                if let Some(n) = s.strip_prefix("custom:") {
                    n.parse::<u16>().ok().map(Channel::Custom)
                } else {
                    None
                }
            }
        }
    }

    /// String representation for display/serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            Channel::Trail => "trail",
            Channel::Causal => "causal",
            Channel::DerivesDe => "derives_de",
            Channel::Episodic => "episodic",
            Channel::Alarm => "alarm",
            Channel::SemanticSim => "semantic_sim",
            Channel::Reasoning => "reasoning",
            Channel::Custom(_) => "custom",
        }
    }
}

impl std::fmt::Display for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Channel::Custom(n) => write!(f, "custom:{}", n),
            other => write!(f, "{}", other.as_str()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_de_is_permanent() {
        assert_eq!(Channel::DerivesDe.tau_decay(), 0.0);
    }

    #[test]
    fn derives_de_not_prunable() {
        assert!(!Channel::DerivesDe.is_prunable());
    }

    #[test]
    fn all_others_prunable() {
        assert!(Channel::Trail.is_prunable());
        assert!(Channel::Episodic.is_prunable());
        assert!(Channel::Alarm.is_prunable());
    }

    #[test]
    fn parse_channel_names() {
        assert_eq!(Channel::from_str_name("trail"), Some(Channel::Trail));
        assert_eq!(Channel::from_str_name("CAUSAL"), Some(Channel::Causal));
        assert_eq!(
            Channel::from_str_name("derives_de"),
            Some(Channel::DerivesDe)
        );
        assert_eq!(
            Channel::from_str_name("custom:42"),
            Some(Channel::Custom(42))
        );
        assert_eq!(Channel::from_str_name("unknown"), None);
    }

    #[test]
    fn decay_ordering() {
        // Episodic should decay fastest, DerivesDe should be permanent
        assert!(Channel::Episodic.tau_decay() > Channel::Trail.tau_decay());
        assert!(Channel::Trail.tau_decay() > Channel::DerivesDe.tau_decay());
    }
}
