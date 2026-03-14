use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Opaque, type-safe node identifier.
/// Deterministic from label via UUID5 — two identical labels yield the same NodeId.
/// This enables automatic deduplication without prior lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(Uuid);

/// Opaque, type-safe edge identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(Uuid);

impl NodeId {
    /// Create a random NodeId (for cases where determinism isn't needed).
    pub fn random() -> Self {
        NodeId(Uuid::new_v4())
    }

    /// Deterministic NodeId from label — core deduplication mechanism.
    /// Two identical labels always produce the same NodeId.
    /// Uses UUID5 with OID namespace for collision resistance.
    #[inline]
    pub fn from_label(label: &str) -> Self {
        NodeId(Uuid::new_v5(&Uuid::NAMESPACE_OID, label.as_bytes()))
    }

    /// Access the inner UUID (for serialization/storage).
    #[inline]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl EdgeId {
    /// Create a random EdgeId.
    pub fn random() -> Self {
        EdgeId(Uuid::new_v4())
    }

    /// Access the inner UUID.
    #[inline]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n:{}", &self.0.to_string()[..8])
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e:{}", &self.0.to_string()[..8])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_node_id() {
        let a = NodeId::from_label("default:ChromoQ");
        let b = NodeId::from_label("default:ChromoQ");
        assert_eq!(a, b);
    }

    #[test]
    fn different_labels_different_ids() {
        let a = NodeId::from_label("default:ChromoQ");
        let b = NodeId::from_label("default:EGFP");
        assert_ne!(a, b);
    }

    #[test]
    fn random_ids_are_unique() {
        let a = NodeId::random();
        let b = NodeId::random();
        assert_ne!(a, b);
    }

    #[test]
    fn display_format() {
        let id = NodeId::from_label("test");
        let s = format!("{}", id);
        assert!(s.starts_with("n:"));
        assert_eq!(s.len(), 10); // "n:" + 8 hex chars
    }
}
