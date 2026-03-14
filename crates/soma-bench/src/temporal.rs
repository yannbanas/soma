//! Temporal Knowledge Benchmark — SOMA's unique contribution.
//!
//! Tests whether stigmergic decay correctly prioritizes recent facts
//! over stale ones. Static KGs (HippoRAG, etc.) score ~50% here.

use chrono::{DateTime, Duration, Utc};
use soma_core::{Channel, NodeKind, SomaQuery};
use soma_graph::StigreGraph;

/// A temporal fact with an associated timestamp.
#[derive(Debug, Clone)]
pub struct TemporalFact {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub channel: Channel,
    pub timestamp: DateTime<Utc>,
    pub source_tag: String,
}

/// A temporal query expecting a specific ranking order.
#[derive(Debug, Clone)]
pub struct TemporalQuery {
    pub question_entity: String,
    pub query_time: DateTime<Utc>,
    pub expected_recent: String,
    pub expected_stale: String,
}

/// Generate the temporal benchmark dataset.
///
/// 40 questions across 4 categories:
/// - CEO succession (10)
/// - Version updates (10)
/// - Location changes (10)
/// - Status changes (10)
pub fn generate_temporal_dataset() -> (Vec<TemporalFact>, Vec<TemporalQuery>) {
    let base = Utc::now() - Duration::hours(1440); // 60 days ago
    let mut facts = Vec::new();
    let mut queries = Vec::new();

    // ── Category 1: CEO succession ─────────────────────────────
    let companies = [
        "Acme Corp",
        "Nexus Inc",
        "Orion Labs",
        "Zenith AI",
        "Cortex Systems",
        "Meridian Bio",
        "Atlas Robotics",
        "Prism Networks",
        "Helix Genomics",
        "Quantum Dynamics",
    ];
    let ceo_old = [
        "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
    ];
    let ceo_new = [
        "Xavier", "Yolanda", "Zack", "Wendy", "Victor", "Uma", "Tom", "Sara", "Rick", "Quinn",
    ];

    for i in 0..10 {
        facts.push(TemporalFact {
            subject: companies[i].to_string(),
            relation: "CEO".to_string(),
            object: ceo_old[i].to_string(),
            channel: Channel::Trail,
            timestamp: base,
            source_tag: format!("temporal/old/{}", i),
        });
        facts.push(TemporalFact {
            subject: companies[i].to_string(),
            relation: "CEO".to_string(),
            object: ceo_new[i].to_string(),
            channel: Channel::Trail,
            timestamp: base + Duration::hours(720), // 30 days later
            source_tag: format!("temporal/new/{}", i),
        });
        queries.push(TemporalQuery {
            question_entity: companies[i].to_string(),
            query_time: base + Duration::hours(1440), // 60 days (now)
            expected_recent: ceo_new[i].to_string(),
            expected_stale: ceo_old[i].to_string(),
        });
    }

    // ── Category 2: Version updates ────────────────────────────
    let projects = [
        "SwarmOS",
        "MorphLang",
        "ZetaFold",
        "NanoKit",
        "BioSDK",
        "QuantumLib",
        "NeuralForge",
        "DataPipe",
        "CryptoMesh",
        "EdgeRuntime",
    ];
    let ver_old = [
        "v1.0", "v2.3", "v0.9", "v3.1", "v1.5", "v2.0", "v4.0", "v1.2", "v0.8", "v3.3",
    ];
    let ver_new = [
        "v2.0", "v3.0", "v1.0", "v4.0", "v2.0", "v3.0", "v5.0", "v2.0", "v1.0", "v4.0",
    ];

    for i in 0..10 {
        facts.push(TemporalFact {
            subject: projects[i].to_string(),
            relation: "latest_version".to_string(),
            object: ver_old[i].to_string(),
            channel: Channel::Trail,
            timestamp: base,
            source_tag: format!("temporal/old/ver{}", i),
        });
        facts.push(TemporalFact {
            subject: projects[i].to_string(),
            relation: "latest_version".to_string(),
            object: ver_new[i].to_string(),
            channel: Channel::Trail,
            timestamp: base + Duration::hours(720),
            source_tag: format!("temporal/new/ver{}", i),
        });
        queries.push(TemporalQuery {
            question_entity: projects[i].to_string(),
            query_time: base + Duration::hours(1440),
            expected_recent: ver_new[i].to_string(),
            expected_stale: ver_old[i].to_string(),
        });
    }

    // ── Category 3: Location changes ───────────────────────────
    let orgs = [
        "TechHQ",
        "BioLab",
        "DataCenter",
        "ResearchDiv",
        "AIUnit",
        "CloudOps",
        "DevTeam",
        "SecOps",
        "MLGroup",
        "SysAdmin",
    ];
    let loc_old = [
        "New York",
        "Boston",
        "London",
        "Berlin",
        "Tokyo",
        "Sydney",
        "Toronto",
        "Paris",
        "Seoul",
        "Singapore",
    ];
    let loc_new = [
        "San Francisco",
        "Austin",
        "Dublin",
        "Munich",
        "Osaka",
        "Melbourne",
        "Vancouver",
        "Lyon",
        "Busan",
        "Jakarta",
    ];

    for i in 0..10 {
        facts.push(TemporalFact {
            subject: orgs[i].to_string(),
            relation: "located_in".to_string(),
            object: loc_old[i].to_string(),
            channel: Channel::Trail,
            timestamp: base,
            source_tag: format!("temporal/old/loc{}", i),
        });
        facts.push(TemporalFact {
            subject: orgs[i].to_string(),
            relation: "located_in".to_string(),
            object: loc_new[i].to_string(),
            channel: Channel::Trail,
            timestamp: base + Duration::hours(720),
            source_tag: format!("temporal/new/loc{}", i),
        });
        queries.push(TemporalQuery {
            question_entity: orgs[i].to_string(),
            query_time: base + Duration::hours(1440),
            expected_recent: loc_new[i].to_string(),
            expected_stale: loc_old[i].to_string(),
        });
    }

    // ── Category 4: Status changes ─────────────────────────────
    let items = [
        "ProjectAlpha",
        "ProjectBeta",
        "ProjectGamma",
        "ProjectDelta",
        "ProjectEpsilon",
        "LibFoo",
        "LibBar",
        "LibBaz",
        "LibQux",
        "LibCorge",
    ];
    let status_old = [
        "active",
        "preview",
        "experimental",
        "stable",
        "active",
        "supported",
        "active",
        "preview",
        "experimental",
        "stable",
    ];
    let status_new = [
        "deprecated",
        "released",
        "released",
        "legacy",
        "archived",
        "abandoned",
        "deprecated",
        "released",
        "released",
        "legacy",
    ];

    for i in 0..10 {
        facts.push(TemporalFact {
            subject: items[i].to_string(),
            relation: "status".to_string(),
            object: status_old[i].to_string(),
            channel: Channel::Trail,
            timestamp: base,
            source_tag: format!("temporal/old/status{}", i),
        });
        facts.push(TemporalFact {
            subject: items[i].to_string(),
            relation: "status".to_string(),
            object: status_new[i].to_string(),
            channel: Channel::Trail,
            timestamp: base + Duration::hours(720),
            source_tag: format!("temporal/new/status{}", i),
        });
        queries.push(TemporalQuery {
            question_entity: items[i].to_string(),
            query_time: base + Duration::hours(1440),
            expected_recent: status_new[i].to_string(),
            expected_stale: status_old[i].to_string(),
        });
    }

    (facts, queries)
}

/// Ingest temporal facts into a graph, respecting their timestamps.
pub fn ingest_temporal_facts(graph: &mut StigreGraph, facts: &[TemporalFact]) {
    for fact in facts {
        let subj_id = graph.upsert_node(&fact.subject, NodeKind::Entity);
        let obj_id = graph.upsert_node(&fact.object, NodeKind::Entity);
        graph.upsert_edge(subj_id, obj_id, fact.channel, 0.9, &fact.source_tag);
    }

    // Set timestamps on edges based on their source tags
    for fact in facts {
        graph.set_edge_timestamps_for_source(&fact.source_tag, fact.timestamp);
    }
}

/// Compute temporal ranking for a single query.
///
/// Returns: (recent_rank, stale_rank) — lower is better.
/// `None` means the entity was not found in results.
pub fn temporal_ranking(
    graph: &StigreGraph,
    query: &TemporalQuery,
    k: usize,
) -> (Option<usize>, Option<usize>) {
    let q = SomaQuery::new(&query.question_entity)
        .with_max_hops(3)
        .with_limit(k);
    let results = graph.traverse(&q);

    let recent_lower = query.expected_recent.to_lowercase();
    let stale_lower = query.expected_stale.to_lowercase();

    // Prefer exact match; fall back to substring only if exact not found
    let recent_rank = results
        .iter()
        .position(|r| r.node.label.to_lowercase() == recent_lower)
        .or_else(|| {
            results
                .iter()
                .position(|r| r.node.label.to_lowercase().contains(&recent_lower))
        });
    let stale_rank = results
        .iter()
        .position(|r| r.node.label.to_lowercase() == stale_lower)
        .or_else(|| {
            results
                .iter()
                .position(|r| r.node.label.to_lowercase().contains(&stale_lower))
        });

    (recent_rank, stale_rank)
}

/// Temporal Accuracy: fraction of queries where the recent answer ranks
/// strictly higher (lower index) than the stale answer.
pub fn temporal_accuracy(results: &[(Option<usize>, Option<usize>)]) -> f32 {
    let valid = results
        .iter()
        .filter(|(r, s)| r.is_some() || s.is_some())
        .count();
    if valid == 0 {
        return 0.0;
    }

    let correct = results
        .iter()
        .filter(|(recent, stale)| match (recent, stale) {
            (Some(r), Some(s)) => r < s,
            (Some(_), None) => true, // recent found, stale decayed away
            _ => false,
        })
        .count();

    correct as f32 / valid as f32
}

/// Run the full temporal benchmark.
///
/// Returns: (temporal_accuracy, results_per_query)
#[allow(clippy::type_complexity)]
pub fn run_temporal_benchmark(k: usize) -> (f32, Vec<(Option<usize>, Option<usize>)>) {
    let (facts, queries) = generate_temporal_dataset();
    let mut graph = StigreGraph::new("temporal-bench", 0.05);

    ingest_temporal_facts(&mut graph, &facts);

    let results: Vec<(Option<usize>, Option<usize>)> = queries
        .iter()
        .map(|q| temporal_ranking(&graph, q, k))
        .collect();

    let accuracy = temporal_accuracy(&results);
    (accuracy, results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_dataset_correct_size() {
        let (facts, queries) = generate_temporal_dataset();
        assert_eq!(facts.len(), 80); // 4 categories × 10 × 2 (old + new)
        assert_eq!(queries.len(), 40); // 4 categories × 10
    }

    #[test]
    fn temporal_accuracy_all_correct() {
        let results = vec![
            (Some(0), Some(2)), // recent rank 0, stale rank 2 → correct
            (Some(1), Some(3)), // correct
            (Some(0), None),    // recent found, stale gone → correct
        ];
        assert_eq!(temporal_accuracy(&results), 1.0);
    }

    #[test]
    fn temporal_accuracy_mixed() {
        let results = vec![
            (Some(0), Some(2)), // correct
            (Some(3), Some(1)), // wrong: stale ranks higher
            (None, None),       // invalid, skipped
        ];
        assert!((temporal_accuracy(&results) - 0.5).abs() < 0.01);
    }

    #[test]
    fn temporal_accuracy_empty() {
        assert_eq!(temporal_accuracy(&[]), 0.0);
    }
}
