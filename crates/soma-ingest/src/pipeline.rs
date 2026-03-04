use soma_core::{Channel, NodeKind, SomaError};
use soma_graph::StigreGraph;
use soma_llm::OllamaClient;

use crate::chunker::Chunker;
use crate::ner;
use crate::patterns::{ExtractedTriplet, PatternExtractor};
use crate::source::IngestSource;

/// A raw triplet extracted from text.
#[derive(Debug, Clone)]
pub struct Triplet {
    pub subject: String,
    pub object: String,
    pub channel: Channel,
}

/// Result of an ingestion operation.
#[derive(Debug, Clone)]
pub struct IngestResult {
    pub chunks_processed: usize,
    pub triplets_extracted: usize,
    pub nodes_created: usize,
    pub edges_created: usize,
    /// All nodes created during ingestion (for WAL persistence).
    pub created_nodes: Vec<soma_core::SomaNode>,
    /// All edges created during ingestion (for WAL persistence).
    pub created_edges: Vec<soma_core::StigreEdge>,
}

/// The main ingestion pipeline.
///
/// Flow: Source → Parse → Chunk → Extract (L0/L1/L2) → Deduplicate → Insert Graph
pub struct IngestPipeline {
    chunker: Chunker,
    llm_client: Option<OllamaClient>,
}

impl IngestPipeline {
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        IngestPipeline {
            chunker: Chunker::new(chunk_size, overlap),
            llm_client: None,
        }
    }

    /// Attach an Ollama client for L2 extraction.
    pub fn with_llm(mut self, client: OllamaClient) -> Self {
        self.llm_client = Some(client);
        self
    }

    pub fn default_config() -> Self {
        Self::new(5, 1)
    }

    /// Ingest a source into the graph.
    ///
    /// 1. Extract raw text from source
    /// 2. Chunk into overlapping segments
    /// 3. Extract triplets via L0 (structured) + L1 (patterns)
    /// 4. Insert into graph (idempotent upserts)
    pub fn ingest(
        &self,
        source: &IngestSource,
        graph: &mut StigreGraph,
        source_tag: &str,
    ) -> Result<IngestResult, SomaError> {
        let text = source.to_text()?;
        self.ingest_text(&text, graph, source_tag)
    }

    /// Ingest raw text directly.
    pub fn ingest_text(
        &self,
        text: &str,
        graph: &mut StigreGraph,
        source_tag: &str,
    ) -> Result<IngestResult, SomaError> {
        let chunks = self.chunker.chunk(text);

        let mut total_triplets = 0;
        let mut total_nodes = 0;
        let mut total_edges = 0;
        let mut created_nodes: Vec<soma_core::SomaNode> = Vec::new();
        let mut created_edges: Vec<soma_core::StigreEdge> = Vec::new();

        let workspace = graph.workspace().to_string();

        for chunk in &chunks {
            // L0: attempt structured extraction (for JSON/structured sources)
            // L1: pattern-based extraction
            let triplets = PatternExtractor::extract(chunk);
            total_triplets += triplets.len();

            for triplet in &triplets {
                // Determine node kinds from channel context
                let (subj_kind, obj_kind) = infer_node_kinds(&triplet);

                // Upsert nodes
                let subj_id = graph.upsert_node(&triplet.subject, subj_kind);
                let obj_id = graph.upsert_node(&triplet.object, obj_kind);
                total_nodes += 2;
                created_nodes.push(soma_core::SomaNode::new(&workspace, &triplet.subject, subj_kind));
                created_nodes.push(soma_core::SomaNode::new(&workspace, &triplet.object, obj_kind));

                // Upsert edge
                if let Some(_edge_id) = graph.upsert_edge(
                    subj_id,
                    obj_id,
                    triplet.channel,
                    0.7, // default confidence for pattern extraction
                    source_tag,
                ) {
                    total_edges += 1;
                    created_edges.push(soma_core::StigreEdge::new(
                        subj_id, obj_id, triplet.channel, 0.7, source_tag.to_string(),
                    ));
                }
            }

            // L1.5: Automatic NER — extract named entities and create co-occurrence edges
            let entities = ner::extract_entities(chunk);
            if entities.len() >= 2 {
                // Create Entity nodes for each named entity
                let entity_ids: Vec<_> = entities
                    .iter()
                    .map(|e| {
                        let id = graph.upsert_node(&e.name, NodeKind::Entity);
                        created_nodes.push(soma_core::SomaNode::new(
                            &workspace,
                            &e.name,
                            NodeKind::Entity,
                        ));
                        total_nodes += 1;
                        id
                    })
                    .collect();

                // Connect co-occurring entities with Trail edges
                for (i, j) in (0..entity_ids.len())
                    .flat_map(|i| ((i + 1)..entity_ids.len()).map(move |j| (i, j)))
                {
                    if let Some(_) = graph.upsert_edge(
                        entity_ids[i],
                        entity_ids[j],
                        Channel::Trail,
                        0.5, // co-occurrence confidence
                        source_tag,
                    ) {
                        total_edges += 1;
                        created_edges.push(soma_core::StigreEdge::new(
                            entity_ids[i],
                            entity_ids[j],
                            Channel::Trail,
                            0.5,
                            source_tag.to_string(),
                        ));
                    }
                }
            } else if entities.len() == 1 {
                // Single entity — still create the node
                graph.upsert_node(&entities[0].name, NodeKind::Entity);
                created_nodes.push(soma_core::SomaNode::new(
                    &workspace,
                    &entities[0].name,
                    NodeKind::Entity,
                ));
                total_nodes += 1;
            }

            // L2: if L0+L1 yielded < 3 triplets on a long chunk, try LLM extraction
            if triplets.len() < 3 && chunk.len() > 50 {
                if let Some(ref llm) = self.llm_client {
                    match llm.extract_triplets(chunk) {
                        Ok(llm_triplets) if !llm_triplets.is_empty() => {
                            tracing::debug!(
                                "[ingest:L2] LLM extracted {} triplets from chunk",
                                llm_triplets.len()
                            );
                            for lt in &llm_triplets {
                                let channel = channel_from_relation(&lt.relation);
                                let (subj_kind, obj_kind) =
                                    infer_node_kinds_from_channel(channel);

                                let subj_id = graph.upsert_node(&lt.subject, subj_kind);
                                let obj_id = graph.upsert_node(&lt.object, obj_kind);
                                total_nodes += 2;
                                created_nodes.push(soma_core::SomaNode::new(&workspace, &lt.subject, subj_kind));
                                created_nodes.push(soma_core::SomaNode::new(&workspace, &lt.object, obj_kind));

                                let confidence = lt.confidence.clamp(0.0, 1.0);
                                if let Some(_edge_id) = graph.upsert_edge(
                                    subj_id,
                                    obj_id,
                                    channel,
                                    confidence,
                                    source_tag,
                                ) {
                                    total_edges += 1;
                                    created_edges.push(soma_core::StigreEdge::new(
                                        subj_id, obj_id, channel, confidence, source_tag.to_string(),
                                    ));
                                }
                                total_triplets += 1;
                            }
                        }
                        Ok(_) => {
                            // LLM returned empty — fall back to Event node
                            let node = create_event_node(graph, chunk, source_tag);
                            created_nodes.push(node);
                            total_nodes += 1;
                        }
                        Err(e) => {
                            tracing::warn!("[ingest:L2] LLM extraction failed: {}", e);
                            let node = create_event_node(graph, chunk, source_tag);
                            created_nodes.push(node);
                            total_nodes += 1;
                        }
                    }
                } else {
                    // No LLM client — existing behavior
                    let node = create_event_node(graph, chunk, source_tag);
                    created_nodes.push(node);
                    total_nodes += 1;
                }
            }
        }

        Ok(IngestResult {
            chunks_processed: chunks.len(),
            triplets_extracted: total_triplets,
            nodes_created: total_nodes,
            edges_created: total_edges,
            created_nodes,
            created_edges,
        })
    }
}

/// Create an Event node for a chunk that couldn't be extracted into triplets.
fn create_event_node(graph: &mut StigreGraph, chunk: &str, source_tag: &str) -> soma_core::SomaNode {
    let label = if chunk.len() > 80 {
        // Find a char boundary at or before byte 77
        let end = chunk
            .char_indices()
            .take_while(|(i, _)| *i <= 77)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("{}...", &chunk[..end])
    } else {
        chunk.to_string()
    };
    graph.upsert_node_with_tags(&label, NodeKind::Event, vec![source_tag.to_string()]);
    soma_core::SomaNode::new(graph.workspace(), &label, NodeKind::Event)
        .with_tags(vec![source_tag.to_string()])
}

/// Map LLM relation strings to Channel enum.
fn channel_from_relation(relation: &str) -> Channel {
    let r = relation.to_lowercase();
    if r.contains("derives") || r.contains("based on") || r.contains("version") {
        Channel::DerivesDe
    } else if r.contains("cause") || r.contains("trigger") || r.contains("produce") {
        Channel::Causal
    } else if r.contains("avoid") || r.contains("incompatible") || r.contains("error") {
        Channel::Alarm
    } else {
        Channel::Trail // default for "is a", "uses", "has", etc.
    }
}

/// Infer node kinds from a Channel (for L2 triplets).
fn infer_node_kinds_from_channel(channel: Channel) -> (NodeKind, NodeKind) {
    match channel {
        Channel::Alarm => (NodeKind::Entity, NodeKind::Warning),
        Channel::Episodic => (NodeKind::Event, NodeKind::Event),
        Channel::DerivesDe => (NodeKind::Entity, NodeKind::Entity),
        _ => (NodeKind::Entity, NodeKind::Concept),
    }
}

/// Infer node kinds from the triplet's channel and content.
fn infer_node_kinds(triplet: &ExtractedTriplet) -> (NodeKind, NodeKind) {
    match triplet.channel {
        Channel::Alarm => (NodeKind::Entity, NodeKind::Warning),
        Channel::Episodic => (NodeKind::Event, NodeKind::Event),
        Channel::DerivesDe => (NodeKind::Entity, NodeKind::Entity),
        Channel::Causal => {
            // Check if object looks like a measurement
            if triplet.object.contains("nm")
                || triplet.object.contains("pLDDT")
                || triplet.object.chars().any(|c| c.is_ascii_digit())
            {
                (NodeKind::Entity, NodeKind::Measurement)
            } else {
                (NodeKind::Entity, NodeKind::Entity)
            }
        }
        _ => (NodeKind::Entity, NodeKind::Concept),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_raw_text() {
        let pipeline = IngestPipeline::default_config();
        let mut graph = StigreGraph::new("test", 0.05);

        let result = pipeline
            .ingest_text(
                "ChromoQ is a fluorescent protein. ChromoQ derives from EGFP. EGFP is a green protein.",
                &mut graph,
                "test",
            )
            .unwrap();

        assert!(result.triplets_extracted > 0);
        assert!(graph.node_count() > 0);
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn ingest_with_alarm() {
        let pipeline = IngestPipeline::default_config();
        let mut graph = StigreGraph::new("test", 0.05);

        let result = pipeline
            .ingest_text(
                "AVOID using deprecated OpenGL calls. ChromoQ is a stable variant.",
                &mut graph,
                "test",
            )
            .unwrap();

        assert!(result.triplets_extracted >= 1);
    }

    #[test]
    fn ingest_empty_text() {
        let pipeline = IngestPipeline::default_config();
        let mut graph = StigreGraph::new("test", 0.05);

        let result = pipeline.ingest_text("", &mut graph, "test").unwrap();
        assert_eq!(result.chunks_processed, 0);
        assert_eq!(result.triplets_extracted, 0);
    }

    #[test]
    fn ingest_source_raw() {
        let pipeline = IngestPipeline::default_config();
        let mut graph = StigreGraph::new("test", 0.05);
        let source = IngestSource::RawText("Rust is a systems programming language.".to_string());

        let result = pipeline.ingest(&source, &mut graph, "test").unwrap();
        assert_eq!(result.chunks_processed, 1, "Single sentence should be 1 chunk");
        assert!(result.triplets_extracted >= 1, "Should extract 'is a' triplet");
        assert!(graph.node_count() >= 1, "Graph should have at least 1 node");
    }

    #[test]
    fn ingest_without_llm_creates_event_nodes() {
        // Verify that without LLM, long chunks with <3 triplets still create Event nodes
        let pipeline = IngestPipeline::default_config();
        let mut graph = StigreGraph::new("test", 0.05);
        let text = "This is a long paragraph of text that does not contain any obvious knowledge relations but is still meaningful content that should be preserved in the graph.";
        let result = pipeline.ingest_text(text, &mut graph, "test").unwrap();
        assert!(
            graph.node_count() >= 1,
            "Should create at least one Event node for unextractable text"
        );
        assert_eq!(result.chunks_processed, 1);
    }

    #[test]
    fn channel_from_relation_mapping() {
        assert!(matches!(channel_from_relation("derives from"), Channel::DerivesDe));
        assert!(matches!(channel_from_relation("is based on"), Channel::DerivesDe));
        assert!(matches!(channel_from_relation("causes"), Channel::Causal));
        assert!(matches!(channel_from_relation("triggers"), Channel::Causal));
        assert!(matches!(channel_from_relation("AVOID"), Channel::Alarm));
        assert!(matches!(channel_from_relation("is incompatible with"), Channel::Alarm));
        assert!(matches!(channel_from_relation("is a"), Channel::Trail));
        assert!(matches!(channel_from_relation("uses"), Channel::Trail));
    }
}
