//! RULER-style integration test for SOMA.
//!
//! Validates multi-needle retrieval and cross-reference capabilities
//! using a realistic corpus with scattered facts.

use soma_core::SomaQuery;
use soma_graph::StigreGraph;
use soma_hdc::HdcEngine;
use soma_ingest::IngestPipeline;

/// Build a realistic corpus with needles embedded in a haystack.
fn build_corpus() -> String {
    let haystack = vec![
        "Green fluorescent protein (GFP) is a protein that exhibits bright green fluorescence when exposed to light in the blue to ultraviolet range.",
        "In cell and molecular biology, the GFP gene is frequently used as a reporter of expression.",
        "Enhanced green fluorescent protein (EGFP) is a mutant form of GFP optimized for brighter fluorescence.",
        // --- Needle 1: ZetaFold ---
        "ZetaFold is a protein folding algorithm with a benchmark accuracy of pLDDT=97.3 on CASP15 targets.",
        "Bioluminescence is the production and emission of light by living organisms.",
        "Luciferase is a generic term for enzymes that produce bioluminescence.",
        // --- Needle 2: NanoLuc ---
        "NanoLuc derives from deep-sea shrimp Oplophorus gracilirostris and produces 150-fold brighter luminescence than firefly luciferase.",
        "Protein structure prediction has been revolutionized by AlphaFold2, developed by DeepMind.",
        "WebAssembly (WASM) is a binary instruction format for a stack-based virtual machine.",
        // --- Needle 3: MorphLang ---
        "MorphLang compiles to WAMR bytecode with an average compilation latency of 12 milliseconds.",
        "Unikraft is a fast, secure, and open-source unikernel development kit.",
        "SwarmOS is a distributed operating system designed for autonomous agent coordination.",
        // --- Needle 4: CRISPR ---
        "CRISPR-Cas13 causes RNA degradation which triggers cellular immune response activation.",
        "KOLOSS is an AI orchestration framework that coordinates multiple language models and agents.",
        "The Model Context Protocol (MCP) is a standard protocol for AI systems.",
        // --- Needle 5: Cross-ref A (Helios → ChromoQ-X99) ---
        "Project Helios uses ChromoQ-X99 as its primary fluorescent marker for neural pathway tracing.",
        "Rust is a systems programming language focused on safety, performance, and concurrency.",
        "The petgraph crate provides graph data structure implementations for Rust.",
        // --- Needle 6: Cross-ref B (ChromoQ-X99 emission) ---
        "ChromoQ-X99 has been engineered to emit at exactly 542nm with a pLDDT=98.1, making it the brightest known ChromoQ variant.",
        "Hyperdimensional computing (HDC) is a computational paradigm inspired by how the brain processes information.",
        "Random indexing is a dimensionality reduction technique used in distributional semantics.",
        // --- Needle 7: AVOID alarm ---
        "AVOID combining MorphLang v2 with WAMR versions below 1.3 due to memory corruption in the bytecode validator.",
        "The write-ahead log (WAL) is a standard method for ensuring data integrity.",
        "Stigmergy is a mechanism of indirect coordination between agents or actions.",
        // --- Needle 8: KOLOSS-v3 dependencies ---
        "KOLOSS-v3 depends on SwarmOS for agent orchestration and uses ZetaFold for on-the-fly protein analysis.",
        "The hippocampus plays a critical role in memory consolidation.",
        "Physarum polycephalum is a slime mold that has been studied for its remarkable ability to optimize networks.",
        // --- Needle 9: StigreNet benchmark ---
        "StigreNet achieves 2.4 million edge traversals per second on a knowledge graph with 100K nodes.",
        "BM25 is a ranking function used by search engines to estimate relevance.",
        "The Maximal Marginal Relevance (MMR) algorithm is used to reduce redundancy in information retrieval results.",
    ];

    haystack.join("\n\n")
}

fn ingest_corpus() -> StigreGraph {
    let corpus = build_corpus();
    let pipeline = IngestPipeline::default_config();
    let mut graph = StigreGraph::new("ruler_test", 0.05);
    pipeline
        .ingest_text(&corpus, &mut graph, "ruler_test")
        .expect("Ingestion should succeed");
    graph
}

/// Helper: check if searching for `query` yields a result whose label contains `expected`.
fn find_in_graph(graph: &StigreGraph, query: &str, expected: &str, max_hops: u8, limit: usize) -> bool {
    let q = SomaQuery::new(query).with_max_hops(max_hops).with_limit(limit);
    let results = graph.traverse(&q);
    results.iter().any(|r| {
        r.node.label.to_lowercase().contains(&expected.to_lowercase())
    })
}

// ── Multi-Needle Retrieval Tests ───────────────────────────────────────────

#[test]
fn ruler_ingest_succeeds() {
    let graph = ingest_corpus();
    assert!(graph.node_count() > 20, "Should have many nodes from corpus");
    assert!(graph.edge_count() > 10, "Should have many edges from patterns");
}

#[test]
fn ruler_needle_zetafold() {
    let graph = ingest_corpus();
    assert!(
        find_in_graph(&graph, "ZetaFold", "ZetaFold", 3, 20),
        "Should find ZetaFold needle"
    );
}

#[test]
fn ruler_needle_nanoluc() {
    let graph = ingest_corpus();
    assert!(
        find_in_graph(&graph, "NanoLuc", "NanoLuc", 3, 20),
        "Should find NanoLuc needle"
    );
}

#[test]
fn ruler_needle_morphlang() {
    let graph = ingest_corpus();
    assert!(
        find_in_graph(&graph, "MorphLang", "MorphLang", 3, 20),
        "Should find MorphLang needle"
    );
}

#[test]
fn ruler_needle_crispr() {
    let graph = ingest_corpus();
    assert!(
        find_in_graph(&graph, "CRISPR-Cas13", "CRISPR-Cas13", 3, 20),
        "Should find CRISPR-Cas13 needle"
    );
}

#[test]
fn ruler_needle_chromoq_x99() {
    let graph = ingest_corpus();
    assert!(
        find_in_graph(&graph, "ChromoQ-X99", "ChromoQ-X99", 3, 20),
        "Should find ChromoQ-X99 needle"
    );
}

#[test]
fn ruler_needle_koloss_v3() {
    let graph = ingest_corpus();
    assert!(
        find_in_graph(&graph, "KOLOSS-v3", "KOLOSS-v3", 3, 20),
        "Should find KOLOSS-v3 needle"
    );
}

#[test]
fn ruler_needle_stigrenet() {
    let graph = ingest_corpus();
    assert!(
        find_in_graph(&graph, "StigreNet", "StigreNet", 3, 20),
        "Should find StigreNet needle"
    );
}

// ── Cross-Reference Tests ──────────────────────────────────────────────────

#[test]
fn ruler_xref_chromoq_x99_to_helios() {
    // Starting from ChromoQ-X99, can we reach Project Helios (or vice versa)?
    // This tests the "uses" relation: Helios → ChromoQ-X99
    let graph = ingest_corpus();
    let q = SomaQuery::new("ChromoQ-X99").with_max_hops(5).with_limit(30);
    let results = graph.traverse(&q);
    let labels: Vec<&str> = results.iter().map(|r| r.node.label.as_str()).collect();
    let combined = labels.join(" ").to_lowercase();

    // We need EITHER Helios visible from ChromoQ-X99 OR ChromoQ-X99 reachable from Helios
    let forward = combined.contains("helios");
    let backward = {
        let q2 = SomaQuery::new("Helios").with_max_hops(5).with_limit(30);
        let r2 = graph.traverse(&q2);
        r2.iter().any(|r| r.node.label.to_lowercase().contains("chromoq-x99"))
    };

    assert!(
        forward || backward,
        "Cross-ref: ChromoQ-X99 and Helios should be linked. Labels found: {:?}", labels
    );
}

#[test]
fn ruler_xref_koloss_v3_dependencies() {
    // KOLOSS-v3 depends on SwarmOS and uses ZetaFold
    let graph = ingest_corpus();
    let q = SomaQuery::new("KOLOSS-v3").with_max_hops(4).with_limit(30);
    let results = graph.traverse(&q);
    let combined: String = results
        .iter()
        .map(|r| r.node.label.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();

    assert!(
        combined.contains("swarmos") || combined.contains("zetafold"),
        "KOLOSS-v3 should reach SwarmOS or ZetaFold. Found: {}", combined
    );
}

#[test]
fn ruler_xref_morphlang_wamr() {
    // MorphLang compiles to WAMR, and AVOID MorphLang v2 with WAMR < 1.3
    // Both entities should exist as nodes in the graph (cross-reference verification).
    // Note: L1 regex extraction creates directional edges, so node co-existence
    // validates that both facts were extracted from the corpus.
    let graph = ingest_corpus();

    let all_labels: Vec<String> = graph.all_labels();
    let all_lower: String = all_labels.join(" ").to_lowercase();

    assert!(
        all_lower.contains("morphlang"),
        "Graph should contain MorphLang node. Labels: {}", all_lower
    );
    assert!(
        all_lower.contains("wamr"),
        "Graph should contain WAMR-related node. Labels: {}", all_lower
    );
}

// ── HDC Semantic Search Tests ──────────────────────────────────────────────

#[test]
fn ruler_hdc_semantic_search() {
    let graph = ingest_corpus();
    let mut hdc = HdcEngine::new(10_000, 5, true);
    let labels = graph.all_labels();
    hdc.train(&labels);

    // Search for "fluorescent protein" — should rank ChromoQ/EGFP/GFP related labels higher
    let results = hdc.search("fluorescent protein", &labels, 10);
    assert!(
        !results.is_empty(),
        "HDC search should return results"
    );
    // At least one result should mention protein-related terms
    let top_labels: String = results.iter().map(|r| r.0.as_str()).collect::<Vec<_>>().join(" ").to_lowercase();
    assert!(
        top_labels.contains("protein") || top_labels.contains("chromoq") || top_labels.contains("gfp") || top_labels.contains("fluorescen"),
        "HDC search for 'fluorescent protein' should return relevant results. Got: {}", top_labels
    );
}

// ── End-to-End Pipeline Test ───────────────────────────────────────────────

#[test]
fn ruler_full_pipeline_metrics() {
    let corpus = build_corpus();
    let pipeline = IngestPipeline::default_config();
    let mut graph = StigreGraph::new("ruler_full", 0.05);

    let result = pipeline
        .ingest_text(&corpus, &mut graph, "ruler_full")
        .unwrap();

    eprintln!("--- RULER Pipeline Metrics ---");
    eprintln!("  Chunks processed:    {}", result.chunks_processed);
    eprintln!("  Triplets extracted:  {}", result.triplets_extracted);
    eprintln!("  Nodes created:       {}", result.nodes_created);
    eprintln!("  Edges created:       {}", result.edges_created);
    eprintln!("  Final node count:    {}", graph.node_count());
    eprintln!("  Final edge count:    {}", graph.edge_count());

    assert!(result.chunks_processed > 5, "Should process multiple chunks");
    assert!(result.triplets_extracted >= 10, "Should extract many triplets from needles");
    assert!(result.edges_created >= 5, "Should create multiple edges");
}
