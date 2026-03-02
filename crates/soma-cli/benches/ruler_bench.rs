//! RULER-style benchmark for SOMA.
//!
//! Inspired by the RULER benchmark methodology (an evolution beyond simple
//! Needle-In-A-Haystack), this benchmark tests:
//!
//! 1. **Multi-Needle Retrieval (MNR)**: 10 different "needle" facts are scattered
//!    throughout a large corpus. SOMA must retrieve each one individually.
//!
//! 2. **Cross-Reference Retrieval (CRR)**: The query requires combining information
//!    from two separate facts in the corpus to produce a correct answer.
//!    This tests graph traversal depth and relational reasoning.
//!
//! Metrics:
//! - Retrieval accuracy (hit@K): can the system find the target within K results?
//! - Latency per query (measured by criterion)
//! - Cross-reference hit rate: can graph traversal connect two distant facts?

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use soma_core::SomaQuery;
use soma_graph::StigreGraph;
use soma_hdc::HdcEngine;
use soma_ingest::IngestPipeline;

// ── Corpus generation ──────────────────────────────────────────────────────

/// Haystack: 30 paragraphs of realistic scientific padding text.
/// The needles are injected at specific positions within this haystack.
const HAYSTACK_PARAGRAPHS: &[&str] = &[
    "Green fluorescent protein (GFP) is a protein that exhibits bright green fluorescence when exposed to light in the blue to ultraviolet range. The label GFP traditionally refers to the protein first isolated from the jellyfish Aequorea victoria. GFP from Aequorea victoria has a major excitation peak at a wavelength of 395 nm and a minor one at 475 nm.",
    "In cell and molecular biology, the GFP gene is frequently used as a reporter of expression. It has been used in modified forms to make biosensors. GFP can be introduced into animals, plants, fungi and bacteria to create fluorescent organisms.",
    "Enhanced green fluorescent protein (EGFP) is a mutant form of GFP that has been optimized for brighter fluorescence and higher expression in mammalian cells. EGFP has an excitation peak at 488 nm and emission peak at 507 nm.",
    "The discovery of GFP led to a revolution in biology. Osamu Shimomura first isolated GFP from Aequorea victoria in 1962. Martin Chalfie demonstrated the use of GFP as a biological marker.",
    "Bioluminescence is the production and emission of light by living organisms. It occurs widely in marine vertebrates and invertebrates, as well as in some fungi, microorganisms, and terrestrial arthropods such as fireflies.",
    "Luciferase is a generic term for enzymes that produce bioluminescence. The most commonly used luciferase in research is the firefly luciferase from Photinus pyralis.",
    "Protein structure prediction has been revolutionized by AlphaFold2, developed by DeepMind. AlphaFold2 uses a neural network architecture to predict protein structures with near-experimental accuracy.",
    "WebAssembly (WASM) is a binary instruction format for a stack-based virtual machine. WASM is designed as a portable compilation target for programming languages, enabling deployment on the web.",
    "Unikraft is a fast, secure, and open-source unikernel development kit. It provides a modular approach to building lightweight virtual machines that run a single application.",
    "SwarmOS is a distributed operating system designed for autonomous agent coordination. It uses stigmergic communication patterns where agents leave traces in shared memory.",
    "KOLOSS is an AI orchestration framework that coordinates multiple language models and agents. It uses SOMA as its persistent memory layer, querying knowledge through the MCP protocol.",
    "The Model Context Protocol (MCP) is a standard protocol for AI systems to communicate with external tools and data sources. MCP uses JSON-RPC 2.0 over stdio or TCP transport.",
    "Rust is a systems programming language focused on safety, performance, and concurrency. It achieves memory safety without a garbage collector through its ownership system.",
    "The petgraph crate provides graph data structure implementations for Rust. It supports directed and undirected graphs with customizable node and edge weights.",
    "Hyperdimensional computing (HDC) is a computational paradigm inspired by how the brain processes information. It uses high-dimensional vectors typically with 10000 dimensions.",
    "Random indexing is a dimensionality reduction technique used in distributional semantics. It creates low-dimensional word vectors by accumulating context information.",
    "The write-ahead log (WAL) is a standard method for ensuring data integrity. Changes are first recorded in the WAL before being applied to the main data store.",
    "Zstandard (zstd) is a fast lossless compression algorithm developed by Facebook. It provides compression ratios comparable to zlib while being significantly faster.",
    "Stigmergy is a mechanism of indirect coordination between agents or actions. The principle is that the trace left in the environment by an individual action stimulates subsequent action.",
    "The hippocampus plays a critical role in memory consolidation, the process by which short-term memories are transformed into long-term memories.",
    "Physarum polycephalum is a slime mold that has been studied for its remarkable ability to optimize networks. When placed in a maze, Physarum can find the shortest path.",
    "FSRS (Free Spaced Repetition Scheduler) is a modern spaced repetition algorithm that uses a neural network to optimize review intervals.",
    "BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query. It considers term frequency and inverse document frequency.",
    "The Maximal Marginal Relevance (MMR) algorithm is used to reduce redundancy in information retrieval results. MMR iteratively selects documents that are both relevant and diverse.",
    "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. Django follows the model-template-view architectural pattern.",
    "Sparse distributed representations (SDR) are a way of encoding information that mimics the brain's neocortex. In an SDR, a small fraction of bits are active.",
    "Grid cells are neurons found in the entorhinal cortex that fire in a regular hexagonal pattern as an animal navigates its environment.",
    "Self-organized criticality (SOC) is a property of dynamical systems that have a critical point as an attractor. The system naturally evolves toward this critical state.",
    "The reconsolidation of memory is a process where previously consolidated memories become labile when reactivated, requiring a new consolidation process to persist.",
    "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a second task.",
];

/// 10 needle facts — each contains a unique, verifiable piece of information.
const NEEDLES: &[&str] = &[
    // Needle 0: Unique entity + measurement
    "ZetaFold is a protein folding algorithm with a benchmark accuracy of pLDDT=97.3 on CASP15 targets.",
    // Needle 1: Unique relation
    "NanoLuc derives from deep-sea shrimp Oplophorus gracilirostris and produces 150-fold brighter luminescence than firefly luciferase.",
    // Needle 2: Unique entity + property
    "MorphLang compiles to WAMR bytecode with an average compilation latency of 12 milliseconds.",
    // Needle 3: Unique causal chain
    "CRISPR-Cas13 causes RNA degradation which triggers cellular immune response activation.",
    // Needle 4: Entity with unique numeric
    "Quantum dot QD-705 has emission at 705nm and a quantum yield of 0.91 making it ideal for deep tissue imaging.",
    // Needle 5: Cross-ref part A — this fact alone is not enough
    "Project Helios uses ChromoQ-X99 as its primary fluorescent marker for neural pathway tracing.",
    // Needle 6: Cross-ref part B — must combine with needle 5
    "ChromoQ-X99 has been engineered to emit at exactly 542nm with a pLDDT=98.1, making it the brightest known ChromoQ variant.",
    // Needle 7: Unique warning/alarm
    "AVOID combining MorphLang v2 with WAMR versions below 1.3 due to memory corruption in the bytecode validator.",
    // Needle 8: Unique dependency chain
    "KOLOSS-v3 depends on SwarmOS for agent orchestration and uses ZetaFold for on-the-fly protein analysis.",
    // Needle 9: Unique measurement
    "StigreNet achieves 2.4 million edge traversals per second on a knowledge graph with 100K nodes.",
];

/// Build the RULER corpus: haystack paragraphs with needles scattered throughout.
fn build_ruler_corpus() -> String {
    let mut paragraphs: Vec<String> = HAYSTACK_PARAGRAPHS.iter().map(|s| s.to_string()).collect();

    // Scatter needles at specific positions within the haystack
    let insertion_points = [2, 5, 8, 11, 14, 17, 19, 22, 25, 28];
    for (i, &needle) in NEEDLES.iter().enumerate() {
        let pos = insertion_points[i].min(paragraphs.len());
        paragraphs.insert(pos + i, needle.to_string()); // +i because we shift
    }

    paragraphs.join("\n\n")
}

/// Ingest the RULER corpus into a graph and return it.
fn ingest_ruler_corpus() -> StigreGraph {
    let corpus = build_ruler_corpus();
    let pipeline = IngestPipeline::default_config();
    let mut graph = StigreGraph::new("ruler", 0.05);
    pipeline.ingest_text(&corpus, &mut graph, "ruler").unwrap();
    graph
}

/// Ingest the RULER corpus into both graph and HDC engine.
fn ingest_ruler_full() -> (StigreGraph, HdcEngine) {
    let corpus = build_ruler_corpus();
    let pipeline = IngestPipeline::default_config();
    let mut graph = StigreGraph::new("ruler", 0.05);
    pipeline.ingest_text(&corpus, &mut graph, "ruler").unwrap();

    // Train HDC on all node labels
    let mut hdc = HdcEngine::new(10_000, 5, true);
    let labels = graph.all_labels();
    hdc.train(&labels);

    (graph, hdc)
}

// ── Multi-Needle Retrieval queries ─────────────────────────────────────────

/// Queries targeting each needle. The expected "hit" label is the key entity.
const NEEDLE_QUERIES: &[(&str, &str)] = &[
    ("ZetaFold", "ZetaFold"),                    // Needle 0
    ("NanoLuc", "NanoLuc"),                       // Needle 1
    ("MorphLang", "MorphLang"),                   // Needle 2
    ("CRISPR-Cas13", "CRISPR-Cas13"),             // Needle 3
    ("QD-705", "QD-705"),                         // Needle 4
    ("Project Helios", "Helios"),                 // Needle 5
    ("ChromoQ-X99", "ChromoQ-X99"),               // Needle 6
    ("WAMR versions", "WAMR"),                    // Needle 7 (alarm)
    ("KOLOSS-v3", "KOLOSS-v3"),                   // Needle 8
    ("StigreNet", "StigreNet"),                   // Needle 9
];

/// Cross-reference queries that require combining two needles.
/// Query asks something that only makes sense by linking needle 5 + needle 6.
const CROSS_REF_QUERIES: &[(&str, &[&str])] = &[
    // Q: What emission wavelength does Project Helios's marker have?
    // Requires: Needle 5 (Helios uses ChromoQ-X99) + Needle 6 (ChromoQ-X99 emits at 542nm)
    ("ChromoQ-X99", &["542nm", "Helios", "ChromoQ-X99"]),
    // Q: What orchestration system does KOLOSS-v3 use, and what does KOLOSS-v3 use for protein analysis?
    // Requires: Needle 8 (KOLOSS-v3 depends on SwarmOS, uses ZetaFold)
    ("KOLOSS-v3", &["SwarmOS", "ZetaFold", "KOLOSS-v3"]),
    // Q: What compiles to WAMR and what WAMR version is dangerous?
    // Requires: Needle 2 (MorphLang → WAMR) + Needle 7 (AVOID WAMR < 1.3)
    ("WAMR", &["MorphLang", "WAMR"]),
];

// ── Accuracy helper ────────────────────────────────────────────────────────

/// Check if any of the top-K results contain the expected label (case-insensitive substring).
fn hit_at_k(graph: &StigreGraph, query_str: &str, expected: &str, k: usize) -> bool {
    let query = SomaQuery::new(query_str).with_max_hops(4).with_limit(k);
    let results = graph.traverse(&query);
    results.iter().any(|r| {
        r.node.label.to_lowercase().contains(&expected.to_lowercase())
    })
}

/// Check cross-reference: do the results collectively contain all expected terms?
fn cross_ref_hit(graph: &StigreGraph, query_str: &str, expected_terms: &[&str], k: usize) -> bool {
    let query = SomaQuery::new(query_str).with_max_hops(5).with_limit(k);
    let results = graph.traverse(&query);
    let all_labels: String = results.iter().map(|r| r.node.label.as_str()).collect::<Vec<_>>().join(" ");
    let all_lower = all_labels.to_lowercase();
    expected_terms.iter().all(|term| all_lower.contains(&term.to_lowercase()))
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

fn bench_ruler_ingest(c: &mut Criterion) {
    let corpus = build_ruler_corpus();
    c.bench_function("ruler/ingest_full_corpus", |b| {
        b.iter_batched(
            || (IngestPipeline::default_config(), StigreGraph::new("ruler", 0.05)),
            |(pipeline, mut graph)| {
                let result = pipeline.ingest_text(black_box(&corpus), &mut graph, "ruler").unwrap();
                black_box((result.triplets_extracted, graph.node_count(), graph.edge_count()));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_ruler_multi_needle(c: &mut Criterion) {
    let graph = ingest_ruler_corpus();

    let mut group = c.benchmark_group("ruler/multi_needle");
    for (i, &(query_str, _expected)) in NEEDLE_QUERIES.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("needle", i),
            &query_str,
            |b, &q| {
                let query = SomaQuery::new(q).with_max_hops(4).with_limit(20);
                b.iter(|| {
                    let results = graph.traverse(black_box(&query));
                    black_box(results.len());
                });
            },
        );
    }
    group.finish();
}

fn bench_ruler_cross_reference(c: &mut Criterion) {
    let graph = ingest_ruler_corpus();

    let mut group = c.benchmark_group("ruler/cross_reference");
    for (i, &(query_str, _expected_terms)) in CROSS_REF_QUERIES.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("xref", i),
            &query_str,
            |b, &q| {
                let query = SomaQuery::new(q).with_max_hops(5).with_limit(30);
                b.iter(|| {
                    let results = graph.traverse(black_box(&query));
                    black_box(results.len());
                });
            },
        );
    }
    group.finish();
}

fn bench_ruler_hdc_search(c: &mut Criterion) {
    let (_graph, hdc) = ingest_ruler_full();
    let all_labels: Vec<String> = _graph.all_labels();

    let mut group = c.benchmark_group("ruler/hdc_search");
    let queries = [
        "fluorescent protein",
        "protein folding accuracy",
        "bytecode compilation",
        "bioluminescence enzyme",
        "knowledge graph traversal",
    ];

    for q in queries {
        group.bench_with_input(BenchmarkId::new("query", q), &q, |b, &q| {
            b.iter(|| {
                let results = hdc.search(black_box(q), black_box(&all_labels), 10);
                black_box(results.len());
            });
        });
    }
    group.finish();
}

fn bench_ruler_accuracy_report(c: &mut Criterion) {
    // This benchmark also prints an accuracy report so we can see the results.
    let graph = ingest_ruler_corpus();

    c.bench_function("ruler/accuracy_check", |b| {
        b.iter(|| {
            let mut hits = 0;
            for &(query_str, expected) in NEEDLE_QUERIES {
                if hit_at_k(&graph, query_str, expected, 20) {
                    hits += 1;
                }
            }

            let mut xref_hits = 0;
            for &(query_str, expected_terms) in CROSS_REF_QUERIES {
                if cross_ref_hit(&graph, query_str, expected_terms, 30) {
                    xref_hits += 1;
                }
            }

            black_box((hits, xref_hits));
        });
    });

    // Print accuracy report (outside of benchmark loop)
    eprintln!("\n╔══════════════════════════════════════════╗");
    eprintln!("║       RULER Accuracy Report              ║");
    eprintln!("╠══════════════════════════════════════════╣");

    let mut needle_hits = 0;
    for (i, &(query_str, expected)) in NEEDLE_QUERIES.iter().enumerate() {
        let found = hit_at_k(&graph, query_str, expected, 20);
        if found { needle_hits += 1; }
        eprintln!("║  Needle {:2}: {:15} → {}  ║",
            i, query_str,
            if found { "HIT " } else { "MISS" }
        );
    }
    eprintln!("╠══════════════════════════════════════════╣");
    eprintln!("║  Multi-Needle Score: {}/{}               ║", needle_hits, NEEDLE_QUERIES.len());
    eprintln!("╠══════════════════════════════════════════╣");

    let mut xref_hits = 0;
    for (i, &(query_str, expected_terms)) in CROSS_REF_QUERIES.iter().enumerate() {
        let found = cross_ref_hit(&graph, query_str, expected_terms, 30);
        if found { xref_hits += 1; }
        eprintln!("║  XRef {:2}:   {:15} → {}  ║",
            i, query_str,
            if found { "HIT " } else { "MISS" }
        );
    }
    eprintln!("╠══════════════════════════════════════════╣");
    eprintln!("║  Cross-Ref Score:    {}/{}               ║", xref_hits, CROSS_REF_QUERIES.len());
    eprintln!("╠══════════════════════════════════════════╣");

    let stats = graph.stats();
    eprintln!("║  Graph: {} nodes, {} edges            ║", stats.nodes, stats.edges);
    eprintln!("╚══════════════════════════════════════════╝\n");
}

criterion_group!(
    benches,
    bench_ruler_ingest,
    bench_ruler_multi_needle,
    bench_ruler_cross_reference,
    bench_ruler_hdc_search,
    bench_ruler_accuracy_report,
);
criterion_main!(benches);
