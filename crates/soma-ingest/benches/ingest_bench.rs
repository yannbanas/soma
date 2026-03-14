use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use soma_graph::StigreGraph;
use soma_ingest::{Chunker, IngestPipeline, PatternExtractor};

const SHORT_TEXT: &str = "ChromoQ is a fluorescent protein. ChromoQ derives from EGFP. EGFP is a green protein. AlphaFold2 predicts protein structures.";

const MEDIUM_TEXT: &str = "\
Green fluorescent protein (GFP) is a protein that exhibits bright green fluorescence. \
The label GFP traditionally refers to the protein first isolated from the jellyfish Aequorea victoria. \
GFP from Aequorea victoria has a major excitation peak at a wavelength of 395 nm and a minor one at 475 nm. \
Its emission peak is at 509 nm, which is in the lower green portion of the visible spectrum. \
In cell and molecular biology, the GFP gene is frequently used as a reporter of expression. \
Enhanced green fluorescent protein (EGFP) is a mutant form of GFP that has been optimized for brighter fluorescence. \
EGFP has an excitation peak at 488 nm and emission peak at 507 nm. \
ChromoQ is a novel fluorescent protein derived from EGFP through directed evolution. \
ChromoQ variant X47 has a predicted structure confidence of pLDDT=94.2, with emission at 523nm. \
AlphaFold2 was used to predict the three-dimensional structure of ChromoQ with high accuracy. \
The discovery of GFP led to a revolution in biology. \
Osamu Shimomura first isolated GFP from Aequorea victoria in 1962. \
Bioluminescence is the production and emission of light by living organisms. \
Luciferase is a generic term for enzymes that produce bioluminescence. \
Protein structure prediction has been revolutionized by AlphaFold2. \
WebAssembly (WASM) is a binary instruction format for a stack-based virtual machine. \
Rust is a systems programming language focused on safety and performance. \
The petgraph crate provides graph data structure implementations for Rust. \
Hyperdimensional computing (HDC) is a computational paradigm inspired by the brain. \
Random indexing is a dimensionality reduction technique used in distributional semantics.";

fn bench_pattern_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest/patterns");

    group.bench_function("short_text", |b| {
        b.iter(|| {
            let triplets = PatternExtractor::extract(black_box(SHORT_TEXT));
            black_box(triplets.len());
        });
    });

    group.bench_function("medium_text", |b| {
        b.iter(|| {
            let triplets = PatternExtractor::extract(black_box(MEDIUM_TEXT));
            black_box(triplets.len());
        });
    });

    // Large text: repeat medium text 50x
    let large_text = MEDIUM_TEXT.repeat(50);
    group.bench_function("large_text_50x", |b| {
        b.iter(|| {
            let triplets = PatternExtractor::extract(black_box(&large_text));
            black_box(triplets.len());
        });
    });

    group.finish();
}

fn bench_chunking(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest/chunking");

    let large_text = MEDIUM_TEXT.repeat(50);
    for (chunk_size, overlap) in [(3, 1), (5, 1), (10, 2)] {
        let chunker = Chunker::new(chunk_size, overlap);
        group.bench_with_input(
            BenchmarkId::new(format!("cs{}_ov{}", chunk_size, overlap), large_text.len()),
            &large_text,
            |b, text| {
                b.iter(|| {
                    let chunks = chunker.chunk(black_box(text));
                    black_box(chunks.len());
                });
            },
        );
    }
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest/pipeline");

    group.bench_function("short_text", |b| {
        b.iter_batched(
            || {
                (
                    IngestPipeline::default_config(),
                    StigreGraph::new("bench", 0.05),
                )
            },
            |(pipeline, mut graph)| {
                let result = pipeline
                    .ingest_text(black_box(SHORT_TEXT), &mut graph, "bench")
                    .unwrap();
                black_box(result.triplets_extracted);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("medium_text", |b| {
        b.iter_batched(
            || {
                (
                    IngestPipeline::default_config(),
                    StigreGraph::new("bench", 0.05),
                )
            },
            |(pipeline, mut graph)| {
                let result = pipeline
                    .ingest_text(black_box(MEDIUM_TEXT), &mut graph, "bench")
                    .unwrap();
                black_box(result.triplets_extracted);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    let large_text = MEDIUM_TEXT.repeat(20);
    group.bench_function("large_text_20x", |b| {
        b.iter_batched(
            || {
                (
                    IngestPipeline::default_config(),
                    StigreGraph::new("bench", 0.05),
                )
            },
            |(pipeline, mut graph)| {
                let result = pipeline
                    .ingest_text(black_box(&large_text), &mut graph, "bench")
                    .unwrap();
                black_box(result.triplets_extracted);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_pipeline_incremental(c: &mut Criterion) {
    c.bench_function("ingest/incremental_100_docs", |b| {
        b.iter_batched(
            || (IngestPipeline::default_config(), StigreGraph::new("bench", 0.05)),
            |(pipeline, mut graph)| {
                for i in 0..100 {
                    let text = format!(
                        "Protein_{} is a fluorescent marker. Protein_{} derives from EGFP. Protein_{} uses ChromoQ.",
                        i, i, i
                    );
                    pipeline.ingest_text(&text, &mut graph, "bench").unwrap();
                }
                black_box(graph.node_count());
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_pattern_extraction,
    bench_chunking,
    bench_full_pipeline,
    bench_pipeline_incremental,
);
criterion_main!(benches);
