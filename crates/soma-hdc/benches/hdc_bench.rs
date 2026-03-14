use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use soma_hdc::HdcEngine;

/// Build a trained engine with a realistic corpus.
fn build_trained_engine(dim: usize, corpus_size: usize) -> HdcEngine {
    let mut engine = HdcEngine::new(dim, 5, true);
    let corpus: Vec<String> = (0..corpus_size)
        .map(|i| match i % 5 {
            0 => format!(
                "ChromoQ variant {} is a fluorescent protein derived from EGFP",
                i
            ),
            1 => format!(
                "AlphaFold2 predicts the structure of protein {} with high accuracy",
                i
            ),
            2 => format!(
                "Rust programming language ensures memory safety through ownership system {}",
                i
            ),
            3 => format!(
                "WebAssembly enables high performance computation in the browser context {}",
                i
            ),
            _ => format!(
                "Bioluminescence in organism {} involves luciferin and luciferase enzymes",
                i
            ),
        })
        .collect();
    engine.train(&corpus);
    engine
}

fn bench_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc/train");
    for corpus_size in [10, 50, 200] {
        group.bench_with_input(
            BenchmarkId::new("D10000", corpus_size),
            &corpus_size,
            |b, &size| {
                let corpus: Vec<String> = (0..size)
                    .map(|i| {
                        format!(
                            "Sentence {} about protein {} and fluorescence properties.",
                            i,
                            i * 7
                        )
                    })
                    .collect();
                b.iter(|| {
                    let mut engine = HdcEngine::new(10_000, 5, true);
                    engine.train(black_box(&corpus));
                    black_box(engine.vocab_size());
                });
            },
        );
    }
    group.finish();
}

fn bench_similarity(c: &mut Criterion) {
    let engine = build_trained_engine(10_000, 50);
    c.bench_function("hdc/similarity_D10000", |b| {
        b.iter(|| {
            let sim = engine.similarity(black_box("chromoq"), black_box("egfp"));
            black_box(sim);
        });
    });
}

fn bench_most_similar(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc/most_similar");
    let engine = build_trained_engine(10_000, 100);

    for k in [5, 10, 20] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let results = engine.most_similar(black_box("protein"), k);
                black_box(results.len());
            });
        });
    }
    group.finish();
}

fn bench_encode_sentence(c: &mut Criterion) {
    let engine = build_trained_engine(10_000, 50);
    c.bench_function("hdc/encode_sentence_D10000", |b| {
        b.iter(|| {
            let vec = engine.encode_sentence(black_box(
                "ChromoQ protein structure prediction using AlphaFold2",
            ));
            black_box(vec.len());
        });
    });
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc/search");
    let engine = build_trained_engine(10_000, 100);

    let candidates: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "candidate sentence {} about various topics including proteins",
                i
            )
        })
        .collect();

    for k in [5, 10, 50] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let results = engine.search(
                    black_box("fluorescent protein structure"),
                    black_box(&candidates),
                    k,
                );
                black_box(results.len());
            });
        });
    }
    group.finish();
}

fn bench_snapshot_roundtrip(c: &mut Criterion) {
    let engine = build_trained_engine(10_000, 50);
    c.bench_function("hdc/snapshot_roundtrip", |b| {
        b.iter(|| {
            let snap = engine.to_snapshot();
            let restored = HdcEngine::from_snapshot(snap);
            black_box(restored.vocab_size());
        });
    });
}

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc/dim_scaling");
    for dim in [1000, 5000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            let engine = build_trained_engine(dim, 20);
            b.iter(|| {
                let sim = engine.similarity(black_box("chromoq"), black_box("protein"));
                black_box(sim);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_train,
    bench_similarity,
    bench_most_similar,
    bench_encode_sentence,
    bench_search,
    bench_snapshot_roundtrip,
    bench_dimension_scaling,
);
criterion_main!(benches);
