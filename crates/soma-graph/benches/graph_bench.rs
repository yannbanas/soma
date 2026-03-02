use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use soma_core::{Channel, NodeKind, SomaQuery};
use soma_graph::StigreGraph;

/// Build a graph with N nodes in a chain: A0 → A1 → ... → AN-1
fn build_chain_graph(n: usize) -> StigreGraph {
    let mut g = StigreGraph::new("bench", 0.05);
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        ids.push(g.upsert_node(&format!("node_{}", i), NodeKind::Entity));
    }
    for i in 0..n - 1 {
        g.upsert_edge(ids[i], ids[i + 1], Channel::Trail, 0.9, "bench");
    }
    g
}

/// Build a star graph with N leaves connected to a central hub.
fn build_star_graph(n: usize) -> StigreGraph {
    let mut g = StigreGraph::new("bench", 0.05);
    let hub = g.upsert_node("hub", NodeKind::Entity);
    for i in 0..n {
        let leaf = g.upsert_node(&format!("leaf_{}", i), NodeKind::Entity);
        g.upsert_edge(hub, leaf, Channel::Trail, 0.8, "bench");
    }
    g
}

/// Build a dense graph where every node connects to every other node.
fn build_dense_graph(n: usize) -> StigreGraph {
    let mut g = StigreGraph::new("bench", 0.05);
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        ids.push(g.upsert_node(&format!("dense_{}", i), NodeKind::Entity));
    }
    for i in 0..n {
        for j in 0..n {
            if i != j {
                g.upsert_edge(ids[i], ids[j], Channel::Trail, 0.7, "bench");
            }
        }
    }
    g
}

fn bench_upsert_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/upsert_node");
    for size in [100, 1000, 5000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let mut g = StigreGraph::new("bench", 0.05);
                for i in 0..n {
                    g.upsert_node(black_box(&format!("node_{}", i)), NodeKind::Entity);
                }
            });
        });
    }
    group.finish();
}

fn bench_upsert_node_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/upsert_dedup");
    group.bench_function("1000_inserts_then_1000_dedup", |b| {
        b.iter(|| {
            let mut g = StigreGraph::new("bench", 0.05);
            // First pass: create
            for i in 0..1000 {
                g.upsert_node(&format!("node_{}", i), NodeKind::Entity);
            }
            // Second pass: dedup (should be O(1) per call)
            for i in 0..1000 {
                g.upsert_node(black_box(&format!("node_{}", i)), NodeKind::Entity);
            }
        });
    });
    group.finish();
}

fn bench_upsert_edge(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/upsert_edge");
    group.bench_function("1000_edges_chain", |b| {
        b.iter(|| {
            let mut g = StigreGraph::new("bench", 0.05);
            let mut ids = Vec::with_capacity(1001);
            for i in 0..1001 {
                ids.push(g.upsert_node(&format!("n_{}", i), NodeKind::Entity));
            }
            for i in 0..1000 {
                g.upsert_edge(ids[i], ids[i + 1], Channel::Trail, 0.8, "bench");
            }
        });
    });
    group.finish();
}

fn bench_traverse_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/traverse_chain");
    for depth in [10, 50, 200, 1000] {
        let g = build_chain_graph(depth);
        group.bench_with_input(BenchmarkId::from_parameter(depth), &g, |b, g| {
            let query = SomaQuery::new("node_0").with_max_hops(10).with_limit(50);
            b.iter(|| {
                let results = g.traverse(black_box(&query));
                black_box(results.len());
            });
        });
    }
    group.finish();
}

fn bench_traverse_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/traverse_star");
    for leaves in [50, 200, 1000] {
        let g = build_star_graph(leaves);
        group.bench_with_input(BenchmarkId::from_parameter(leaves), &g, |b, g| {
            let query = SomaQuery::new("hub").with_max_hops(2).with_limit(100);
            b.iter(|| {
                let results = g.traverse(black_box(&query));
                black_box(results.len());
            });
        });
    }
    group.finish();
}

fn bench_traverse_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/traverse_dense");
    for size in [10, 30, 50] {
        let g = build_dense_graph(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &g, |b, g| {
            let query = SomaQuery::new("dense_0").with_max_hops(3).with_limit(100);
            b.iter(|| {
                let results = g.traverse(black_box(&query));
                black_box(results.len());
            });
        });
    }
    group.finish();
}

fn bench_prune_dead_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/prune");
    group.bench_function("prune_1000_episodic", |b| {
        b.iter_batched(
            || {
                let mut g = StigreGraph::new("bench", 0.5); // high threshold
                let mut ids = Vec::new();
                for i in 0..1001 {
                    ids.push(g.upsert_node(&format!("p_{}", i), NodeKind::Entity));
                }
                for i in 0..1000 {
                    g.upsert_edge(ids[i], ids[i + 1], Channel::Episodic, 0.1, "bench");
                }
                g
            },
            |mut g| {
                let pruned = g.prune_dead_edges();
                black_box(pruned);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_stats(c: &mut Criterion) {
    let g = build_chain_graph(1000);
    c.bench_function("graph/stats_1000", |b| {
        b.iter(|| {
            let stats = g.stats();
            black_box(stats);
        });
    });
}

criterion_group!(
    benches,
    bench_upsert_node,
    bench_upsert_node_dedup,
    bench_upsert_edge,
    bench_traverse_chain,
    bench_traverse_star,
    bench_traverse_dense,
    bench_prune_dead_edges,
    bench_stats,
);
criterion_main!(benches);
