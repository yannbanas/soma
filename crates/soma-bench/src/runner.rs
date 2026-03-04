//! Benchmark runner: orchestrates ingest → search → score.
//!
//! Uses a shared graph approach (all paragraphs ingested into one graph),
//! matching SOMA's real usage pattern. Search uses 5-path hybrid:
//! graph BFS + HDC + fuzzy + PPR + neural embeddings.

use std::collections::HashMap;

use crate::loader::BenchQuestion;
use crate::metrics;
use serde::Serialize;
use soma_core::{fuzzy_label_search, rrf_merge_with_sources, Channel, NodeKind, SomaQuery};
use soma_graph::StigreGraph;
use soma_hdc::HdcEngine;
use soma_ingest::IngestPipeline;
use soma_llm::OllamaClient;

/// Configuration for a benchmark run.
pub struct BenchConfig {
    /// Top-K cutoff for Recall@K.
    pub retrieval_k: Vec<usize>,
    /// Maximum hops for graph traversal.
    pub max_hops: u8,
    /// Maximum search results.
    pub limit: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            retrieval_k: vec![2, 5, 10],
            max_hops: 4,
            limit: 20,
        }
    }
}

/// Aggregated results of a benchmark run.
#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub dataset: String,
    pub num_questions: usize,
    pub entity_recall_at_2: f32,
    pub entity_recall_at_5: f32,
    pub entity_recall_at_10: f32,
    pub path_recall_avg: f32,
    pub mrr: f32,
    pub avg_latency_us: f64,
}

/// Run a benchmark (backward-compatible, no LLM/embeddings).
pub fn run_benchmark(
    questions: &[BenchQuestion],
    config: &BenchConfig,
    dataset_name: &str,
) -> BenchReport {
    run_benchmark_full(questions, config, dataset_name, None, None)
}

/// Run with optional LLM extraction only.
pub fn run_benchmark_with_llm(
    questions: &[BenchQuestion],
    config: &BenchConfig,
    dataset_name: &str,
    llm_client: Option<&OllamaClient>,
) -> BenchReport {
    run_benchmark_full(questions, config, dataset_name, llm_client, None)
}

/// Full benchmark: optional LLM extraction + optional neural embeddings.
pub fn run_benchmark_full(
    questions: &[BenchQuestion],
    config: &BenchConfig,
    dataset_name: &str,
    llm_client: Option<&OllamaClient>,
    embed_client: Option<&OllamaClient>,
) -> BenchReport {
    // ── Phase 1: Build shared graph ──────────────────────────────
    let mut graph = StigreGraph::new("bench", 0.05);
    let mut hdc = HdcEngine::new(10_000, 5, true);
    let mut pipeline = IngestPipeline::default_config();
    if let Some(llm) = llm_client {
        pipeline = pipeline.with_llm(llm.clone());
    }

    // Track title → entities for cross-paragraph linking
    let mut title_entities: HashMap<String, Vec<String>> = HashMap::new();

    let total_paragraphs: usize = questions.iter().map(|q| q.all_paragraphs.len()).sum();
    let mut ingested = 0usize;
    for question in questions {
        for (title, text) in &question.all_paragraphs {
            // Inject title as an explicit Entity node
            let title_id = graph.upsert_node(title, NodeKind::Entity);
            let full_text = format!("{}: {}", title, text);
            let _ = pipeline.ingest_text(&full_text, &mut graph, "bench");

            // Connect title node to NER entities from its text
            let ner_entities = soma_ingest::ner::extract_entities(text);
            let mut ent_names = Vec::new();
            for ent in &ner_entities {
                let ent_id = graph.upsert_node(&ent.name, NodeKind::Entity);
                if ent_id != title_id {
                    graph.upsert_edge(title_id, ent_id, Channel::Trail, 0.7, "bench");
                }
                ent_names.push(ent.name.clone());
            }
            title_entities.insert(title.clone(), ent_names);

            ingested += 1;
            if llm_client.is_some() && ingested % 20 == 0 {
                eprint!(
                    "\r  Ingesting with LLM: {}/{} paragraphs...",
                    ingested, total_paragraphs
                );
            }
        }
    }
    if llm_client.is_some() && total_paragraphs > 20 {
        eprintln!(
            "\r  Ingesting with LLM: {}/{} paragraphs... done.",
            total_paragraphs, total_paragraphs
        );
    }

    // ── Cross-paragraph title linking ────────────────────────────
    // Connect titles that share entities → creates multi-hop paths
    let titles: Vec<String> = title_entities.keys().cloned().collect();
    for i in 0..titles.len() {
        for j in (i + 1)..titles.len() {
            let ents_i = &title_entities[&titles[i]];
            let ents_j = &title_entities[&titles[j]];
            let shared = ents_i
                .iter()
                .any(|e| ents_j.iter().any(|f| e.to_lowercase() == f.to_lowercase()));
            if shared {
                let id_i = graph.upsert_node(&titles[i], NodeKind::Entity);
                let id_j = graph.upsert_node(&titles[j], NodeKind::Entity);
                graph.upsert_edge(id_i, id_j, Channel::Trail, 0.6, "bench");
            }
        }
    }

    let labels = graph.all_labels();
    if !labels.is_empty() {
        hdc.train(&labels);
    }

    // ── Phase 1b: Pre-compute neural embeddings (Entity/Concept only) ─
    let label_embeddings: HashMap<String, Vec<f64>> = if let Some(ec) = embed_client {
        let filtered: Vec<String> = graph
            .all_nodes()
            .filter(|n| matches!(n.kind, NodeKind::Entity | NodeKind::Concept))
            .map(|n| n.label.clone())
            .filter(|l| l.len() >= 3 && l.len() <= 100)
            .collect();
        let mut emb_map = HashMap::new();
        let batch_size = 100;
        let total_batches = (filtered.len() + batch_size - 1) / batch_size;
        for (batch_idx, chunk) in filtered.chunks(batch_size).enumerate() {
            let batch: Vec<String> = chunk.to_vec();
            match ec.embed_batch(&batch) {
                Ok(Some(vecs)) => {
                    for (label, vec) in batch.iter().zip(vecs.into_iter()) {
                        emb_map.insert(label.clone(), vec);
                    }
                }
                _ => {}
            }
            if total_batches > 2 {
                eprint!(
                    "\r  Embedding labels: {}/{}...",
                    (batch_idx + 1) * batch_size,
                    filtered.len()
                );
            }
        }
        if total_batches > 2 {
            eprintln!("\r  Embedding labels: {}/{} done.", filtered.len(), filtered.len());
        }
        emb_map
    } else {
        HashMap::new()
    };

    // ── Phase 2: Query & score ───────────────────────────────────
    let mut total_recall_2 = 0.0f32;
    let mut total_recall_5 = 0.0f32;
    let mut total_recall_10 = 0.0f32;
    let mut total_path_recall = 0.0f32;
    let mut all_ranks: Vec<Option<usize>> = Vec::new();
    let mut total_latency_us = 0u128;
    let mut count = 0usize;

    for question in questions {
        let start = std::time::Instant::now();
        let all_labels = graph.all_labels();

        // Extract entities from question for targeted search
        let query_entities = soma_graph::extract_query_entities(&question.question);

        // Path 1: Graph BFS (try each extracted entity as seed)
        let mut graph_list: Vec<(String, f32)> = Vec::new();
        for entity in &query_entities {
            let q = SomaQuery::new(entity)
                .with_max_hops(config.max_hops)
                .with_limit(config.limit);
            let results = graph.traverse(&q);
            for r in &results {
                if !graph_list.iter().any(|(l, _)| l == &r.node.label) {
                    graph_list.push((r.node.label.clone(), r.score));
                }
            }
        }
        // Also try fuzzy-matched labels as BFS seeds (per entity)
        let mut fuzzy_seeds: Vec<(String, f32)> = Vec::new();
        for entity in &query_entities {
            for (label, score) in fuzzy_label_search(entity, &all_labels, 3) {
                if !fuzzy_seeds.iter().any(|(l, _)| l == &label) {
                    fuzzy_seeds.push((label, score));
                }
            }
        }
        for (seed_label, seed_score) in &fuzzy_seeds {
            if *seed_score >= 0.5 {
                let q = SomaQuery::new(seed_label)
                    .with_max_hops(config.max_hops)
                    .with_limit(config.limit);
                let results = graph.traverse(&q);
                for r in &results {
                    if !graph_list.iter().any(|(l, _)| l == &r.node.label) {
                        graph_list.push((r.node.label.clone(), r.score));
                    }
                }
            }
        }

        // Path 2: HDC semantic
        let hdc_list = hdc.search_labels(&question.question, &all_labels, config.limit);

        // Path 3: Fuzzy label match
        let fuzzy_list = fuzzy_label_search(&question.question, &all_labels, config.limit);

        // Path 4: PPR (Personalized PageRank)
        let mut ppr_seeds = query_entities.clone();
        for (label, score) in fuzzy_label_search(&question.question, &all_labels, 5) {
            if score >= 0.7 && !ppr_seeds.contains(&label) {
                ppr_seeds.push(label);
            }
        }
        let ppr_results = graph.ppr(&ppr_seeds, 0.15, 50, 1e-6, None);
        let ppr_list: Vec<(String, f32)> = ppr_results
            .iter()
            .take(config.limit)
            .map(|(_, label, score)| (label.clone(), *score))
            .collect();

        // Path 5: Neural embedding similarity
        let neural_list: Vec<(String, f32)> = if !label_embeddings.is_empty() {
            if let Some(ec) = embed_client {
                match ec.embed(&question.question) {
                    Ok(Some(q_emb)) => {
                        let mut scores: Vec<(String, f32)> = label_embeddings
                            .iter()
                            .map(|(label, emb)| {
                                let sim = cosine_similarity(&q_emb, emb);
                                (label.clone(), sim as f32)
                            })
                            .collect();
                        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        scores.truncate(config.limit);
                        scores
                    }
                    _ => Vec::new(),
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // RRF merge: PPR gets 2x weight (most effective path)
        let mut ranked_lists: Vec<(&str, Vec<(String, f32)>)> = vec![
            ("graph", graph_list),
            ("hdc", hdc_list),
            ("fuzzy", fuzzy_list),
            ("ppr", ppr_list.clone()),
            ("ppr2", ppr_list), // PPR 2x boost
        ];
        if !neural_list.is_empty() {
            ranked_lists.push(("neural", neural_list));
        }
        let hybrid_results = rrf_merge_with_sources(&ranked_lists, 60.0);

        // Extract retrieved labels
        let retrieved: Vec<String> = hybrid_results
            .iter()
            .take(config.limit)
            .map(|hr| hr.label.clone())
            .collect();

        let elapsed = start.elapsed();
        total_latency_us += elapsed.as_micros();

        // Compute gold entities
        let gold_entities: Vec<String> = question
            .supporting_paragraphs
            .iter()
            .flat_map(|p| p.entities.clone())
            .collect();

        // Entity Recall@K
        let (r2, _, _) = metrics::entity_recall_at_k(&gold_entities, &retrieved, 2);
        let (r5, _, _) = metrics::entity_recall_at_k(&gold_entities, &retrieved, 5);
        let (r10, _, _) = metrics::entity_recall_at_k(&gold_entities, &retrieved, 10);
        total_recall_2 += r2;
        total_recall_5 += r5;
        total_recall_10 += r10;

        // Path Recall (use supporting paragraph titles as chain)
        let chain: Vec<String> = question
            .supporting_paragraphs
            .iter()
            .map(|p| p.title.clone())
            .collect();
        total_path_recall += metrics::path_recall(&chain, &retrieved);

        // MRR: rank of first gold entity
        let first_gold_rank = gold_entities.iter().find_map(|g| {
            let g_lower = g.to_lowercase();
            retrieved.iter().position(|r: &String| {
                let r_lower = r.to_lowercase();
                r_lower.contains(&g_lower) || g_lower.contains(&r_lower)
            })
        });
        all_ranks.push(first_gold_rank);

        count += 1;
    }

    let n = count.max(1) as f32;

    BenchReport {
        dataset: dataset_name.to_string(),
        num_questions: count,
        entity_recall_at_2: total_recall_2 / n,
        entity_recall_at_5: total_recall_5 / n,
        entity_recall_at_10: total_recall_10 / n,
        path_recall_avg: total_path_recall / n,
        mrr: metrics::mean_reciprocal_rank(&all_ranks),
        avg_latency_us: total_latency_us as f64 / count.max(1) as f64,
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{DatasetKind, SupportingParagraph};

    fn make_question() -> BenchQuestion {
        BenchQuestion {
            id: "test-1".to_string(),
            question: "Who founded Acme Corp?".to_string(),
            answer: "Alice Smith".to_string(),
            supporting_paragraphs: vec![SupportingParagraph {
                title: "Acme Corp".to_string(),
                text: "Acme Corp was founded by Alice Smith in 2020.".to_string(),
                entities: vec!["Acme Corp".to_string(), "Alice Smith".to_string()],
            }],
            all_paragraphs: vec![
                (
                    "Acme Corp".to_string(),
                    "Acme Corp was founded by Alice Smith in 2020. It produces widgets."
                        .to_string(),
                ),
                (
                    "Bob Industries".to_string(),
                    "Bob Industries is a competitor of Acme Corp.".to_string(),
                ),
            ],
            num_hops: 1,
            dataset: DatasetKind::MuSiQue,
        }
    }

    #[test]
    fn run_single_question() {
        let questions = vec![make_question()];
        let config = BenchConfig::default();
        let report = run_benchmark(&questions, &config, "test");
        assert_eq!(report.num_questions, 1);
        assert!(report.avg_latency_us > 0.0);
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }
}
