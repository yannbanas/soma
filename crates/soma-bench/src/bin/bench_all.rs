//! bench_all — Run all SOMA benchmarks and generate publication-ready outputs.
//!
//! Usage: cargo run --bin bench_all [-- OPTIONS]
//!
//! Options:
//!   --limit N         Number of questions per dataset (default: 50)
//!   --temporal-k N    Top-K for temporal benchmark (default: 20)
//!   --verbose / -v    Include per-query details
//!   --skip-datasets   Skip MuSiQue/HotpotQA (only run temporal + ablation)
//!
//! Outputs (in `bench_results/`):
//!   - RESULTS.md            — Full markdown report
//!   - temporal_chart.svg    — Temporal accuracy bar chart
//!   - retrieval_chart.svg   — Entity Recall comparison chart
//!   - summary.json          — Machine-readable results

use soma_bench::ablation::{self, AblationConfig};
use soma_bench::loader;
use soma_bench::runner::{self, BenchConfig};
use soma_bench::temporal::{generate_temporal_dataset, run_temporal_benchmark};
use soma_core::LlmSection;
use soma_llm::OllamaClient;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose" || a == "-v");
    let skip_datasets = args.iter().any(|a| a == "--skip-datasets");
    let no_llm = args.iter().any(|a| a == "--no-llm");
    let no_embed = args.iter().any(|a| a == "--no-embed");

    let limit = parse_arg(&args, "--limit").unwrap_or(50);
    let temporal_k = parse_arg(&args, "--temporal-k").unwrap_or(20);

    // ── Ollama LLM client (cogito for L2 extraction) ────────────
    let llm_client = if no_llm {
        println!("[LLM] Disabled via --no-llm flag");
        None
    } else {
        let config = LlmSection {
            enabled: true,
            provider: "ollama".to_string(),
            endpoint: "http://localhost:11434".to_string(),
            model: "cogito:8b".to_string(),
            embedding_model: Some("embeddinggemma:300m".to_string()),
            timeout_ms: 30_000,
        };
        let client = OllamaClient::from_config(&config);
        if client.is_available() {
            println!("[LLM] Ollama cogito:8b connected (L2 extraction enabled)");
            Some(client)
        } else {
            println!("[LLM] Ollama not reachable — running without L2 extraction");
            None
        }
    };

    // ── Ollama embedding client (embeddinggemma:300m for 5th search path) ──
    let embed_client = if no_embed {
        println!("[Embed] Disabled via --no-embed flag");
        None
    } else {
        let config = LlmSection {
            enabled: true,
            provider: "ollama".to_string(),
            endpoint: "http://localhost:11434".to_string(),
            model: "embeddinggemma:300m".to_string(),
            embedding_model: Some("embeddinggemma:300m".to_string()),
            timeout_ms: 30_000,
        };
        let client = OllamaClient::from_config(&config);
        if client.is_available() {
            println!("[Embed] embeddinggemma:300m connected (neural search enabled)");
            Some(client)
        } else {
            println!("[Embed] Ollama not reachable — running without neural embeddings");
            None
        }
    };

    let out_dir = Path::new("bench_results");
    fs::create_dir_all(out_dir).expect("Failed to create bench_results/");

    let mut md = String::new();
    let mut json = serde_json::Map::new();
    let total_start = Instant::now();

    md.push_str("# SOMA Benchmark Results\n\n");
    let mut config_parts = vec!["NER (L1.5)", "Pattern extraction (L1)"];
    if llm_client.is_some() {
        config_parts.push("LLM cogito:8b (L2)");
    }
    if embed_client.is_some() {
        config_parts.push("Neural embeddings (embeddinggemma:300m)");
    }
    config_parts.push("5-path hybrid search + weighted RRF");
    md.push_str(&format!(
        "**Configuration**: {}\n\n",
        config_parts.join(" + ")
    ));

    // ═══════════════════════════════════════════════════════════════
    // 1. TEMPORAL BENCHMARK (always runs — synthetic, no data needed)
    // ═══════════════════════════════════════════════════════════════
    println!(
        "[1/4] Running temporal benchmark (40 queries, k={})...",
        temporal_k
    );
    let (temp_acc, temp_results) = run_temporal_benchmark(temporal_k);
    let temp_data = build_temporal_section(&mut md, &temp_results, temp_acc, temporal_k, verbose);
    json.insert("temporal".into(), temp_data);
    println!("  => Temporal Accuracy: {:.1}%", temp_acc * 100.0);

    // ═══════════════════════════════════════════════════════════════
    // 2. MuSiQue BENCHMARK
    // ═══════════════════════════════════════════════════════════════
    let musique_path = Path::new("crates/soma-bench/data/musique_ans_v1.0_dev.jsonl");
    if !skip_datasets && musique_path.exists() {
        println!("[2/4] Running MuSiQue benchmark ({} questions)...", limit);
        let start = Instant::now();
        match loader::load_musique(musique_path, limit) {
            Ok(questions) => {
                let config = BenchConfig::default();
                let report = runner::run_benchmark_full(
                    &questions,
                    &config,
                    "MuSiQue",
                    llm_client.as_ref(),
                    embed_client.as_ref(),
                );
                let elapsed = start.elapsed();
                build_dataset_section(&mut md, &report, elapsed);
                json.insert("musique".into(), report_to_json(&report, elapsed));
                println!(
                    "  => ER@5={:.1}% ER@10={:.1}% PathRecall={:.1}% MRR={:.3} ({:.1}s)",
                    report.entity_recall_at_5 * 100.0,
                    report.entity_recall_at_10 * 100.0,
                    report.path_recall_avg * 100.0,
                    report.mrr,
                    elapsed.as_secs_f64()
                );
            }
            Err(e) => {
                eprintln!("  => MuSiQue load error: {}", e);
                md.push_str(&format!("## MuSiQue\n\n*Load error: {}*\n\n", e));
            }
        }
    } else if !skip_datasets {
        println!(
            "[2/4] Skipping MuSiQue — dataset not found at {}",
            musique_path.display()
        );
        md.push_str("## MuSiQue\n\n*Dataset not found. See `crates/soma-bench/data/README.md` for download instructions.*\n\n");
    } else {
        println!("[2/4] Skipping MuSiQue (--skip-datasets)");
    }

    // ═══════════════════════════════════════════════════════════════
    // 3. HotpotQA BENCHMARK
    // ═══════════════════════════════════════════════════════════════
    let hotpot_path = Path::new("crates/soma-bench/data/hotpot_dev_distractor_v1.json");
    if !skip_datasets && hotpot_path.exists() {
        println!("[3/4] Running HotpotQA benchmark ({} questions)...", limit);
        let start = Instant::now();
        match loader::load_hotpotqa(hotpot_path, limit) {
            Ok(questions) => {
                let config = BenchConfig::default();
                let report = runner::run_benchmark_full(
                    &questions,
                    &config,
                    "HotpotQA",
                    llm_client.as_ref(),
                    embed_client.as_ref(),
                );
                let elapsed = start.elapsed();
                build_dataset_section(&mut md, &report, elapsed);
                json.insert("hotpotqa".into(), report_to_json(&report, elapsed));
                println!(
                    "  => ER@5={:.1}% ER@10={:.1}% PathRecall={:.1}% MRR={:.3} ({:.1}s)",
                    report.entity_recall_at_5 * 100.0,
                    report.entity_recall_at_10 * 100.0,
                    report.path_recall_avg * 100.0,
                    report.mrr,
                    elapsed.as_secs_f64()
                );
            }
            Err(e) => {
                eprintln!("  => HotpotQA load error: {}", e);
                md.push_str(&format!("## HotpotQA\n\n*Load error: {}*\n\n", e));
            }
        }
    } else if !skip_datasets {
        println!(
            "[3/4] Skipping HotpotQA — dataset not found at {}",
            hotpot_path.display()
        );
        md.push_str("## HotpotQA\n\n*Dataset not found. See `crates/soma-bench/data/README.md` for download instructions.*\n\n");
    } else {
        println!("[3/4] Skipping HotpotQA (--skip-datasets)");
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. ABLATION STUDY (uses MuSiQue if available, else synthetic)
    // ═══════════════════════════════════════════════════════════════
    println!("[4/4] Running ablation study (8 configurations)...");
    let ablation_start = Instant::now();
    // Ablation tests search paths, not LLM ingestion — run without LLM for speed
    let ablation_data = run_ablation_section(&mut md, limit, None);
    let ablation_elapsed = ablation_start.elapsed();
    if let Some(data) = ablation_data {
        json.insert("ablation".into(), data);
    }
    println!(
        "  => Ablation done ({:.1}s)",
        ablation_elapsed.as_secs_f64()
    );

    // ═══════════════════════════════════════════════════════════════
    // WRITE OUTPUT FILES
    // ═══════════════════════════════════════════════════════════════
    let total_elapsed = total_start.elapsed();
    md.push_str(&format!(
        "\n---\n*Generated by `bench_all` in {:.1}s*\n",
        total_elapsed.as_secs_f64()
    ));

    // RESULTS.md
    let md_path = out_dir.join("RESULTS.md");
    let mut f = fs::File::create(&md_path).expect("Failed to create RESULTS.md");
    f.write_all(md.as_bytes())
        .expect("Failed to write RESULTS.md");
    println!("\nWrote {}", md_path.display());

    // temporal_chart.svg
    let categories = [
        "CEO Succession",
        "Version Updates",
        "Location Changes",
        "Status Changes",
    ];
    let mut cat_acc = [0.0f32; 4];
    for (i, chunk) in temp_results.chunks(10).enumerate() {
        let total = chunk
            .iter()
            .filter(|(r, s)| r.is_some() || s.is_some())
            .count();
        let correct = chunk
            .iter()
            .filter(|(r, s)| match (r, s) {
                (Some(rv), Some(sv)) => rv < sv,
                (Some(_), None) => true,
                _ => false,
            })
            .count();
        cat_acc[i] = if total > 0 {
            correct as f32 / total as f32 * 100.0
        } else {
            0.0
        };
    }
    let svg = generate_bar_chart_svg(&categories, &cat_acc, temp_acc * 100.0);
    let svg_path = out_dir.join("temporal_chart.svg");
    fs::write(&svg_path, &svg).expect("Failed to write SVG");
    println!("Wrote {}", svg_path.display());

    // retrieval_chart.svg (if we have dataset results)
    if let (Some(m), Some(h)) = (json.get("musique"), json.get("hotpotqa")) {
        let svg2 = generate_retrieval_chart_svg(m, h);
        let svg2_path = out_dir.join("retrieval_chart.svg");
        fs::write(&svg2_path, &svg2).expect("Failed to write retrieval SVG");
        println!("Wrote {}", svg2_path.display());
    }

    // summary.json
    let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(json))
        .expect("JSON serialization failed");
    let json_path = out_dir.join("summary.json");
    fs::write(&json_path, &json_str).expect("Failed to write JSON");
    println!("Wrote {}", json_path.display());

    println!(
        "\n=== SOMA Benchmark Complete ({:.1}s) ===",
        total_elapsed.as_secs_f64()
    );
}

// ── Argument parsing ─────────────────────────────────────────────

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .skip_while(|a| a.as_str() != flag)
        .nth(1)
        .and_then(|v| v.parse().ok())
}

// ── Temporal section ─────────────────────────────────────────────

fn build_temporal_section(
    md: &mut String,
    results: &[(Option<usize>, Option<usize>)],
    accuracy: f32,
    k: usize,
    verbose: bool,
) -> serde_json::Value {
    let categories = [
        "CEO Succession",
        "Version Updates",
        "Location Changes",
        "Status Changes",
    ];
    let mut cat_correct = [0u32; 4];
    let mut cat_total = [0u32; 4];

    for (i, (recent, stale)) in results.iter().enumerate() {
        let cat = i / 10;
        if recent.is_some() || stale.is_some() {
            cat_total[cat] += 1;
            match (recent, stale) {
                (Some(r), Some(s)) if r < s => cat_correct[cat] += 1,
                (Some(_), None) => cat_correct[cat] += 1,
                _ => {}
            }
        }
    }

    let mrr_sum: f32 = results
        .iter()
        .filter_map(|(r, _)| r.map(|rank| 1.0 / (rank as f32 + 1.0)))
        .sum();
    let mrr_count = results.iter().filter(|(r, _)| r.is_some()).count();
    let mrr = if mrr_count > 0 {
        mrr_sum / mrr_count as f32
    } else {
        0.0
    };

    md.push_str("## Temporal Knowledge Benchmark\n\n");
    md.push_str(
        "Tests whether stigmergic decay correctly prioritises recent facts over stale ones.\n",
    );
    md.push_str("Static KGs (HippoRAG, GraphRAG, etc.) score ~50% (random baseline).\n\n");
    md.push_str("![Temporal Accuracy by Category](temporal_chart.svg)\n\n");
    md.push_str("| Category | Correct | Total | Accuracy |\n");
    md.push_str("|----------|---------|-------|----------|\n");
    for (i, cat) in categories.iter().enumerate() {
        let acc = if cat_total[i] > 0 {
            cat_correct[i] as f32 / cat_total[i] as f32 * 100.0
        } else {
            0.0
        };
        md.push_str(&format!(
            "| {} | {} | {} | {:.1}% |\n",
            cat, cat_correct[i], cat_total[i], acc
        ));
    }
    let total_correct: u32 = cat_correct.iter().sum();
    let total_valid: u32 = cat_total.iter().sum();
    md.push_str(&format!(
        "| **Overall** | **{}** | **{}** | **{:.1}%** |\n\n",
        total_correct,
        total_valid,
        accuracy * 100.0
    ));
    md.push_str(&format!(
        "- **Temporal Accuracy** (k={}): {:.1}%\n",
        k,
        accuracy * 100.0
    ));
    md.push_str(&format!("- **Temporal MRR**: {:.3}\n\n", mrr));

    if verbose {
        let (_, queries) = generate_temporal_dataset();
        md.push_str("<details>\n<summary>Per-query details (click to expand)</summary>\n\n");
        md.push_str("| # | Entity | Recent | Stale | R.Rank | S.Rank | Result |\n");
        md.push_str("|---|--------|--------|-------|--------|--------|--------|\n");
        for (i, ((rr, sr), q)) in results.iter().zip(queries.iter()).enumerate() {
            let result = match (rr, sr) {
                (Some(r), Some(s)) if r < s => "OK",
                (Some(_), None) => "OK",
                _ => "FAIL",
            };
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                i + 1,
                q.question_entity,
                q.expected_recent,
                q.expected_stale,
                rr.map(|r| r.to_string()).unwrap_or("-".into()),
                sr.map(|s| s.to_string()).unwrap_or("-".into()),
                result,
            ));
        }
        md.push_str("\n</details>\n\n");
    }

    // Build JSON
    let mut cat_json = serde_json::Map::new();
    for (i, cat) in categories.iter().enumerate() {
        let mut c = serde_json::Map::new();
        c.insert("correct".into(), cat_correct[i].into());
        c.insert("total".into(), cat_total[i].into());
        cat_json.insert(cat.to_string(), serde_json::Value::Object(c));
    }
    let mut obj = serde_json::Map::new();
    obj.insert("accuracy".into(), serde_json::Value::from(accuracy));
    obj.insert("mrr".into(), serde_json::Value::from(mrr));
    obj.insert("k".into(), serde_json::Value::from(k as u64));
    obj.insert("total_queries".into(), serde_json::Value::from(total_valid));
    obj.insert("correct".into(), serde_json::Value::from(total_correct));
    obj.insert("categories".into(), serde_json::Value::Object(cat_json));
    serde_json::Value::Object(obj)
}

// ── Dataset benchmark section ────────────────────────────────────

fn build_dataset_section(
    md: &mut String,
    report: &runner::BenchReport,
    elapsed: std::time::Duration,
) {
    md.push_str(&format!("## {} Benchmark\n\n", report.dataset));
    md.push_str(&format!(
        "**{}** questions evaluated.\n\n",
        report.num_questions
    ));
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!(
        "| Entity Recall@2 | {:.1}% |\n",
        report.entity_recall_at_2 * 100.0
    ));
    md.push_str(&format!(
        "| Entity Recall@5 | {:.1}% |\n",
        report.entity_recall_at_5 * 100.0
    ));
    md.push_str(&format!(
        "| Entity Recall@10 | {:.1}% |\n",
        report.entity_recall_at_10 * 100.0
    ));
    md.push_str(&format!(
        "| Path Recall | {:.1}% |\n",
        report.path_recall_avg * 100.0
    ));
    md.push_str(&format!("| MRR | {:.3} |\n", report.mrr));
    md.push_str(&format!(
        "| Avg Latency | {:.0} \u{00B5}s |\n",
        report.avg_latency_us
    ));
    md.push_str(&format!(
        "| Total Time | {:.1}s |\n\n",
        elapsed.as_secs_f64()
    ));
}

fn report_to_json(report: &runner::BenchReport, elapsed: std::time::Duration) -> serde_json::Value {
    serde_json::json!({
        "dataset": report.dataset,
        "num_questions": report.num_questions,
        "entity_recall_at_2": report.entity_recall_at_2,
        "entity_recall_at_5": report.entity_recall_at_5,
        "entity_recall_at_10": report.entity_recall_at_10,
        "path_recall": report.path_recall_avg,
        "mrr": report.mrr,
        "avg_latency_us": report.avg_latency_us,
        "total_time_s": elapsed.as_secs_f64(),
    })
}

// ── Ablation section ─────────────────────────────────────────────

fn run_ablation_section(
    md: &mut String,
    limit: usize,
    llm_client: Option<&OllamaClient>,
) -> Option<serde_json::Value> {
    use soma_graph::StigreGraph;
    use soma_hdc::HdcEngine;
    use soma_ingest::IngestPipeline;

    // Try MuSiQue first, fall back to synthetic
    let musique_path = Path::new("crates/soma-bench/data/musique_ans_v1.0_dev.jsonl");
    let questions = if musique_path.exists() {
        match loader::load_musique(musique_path, limit.min(30)) {
            Ok(q) => {
                println!("  Using MuSiQue ({} questions) for ablation...", q.len());
                q
            }
            Err(_) => {
                println!("  MuSiQue load failed, using synthetic data...");
                generate_synthetic_questions()
            }
        }
    } else {
        println!("  No dataset found, using synthetic data for ablation...");
        generate_synthetic_questions()
    };

    if questions.is_empty() {
        md.push_str("## Ablation Study\n\n*No data available.*\n\n");
        return None;
    }

    // Build a shared graph + HDC from all questions' paragraphs
    let mut graph = StigreGraph::new("ablation", 0.05);
    let mut hdc = HdcEngine::new(10_000, 5, true);
    let mut pipeline = IngestPipeline::default_config();
    if let Some(llm) = llm_client {
        pipeline = pipeline.with_llm(llm.clone());
    }

    for q in &questions {
        for (title, text) in &q.all_paragraphs {
            let full = format!("{}: {}", title, text);
            let _ = pipeline.ingest_text(&full, &mut graph, "ablation");
        }
    }
    let labels = graph.all_labels();
    if !labels.is_empty() {
        hdc.train(&labels);
    }

    let reports = ablation::run_ablation(&questions, &graph, &hdc);

    md.push_str("## Ablation Study\n\n");
    md.push_str(&format!(
        "**{}** questions, shared graph ({} nodes).\n\n",
        questions.len(),
        labels.len()
    ));
    md.push_str("| Configuration | ER@5 | ER@10 | Path Recall |\n");
    md.push_str("|---------------|------|-------|-------------|\n");

    let mut json_ablation = serde_json::Map::new();
    for config in AblationConfig::all_configs() {
        if let Some(report) = reports.get(&config.name) {
            md.push_str(&format!(
                "| {} | {:.1}% | {:.1}% | {:.1}% |\n",
                report.config_name,
                report.entity_recall_at_5 * 100.0,
                report.entity_recall_at_10 * 100.0,
                report.path_recall_avg * 100.0,
            ));
            json_ablation.insert(
                config.name.clone(),
                serde_json::json!({
                    "entity_recall_at_5": report.entity_recall_at_5,
                    "entity_recall_at_10": report.entity_recall_at_10,
                    "path_recall": report.path_recall_avg,
                    "num_questions": report.num_questions,
                }),
            );
        }
    }
    md.push('\n');

    if let Some(best) = reports.values().max_by(|a, b| {
        a.entity_recall_at_10
            .partial_cmp(&b.entity_recall_at_10)
            .unwrap()
    }) {
        md.push_str(&format!(
            "> **Best config**: `{}` with ER@10={:.1}%\n\n",
            best.config_name,
            best.entity_recall_at_10 * 100.0
        ));
    }

    Some(serde_json::Value::Object(json_ablation))
}

fn generate_synthetic_questions() -> Vec<loader::BenchQuestion> {
    let data = vec![
        (
            "q1",
            "Who founded Acme Corp?",
            "Alice Smith",
            "Acme Corp",
            "Acme Corp was founded by Alice Smith in 2020. It produces widgets.",
        ),
        (
            "q2",
            "Where is Nexus Inc headquartered?",
            "San Francisco",
            "Nexus Inc",
            "Nexus Inc is headquartered in San Francisco. It was founded in 2015 by Bob Chen.",
        ),
        (
            "q3",
            "What did Orion Labs develop?",
            "quantum sensor",
            "Orion Labs",
            "Orion Labs developed a breakthrough quantum sensor. The lab was established in Berlin.",
        ),
        (
            "q4",
            "Who leads Zenith AI research?",
            "Dr. Carol Wu",
            "Zenith AI",
            "Zenith AI research is led by Dr. Carol Wu. The team focuses on reinforcement learning.",
        ),
        (
            "q5",
            "What framework does Cortex use?",
            "TensorFlow",
            "Cortex Systems",
            "Cortex Systems uses TensorFlow for its production ML pipeline. Founded in 2018 in Austin.",
        ),
    ];

    data.into_iter()
        .map(|(id, question, answer, title, text)| {
            let entities = vec![title.to_string(), answer.to_string()];
            loader::BenchQuestion {
                id: id.to_string(),
                question: question.to_string(),
                answer: answer.to_string(),
                supporting_paragraphs: vec![loader::SupportingParagraph {
                    title: title.to_string(),
                    text: text.to_string(),
                    entities,
                }],
                all_paragraphs: vec![(title.to_string(), text.to_string())],
                num_hops: 1,
                dataset: loader::DatasetKind::MuSiQue,
            }
        })
        .collect()
}

// ── SVG Charts ───────────────────────────────────────────────────

fn generate_bar_chart_svg(_categories: &[&str; 4], accuracies: &[f32; 4], overall: f32) -> String {
    let width = 600;
    let height = 400;
    let ml = 160;
    let mt = 50;
    let cw = 380;
    let ch = 310;
    let bar_count = 5;
    let bar_h = ch / (bar_count + 1);
    let gap = bar_h / 4;
    let bh = bar_h - gap;

    let mut s = String::new();
    s.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" font-family="Arial, sans-serif">
  <style>
    .title {{ font-size: 16px; font-weight: bold; fill: #1a1a2e; text-anchor: middle; }}
    .label {{ font-size: 12px; fill: #333; text-anchor: end; dominant-baseline: middle; }}
    .value {{ font-size: 12px; fill: #333; dominant-baseline: middle; }}
    .bar {{ rx: 3; ry: 3; }}
    .bar-cat {{ fill: #4361ee; }}
    .bar-overall {{ fill: #e63946; }}
    .grid {{ stroke: #e0e0e0; stroke-width: 0.5; }}
    .axis {{ stroke: #666; stroke-width: 1; }}
    .tick {{ font-size: 10px; fill: #666; text-anchor: middle; }}
  </style>
  <text x="{}" y="30" class="title">SOMA Temporal Knowledge Benchmark</text>
"#,
        width, height, width / 2
    ));

    for pct in (0..=100).step_by(25) {
        let x = ml as f32 + (pct as f32 / 100.0) * cw as f32;
        s.push_str(&format!(
            r#"  <line x1="{:.0}" y1="{}" x2="{:.0}" y2="{}" class="grid"/>
  <text x="{:.0}" y="{}" class="tick">{}%</text>
"#,
            x,
            mt,
            x,
            mt + ch,
            x,
            mt + ch + 20,
            pct
        ));
    }

    s.push_str(&format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis"/>
"#,
        ml,
        mt,
        ml,
        mt + ch
    ));

    let all_labels = [
        "CEO Succession",
        "Version Updates",
        "Location Changes",
        "Status Changes",
        "Overall",
    ];
    let all_vals = [
        accuracies[0],
        accuracies[1],
        accuracies[2],
        accuracies[3],
        overall,
    ];

    for (i, (lbl, val)) in all_labels.iter().zip(all_vals.iter()).enumerate() {
        let y = mt + (i as i32) * bar_h + gap / 2;
        let bw = (val / 100.0) * cw as f32;
        let cls = if i < 4 {
            "bar bar-cat"
        } else {
            "bar bar-overall"
        };
        let cy = y as f32 + bh as f32 / 2.0;
        s.push_str(&format!(
            r#"  <text x="{}" y="{:.0}" class="label">{}</text>
  <rect x="{}" y="{}" width="{:.0}" height="{}" class="{}"/>
  <text x="{:.0}" y="{:.0}" class="value">{:.1}%</text>
"#,
            ml - 10,
            cy,
            lbl,
            ml,
            y,
            bw,
            bh,
            cls,
            ml as f32 + bw + 8.0,
            cy,
            val
        ));
    }

    // Baseline 50% dashed line
    let bx = ml as f32 + 0.5 * cw as f32;
    s.push_str(&format!(
        "  <line x1=\"{:.0}\" y1=\"{}\" x2=\"{:.0}\" y2=\"{}\" stroke=\"#999\" stroke-width=\"1.5\" stroke-dasharray=\"6,3\"/>\n\
         <text x=\"{:.0}\" y=\"{}\" font-size=\"10\" fill=\"#999\" text-anchor=\"start\">Static KG baseline (50%)</text>\n",
        bx, mt - 5, bx, mt + ch, bx + 4.0, mt - 8
    ));
    s.push_str("</svg>\n");
    s
}

fn generate_retrieval_chart_svg(musique: &serde_json::Value, hotpot: &serde_json::Value) -> String {
    let metrics = [
        "entity_recall_at_2",
        "entity_recall_at_5",
        "entity_recall_at_10",
        "path_recall",
        "mrr",
    ];
    let labels = ["ER@2", "ER@5", "ER@10", "PathRecall", "MRR"];
    let width = 650;
    let height = 350;
    let ml = 100;
    let mt = 50;
    let cw = 480;
    let ch = 250;
    let group_w = cw / metrics.len() as i32;
    let bar_w = group_w / 3;

    let mut s = String::new();
    s.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" font-family="Arial, sans-serif">
  <style>
    .title {{ font-size: 16px; font-weight: bold; fill: #1a1a2e; text-anchor: middle; }}
    .musique {{ fill: #4361ee; }}
    .hotpot {{ fill: #f77f00; }}
    .tick {{ font-size: 10px; fill: #666; }}
    .val {{ font-size: 9px; fill: #333; text-anchor: middle; }}
    .grid {{ stroke: #e0e0e0; stroke-width: 0.5; }}
    .legend {{ font-size: 11px; fill: #333; }}
  </style>
  <text x="{}" y="30" class="title">SOMA Multi-hop QA Retrieval</text>
"#,
        width, height, width / 2
    ));

    // Y-axis grid
    for pct in (0..=100).step_by(25) {
        let y = mt + ch - (pct as f32 / 100.0 * ch as f32) as i32;
        s.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid"/>
  <text x="{}" y="{}" class="tick" text-anchor="end">{}%</text>
"#,
            ml,
            y,
            ml + cw,
            y,
            ml - 5,
            y + 4,
            pct
        ));
    }

    for (i, (metric, label)) in metrics.iter().zip(labels.iter()).enumerate() {
        let gx = ml + (i as i32) * group_w + group_w / 2;
        let m_val = musique.get(metric).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let h_val = hotpot.get(metric).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        let m_h = (m_val * ch as f32).max(1.0) as i32;
        let h_h = (h_val * ch as f32).max(1.0) as i32;
        let m_y = mt + ch - m_h;
        let h_y = mt + ch - h_h;

        // MuSiQue bar
        s.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="{}" height="{}" class="musique" rx="2"/>
  <text x="{}" y="{}" class="val">{:.0}%</text>
"#,
            gx - bar_w,
            m_y,
            bar_w,
            m_h,
            gx - bar_w / 2,
            m_y - 4,
            m_val * 100.0
        ));
        // HotpotQA bar
        s.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="{}" height="{}" class="hotpot" rx="2"/>
  <text x="{}" y="{}" class="val">{:.0}%</text>
"#,
            gx,
            h_y,
            bar_w,
            h_h,
            gx + bar_w / 2,
            h_y - 4,
            h_val * 100.0
        ));
        // X label
        s.push_str(&format!(
            r#"  <text x="{}" y="{}" class="tick" text-anchor="middle">{}</text>
"#,
            gx,
            mt + ch + 18,
            label
        ));
    }

    // Legend
    s.push_str(&format!(
        r#"  <rect x="{}" y="38" width="12" height="12" class="musique" rx="2"/>
  <text x="{}" y="49" class="legend">MuSiQue</text>
  <rect x="{}" y="38" width="12" height="12" class="hotpot" rx="2"/>
  <text x="{}" y="49" class="legend">HotpotQA</text>
"#,
        width - 180,
        width - 165,
        width - 100,
        width - 85
    ));

    s.push_str("</svg>\n");
    s
}
