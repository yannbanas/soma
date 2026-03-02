use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use colored::Colorize;
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use soma_bio::{BioConfig, BioScheduler};
use chrono::Utc;
use soma_core::{Channel, NodeKind, SomaConfig, SomaQuery, fuzzy_label_search, rrf_merge_with_sources};
use soma_graph::StigreGraph;
use soma_hdc::HdcEngine;
use soma_ingest::{IngestPipeline, IngestSource};
use soma_llm::OllamaClient;
use soma_mcp::McpServer;
use soma_store::Store;

#[derive(Parser)]
#[command(
    name = "soma",
    about = "SOMA — Stigmergic Ontological Memory Architecture",
    version = "0.1.0",
    author = "Yann Banas"
)]
struct Cli {
    /// Path to soma.toml configuration file
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Workspace to operate on
    #[arg(long, short = 'w', global = true, default_value = "default")]
    workspace: String,

    /// Output format: text or json (for scripting/piping)
    #[arg(long, global = true, default_value = "text")]
    format: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add text or a note to SOMA memory
    Add {
        /// Content to add
        content: String,
        /// Tags (comma-separated)
        #[arg(long, short = 't')]
        tags: Option<String>,
        /// Channel type
        #[arg(long, short = 'c', default_value = "trail")]
        channel: String,
        /// Source identifier
        #[arg(long, short = 's', default_value = "cli")]
        source: String,
    },
    /// Ingest a file into SOMA
    Ingest {
        /// File path to ingest
        #[arg(long, short = 'f')]
        file: PathBuf,
        /// Enable chunking
        #[arg(long)]
        chunk: bool,
    },
    /// Search the knowledge graph
    Search {
        /// Search query
        query: String,
        /// Number of results
        #[arg(long, short = 'k', default_value = "10")]
        limit: usize,
        /// Channel filter
        #[arg(long)]
        channel: Option<String>,
        /// Maximum hops
        #[arg(long, default_value = "3")]
        max_hops: u8,
        /// Disable hybrid search (graph-only, like legacy mode)
        #[arg(long)]
        no_semantic: bool,
    },
    /// Create a relation between two entities
    Relate {
        /// Source entity
        from: String,
        /// Target entity
        to: String,
        /// Channel type
        #[arg(long, short = 'c', default_value = "trail")]
        channel: String,
        /// Confidence [0.0, 1.0]
        #[arg(long, default_value = "0.8")]
        confidence: f32,
    },
    /// Display graph statistics
    Stats,
    /// Manage workspaces
    Workspace {
        /// Action: create, list, switch
        action: String,
        /// Workspace name
        name: Option<String>,
    },
    /// Trigger manual sleep consolidation
    Sleep,
    /// Start MCP server (stdio)
    #[command(name = "mcp-stdio")]
    McpStdio,
    /// Start MCP server (TCP)
    #[command(name = "mcp-tcp")]
    McpTcp {
        #[arg(long, default_value = "3333")]
        port: u16,
    },
    /// Embed all graph labels using Ollama neural embeddings
    Embed {
        /// Only embed labels without existing neural embeddings
        #[arg(long)]
        incremental: bool,
    },
    /// Inspect a node and its neighbors
    Show {
        /// Node label to inspect
        label: String,
    },
    /// List all nodes in the graph
    List {
        /// Filter by kind (entity, concept, event, measurement, procedure, warning)
        #[arg(long)]
        kind: Option<String>,
        /// Filter by tag
        #[arg(long)]
        tag: Option<String>,
        /// Maximum results
        #[arg(long, short = 'k', default_value = "50")]
        limit: usize,
    },
    /// Export the knowledge graph
    Export {
        /// Export format: json, dot, csv
        #[arg(long, default_value = "json")]
        format: String,
        /// Output file (stdout if absent)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },
    /// Forget (archive) a node
    Forget {
        /// Label of node to forget
        label: String,
    },
    /// Flag an entity with an alarm
    Alarm {
        /// Entity to flag
        label: String,
        /// Reason for the alarm
        reason: String,
    },
    /// Reinforce edges between two entities
    Reinforce {
        /// Source entity
        from: String,
        /// Target entity
        to: String,
    },
    /// Retrieve LLM-ready context from the graph
    Context {
        /// Query for context retrieval
        query: String,
    },
    /// Start daemon with biological scheduler
    Daemon,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Load and validate config
    let config = load_config(cli.config.as_deref())?;
    config.validate().map_err(|e| format!("invalid config: {}", e))?;
    let data_dir = config.resolved_data_dir();

    // Open store
    let store = Store::open(&data_dir, &cli.workspace)?;

    // Initialize graph
    let mut graph = StigreGraph::new(&cli.workspace, config.bio.prune_threshold);

    // Try to recover from snapshot + WAL
    let mut hdc_snapshot_json: Option<String> = None;
    if let Ok(Some(snapshot)) = store.read_snapshot() {
        if let Ok(petgraph_data) = serde_json::from_str(&snapshot.graph_json) {
            graph.set_inner(petgraph_data);
            eprintln!(
                "{} Loaded snapshot: {} nodes, {} edges",
                "✓".green(),
                snapshot.meta.node_count,
                snapshot.meta.edge_count
            );
        }
        hdc_snapshot_json = snapshot.hdc_json;
    }

    // Replay WAL entries
    let wal_entries = store.read_wal()?;
    if !wal_entries.is_empty() {
        eprintln!(
            "{} Replaying {} WAL entries...",
            "→".cyan(),
            wal_entries.len()
        );
        for entry in &wal_entries {
            replay_wal_entry(&mut graph, entry);
        }
    }

    // Initialize HDC engine (restore from snapshot if available)
    let mut hdc = if let Some(ref hdc_json) = hdc_snapshot_json {
        if let Ok(snap) = serde_json::from_str::<soma_hdc::HdcSnapshot>(hdc_json) {
            let engine = HdcEngine::from_snapshot(snap);
            let nc = engine.neural_count();
            if nc > 0 {
                eprintln!("{} Restored {} neural embeddings", "✓".green(), nc);
            }
            engine
        } else {
            HdcEngine::new(config.hdc.dimension, config.hdc.window_size, config.hdc.tfidf)
        }
    } else {
        HdcEngine::new(config.hdc.dimension, config.hdc.window_size, config.hdc.tfidf)
    };

    // Auto-train HDC on any graph labels not in vocab (covers WAL replay gap)
    let all_graph_labels = graph.all_labels();
    if !all_graph_labels.is_empty() {
        let vocab_before = hdc.vocab_size();
        hdc.train(&all_graph_labels);
        let new_tokens = hdc.vocab_size() - vocab_before;
        if new_tokens > 0 {
            eprintln!("{} HDC trained {} new tokens from graph labels", "✓".green(), new_tokens);
        }
    }

    // Initialize Ollama client (optional, graceful degradation)
    let llm_client = if config.llm.enabled {
        let client = OllamaClient::from_config(&config.llm);
        if client.is_available() {
            let emb_model = config
                .llm
                .embedding_model
                .as_deref()
                .unwrap_or(&config.llm.model);
            eprintln!(
                "{} Ollama connected (gen: {}, emb: {})",
                "✓".green(),
                config.llm.model,
                emb_model
            );
            Some(client)
        } else {
            eprintln!("{} Ollama not available, L2 disabled", "⚠".yellow());
            None
        }
    } else {
        None
    };

    // Wrap in Arc<RwLock> for shared access
    let graph = Arc::new(RwLock::new(graph));
    let hdc = Arc::new(RwLock::new(hdc));
    let store = Arc::new(RwLock::new(store));
    let json_mode = cli.format == "json";

    match cli.command {
        Commands::Add {
            content,
            tags,
            channel,
            source,
        } => {
            cmd_add(&graph, &hdc, &store, &content, tags, &channel, &source, &llm_client).await?;
            maybe_snapshot(&graph, &hdc, &store).await?;
        }
        Commands::Ingest { file, chunk: _ } => {
            cmd_ingest(&graph, &hdc, &store, &file, &llm_client).await?;
            maybe_snapshot(&graph, &hdc, &store).await?;
        }
        Commands::Search {
            query,
            limit,
            channel,
            max_hops,
            no_semantic,
        } => {
            cmd_search(&graph, &hdc, &query, limit, channel, max_hops, no_semantic, json_mode).await?;
        }
        Commands::Relate {
            from,
            to,
            channel,
            confidence,
        } => {
            cmd_relate(&graph, &store, &from, &to, &channel, confidence).await?;
            maybe_snapshot(&graph, &hdc, &store).await?;
        }
        Commands::Stats => {
            cmd_stats(&graph, &hdc, json_mode).await?;
        }
        Commands::Workspace { action, name } => {
            cmd_workspace(&action, name.as_deref(), &data_dir).await?;
        }
        Commands::Sleep => {
            eprintln!("{} Manual consolidation triggered", "◉".yellow());
            eprintln!("  (auto-consolidation runs every 6h in daemon mode)");
        }
        Commands::Embed { incremental } => {
            cmd_embed(&graph, &hdc, &store, &llm_client, incremental).await?;
        }
        Commands::Show { label } => {
            cmd_show(&graph, &label, json_mode).await?;
        }
        Commands::List { kind, tag, limit } => {
            cmd_list(&graph, kind, tag, limit, json_mode).await?;
        }
        Commands::Export { format, output } => {
            cmd_export(&graph, &format, output).await?;
        }
        Commands::Forget { label } => {
            cmd_forget(&graph, &store, &label).await?;
            maybe_snapshot(&graph, &hdc, &store).await?;
        }
        Commands::Alarm { label, reason } => {
            cmd_alarm(&graph, &store, &label, &reason).await?;
            maybe_snapshot(&graph, &hdc, &store).await?;
        }
        Commands::Reinforce { from, to } => {
            cmd_reinforce(&graph, &store, &from, &to).await?;
            maybe_snapshot(&graph, &hdc, &store).await?;
        }
        Commands::Context { query } => {
            cmd_context(&graph, &hdc, &query, json_mode).await?;
        }
        Commands::McpStdio => {
            let mut server = McpServer::new(graph, hdc, store);
            if let Some(ref llm) = llm_client {
                server = server.with_llm(llm.clone());
            }
            server.run_stdio().await?;
        }
        Commands::McpTcp { port } => {
            let mut server = McpServer::new(graph, hdc, store);
            if let Some(ref llm) = llm_client {
                server = server.with_llm(llm.clone());
            }
            server.run_tcp(port).await?;
        }
        Commands::Daemon => {
            cmd_daemon(graph.clone(), hdc.clone(), store.clone(), &config).await?;
        }
    }

    Ok(())
}

fn load_config(path: Option<&std::path::Path>) -> Result<SomaConfig, Box<dyn std::error::Error>> {
    if let Some(path) = path {
        let content = std::fs::read_to_string(path)?;
        let config: SomaConfig = toml::from_str(&content)?;
        Ok(config)
    } else {
        // Try default locations
        let candidates = [
            PathBuf::from("soma.toml"),
            dirs_home().join(".config/soma/soma.toml"),
        ];
        for candidate in &candidates {
            if candidate.exists() {
                let content = std::fs::read_to_string(candidate)?;
                let config: SomaConfig = toml::from_str(&content)?;
                return Ok(config);
            }
        }
        Ok(SomaConfig::default())
    }
}

fn dirs_home() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:\\Users\\default"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
    }
}

fn replay_wal_entry(graph: &mut StigreGraph, entry: &soma_store::WalEntry) {
    match entry {
        soma_store::WalEntry::NodeUpsert(node) => {
            graph.upsert_node(&node.label, node.kind.clone());
        }
        soma_store::WalEntry::EdgeUpsert(edge) => {
            graph.upsert_edge(edge.from, edge.to, edge.channel, edge.confidence, &edge.source);
        }
        soma_store::WalEntry::EdgeReinforce { id, .. } => {
            graph.reinforce_edge(*id);
        }
        soma_store::WalEntry::EdgePrune(_) | soma_store::WalEntry::NodeArchive(_) => {
            // These are informational during replay
        }
        soma_store::WalEntry::ConsolidationEvent { .. } => {
            // Informational
        }
    }
}

/// Auto-snapshot when WAL exceeds threshold.
async fn maybe_snapshot(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    store: &Arc<RwLock<Store>>,
) -> Result<(), Box<dyn std::error::Error>> {
    const SNAPSHOT_THRESHOLD: u64 = 10;
    let should = {
        let s = store.read().await;
        s.should_snapshot(SNAPSHOT_THRESHOLD)
    };
    if should {
        save_snapshot(graph, hdc, store).await?;
    }
    Ok(())
}

/// Save a full snapshot (graph + HDC state).
async fn save_snapshot(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    store: &Arc<RwLock<Store>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let g = graph.read().await;
    let h = hdc.read().await;
    let graph_json = serde_json::to_string(g.inner())?;
    let hdc_snapshot = h.to_snapshot();
    let hdc_json = serde_json::to_string(&hdc_snapshot)?;
    let stats = g.stats();
    let mut s = store.write().await;
    s.write_snapshot(&graph_json, Some(&hdc_json), stats.nodes, stats.edges)?;
    eprintln!("{} Snapshot saved ({} nodes, {} edges, HDC vocab={})",
        "✓".green(), stats.nodes, stats.edges, h.vocab_size());
    Ok(())
}

async fn cmd_add(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    store: &Arc<RwLock<Store>>,
    content: &str,
    tags: Option<String>,
    _channel: &str,
    source: &str,
    llm_client: &Option<OllamaClient>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tag_list: Vec<String> = tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let mut pipeline = IngestPipeline::default_config();
    if let Some(ref llm) = llm_client {
        pipeline = pipeline.with_llm(llm.clone());
    }
    let mut g = graph.write().await;
    let result = pipeline.ingest_text(content, &mut g, source)?;

    // If no triplets extracted, create a direct node
    if result.triplets_extracted == 0 {
        let label = if content.len() > 80 {
            format!("{}...", &content[..77])
        } else {
            content.to_string()
        };
        g.upsert_node_with_tags(&label, NodeKind::Event, tag_list.clone());
    }
    drop(g);

    // Auto-train HDC on new labels (train them all as sentences for context)
    let new_labels: Vec<String> = result.created_nodes.iter().map(|n| n.label.clone()).collect();
    let hdc_trained = if !new_labels.is_empty() {
        let mut h = hdc.write().await;
        let vocab_before = h.vocab_size();
        h.train(&new_labels);
        h.vocab_size() > vocab_before
    } else {
        false
    };

    // Write all mutations to WAL
    {
        let g = graph.read().await;
        let mut s = store.write().await;
        for node in &result.created_nodes {
            s.write_wal(&soma_store::WalEntry::NodeUpsert(node.clone()))?;
        }
        for edge in &result.created_edges {
            s.write_wal(&soma_store::WalEntry::EdgeUpsert(edge.clone()))?;
        }
        // If no triplets, write the fallback Event node
        if result.triplets_extracted == 0 {
            let node = soma_core::SomaNode::new(g.workspace(), content, NodeKind::Event)
                .with_tags(tag_list);
            s.write_wal(&soma_store::WalEntry::NodeUpsert(node))?;
        }
    }

    // Force snapshot if HDC was trained (ensures vocab persists between invocations)
    if hdc_trained {
        save_snapshot(graph, hdc, store).await?;
    }

    eprintln!(
        "{} Added: {} triplets, {} nodes, {} edges",
        "✓".green(),
        result.triplets_extracted,
        result.nodes_created,
        result.edges_created
    );
    Ok(())
}

async fn cmd_ingest(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    store: &Arc<RwLock<Store>>,
    file: &std::path::Path,
    llm_client: &Option<OllamaClient>,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = IngestSource::File(file.to_path_buf());
    let mut pipeline = IngestPipeline::default_config();
    if let Some(ref llm) = llm_client {
        pipeline = pipeline.with_llm(llm.clone());
    }
    let mut g = graph.write().await;

    let start = std::time::Instant::now();
    let result = pipeline.ingest(&source, &mut g, &file.to_string_lossy())?;
    let elapsed = start.elapsed();
    drop(g);

    // Auto-train HDC on new labels
    let new_labels: Vec<String> = result.created_nodes.iter().map(|n| n.label.clone()).collect();
    let hdc_trained = if !new_labels.is_empty() {
        let mut h = hdc.write().await;
        let vocab_before = h.vocab_size();
        h.train(&new_labels);
        h.vocab_size() > vocab_before
    } else {
        false
    };

    // Write all mutations to WAL
    {
        let mut s = store.write().await;
        for node in &result.created_nodes {
            s.write_wal(&soma_store::WalEntry::NodeUpsert(node.clone()))?;
        }
        for edge in &result.created_edges {
            s.write_wal(&soma_store::WalEntry::EdgeUpsert(edge.clone()))?;
        }
    }

    // Force snapshot if HDC was trained
    if hdc_trained {
        save_snapshot(graph, hdc, store).await?;
    }

    eprintln!(
        "{} Ingested {}: {} chunks, {} triplets, {} nodes, {} edges ({:.0}ms)",
        "✓".green(),
        file.display(),
        result.chunks_processed,
        result.triplets_extracted,
        result.nodes_created,
        result.edges_created,
        elapsed.as_millis()
    );
    Ok(())
}

async fn cmd_search(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    query: &str,
    limit: usize,
    channel: Option<String>,
    max_hops: u8,
    no_semantic: bool,
    json_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let channels: Vec<Channel> = channel
        .as_deref()
        .and_then(Channel::from_str_name)
        .map(|c| vec![c])
        .unwrap_or_default();

    let q = SomaQuery::new(query)
        .with_max_hops(max_hops)
        .with_channels(channels)
        .with_limit(limit);

    let g = graph.read().await;
    let start = std::time::Instant::now();

    // Path 1: Graph BFS traverse
    let graph_results = g.traverse(&q);
    let graph_list: Vec<(String, f32)> = graph_results
        .iter()
        .map(|r| (r.node.label.clone(), r.score))
        .collect();

    if no_semantic {
        // Legacy mode: graph-only results
        let elapsed = start.elapsed();
        display_search_results(query, &graph_results, elapsed);
        return Ok(());
    }

    // Path 2: HDC/Neural semantic search
    let all_labels = g.all_labels();
    let h = hdc.read().await;
    let hdc_list = h.search_labels(query, &all_labels, limit);

    // Path 3: Fuzzy label search
    let fuzzy_list = fuzzy_label_search(query, &all_labels, limit);

    drop(h);
    drop(g);

    // RRF Merge with source tracking
    let ranked_lists: Vec<(&str, Vec<(String, f32)>)> = vec![
        ("graph", graph_list),
        ("hdc", hdc_list),
        ("fuzzy", fuzzy_list),
    ];
    let hybrid_results = rrf_merge_with_sources(&ranked_lists, 60.0);
    let elapsed = start.elapsed();

    // Build final QueryResult list with source attribution
    let g = graph.read().await;
    let mut final_results: Vec<soma_core::QueryResult> = Vec::new();

    for hr in hybrid_results.iter().take(limit) {
        // Try to find the node in graph
        if let Some(node) = g.get_node_by_label(&hr.label) {
            // Find matching graph result for hops/path info
            let graph_match = graph_results.iter().find(|r| r.node.label == hr.label);
            let (path, hops) = match graph_match {
                Some(gr) => (gr.path.clone(), gr.hops),
                None => (Vec::new(), 0),
            };

            let qr = soma_core::QueryResult::new(node.clone(), path, hr.score, hops)
                .with_sources(hr.sources.clone());
            final_results.push(qr);
        }
    }

    if json_mode {
        let json_results: Vec<serde_json::Value> = final_results.iter().map(|r| {
            serde_json::json!({
                "label": r.node.label,
                "kind": r.node.kind.as_str(),
                "score": r.score,
                "hops": r.hops,
                "sources": r.sources,
                "tags": r.node.tags,
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&json_results)?);
    } else if final_results.is_empty() {
        eprintln!("{} No results for '{}'", "✗".red(), query);
    } else {
        eprintln!(
            "{} {} results for '{}' ({:.1}ms) {}\n",
            "●".cyan(),
            final_results.len(),
            query,
            elapsed.as_secs_f64() * 1000.0,
            "[hybrid: graph+hdc+fuzzy]".dimmed(),
        );
        for (i, r) in final_results.iter().enumerate() {
            let source_str = if r.sources.is_empty() {
                String::new()
            } else {
                format!(" [{}]", r.sources.join("+"))
            };
            let tags_str = if r.node.tags.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", r.node.tags.join(", "))
            };
            println!(
                "  {} {} {} (score={:.4}, hops={}){}{}",
                format!("{}.", i + 1).dimmed(),
                r.node.label,
                format!("({})", r.node.kind).dimmed(),
                r.score,
                r.hops,
                source_str.yellow(),
                tags_str.dimmed(),
            );
        }
    }
    Ok(())
}

fn display_search_results(
    query: &str,
    results: &[soma_core::QueryResult],
    elapsed: std::time::Duration,
) {
    if results.is_empty() {
        eprintln!("{} No results for '{}'", "✗".red(), query);
    } else {
        eprintln!(
            "{} {} results for '{}' ({:.1}ms) {}\n",
            "●".cyan(),
            results.len(),
            query,
            elapsed.as_secs_f64() * 1000.0,
            "[graph-only]".dimmed(),
        );
        for (i, r) in results.iter().enumerate() {
            let channel_str = if r.path.is_empty() {
                "start".to_string()
            } else {
                r.path
                    .last()
                    .map(|e| e.channel.to_string())
                    .unwrap_or_default()
            };
            let tags_str = if r.node.tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", r.node.tags.join(", "))
            };
            println!(
                "  {} {} {} (score={:.3}, hops={}, {}){}",
                format!("{}.", i + 1).dimmed(),
                r.node.label,
                format!("({})", r.node.kind).dimmed(),
                r.score,
                r.hops,
                channel_str.yellow(),
                tags_str.dimmed(),
            );
        }
    }
}

async fn cmd_relate(
    graph: &Arc<RwLock<StigreGraph>>,
    store: &Arc<RwLock<Store>>,
    from: &str,
    to: &str,
    channel: &str,
    confidence: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let ch = Channel::from_str_name(channel)
        .ok_or_else(|| format!("unknown channel: {}", channel))?;

    let mut g = graph.write().await;
    let from_id = g.upsert_node(from, NodeKind::Entity);
    let to_id = g.upsert_node(to, NodeKind::Entity);

    if let Some(edge_id) = g.upsert_edge(from_id, to_id, ch, confidence, "cli:relate") {
        // Write to WAL
        let edge = soma_core::StigreEdge::new(from_id, to_id, ch, confidence, "cli:relate".into());
        let mut s = store.write().await;
        s.write_wal(&soma_store::WalEntry::EdgeUpsert(edge))?;

        eprintln!(
            "{} {} --[{} {:.1}]--> {}  ({})",
            "✓".green(),
            from,
            channel,
            confidence,
            to,
            edge_id
        );
    }
    Ok(())
}

async fn cmd_stats(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    json_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let g = graph.read().await;
    let stats = g.stats();
    let h = hdc.read().await;

    if json_mode {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "workspace": stats.workspace,
            "nodes": stats.nodes,
            "edges": stats.edges,
            "dead_edges": stats.dead_edges,
            "avg_intensity": stats.avg_intensity,
            "hdc_vocab": h.vocab_size(),
            "hdc_dim": h.dim(),
            "neural_embeddings": h.neural_count(),
            "neural_dim": h.neural_dim(),
        }))?);
    } else {
        println!("{}", "═══ SOMA Stats ═══".cyan().bold());
        println!("  Workspace:      {}", stats.workspace);
        println!("  Nodes:          {}", stats.nodes);
        println!("  Edges:          {}", stats.edges);
        println!("  Dead edges:     {}", stats.dead_edges);
        println!(
            "  Avg intensity:  {:.3}",
            stats.avg_intensity
        );
        println!("  HDC vocab:      {}", h.vocab_size());
        println!("  HDC dim:        {}", h.dim());
        println!("  Neural emb:     {}", h.neural_count());
        if let Some(nd) = h.neural_dim() {
            println!("  Neural dim:     {}", nd);
        }
    }
    Ok(())
}

async fn cmd_workspace(
    action: &str,
    name: Option<&str>,
    data_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    match action {
        "list" => {
            let workspaces = Store::list_workspaces(data_dir)?;
            if workspaces.is_empty() {
                println!("  (no workspaces yet)");
            } else {
                for ws in &workspaces {
                    println!("  - {}", ws);
                }
            }
        }
        "create" => {
            let name = name.ok_or("workspace name required")?;
            Store::open(data_dir, name)?;
            eprintln!("{} Created workspace '{}'", "✓".green(), name);
        }
        _ => {
            eprintln!("Unknown workspace action: {}", action);
        }
    }
    Ok(())
}

async fn cmd_embed(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    store: &Arc<RwLock<Store>>,
    llm_client: &Option<OllamaClient>,
    incremental: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(ref llm) = llm_client else {
        eprintln!("{} LLM not enabled or not available", "✗".red());
        eprintln!("  Enable [llm] in soma.toml and ensure Ollama is running");
        return Ok(());
    };

    // Collect all labels from the graph
    let labels = {
        let g = graph.read().await;
        let all_labels: Vec<String> = g.all_labels().into_iter().collect();
        all_labels
    };

    let total = labels.len();
    if total == 0 {
        eprintln!("{} No labels to embed (graph is empty)", "⚠".yellow());
        return Ok(());
    }

    eprintln!(
        "{} Embedding {} labels...",
        "→".cyan(),
        total
    );

    let mut embedded = 0usize;
    let mut skipped = 0usize;

    for label in &labels {
        if incremental {
            let h = hdc.read().await;
            if h.has_neural_embedding(label) {
                skipped += 1;
                continue;
            }
        }

        // Call Ollama embed (blocking) from async context
        let label_clone = label.clone();
        let llm_clone = llm.clone();
        let result = tokio::task::spawn_blocking(move || llm_clone.embed(&label_clone)).await?;

        match result {
            Ok(Some(embedding)) => {
                let embedding_f32: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();
                let mut h = hdc.write().await;
                h.set_neural_embedding(label, embedding_f32);
                embedded += 1;
            }
            Ok(None) => {
                // Ollama returned nothing — skip
            }
            Err(e) => {
                eprintln!("  {} Failed to embed '{}': {}", "⚠".yellow(), label, e);
            }
        }

        if embedded > 0 && embedded % 50 == 0 {
            eprintln!("  ... {}/{} embedded", embedded, total);
        }
    }

    eprintln!(
        "{} Embedded {}/{} labels{}",
        "✓".green(),
        embedded,
        total,
        if skipped > 0 {
            format!(" ({} skipped, already embedded)", skipped)
        } else {
            String::new()
        }
    );

    // Persist HDC snapshot (includes neural embeddings)
    if embedded > 0 {
        let h = hdc.read().await;
        let hdc_snapshot = h.to_snapshot();
        let hdc_json = serde_json::to_string(&hdc_snapshot)?;
        let g = graph.read().await;
        let graph_json = serde_json::to_string(g.inner())?;
        let stats = g.stats();
        let mut s = store.write().await;
        s.write_snapshot(&graph_json, Some(&hdc_json), stats.nodes, stats.edges)?;
        eprintln!("{} HDC snapshot saved (with neural embeddings)", "✓".green());
    }

    Ok(())
}

async fn cmd_show(
    graph: &Arc<RwLock<StigreGraph>>,
    label: &str,
    json_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let g = graph.read().await;
    let Some(node) = g.get_node_by_label(label) else {
        eprintln!("{} Node '{}' not found", "✗".red(), label);
        return Ok(());
    };

    let node_id = node.id;
    let outgoing = g.outgoing_edges(node_id);
    let incoming = g.incoming_edges(node_id);

    if json_mode {
        let out_json: Vec<serde_json::Value> = outgoing.iter().map(|e| {
            let to_label = g.get_node(e.to).map(|n| n.label.as_str()).unwrap_or("?");
            serde_json::json!({
                "to": to_label,
                "channel": e.channel.to_string(),
                "intensity": e.confidence,
                "uses": e.uses,
                "source": e.source,
            })
        }).collect();
        let in_json: Vec<serde_json::Value> = incoming.iter().map(|e| {
            let from_label = g.get_node(e.from).map(|n| n.label.as_str()).unwrap_or("?");
            serde_json::json!({
                "from": from_label,
                "channel": e.channel.to_string(),
                "intensity": e.confidence,
                "uses": e.uses,
                "source": e.source,
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "label": node.label,
            "kind": node.kind.as_str(),
            "created_at": node.created_at.to_rfc3339(),
            "last_seen": node.last_seen.to_rfc3339(),
            "tags": node.tags,
            "outgoing": out_json,
            "incoming": in_json,
        }))?);
    } else {
        println!("{}", format!("═══ {} ═══", node.label).cyan().bold());
        println!("  Kind:       {}", node.kind);
        println!("  Created:    {}", node.created_at.format("%Y-%m-%d %H:%M"));
        println!("  Last seen:  {}", node.last_seen.format("%Y-%m-%d %H:%M"));
        if !node.tags.is_empty() {
            println!("  Tags:       {}", node.tags.join(", "));
        }

        if !outgoing.is_empty() {
            println!("\n  {} ({}):", "Outgoing".yellow(), outgoing.len());
            for e in &outgoing {
                let to_label = g.get_node(e.to).map(|n| n.label.as_str()).unwrap_or("?");
                println!(
                    "    → {} [{}] intensity={:.2} uses={} ({})",
                    to_label, e.channel, e.confidence, e.uses, e.source
                );
            }
        }

        if !incoming.is_empty() {
            println!("\n  {} ({}):", "Incoming".yellow(), incoming.len());
            for e in &incoming {
                let from_label = g.get_node(e.from).map(|n| n.label.as_str()).unwrap_or("?");
                println!(
                    "    ← {} [{}] intensity={:.2} uses={} ({})",
                    from_label, e.channel, e.confidence, e.uses, e.source
                );
            }
        }

        if outgoing.is_empty() && incoming.is_empty() {
            println!("\n  {}", "(no edges)".dimmed());
        }
    }
    Ok(())
}

async fn cmd_list(
    graph: &Arc<RwLock<StigreGraph>>,
    kind: Option<String>,
    tag: Option<String>,
    limit: usize,
    json_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let g = graph.read().await;

    let kind_filter = kind.as_deref().and_then(NodeKind::from_str_name);
    let mut nodes: Vec<_> = g.all_nodes()
        .filter(|n| {
            if let Some(k) = &kind_filter {
                if n.kind != *k { return false; }
            }
            if let Some(ref t) = tag {
                if !n.tags.iter().any(|nt| nt == t) { return false; }
            }
            true
        })
        .collect();

    // Sort by last_seen descending
    nodes.sort_by(|a, b| b.last_seen.cmp(&a.last_seen));
    nodes.truncate(limit);

    if json_mode {
        let json_nodes: Vec<serde_json::Value> = nodes.iter().map(|n| {
            serde_json::json!({
                "label": n.label,
                "kind": n.kind.as_str(),
                "created_at": n.created_at.to_rfc3339(),
                "last_seen": n.last_seen.to_rfc3339(),
                "tags": n.tags,
            })
        }).collect();
        println!("{}", serde_json::to_string_pretty(&json_nodes)?);
    } else {
        if nodes.is_empty() {
            eprintln!("{} No nodes found", "✗".red());
        } else {
            println!("{} ({} nodes)\n", "═══ SOMA Nodes ═══".cyan().bold(), nodes.len());
            for n in &nodes {
                let tags_str = if n.tags.is_empty() {
                    String::new()
                } else {
                    format!(" {{{}}}", n.tags.join(", "))
                };
                println!(
                    "  {} {} {}{}",
                    n.label,
                    format!("({})", n.kind).dimmed(),
                    n.last_seen.format("%Y-%m-%d").to_string().dimmed(),
                    tags_str.dimmed(),
                );
            }
        }
    }
    Ok(())
}

async fn cmd_export(
    graph: &Arc<RwLock<StigreGraph>>,
    format: &str,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let g = graph.read().await;

    let content = match format {
        "json" => {
            serde_json::to_string_pretty(g.inner())?
        }
        "dot" => {
            let mut lines = vec!["digraph soma {".to_string()];
            lines.push("  rankdir=LR;".to_string());
            for node in g.all_nodes() {
                lines.push(format!(
                    "  \"{}\" [label=\"{}\\n({})\"];",
                    node.label, node.label, node.kind
                ));
            }
            for edge in g.all_edges() {
                let from_label = g.get_node(edge.from).map(|n| n.label.as_str()).unwrap_or("?");
                let to_label = g.get_node(edge.to).map(|n| n.label.as_str()).unwrap_or("?");
                lines.push(format!(
                    "  \"{}\" -> \"{}\" [label=\"{} ({:.2})\"];",
                    from_label, to_label, edge.channel, edge.confidence
                ));
            }
            lines.push("}".to_string());
            lines.join("\n")
        }
        "csv" => {
            let mut lines = vec!["from,to,channel,intensity,uses,source".to_string()];
            for edge in g.all_edges() {
                let from_label = g.get_node(edge.from).map(|n| n.label.as_str()).unwrap_or("?");
                let to_label = g.get_node(edge.to).map(|n| n.label.as_str()).unwrap_or("?");
                lines.push(format!(
                    "\"{}\",\"{}\",{},{:.4},{},\"{}\"",
                    from_label, to_label, edge.channel, edge.confidence, edge.uses, edge.source
                ));
            }
            lines.join("\n")
        }
        _ => {
            return Err(format!("unknown export format: {} (use json, dot, csv)", format).into());
        }
    };

    if let Some(path) = output {
        std::fs::write(&path, &content)?;
        eprintln!("{} Exported to {} ({} format)", "✓".green(), path.display(), format);
    } else {
        println!("{}", content);
    }
    Ok(())
}

async fn cmd_forget(
    graph: &Arc<RwLock<StigreGraph>>,
    store: &Arc<RwLock<Store>>,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut g = graph.write().await;
    if let Some(node_id) = g.remove_node_by_label(label) {
        let mut s = store.write().await;
        s.write_wal(&soma_store::WalEntry::NodeArchive(node_id))?;
        eprintln!("{} Archived '{}' ({})", "✓".green(), label, node_id);
    } else {
        eprintln!("{} Node '{}' not found", "✗".red(), label);
    }
    Ok(())
}

async fn cmd_alarm(
    graph: &Arc<RwLock<StigreGraph>>,
    store: &Arc<RwLock<Store>>,
    label: &str,
    reason: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut g = graph.write().await;
    let entity_id = g.upsert_node(label, NodeKind::Entity);
    let warning_id = g.upsert_node(reason, NodeKind::Warning);
    let edge_id = g.upsert_edge(entity_id, warning_id, Channel::Alarm, 0.9, "cli:alarm");

    // Write to WAL
    {
        let entity = soma_core::SomaNode::new(g.workspace(), label, NodeKind::Entity);
        let warning = soma_core::SomaNode::new(g.workspace(), reason, NodeKind::Warning);
        let mut s = store.write().await;
        s.write_wal(&soma_store::WalEntry::NodeUpsert(entity))?;
        s.write_wal(&soma_store::WalEntry::NodeUpsert(warning))?;
        if let Some(_eid) = edge_id {
            let edge = soma_core::StigreEdge::new(
                entity_id, warning_id, Channel::Alarm, 0.9, "cli:alarm".into(),
            );
            s.write_wal(&soma_store::WalEntry::EdgeUpsert(edge))?;
        }
    }

    eprintln!(
        "{} Alarm: {} → {} [{}]",
        "⚠".yellow(),
        label,
        reason,
        edge_id.map(|e| e.to_string()).unwrap_or_default()
    );
    Ok(())
}

async fn cmd_reinforce(
    graph: &Arc<RwLock<StigreGraph>>,
    store: &Arc<RwLock<Store>>,
    from: &str,
    to: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let g = graph.read().await;
    let from_id = g.node_id_by_label(from)
        .ok_or_else(|| format!("node '{}' not found", from))?;

    // Find all edges from → to
    let edges = g.outgoing_edges(from_id);
    let matching: Vec<_> = edges.iter()
        .filter(|e| {
            g.get_node(e.to).map(|n| n.label == to).unwrap_or(false)
        })
        .map(|e| e.id)
        .collect();
    drop(g);

    if matching.is_empty() {
        eprintln!("{} No edges from '{}' to '{}'", "✗".red(), from, to);
        return Ok(());
    }

    let mut g = graph.write().await;
    let mut reinforced = 0;
    for eid in &matching {
        if g.reinforce_edge(*eid) {
            // WAL
            let mut s = store.write().await;
            s.write_wal(&soma_store::WalEntry::EdgeReinforce {
                id: *eid,
                delta: 0.1,
                ts: Utc::now(),
            })?;
            reinforced += 1;
        }
    }
    eprintln!("{} Reinforced {} edge(s) from '{}' to '{}'", "✓".green(), reinforced, from, to);
    Ok(())
}

async fn cmd_context(
    graph: &Arc<RwLock<StigreGraph>>,
    hdc: &Arc<RwLock<HdcEngine>>,
    query: &str,
    json_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let g = graph.read().await;
    let q = SomaQuery::new(query)
        .with_max_hops(3)
        .with_min_intensity(0.1)
        .with_limit(20);

    // Path 1: Graph BFS
    let graph_results = g.traverse(&q);
    let graph_list: Vec<(String, f32)> = graph_results
        .iter()
        .map(|r| (r.node.label.clone(), r.score))
        .collect();

    // Path 2: HDC/Neural
    let all_labels = g.all_labels();
    let h = hdc.read().await;
    let hdc_list = h.search_labels(query, &all_labels, 20);
    drop(h);

    // Path 3: Fuzzy
    let fuzzy_list = fuzzy_label_search(query, &all_labels, 20);

    // RRF Merge
    let ranked_lists: Vec<(&str, Vec<(String, f32)>)> = vec![
        ("graph", graph_list),
        ("hdc", hdc_list),
        ("fuzzy", fuzzy_list),
    ];
    let hybrid_results = rrf_merge_with_sources(&ranked_lists, 60.0);

    // Format as LLM-ready context
    let now = Utc::now().format("%Y-%m-%d");
    let workspace = g.workspace();

    let mut lines = vec![
        format!("[SOMA MEMORY — {} — workspace: {}]", now, workspace),
        String::new(),
    ];

    let facts: Vec<_> = hybrid_results.iter()
        .take(20)
        .filter_map(|hr| {
            g.get_node_by_label(&hr.label).map(|node| (node, hr))
        })
        .collect();

    if !facts.is_empty() {
        lines.push("Relevant facts (hybrid: graph+hdc+fuzzy):".to_string());
        for (node, hr) in &facts {
            let source_str = format!("[{}]", hr.sources.join("+"));
            lines.push(format!(
                "  - {} ({}, score={:.3}) {}",
                node.label, node.kind, hr.score, source_str,
            ));
        }
    }

    // Alarms
    let alarms: Vec<_> = g.all_nodes()
        .filter(|n| n.kind == NodeKind::Warning)
        .collect();
    if !alarms.is_empty() {
        lines.push(String::new());
        lines.push("Active alarms:".to_string());
        for a in &alarms {
            lines.push(format!("  ! {}", a.label));
        }
    }

    let context = lines.join("\n");

    if json_mode {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "context": context,
            "facts": facts.len(),
            "alarms": alarms.len(),
        }))?);
    } else {
        println!("{}", context);
    }
    Ok(())
}

async fn cmd_daemon(
    graph: Arc<RwLock<StigreGraph>>,
    hdc: Arc<RwLock<HdcEngine>>,
    store: Arc<RwLock<Store>>,
    config: &SomaConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("{}", "═══ SOMA Daemon ═══".cyan().bold());
    eprintln!("  Biological scheduler: active");
    eprintln!("  Evaporation:    every 1h");
    eprintln!("  Physarum:       every {}h", config.bio.physarum_interval_hours);
    eprintln!("  Consolidation:  every {}h", config.bio.consolidation_interval_hours);
    eprintln!("  Pruning:        every {}h", config.bio.pruning_interval_hours);
    eprintln!("  Press Ctrl+C for graceful shutdown");
    eprintln!();

    let bio_config = BioConfig::from_soma_config(config);
    let scheduler = BioScheduler::new(bio_config);

    // Run biological loops (returns on Ctrl+C)
    scheduler.run(graph.clone(), store.clone()).await;

    // Graceful shutdown: save final snapshot
    eprintln!("{} Saving final snapshot...", "→".cyan());
    save_snapshot(&graph, &hdc, &store).await?;
    eprintln!("{} Daemon shutdown complete", "✓".green());

    Ok(())
}
