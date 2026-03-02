<p align="center">
  <img src="logo_soma.png" alt="SOMA Logo"/>
</p>

<p align="center">
  <strong>Stigmergic Ontological Memory Architecture</strong><br/>
  A persistent, biologically-inspired knowledge graph for AI agents.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-pure-orange.svg?logo=rust" alt="Rust"/></a>
  <a href="#testing"><img src="https://img.shields.io/badge/tests-109%20passing-brightgreen.svg" alt="Tests: 109 passing"/></a>
  <a href="#ruler-multi-needle-retrieval"><img src="https://img.shields.io/badge/RULER-10%2F10-brightgreen.svg" alt="RULER: 10/10"/></a>
  <a href="#mcp-integration"><img src="https://img.shields.io/badge/MCP-10%20tools-8A2BE2.svg" alt="MCP: 10 tools"/></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/crates-9-blue.svg" alt="9 crates"/></a>
  <img src="https://img.shields.io/badge/cloud%20dependencies-0-critical.svg" alt="Zero cloud deps"/>
  <a href="#supported-platforms"><img src="https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos%20%7C%20wasm-lightgrey.svg" alt="Platforms"/></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#cli-reference">CLI</a> &bull;
  <a href="#mcp-integration">MCP</a> &bull;
  <a href="#configuration">Config</a>
</p>

---

SOMA gives your AI agents **long-term memory that persists across sessions**. It ingests text, extracts structured relationships (via pattern matching or a local LLM), stores them in a knowledge graph with biologically-modeled decay, and serves them over [MCP](https://modelcontextprotocol.io/) or a full-featured CLI.

Built entirely in Rust. Zero cloud dependencies. Single binary. Cold start under 50 ms.

## Why SOMA?

Every conversation with an AI agent starts from zero. Context is lost between sessions, insights are forgotten, and you end up re-explaining the same things over and over.

SOMA solves this by acting as **shared, persistent memory** that any agent can read from and write to:

- **Claude** queries SOMA via MCP to recall decisions made weeks ago
- **Ollama agents** store discoveries and retrieve prior context on startup
- **You** search your entire knowledge base from the terminal with `soma search`

One tool. All your memory. Fully local.

## Features

- **Living Knowledge Graph** -- Edges carry intensity that decays over time (stigmergic evaporation). Frequently accessed knowledge strengthens; unused knowledge naturally fades.
- **Hybrid Search** -- Three-path retrieval combining graph BFS, HDC cosine similarity, and fuzzy label matching, merged via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
- **Pattern-Based Extraction** -- 15 regex patterns extract subject-relation-object triplets from plain text, no LLM required.
- **Optional LLM Extraction** -- When patterns aren't enough, SOMA delegates to a local Ollama model for deeper extraction. Privacy-first: nothing leaves your machine.
- **Hyperdimensional Computing** -- Random Indexing with D=10,000 dimensions, TF-IDF weighting, and optional neural embeddings via Ollama for semantic search.
- **Biological Scheduler** -- Four async background loops model real neural processes: synaptic evaporation, Physarum-inspired path optimization, sleep consolidation, and pruning.
- **Crash-Safe Persistence** -- Append-only WAL with `fsync` on every write, plus zstd-compressed snapshots with atomic rename. No data loss on crash or power failure.
- **MCP Protocol** -- 10 tools over JSON-RPC 2.0 (stdio or TCP) for seamless AI agent integration.
- **Complete CLI** -- 13 commands with `--format json` output for scripting and pipelines.
- **Graceful Shutdown** -- The daemon catches `Ctrl+C` and writes a final snapshot before exiting.
- **Config Validation** -- Invalid settings are caught and rejected at startup with clear error messages.

## Quick Start

### Install

```bash
git clone https://github.com/yannbanas/soma.git
cd soma
cargo build --release
```

The binary is at `target/release/soma` (or `soma.exe` on Windows).

### Basic Usage

```bash
# Add knowledge
soma add "ChromoQ derives from EGFP through directed evolution"

# Search (hybrid: graph + HDC + fuzzy, merged via RRF)
soma search ChromoQ

# Inspect a node and all its connections
soma show ChromoQ

# List all nodes, with optional filters
soma list
soma list --kind entity --tag protein

# Create an explicit relation
soma relate ChromoQ EGFP -c derives_de --confidence 0.95

# Reinforce a relation (increases edge intensity)
soma reinforce ChromoQ EGFP

# Flag something with a permanent warning
soma alarm EGFP "unstable at pH < 6.5"

# Archive a node you no longer need
soma forget "old-experiment"

# Get an LLM-ready context block for a topic
soma context "fluorescent protein pipeline"

# Ingest a text file (chunked, with automatic triplet extraction)
soma ingest -f corpus.txt

# Export the full graph
soma export --format dot -o graph.dot
soma export --format csv -o edges.csv

# View statistics
soma stats
```

### JSON Output

All read commands support `--format json` for machine-readable output:

```bash
soma search ChromoQ --format json
soma stats --format json
soma list --format json
soma show ChromoQ --format json
```

### Daemon Mode

Start the biological scheduler as a long-running process. It handles evaporation, consolidation, and pruning in the background. Press `Ctrl+C` for a graceful shutdown with a final snapshot.

```bash
soma daemon
```

### MCP Server

Expose SOMA as an MCP tool server for AI agents:

```bash
# stdio transport (for Claude Desktop, etc.)
soma mcp-stdio

# TCP transport (for networked agents)
soma mcp-tcp --port 3333
```

## Architecture

SOMA is organized as a Cargo workspace with 9 crates:

```
soma-core       Core types, IDs, channels, config, hybrid search (RRF)
soma-graph      petgraph-backed directed graph with O(1) label lookups
soma-hdc        Hyperdimensional computing: Random Indexing, TF-IDF, neural embeddings
soma-store      WAL (append-only + fsync) and zstd-compressed snapshots
soma-bio        Biological scheduler (evaporation, Physarum, consolidation, pruning)
soma-ingest     Text chunker + L0/L1 pattern extraction + L2 Ollama pipeline
soma-llm        Ollama HTTP client for generation and embeddings
soma-mcp        MCP server (stdio + TCP, 10 tools)
soma-cli        CLI frontend (13 commands) + daemon entry point
```

### Dependency Graph

```
soma-cli ─→ soma-mcp ─→ soma-core
   │            │
   ├─→ soma-bio │
   │      │     │
   │      ▼     ▼
   ├─→ soma-graph ←─ soma-ingest ─→ soma-llm
   │      │                │
   │      ▼                ▼
   └─→ soma-store      soma-hdc
```

## CLI Reference

| Command     | Description                                       |
|-------------|---------------------------------------------------|
| `add`       | Add text or a note to memory                      |
| `ingest`    | Ingest a file (chunked, with triplet extraction)  |
| `search`    | Hybrid search the knowledge graph                 |
| `show`      | Inspect a node and its neighbors                  |
| `list`      | List nodes (filterable by `--kind` and `--tag`)   |
| `relate`    | Create a typed relation between two entities      |
| `reinforce` | Strengthen the edge between two entities          |
| `alarm`     | Attach a permanent warning to an entity           |
| `forget`    | Archive (soft-delete) a node                      |
| `context`   | Retrieve an LLM-ready context block               |
| `export`    | Export the graph as JSON, DOT, or CSV             |
| `stats`     | Display graph statistics                          |
| `daemon`    | Run the biological scheduler in the foreground    |

## MCP Integration

SOMA implements the [Model Context Protocol](https://modelcontextprotocol.io/) and exposes 10 tools over JSON-RPC 2.0:

| Tool              | Description                                    |
|-------------------|------------------------------------------------|
| `soma_add`        | Add text to memory (with optional L2 LLM extraction) |
| `soma_ingest`     | Ingest a file                                  |
| `soma_search`     | Hybrid search (graph + HDC + fuzzy via RRF)    |
| `soma_relate`     | Create a relation between entities             |
| `soma_reinforce`  | Strengthen an edge                             |
| `soma_alarm`      | Attach a warning to an entity                  |
| `soma_forget`     | Archive a node                                 |
| `soma_stats`      | Return graph statistics                        |
| `soma_workspace`  | Create or switch workspaces                    |
| `soma_context`    | Retrieve an LLM-ready context block            |

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "soma": {
      "command": "soma",
      "args": ["mcp-stdio"]
    }
  }
}
```

### Claude Code

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "soma": {
      "command": "soma",
      "args": ["mcp-stdio"]
    }
  }
}
```

## Channel Types

SOMA models different kinds of knowledge with distinct decay characteristics:

| Channel      | Decay Rate | Half-life  | Purpose                  |
|--------------|-----------|------------|--------------------------|
| Trail        | 0.001     | ~40 days   | General knowledge        |
| DerivesDe    | 0.0       | Permanent  | Taxonomic / "is-a" links |
| Causal       | 0.005     | ~6 days    | Cause-effect relations   |
| Episodic     | 0.05      | ~14 hours  | Session-level events     |
| Alarm        | 0.0       | Permanent  | Warnings and red flags   |
| SemanticSim  | 0.01      | ~3 days    | HDC similarity links     |

## Configuration

SOMA works with zero configuration. For customization, create `soma.toml` in your project root or at `~/.config/soma/soma.toml`:

```toml
[soma]
data_dir = "~/.local/share/soma"
default_workspace = "default"

[bio]
prune_threshold = 0.05              # edges below this intensity get pruned
physarum_interval_hours = 2
consolidation_interval_hours = 6
pruning_interval_hours = 24

[hdc]
dimension = 10000                   # hypervector dimensionality
window_size = 5                     # n-gram window for encoding
tfidf = true                        # enable TF-IDF weighting

[ingest]
chunk_size = 5                      # lines per chunk
chunk_overlap = 1                   # overlap between chunks

[llm]
enabled = false                     # set true for L2 extraction + neural embeddings
provider = "ollama"
model = "cogito:8b"                 # triplet extraction model
embedding_model = "nomic-embed-text"  # neural embedding model
endpoint = "http://localhost:11434"
timeout_ms = 30000

[mcp]
transport = "stdio"
tcp_port = 3333
```

All values above are defaults. Only override what you need.

## Benchmarks

```bash
cargo bench                        # all benchmarks
cargo bench --bench ruler_bench    # RULER multi-needle retrieval
cargo bench --bench graph_bench    # graph operations
cargo bench --bench hdc_bench      # HDC similarity
cargo bench --bench ingest_bench   # ingestion pipeline
```

### RULER Multi-Needle Retrieval

The [RULER benchmark](https://arxiv.org/abs/2404.06654) tests retrieval of 10 "needle" facts scattered across a 40-paragraph corpus:

| Metric          | Result                 |
|-----------------|------------------------|
| Score           | **10/10** needles found |
| Graph size      | 121 nodes, 66 edges    |
| Ingest latency  | ~195 us (full corpus)  |
| Query latency   | 155 ns -- 7 us/needle  |

## Testing

109 tests across all crates:

```bash
cargo test                                  # run all 109 tests
cargo test -p soma-core                     # 39 tests — types, channels, config, hybrid search
cargo test -p soma-ingest                   # 18 tests — chunker, pattern extraction
cargo test -p soma-cli --test ruler_test    # 13 tests — RULER integration
cargo test -p soma-hdc                      # 11 tests — HDC + neural embeddings
cargo test -p soma-llm                      # 11 tests — Ollama client
cargo test -p soma-store                    #  8 tests — WAL + snapshots
cargo test -p soma-graph                    #  7 tests — graph operations
cargo test -p soma-bio                      #  2 tests — scheduler
```

## Supported Platforms

- Linux x86_64
- Linux aarch64 (ARM64)
- Windows x86_64
- macOS (x86_64, ARM64)
- WASM (via wasm32 target)

## License

[MIT](LICENSE) -- Yann Banas
