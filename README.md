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
  <a href="#testing"><img src="https://img.shields.io/badge/tests-148%20passing-brightgreen.svg" alt="Tests: 148 passing"/></a>
  <a href="#ruler-multi-needle-retrieval"><img src="https://img.shields.io/badge/RULER-10%2F10-brightgreen.svg" alt="RULER: 10/10"/></a>
  <a href="#mcp-integration"><img src="https://img.shields.io/badge/MCP-10%20tools-8A2BE2.svg" alt="MCP: 10 tools"/></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/crates-13-blue.svg" alt="13 crates"/></a>
  <img src="https://img.shields.io/badge/cloud%20dependencies-0-critical.svg" alt="Zero cloud deps"/>
  <a href="#supported-platforms"><img src="https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos%20%7C%20wasm-lightgrey.svg" alt="Platforms"/></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#cli-reference">CLI</a> &bull;
  <a href="#rest-api">REST API</a> &bull;
  <a href="#mcp-integration">MCP</a> &bull;
  <a href="#docker">Docker</a> &bull;
  <a href="#python-client">Python</a> &bull;
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
- **REST API** -- 13 HTTP endpoints (axum) exposing every operation. Start with `soma daemon --http 8080`.
- **Web Dashboard** -- Built-in D3.js force-directed graph visualization at `http://localhost:8080/`.
- **Watch Mode** -- Auto-ingest files on change with `soma daemon --watch ./notes`. Supports `.txt`, `.md`, `.json`, `.csv`, `.rs`, `.py`, and more.
- **Complete CLI** -- 13 commands with `--format json` output for scripting and pipelines.
- **Docker Ready** -- Multi-stage Alpine image, single binary, `docker compose up` and go.
- **Python Client** -- `soma-memory` package wraps the REST API with a clean Pythonic interface.
- **Graceful Shutdown** -- The daemon catches `Ctrl+C` and writes a final snapshot before exiting.
- **Config Validation** -- Invalid settings are caught and rejected at startup with clear error messages.

## Quick Start

### Install from Source

```bash
git clone https://github.com/yannbanas/soma.git
cd soma
cargo build --release
```

The binary is at `target/release/soma` (or `soma.exe` on Windows).

### Install with Docker

```bash
docker compose up -d
# REST API is now available at http://localhost:8080
curl http://localhost:8080/health
```

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
# Bio scheduler only
soma daemon

# Bio scheduler + REST API on port 8080
soma daemon --http 8080

# Bio scheduler + REST API + auto-ingest on file changes
soma daemon --http 8080 --watch ./notes --recursive
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

SOMA is organized as a Cargo workspace with 13 crates:

```
soma-core       Core types, IDs, channels, config, hybrid search (RRF)
soma-graph      petgraph-backed directed graph with O(1) label lookups
soma-hdc        Hyperdimensional computing: Random Indexing, TF-IDF, neural embeddings
soma-store      WAL (append-only + fsync) and zstd-compressed snapshots
soma-bio        Biological scheduler (evaporation, Physarum, consolidation, pruning)
soma-ingest     Text chunker + L0/L1 pattern extraction + L2 Ollama pipeline
soma-llm        Ollama HTTP client for generation and embeddings
soma-mcp        MCP server (stdio + TCP, 10 tools)
soma-http       REST API server (axum, 11 endpoints)
soma-watch      File watcher for auto-ingest on change (notify)
soma-bench      Benchmark suite (RULER, MuSiQue, HotpotQA, ablation)
soma-cli        CLI frontend (13 commands) + daemon entry point
```

### Dependency Graph

```
soma-cli ──→ soma-http ──→ soma-mcp ──→ soma-core
   │              │            │
   ├─→ soma-watch │  soma-bio  │
   │      │       │    │       │
   │      ▼       ▼    ▼       ▼
   ├─→ soma-graph ←── soma-ingest ──→ soma-llm
   │      │                │
   │      ▼                ▼
   └─→ soma-store       soma-hdc       soma-bench
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

## REST API

Start the daemon with `--http` to expose the REST API:

```bash
soma daemon --http 8080
```

| Method | Endpoint     | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/`          | Web dashboard (D3.js graph viz)      |
| GET    | `/health`    | Server health + uptime + graph size  |
| GET    | `/stats`     | Full graph statistics                |
| GET    | `/search`    | Hybrid search (`?q=...&limit=20`)    |
| GET    | `/context`   | LLM-ready context block (`?q=...`)   |
| GET    | `/api/graph` | Full graph data for visualization    |
| POST   | `/add`       | Add text to memory                   |
| POST   | `/ingest`    | Ingest a file by path                |
| POST   | `/relate`    | Create a typed relation              |
| POST   | `/reinforce` | Strengthen an edge                   |
| POST   | `/alarm`     | Attach a warning to an entity        |
| POST   | `/forget`    | Archive (soft-delete) an entity      |
| POST   | `/sleep`     | Trigger manual consolidation         |

```bash
# Examples
curl http://localhost:8080/health
curl "http://localhost:8080/search?q=ChromoQ&limit=5"
curl -X POST http://localhost:8080/add \
  -H "Content-Type: application/json" \
  -d '{"content": "EGFP emits green fluorescence", "tags": ["protein"]}'
```

## Web Dashboard

SOMA includes a built-in web dashboard with a D3.js force-directed graph visualization. Access it at `http://localhost:8080/` when the daemon is running with `--http`.

- **Interactive graph** -- Drag, zoom, and click nodes to explore your knowledge graph
- **Real-time stats** -- Node count, edge count, uptime, and connection status
- **Search** -- Full hybrid search from the sidebar with result highlighting
- **Add knowledge** -- Create nodes, relations, and alarms directly from the UI
- **Filters** -- Toggle node types and set minimum intensity thresholds
- **Detail panel** -- Click any node to see its kind, tags, edges, and actions (reinforce / forget)
- **Edge labels** -- Channel name and intensity displayed on each relation
- **Export** -- Download the graph as JSON, DOT, or CSV
- **Dark theme** -- Modern dark UI with color-coded node types and channels
- **Tools** -- Trigger sleep consolidation and refresh the graph

```bash
soma daemon --http 8080
# Open http://localhost:8080 in your browser
```

## Docker

### Quick Start

```bash
docker compose up -d
```

This starts SOMA with:
- REST API on port **8080**
- MCP TCP on port **4242**
- Persistent data volume `soma-data`

### Build Manually

```bash
docker build -t soma .
docker run -d -p 8080:8080 -v soma-data:/data soma
```

### docker-compose.yml

```yaml
services:
  soma:
    build: .
    ports:
      - "8080:8080"   # REST API
      - "4242:4242"   # MCP TCP
    volumes:
      - soma-data:/data
    environment:
      - SOMA_DATA_DIR=/data
      - RUST_LOG=info
    restart: unless-stopped

volumes:
  soma-data:
```

## Python Client

The `soma-memory` package provides a Python client for the REST API.

### Install

```bash
pip install ./python/soma-memory
```

### Usage

```python
from soma_memory import SomaClient

with SomaClient("http://localhost:8080") as s:
    # Add knowledge
    s.add("CRISPR edits gene X", source="paper", tags=["bio"])

    # Search
    results = s.search("gene editing", limit=10)

    # Get LLM context
    ctx = s.context("gene editing pipeline")

    # Create relations
    s.relate("CRISPR", "gene_X", channel="causal", confidence=0.9)

    # Reinforce, alarm, forget
    s.reinforce("CRISPR", "gene_X")
    s.alarm("gene_X", reason="off-target effects reported")
    s.forget("old-experiment")

    # Trigger consolidation
    s.sleep()

    # Health & stats
    print(s.health())
    print(s.stats())
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

148 tests across all crates:

```bash
cargo test                                  # run all 148 tests
cargo test -p soma-core                     # 41 tests — types, channels, config, hybrid search
cargo test -p soma-bench                    # 30 tests — benchmark metrics, loaders, runners
cargo test -p soma-ingest                   # 24 tests — chunker, NER, pattern extraction
cargo test -p soma-graph                    # 19 tests — graph operations, traversal
cargo test -p soma-hdc                      # 11 tests — HDC + neural embeddings
cargo test -p soma-llm                      # 11 tests — Ollama client
cargo test -p soma-store                    #  8 tests — WAL + snapshots
cargo test -p soma-bio                      #  2 tests — scheduler
cargo test -p soma-watch                    #  1 test  — file extension filtering
```

## Supported Platforms

- Linux x86_64
- Linux aarch64 (ARM64)
- Windows x86_64
- macOS (x86_64, ARM64)
- WASM (via wasm32 target)

## License

[MIT](LICENSE) -- Yann Banas
