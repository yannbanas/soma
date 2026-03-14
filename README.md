<p align="center">
  <img src="logo_soma.png" alt="SOMA Logo"/>
</p>

<p align="center">
  <strong>Stigmergic Ontological Memory Architecture</strong><br/>
  A persistent, biologically-inspired knowledge graph for AI agents.
</p>

<p align="center">
  <a href="https://github.com/yannbanas/soma/actions/workflows/ci.yml"><img src="https://github.com/yannbanas/soma/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"/></a>
  <a href="https://github.com/yannbanas/soma/releases/latest"><img src="https://img.shields.io/github/v/release/yannbanas/soma?include_prereleases&label=release" alt="Release"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/yannbanas/soma" alt="License"/></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-pure-orange.svg?logo=rust" alt="Rust"/></a>
  <a href="#testing"><img src="https://img.shields.io/badge/tests-186%20passing-brightgreen.svg" alt="Tests: 186 passing"/></a>
  <a href="#ruler-multi-needle-retrieval"><img src="https://img.shields.io/badge/RULER-10%2F10-brightgreen.svg" alt="RULER: 10/10"/></a>
  <a href="#mcp-integration"><img src="https://img.shields.io/badge/MCP-19%20tools-8A2BE2.svg" alt="MCP: 19 tools"/></a>
  <a href="#rest-api"><img src="https://img.shields.io/badge/REST-31%20endpoints-blue.svg" alt="REST: 31 endpoints"/></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/crates-14-blue.svg" alt="14 crates"/></a>
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

### Core Graph Engine
- **Living Knowledge Graph** -- Edges carry intensity that decays over time (stigmergic evaporation). Frequently accessed knowledge strengthens; unused knowledge naturally fades.
- **Hybrid Search + Re-ranking** -- Four-path retrieval: graph BFS, HDC cosine, fuzzy matching, and community search, merged via [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) then re-ranked with temporal boost, IDF specificity, and MMR diversification.
- **Community Detection** -- Louvain algorithm detects thematic clusters in the graph (GraphRAG-style). Communities can be summarized via LLM for global search.
- **Native Cypher Queries** -- `MATCH (a)-[r]->(b) WHERE ...` -- subset of Cypher for ad-hoc graph queries from CLI, REST, or MCP.
- **Provenance Tracking** -- Every edge carries a `provenance` field (`Human`, `AiInferred`, `AiValidated`, `Automated`) for trust-aware filtering.

### AI Agent Integration
- **MCP Protocol** -- 19 tools over JSON-RPC 2.0 (stdio or TCP) for seamless AI agent integration.
- **Feedback Loop** -- AI agents can correct (`soma_correct`), validate (`soma_validate`), and merge (`soma_merge`) knowledge directly.
- **Context Compaction** -- `soma_compact` saves session summaries before context window compaction; `soma_session_restore` retrieves them in new sessions.
- **Graph of Thoughts** -- `soma_think` stores reasoning steps as graph nodes with a dedicated `Reasoning` channel.
- **Explainable Paths** -- `soma_explain` finds and displays the K shortest paths between any two entities.

### Ingestion & Extraction
- **Pattern-Based Extraction** -- 15 regex patterns extract subject-relation-object triplets from plain text, no LLM required.
- **Optional LLM Extraction** -- Delegates to a local Ollama model for deeper extraction. Privacy-first: nothing leaves your machine.
- **Code Ingestion** -- Parse Rust, Python, JS/TS, Go, Java, C/C++ files and extract functions, classes, and dependencies as graph nodes.
- **Plugin System** -- `IngestPlugin` trait for extending ingestion to custom formats (CSV, BibTeX, YAML, etc.) without modifying core code.
- **Watch Mode** -- Auto-ingest files on change with `soma daemon --watch ./notes`.

### Infrastructure
- **Hyperdimensional Computing** -- Random Indexing with D=10,000 dimensions, TF-IDF weighting, and optional neural embeddings via Ollama.
- **Biological Scheduler** -- Four async background loops: synaptic evaporation, Physarum-inspired path optimization, sleep consolidation, and pruning.
- **Crash-Safe Persistence** -- Append-only WAL with `fsync` on every write, plus zstd-compressed snapshots with atomic rename.
- **REST API** -- 31 HTTP endpoints (axum) with SSE streaming, webhooks, and multi-tenancy support.
- **Web Dashboard** -- Built-in D3.js force-directed graph visualization.
- **Complete CLI** -- 20 commands with `--format json` output for scripting and pipelines.
- **Docker Ready** -- Multi-stage Alpine image, single binary, `docker compose up` and go.
- **CI/CD** -- GitHub Actions pipeline: lint, test, audit, build, Docker, benchmarks. Multi-platform release builds on tag push.
- **Python Client** -- `soma-memory` package with 22 methods wrapping the full REST API.

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

# Ingest source code (functions, classes, dependencies)
soma ingest-code ./src

# Run a Cypher query
soma cypher 'MATCH (a)-[r]->(b) WHERE a.label = "ChromoQ" RETURN a, r, b'

# Import a previously exported graph
soma import graph.json
soma import graph.json --merge   # merge into existing data

# Generate HDC embeddings
soma embed

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

SOMA is organized as a Cargo workspace with 14 crates:

```
soma-core       Core types, channels, config, hybrid search (RRF + re-ranking + MMR)
soma-graph      petgraph-backed directed graph, O(1) label lookups, Louvain communities
soma-hdc        Hyperdimensional computing: Random Indexing, TF-IDF, neural embeddings
soma-store      WAL (append-only + fsync) and zstd-compressed snapshots
soma-bio        Biological scheduler (evaporation, Physarum, consolidation, pruning)
soma-ingest     Text chunker + pattern extraction + Ollama pipeline + plugin system
soma-llm        Ollama HTTP client for generation and embeddings
soma-mcp        MCP server (stdio + TCP, 19 tools)
soma-http       REST API server (axum, 31 endpoints, SSE, webhooks, multi-tenancy)
soma-watch      File watcher for auto-ingest on change (notify)
soma-cypher     Native Cypher query parser and executor
soma-bench      Benchmark suite (RULER, MuSiQue, HotpotQA, ablation)
soma-cli        CLI frontend (20 commands) + daemon entry point
```

### Dependency Graph

```
soma-cli ──→ soma-http ──→ soma-mcp ──→ soma-core
   │              │            │            │
   ├─→ soma-watch │  soma-bio  │   soma-cypher
   │      │       │    │       │
   │      ▼       ▼    ▼       ▼
   ├─→ soma-graph ←── soma-ingest ──→ soma-llm
   │      │                │
   │      ▼                ▼
   └─→ soma-store       soma-hdc       soma-bench
```

## CLI Reference

| Command       | Description                                       |
|---------------|---------------------------------------------------|
| `add`         | Add text or a note to memory                      |
| `ingest`      | Ingest a file (chunked, with triplet extraction)  |
| `ingest-code` | Parse source files and extract code structure     |
| `search`      | Hybrid search the knowledge graph                 |
| `show`        | Inspect a node and its neighbors                  |
| `list`        | List nodes (filterable by `--kind` and `--tag`)   |
| `relate`      | Create a typed relation between two entities      |
| `reinforce`   | Strengthen the edge between two entities          |
| `alarm`       | Attach a permanent warning to an entity           |
| `forget`      | Archive (soft-delete) a node                      |
| `context`     | Retrieve an LLM-ready context block               |
| `cypher`      | Run a native Cypher query                         |
| `export`      | Export the graph as JSON, DOT, or CSV             |
| `import`      | Import a graph from JSON (`--merge` to upsert)    |
| `embed`       | Generate HDC embeddings (`--incremental`)         |
| `stats`       | Display graph statistics                          |
| `workspace`   | Create or switch workspaces                       |
| `sleep`       | Trigger manual consolidation                      |
| `watch`       | Watch a directory and auto-ingest on change       |
| `daemon`      | Run bio scheduler + optional HTTP + watch         |
| `mcp-stdio`   | Start MCP server on stdio                         |
| `mcp-tcp`     | Start MCP server on TCP                           |

## MCP Integration

SOMA implements the [Model Context Protocol](https://modelcontextprotocol.io/) and exposes 19 tools over JSON-RPC 2.0:

| Tool                    | Description                                           |
|-------------------------|-------------------------------------------------------|
| `soma_add`              | Add text to memory (with optional LLM extraction)     |
| `soma_ingest`           | Ingest a file                                         |
| `soma_search`           | Hybrid search (graph + HDC + fuzzy + community, RRF)  |
| `soma_relate`           | Create a relation between entities                    |
| `soma_reinforce`        | Strengthen an edge                                    |
| `soma_alarm`            | Attach a warning to an entity                         |
| `soma_forget`           | Archive a node                                        |
| `soma_stats`            | Return graph statistics                               |
| `soma_workspace`        | Create or switch workspaces                           |
| `soma_context`          | Retrieve an LLM-ready context block (token budget)    |
| `soma_cypher`           | Execute a native Cypher query                         |
| `soma_correct`          | Lower confidence of an edge (feedback loop)           |
| `soma_validate`         | Confirm an edge and mark as AI-validated              |
| `soma_compact`          | Save session summary before context compaction        |
| `soma_session_restore`  | Restore context from previous session summaries       |
| `soma_explain`          | Find K shortest paths between two entities            |
| `soma_merge`            | Merge duplicate nodes (transfer edges + meta)         |
| `soma_communities`      | Detect and list thematic communities (Louvain)        |
| `soma_think`            | Record a reasoning step in the Graph of Thoughts      |

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

**Core Operations**

| Method | Endpoint           | Description                          |
|--------|--------------------|--------------------------------------|
| GET    | `/`                | Web dashboard (D3.js graph viz)      |
| GET    | `/health`          | Server health + uptime + graph size  |
| GET    | `/stats`           | Full graph statistics                |
| GET    | `/search`          | Hybrid search (`?q=...&limit=20`)    |
| GET    | `/context`         | LLM-ready context block (`?q=...`)   |
| GET    | `/api/graph`       | Full graph data for visualization    |
| POST   | `/add`             | Add text to memory                   |
| POST   | `/ingest`          | Ingest a file by path                |
| POST   | `/ingest-code`     | Parse and ingest source code         |
| POST   | `/relate`          | Create a typed relation              |
| POST   | `/reinforce`       | Strengthen an edge                   |
| POST   | `/alarm`           | Attach a warning to an entity        |
| POST   | `/forget`          | Archive (soft-delete) an entity      |
| POST   | `/sleep`           | Trigger manual consolidation         |
| POST   | `/snapshot`        | Force a snapshot to disk             |
| POST   | `/cypher`          | Execute a Cypher query               |

**AI Feedback & Reasoning**

| Method | Endpoint             | Description                          |
|--------|----------------------|--------------------------------------|
| POST   | `/correct`           | Lower edge confidence (feedback)     |
| POST   | `/validate`          | Confirm edge as AI-validated         |
| POST   | `/compact`           | Save session summary                 |
| GET    | `/session-restore`   | Restore previous session context     |
| GET    | `/explain`           | Find shortest paths between nodes    |
| POST   | `/merge`             | Merge duplicate nodes                |
| GET    | `/communities`       | Detect communities (Louvain)         |
| POST   | `/think`             | Record a reasoning step              |

**Streaming, Webhooks & Multi-tenancy**

| Method | Endpoint             | Description                          |
|--------|----------------------|--------------------------------------|
| GET    | `/search/stream`     | SSE streaming search results         |
| GET    | `/events`            | SSE event stream (graph mutations)   |
| GET    | `/webhooks`          | List registered webhooks             |
| POST   | `/webhooks`          | Register a webhook                   |
| DELETE | `/webhooks/:id`      | Delete a webhook                     |
| GET    | `/tenants`           | List tenants                         |
| POST   | `/tenants`           | Create a new tenant                  |

```bash
# Examples
curl http://localhost:8080/health
curl "http://localhost:8080/search?q=ChromoQ&limit=5"
curl -X POST http://localhost:8080/add \
  -H "Content-Type: application/json" \
  -d '{"content": "EGFP emits green fluorescence", "tags": ["protein"]}'

# SSE streaming search
curl -N "http://localhost:8080/search/stream?q=ChromoQ"

# Cypher query
curl -X POST http://localhost:8080/cypher \
  -d '{"query": "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 10"}'

# Webhooks
curl -X POST http://localhost:8080/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/hook", "events": ["node_added"], "secret": "s3cret"}'
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

    # Get LLM context (with token budget)
    ctx = s.context("gene editing pipeline")

    # Create relations
    s.relate("CRISPR", "gene_X", channel="causal", confidence=0.9)

    # Reinforce, alarm, forget
    s.reinforce("CRISPR", "gene_X")
    s.alarm("gene_X", reason="off-target effects reported")
    s.forget("old-experiment")

    # AI feedback loop
    s.correct("ChromoQ", "stable at all pH", new_confidence=0.2,
              reason="Disproved by PMID:12345")
    s.validate("EGFP", "green fluorescence", source="UniProt P42212")

    # Session memory
    s.compact(summary="Discussed ChromoQ vs EGFP",
              entities=["ChromoQ", "EGFP"],
              decisions=["Use EGFP for acidic pH"])
    s.session_restore(query="ChromoQ EGFP", limit=5)

    # Cypher queries
    s.cypher("MATCH (a)-[r]->(b) RETURN a LIMIT 10")

    # Reasoning & explanation
    s.explain("ChromoQ", "biophotonique", max_paths=3)
    s.think("If ChromoQ unstable at pH<6.5, EGFP better for lysosomes",
            depends_on=["ChromoQ pH unstable", "lysosomes are acidic"])

    # Communities & merge
    s.communities(min_size=3)
    s.merge(keep="ChromoQ", absorb="chromoQ", reason="same entity")

    # Webhooks
    s.register_webhook(url="https://example.com/hook",
                       events=["node_added"])
    s.list_webhooks()

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
| Reasoning    | 0.08      | ~9 hours   | Graph of Thoughts steps  |

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
embedding_model = "embeddinggemma:300m"  # neural embedding model
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

186 tests across all crates:

```bash
cargo test                                  # run all 186 tests
cargo test -p soma-core                     # types, channels, config, hybrid search, re-ranking, provenance
cargo test -p soma-bench                    # benchmark metrics, loaders, runners
cargo test -p soma-ingest                   # chunker, NER, pattern extraction, plugins
cargo test -p soma-graph                    # graph operations, traversal, communities
cargo test -p soma-hdc                      # HDC + neural embeddings
cargo test -p soma-llm                      # Ollama client
cargo test -p soma-cypher                   # Cypher parser and executor
cargo test -p soma-store                    # WAL + snapshots
cargo test -p soma-bio                      # scheduler
cargo test -p soma-watch                    # file extension filtering
```

## Supported Platforms

- Linux x86_64
- Linux aarch64 (ARM64)
- Windows x86_64
- macOS (x86_64, ARM64)
- WASM (via wasm32 target)

## License

[MIT](LICENSE) -- Yann Banas
