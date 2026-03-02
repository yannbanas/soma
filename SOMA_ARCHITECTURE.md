# SOMA — Architecture

> **Stigmergic Ontological Memory Architecture**  
> Version : 0.1.0 | Rust 2021 | Workspace Cargo | 109 tests | 9 crates

---

## Vue Globale

```
                    ┌─────────────────────────────────────────┐
                    │            SOURCES D'ENTRÉE              │
                    │  Conversations Claude · Sessions Ollama  │
                    │  Fichiers · PDF · Code · JSON · Notes    │
                    └──────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │            soma-ingest                   │
                    │  Parsing · Chunking · Extraction L0/L1   │
                    │  L2 Ollama optionnel (soma-llm)          │
                    └──────────────────┬──────────────────────┘
                                       │ Triplets + Chunks
                          ┌────────────┴────────────┐
                          ▼                         ▼
          ┌───────────────────────┐   ┌─────────────────────────┐
          │     soma-graph        │   │       soma-hdc           │
          │  Knowledge Graph      │   │   Index Sémantique HDC   │
          │  Vivant + Évaporation │◄──►  Random Indexing 10000D  │
          │  BFS Traversal        │   │  + Neural Embeddings     │
          └───────────────────────┘   └─────────────────────────┘
                          │                     │
                          ▼                     ▼
          ┌───────────────────────┐   ┌─────────────────────────┐
          │     soma-store        │   │     soma-core (hybrid)   │
          │  WAL (fsync) +        │   │  RRF Merge (3 paths)     │
          │  Snapshots (sync_all) │   │  Fuzzy Label Search      │
          │  Crash-safe, No ORM   │   │  Source Attribution       │
          └───────────────────────┘   └─────────────────────────┘
                          │
                          │  (background, 24/7)
                          ▼
          ┌───────────────────────┐
          │     soma-bio          │
          │  Évaporation          │
          │  Physarum Reshape     │
          │  Consolidation        │
          │  Élagage + Pruning    │
          │  Graceful Shutdown    │
          └───────────────────────┘
                          │
              ┌───────────┴──────────┐
              ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐
│     soma-mcp        │  │     soma-cli         │
│  Serveur MCP        │  │  13 Commands         │
│  10 tools JSON-RPC  │  │  --format json       │
│  Claude · Ollama    │  │  Hybrid search       │
└─────────────────────┘  └─────────────────────┘
```

---

## Structure du Workspace

```
soma/
├── Cargo.toml              # Workspace root
├── Cargo.lock
├── soma.toml               # Config utilisateur
├── VISION.md
├── ARCHITECTURE.md
├── README.md
│
├── crates/
│   ├── soma-core/          # Types fondamentaux — aucune dépendance interne
│   ├── soma-graph/         # Knowledge graph vivant
│   ├── soma-hdc/           # Index sémantique HDC
│   ├── soma-store/         # Persistance WAL
│   ├── soma-bio/           # Scheduler biologique (évaporation, consolidation)
│   ├── soma-ingest/        # Pipeline d'ingestion universel
│   ├── soma-llm/           # Client Ollama (génération + embeddings)
│   ├── soma-mcp/           # Serveur MCP natif
│   └── soma-cli/           # Interface terminal (13 commandes) + daemon
│
└── tests/
    ├── integration/
    └── fixtures/
```

**Règle de dépendance — sens unique strict :**
```
soma-core → (rien)
soma-graph → soma-core
soma-hdc → soma-core
soma-llm → soma-core
soma-store → soma-core
soma-bio → soma-core, soma-graph, soma-store
soma-ingest → soma-core, soma-graph, soma-hdc, soma-llm
soma-mcp → soma-core, soma-graph, soma-hdc, soma-store, soma-ingest, soma-llm
soma-cli → tout (9 crates)
```

---

## soma-core — Les Types Fondamentaux

### Identifiants

```rust
// IDs opaques et type-safe
// NodeId déterministe sur le label pour déduplication automatique

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(Uuid);

impl NodeId {
    pub fn new() -> Self {
        NodeId(Uuid::new_v4())
    }

    /// Déterministe — deux labels identiques → même NodeId
    /// Permet la déduplication sans lookup préalable
    pub fn from_label(label: &str) -> Self {
        NodeId(Uuid::new_v5(&Uuid::NAMESPACE_OID, label.as_bytes()))
    }
}
```

### Channel — Types de Relations

```rust
/// Le type d'une arête détermine son comportement biologique complet :
/// vitesse d'évaporation, amplitude de renforcement, traitement lors de la consolidation.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Channel {
    /// Connaissance validée, chemin de confiance établi
    Trail,
    /// Relation causale directe : A provoque B
    Causal,
    /// Filiation : B dérive de A (version, mutation, inspiration)
    DerivesDe,
    /// Événement situé dans le temps — s'efface naturellement
    Episodic,
    /// Erreur confirmée, incompatibilité, dead-end — inhibe les chemins
    Alarm,
    /// Similarité sémantique calculée par soma-hdc (auto-générée)
    SemanticSim,
    /// Canal personnalisé déclaré dans soma.toml
    Custom(u16),
}

impl Channel {
    /// Constante de décroissance (intensité perdue par heure, base exponentielle)
    /// τ = 0.0 → permanent | τ = 0.05 → s'efface en ~20h
    pub fn tau_decay(self) -> f32 {
        match self {
            Channel::DerivesDe   => 0.0,     // permanent — la filiation ne s'efface pas
            Channel::Causal      => 0.0001,  // quasi-permanent
            Channel::Trail       => 0.001,   // très lent (~40 jours)
            Channel::Alarm       => 0.005,   // lent (~8 jours — les erreurs persistent)
            Channel::SemanticSim => 0.02,    // modéré (~2 jours)
            Channel::Episodic    => 0.05,    // rapide (~20h)
            Channel::Custom(_)   => 0.01,
        }
    }

    /// Delta ajouté à l'intensité lors d'une traversée (feedback positif)
    pub fn reinforce_delta(self) -> f32 {
        match self {
            Channel::Causal      => 0.50,
            Channel::DerivesDe   => 0.40,
            Channel::Trail       => 0.30,
            Channel::Alarm       => 0.35,  // les erreurs se renforcent si répétées
            Channel::SemanticSim => 0.20,
            Channel::Episodic    => 0.10,
            Channel::Custom(_)   => 0.20,
        }
    }

    /// Vrai si l'arête peut être élagage
    pub fn is_prunable(self) -> bool {
        !matches!(self, Channel::DerivesDe)
    }
}
```

### SomaNode

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaNode {
    pub id:         NodeId,
    /// Texte principal — unique dans un workspace
    pub label:      String,
    pub kind:       NodeKind,
    pub created_at: DateTime<Utc>,
    pub last_seen:  DateTime<Utc>,
    /// Mots-clés pour filtrage rapide
    pub tags:       Vec<String>,
    /// Métadonnées libres
    pub meta:       Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Objet concret : outil, fichier, protéine, personnage
    Entity,
    /// Idée abstraite : performance, fiabilité, technique
    Concept,
    /// Événement daté : session, expérience, incident
    Event,
    /// Valeur mesurée : pLDDT=94.2, λ=523nm, latence=16ms
    Measurement,
    /// Savoir-faire : procédure, recette, pattern
    Procedure,
    /// Problème connu : bug, incompatibilité, contrainte
    Warning,
}
```

### StigreEdge — L'Arête Vivante

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StigreEdge {
    pub id:           EdgeId,
    pub from:         NodeId,
    pub to:           NodeId,
    pub channel:      Channel,

    /// Intensité au moment du dernier touch — [0.0, 1.0]
    /// L'intensité effective (après évaporation) se calcule à la demande
    pub intensity:    f32,

    /// Timestamp du dernier renforcement ou de la création
    pub last_touch:   DateTime<Utc>,

    /// Nombre de traversées ou de confirmations
    pub uses:         u32,

    /// Confiance dans la source — [0.0, 1.0]
    pub confidence:   f32,

    /// D'où vient cette relation : "claude/conv-abc123", "ollama/session-42", "file/paper.pdf"
    pub source:       String,

    /// Étiquette humaine optionnelle : "utilise le plugin X pour Y"
    pub label:        Option<String>,
}

impl StigreEdge {
    /// Intensité effective après évaporation — calcul lazy, pas de timer actif
    pub fn effective_intensity(&self, now: DateTime<Utc>) -> f32 {
        let tau = self.channel.tau_decay();
        if tau == 0.0 {
            return self.intensity;
        }
        let dt_hours = (now - self.last_touch).num_seconds().max(0) as f32 / 3600.0;
        self.intensity * (-tau * dt_hours).exp()
    }

    /// Renforce l'arête — réinitialise l'horloge, combine l'intensité actuelle + delta
    pub fn reinforce(&mut self, now: DateTime<Utc>) {
        let current = self.effective_intensity(now);
        let delta = self.channel.reinforce_delta();
        self.intensity = (current + delta).min(1.0);
        self.last_touch = now;
        self.uses += 1;
    }

    pub fn is_dead(&self, threshold: f32, now: DateTime<Utc>) -> bool {
        self.channel.is_prunable() && self.effective_intensity(now) < threshold
    }
}
```

### SomaQuery

```rust
pub struct SomaQuery {
    /// Point de départ — texte libre (sera matché par label ou HDC)
    pub start:         String,
    /// Canaux autorisés pour la traversée (vide = tous sauf Alarm)
    pub channels:      Vec<Channel>,
    /// Profondeur maximale de traversée
    pub max_hops:      u8,           // défaut: 3
    /// Seuil d'intensité minimale — arêtes en dessous ignorées
    pub min_intensity: f32,          // défaut: 0.15
    /// Si vrai, enrichit les résultats avec similarité sémantique HDC
    pub semantic:      bool,         // défaut: true
    /// Workspace cible
    pub workspace:     String,       // défaut: "default"
    /// Filtre temporel optionnel
    pub since:         Option<DateTime<Utc>>,
    pub until:         Option<DateTime<Utc>>,
}

pub struct QueryResult {
    pub node:   SomaNode,
    pub path:   Vec<StigreEdge>,
    /// Produit des intensités effectives sur le chemin
    pub score:  f32,
    pub hops:   u8,
    /// Which search paths contributed: "graph", "hdc", "fuzzy"
    pub sources: Vec<String>,
}
```

---

## soma-graph — Knowledge Graph Vivant

### StigreGraph

```rust
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

pub struct StigreGraph {
    /// Backend petgraph — graphe orienté
    inner:          DiGraph<SomaNode, StigreEdge>,
    /// Lookup O(1) : label → NodeIndex (pour déduplication)
    label_idx:      HashMap<String, NodeIndex>,
    /// Lookup O(1) : NodeId → NodeIndex
    id_idx:         HashMap<NodeId, NodeIndex>,
    /// Identifiant du workspace
    workspace:      String,
    /// Seuil global de pruning
    prune_threshold: f32,
}

impl StigreGraph {

    /// Upsert idempotent — si le label existe déjà, retourne l'ID existant sans doublon
    pub fn upsert_node(&mut self, label: &str, kind: NodeKind) -> NodeId {
        let node_id = NodeId::from_label(&format!("{}:{}", self.workspace, label));
        if self.id_idx.contains_key(&node_id) {
            // Met à jour last_seen
            let idx = self.id_idx[&node_id];
            self.inner[idx].last_seen = Utc::now();
            return node_id;
        }
        let node = SomaNode {
            id: node_id,
            label: label.to_string(),
            kind,
            created_at: Utc::now(),
            last_seen: Utc::now(),
            tags: vec![],
            meta: None,
        };
        let idx = self.inner.add_node(node);
        self.label_idx.insert(label.to_string(), idx);
        self.id_idx.insert(node_id, idx);
        node_id
    }

    /// Upsert arête — si (from, to, channel) existe déjà, renforce au lieu de dupliquer
    pub fn upsert_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        channel: Channel,
        confidence: f32,
        source: &str,
    ) -> EdgeId {
        let now = Utc::now();
        let from_idx = self.id_idx[&from];
        let to_idx = self.id_idx[&to];

        // Cherche une arête existante (from, to, channel)
        if let Some(edge_idx) = self.inner.find_edge(from_idx, to_idx) {
            let edge = &mut self.inner[edge_idx];
            if edge.channel == channel {
                edge.reinforce(now);
                return edge.id;
            }
        }

        // Nouvelle arête
        let edge = StigreEdge {
            id: EdgeId(Uuid::new_v4()),
            from,
            to,
            channel,
            intensity: confidence.min(1.0),
            last_touch: now,
            uses: 1,
            confidence,
            source: source.to_string(),
            label: None,
        };
        let eid = edge.id;
        self.inner.add_edge(from_idx, to_idx, edge);
        eid
    }

    /// Traversal Dijkstra pondéré par intensité effective
    /// Score d'un chemin = produit des intensités effectives sur les arêtes
    pub fn traverse(&self, query: &SomaQuery) -> Vec<QueryResult> { ... }

    /// Voisins directs d'un nœud, filtrés par canal et intensité minimale
    pub fn neighbors(
        &self,
        node_id: NodeId,
        channels: &[Channel],
        min_intensity: f32,
    ) -> Vec<(NodeId, &StigreEdge)> { ... }

    /// Stats globales du graphe
    pub fn stats(&self) -> GraphStats {
        let now = Utc::now();
        let active_edges = self.inner.edge_count();
        let dead_edges = self.inner.edge_references()
            .filter(|e| e.weight().is_dead(self.prune_threshold, now))
            .count();
        GraphStats {
            nodes: self.inner.node_count(),
            edges: active_edges,
            dead_edges,
            workspace: self.workspace.clone(),
        }
    }
}
```

---

## soma-hdc — Index Sémantique (Random Indexing)

### Principe

```
Chaque token → un vecteur de D=10000 dimensions (f32)

Construction :
  1. Pour chaque token, créer un vecteur de base aléatoire sparse (±1, ~1% actifs)
  2. Pour chaque occurrence dans le corpus, parcourir la fenêtre ±5 mots :
       context_vector[token] += base_vector[voisin] × poids_tfidf(voisin)
  3. Normaliser → vecteur distributional

Résultat :
  "luciferase" et "bioluminescence" → cos_sim ≈ 0.82   (même contexte)
  "WAMR" et "WebAssembly" → cos_sim ≈ 0.75
  "ChromoQ" et "fluorescent_protein" → cos_sim ≈ 0.71
  "trail" et "alarm" → cos_sim ≈ 0.10  (contextes différents — c'est voulu)
```

### HdcEngine

```rust
pub struct HdcEngine {
    /// Dimension des vecteurs
    dim:            usize,       // 10000
    /// Vecteurs de base aléatoires (sparse) par token
    base_vectors:   HashMap<String, Vec<f32>>,
    /// Vecteurs distributionnels (enrichis par co-occurrence)
    context_vectors: HashMap<String, Vec<f32>>,
    /// Fenêtre de co-occurrence
    window:         usize,       // 5
    /// Poids TF-IDF par token
    idf:            HashMap<String, f32>,
}

impl HdcEngine {
    /// Entraîne sur un corpus de phrases (corpus = tes conversations, docs, notes)
    pub fn train(&mut self, sentences: &[String]) { ... }

    /// Top-K voisins sémantiques d'un token
    pub fn most_similar(&self, token: &str, k: usize) -> Vec<(String, f32)> { ... }

    /// Cosine similarity entre deux tokens
    pub fn similarity(&self, a: &str, b: &str) -> f32 { ... }

    /// Encode une phrase entière → vecteur centroïde pondéré TF-IDF
    pub fn encode_sentence(&self, sentence: &str) -> Vec<f32> { ... }

    /// Recherche sémantique : top-K nœuds du graphe par similarité de phrase
    pub fn search(&self, query: &str, candidates: &[String], k: usize) -> Vec<(String, f32)> { ... }
}
```

### Couplage Graphe → HDC

```rust
/// Enrichit les vecteurs HDC avec la topologie du graphe.
/// Un nœud fortement connecté (hub Trail) → vecteur amplifié.
/// Un nœud Alarm → vecteur "isolé" dans l'espace HD.
pub fn enrich_from_topology(hdc: &mut HdcEngine, graph: &StigreGraph) {
    let now = Utc::now();
    for node in graph.all_nodes() {
        let trail_strength: f32 = graph
            .outgoing_edges(node.id)
            .filter(|e| e.channel == Channel::Trail)
            .map(|e| e.effective_intensity(now))
            .sum();

        let has_alarm = graph
            .incoming_edges(node.id)
            .any(|e| e.channel == Channel::Alarm);

        hdc.modulate(&node.label, trail_strength, has_alarm);
    }
}
```

### Neural Embeddings (via Ollama)

```rust
/// HdcEngine supporte aussi des embeddings neuronaux (Ollama)
/// Les embeddings neuronaux sont préférés s'ils existent, sinon fallback HDC.
impl HdcEngine {
    pub fn set_neural_embedding(&mut self, label: &str, embedding: Vec<f32>) { ... }
    pub fn has_neural_embedding(&self, label: &str) -> bool { ... }
    /// Hybrid similarity: neural cosine if both have embeddings, else HDC cosine
    pub fn similarity_hybrid(&self, a: &str, b: &str) -> f32 { ... }
}
```

---

## soma-core (hybrid) — Recherche Hybride RRF

### Principe

La recherche combine 3 chemins indépendants fusionnés par Reciprocal Rank Fusion :

```
Query
  │
  ├──→ [Path 1: Graph BFS]  — traverse le graphe par intensité d'arêtes
  │
  ├──→ [Path 2: HDC/Neural]  — similarité sémantique (cosine) sur tous les labels
  │
  └──→ [Path 3: Fuzzy]  — exact/prefix/contains matching sur les labels
  │
  ▼
[RRF Merge]  score(d) = Σ 1/(k + rank_i(d)),  k=60
  │
  ▼
Résultats avec attribution de sources: [graph+hdc], [fuzzy], [graph+hdc+fuzzy]
```

```rust
/// Reciprocal Rank Fusion — merges N ranked lists into one.
pub fn rrf_merge_with_sources(
    ranked_lists: &[(&str, Vec<(String, f32)>)],
    k: f32,
) -> Vec<HybridResult>;

/// Fuzzy label search — exact (1.0), prefix (0.9), contains (0.7), word-prefix (0.6)
pub fn fuzzy_label_search(query: &str, labels: &[String], limit: usize) -> Vec<(String, f32)>;

pub struct HybridResult {
    pub label: String,
    pub score: f32,
    pub sources: Vec<String>,  // e.g. ["graph", "hdc", "fuzzy"]
    pub node: Option<SomaNode>,
    pub hops: Option<u8>,
}
```

### Stockage mmap

```
~/.local/share/soma/<workspace>/hdc/
├── vectors.bin       # vecteurs f32 — mmap, accès O(1)
├── index.bin         # label_hash(8) + offset(8) + len(4) par entrée
└── vocab.txt         # liste des tokens (debug + audit)

Format vectors.bin :
  [magic: b"SOMA" (4)]
  [version: u16 (2)]
  [dim: u32 (4)]
  [count: u64 (8)]
  [data: f32 × dim × count]
```

---

## soma-store — Persistance WAL

### Principe

Pas de SQLite (overhead), pas d'ORM. Un WAL (Write-Ahead Log) append-only avec fsync — chaque écriture est loggée et flushed au disque, un snapshot compressé (zstd + sync_all + atomic rename) est produit périodiquement.

```
~/.local/share/soma/<workspace>/
├── wal.log          # journal append-only (bincode)
├── snapshot.soma    # dernier snapshot compressé (zstd)
└── meta.toml        # métadonnées workspace (version, ts snapshot)
```

### WalEntry

```rust
#[derive(Serialize, Deserialize)]
pub enum WalEntry {
    NodeUpsert(SomaNode),
    EdgeUpsert(StigreEdge),
    EdgeReinforce {
        id:    EdgeId,
        delta: f32,
        ts:    DateTime<Utc>,
    },
    EdgePrune(EdgeId),
    NodeArchive(NodeId),
    ConsolidationEvent {
        ts:            DateTime<Utc>,
        episodes_merged: u32,
        concepts_created: u32,
    },
}
```

**Au démarrage du daemon :**
1. Charger le dernier `snapshot.soma`
2. Rejouer les `WalEntry` postérieures au snapshot
3. Graphe reconstruit — résistant aux crashs (kill -9)

**Snapshot :** produit toutes les 6h (configurable). Le WAL est tronqué après snapshot réussi.

---

## soma-bio — Scheduler Biologique

```rust
// 4 tâches tokio indépendantes avec graceful shutdown (Ctrl+C)
pub async fn run(&self, graph: Arc<RwLock<StigreGraph>>, store: Arc<RwLock<Store>>) {
    tokio::select! {
        _ = async {
            tokio::join!(
                Self::evaporation_watchdog(graph.clone(), config.clone()),   // 1h
                Self::physarum_reshape(graph.clone(), config.clone()),        // 2h
                Self::sleep_consolidation(graph.clone(), store.clone(), config.clone()), // 6h
                Self::daily_pruning(graph.clone(), store.clone(), config),    // 24h
            );
        } => {},
        _ = tokio::signal::ctrl_c() => {
            info!("[bio] Graceful shutdown — Ctrl+C received");
        }
    }
    // Daemon saves final snapshot after scheduler returns
}
```

### Les 4 Boucles

**Évaporation (watchdog 1h)**  
L'évaporation est *lazy* — `effective_intensity()` est calculé à la demande, pas par timer. Le watchdog vérifie les arêtes dont l'intensité effective est tombée sous le seuil et les marque comme candidates au pruning.

**Physarum Reshape (2h)**  
Inspiré du *Physarum polycephalum* (la moisissure qui optimise les réseaux). Les chemins fréquemment traversés voient leurs arêtes s'épaissir (intensité restaurée). Les chemins jamais empruntés voient leurs arêtes s'amincir (τ_decay augmente temporairement). Le résultat : les chemins efficaces se renforcent, les dead-ends s'estompent.

**Sleep Consolidation (6h — "nuit biologique")**  
Cluster les nœuds `Episodic` récents par similarité sémantique HDC. Pour chaque cluster de taille ≥ 3 : crée un nœud `Concept` qui les résume, relie les membres via `DerivesDe`. Marque les épisodes consolidés (ils peuvent s'évaporer plus vite désormais). Mimique la consolidation hippocampo-corticale du sommeil lent.

**Daily Pruning (24h)**  
Supprime les arêtes `is_dead()`. Archive (n'efface JAMAIS) les nœuds orphelins dans un store d'archive séparé. Un nœud archivé peut être rappelé si une nouvelle information le réfère.

---

## soma-ingest — Pipeline d'Ingestion Universel

### Sources Supportées

```rust
pub enum IngestSource {
    /// Texte libre ou Markdown
    RawText(String),
    /// Fichier (détection automatique du format)
    File(PathBuf),
    /// Conversation Claude (format JSON export)
    ClaudeConversation(PathBuf),
    /// Session Ollama (format structuré ou log)
    OllamaSession { log: String, model: String },
    /// URL web (fetch + extraction)
    Url(String),
    /// Données structurées (JSON/TOML/CSV)
    Structured(serde_json::Value),
}
```

### Pipeline L0 → L1 → L2

```
Input
  │
  ▼
[Parsing]
  Détection format, extraction texte brut, chunking intelligent
  (chunks 3-5 phrases, overlap 1-2 phrases pour conserver le contexte)
  │
  ▼
[L0 — Heuristiques Structurées]  < 0.1ms par chunk
  Sources JSON structurées : phéromones, logs Ollama, exports Claude
  → Extraction directe des triplets depuis les champs structurés
  Couvre ~40% des cas
  │
  ▼ (si L0 insuffisant)
[L1 — Patterns Domaine]  < 1ms par chunk
  Regex compilées une fois (once_cell)
  Patterns généraux : "X est un Y", "X dérive de Y", "X cause Y"
  Patterns domaine depuis soma.toml : "X émet à Ynm", "X incompatible Z"
  Couvre ~45% des cas supplémentaires
  │
  ▼ (si L0+L1 < 3 triplets sur un chunk long)
[L2 — LLM Local Optionnel]  ~200-500ms par chunk (Ollama)
  Prompt structuré → JSON triplets
  Uniquement si activé dans soma.toml et Ollama disponible
  Couvre le reste
  │
  ▼
[Déduplication + Normalisation]
  NodeId déterministe → pas de doublons automatiquement
  Normalisation labels : lowercase, strip ponctuation, lemmatisation légère
  │
  ▼
[Insertion graphe + Mise à jour HDC]
```

### Patterns L1 intégrés

```rust
static PATTERNS: &[PatternDef] = &[
    PatternDef { re: r"(?i)(\w+)\s+(?:est un|est une|is a|is an)\s+(.+?)(?:\.|$)",
                 channel: Channel::Trail, swap: false },
    PatternDef { re: r"(?i)(\w+)\s+(?:dérive de|derives from|based on)\s+(\w+)",
                 channel: Channel::DerivesDe, swap: false },
    PatternDef { re: r"(?i)(\w+)\s+(?:cause|causes?|provoque|triggers?)\s+(\w+)",
                 channel: Channel::Causal, swap: false },
    PatternDef { re: r"(?i)(?:ÉVITER|AVOID|incompatible).{0,50}(\w+)",
                 channel: Channel::Alarm, swap: false },
    PatternDef { re: r"(?i)(\w+)\s+(?:remplace|replaces?)\s+(\w+)",
                 channel: Channel::DerivesDe, swap: false },
    PatternDef { re: r"(?i)(\w+)\s+(?:utilise|uses?)\s+(\w+)",
                 channel: Channel::Trail, swap: false },
    // Patterns mesures (bioinformatique)
    PatternDef { re: r"(?i)(\w+)\s+(?:émission|emission|λ|lambda)\s*[=:]\s*([\d.]+\s*nm)",
                 channel: Channel::Causal, swap: false },
    PatternDef { re: r"(?i)(\w+)\s+pLDDT\s*[=:>]\s*([\d.]+)",
                 channel: Channel::Trail, swap: false },
];
```

---

## soma-mcp — Serveur MCP Natif

### Transport

```
# stdio (défaut — pour Claude desktop, KOLOSS)
somad --mcp-stdio

# TCP (pour agents Ollama, apps Django)
somad --mcp-tcp --port 3333
```

### Les 10 Outils MCP

```
soma_add
  description: Ajoute du texte ou une note dans la mémoire SOMA
  params: content(str), source?(str), tags?([str]), workspace?(str), channel?(str)
  returns: { nodes_created: N, edges_created: M, duration_ms: f }

soma_ingest
  description: Ingère un fichier complet (PDF, Markdown, JSON, log Ollama, export Claude)
  params: path(str), source_type?(str), workspace?(str)
  returns: { chunks: N, nodes: M, edges: K, duration_ms: f }

soma_search
  description: Recherche hybride (sémantique + traversal graphe)
  params: query(str), channels?([str]), max_hops?(int), min_intensity?(float),
          workspace?(str), since?(datetime), limit?(int)
  returns: [{ node, path, score, hops }]

soma_relate
  description: Crée manuellement une relation typée entre deux entités
  params: from(str), to(str), channel(str), confidence?(float), source?(str), workspace?(str)
  returns: { edge_id: str }

soma_reinforce
  description: Renforce une relation après validation externe
  params: from(str), to(str), channel?(str), workspace?(str)
  returns: { new_intensity: float }

soma_alarm
  description: Marque une entité comme dangereuse / erronée / à éviter
  params: label(str), reason(str), source?(str), workspace?(str)
  returns: { alarm_id: str }

soma_forget
  description: Archive (sans effacer) une entité ou une relation
  params: label(str), workspace?(str)
  returns: { archived: bool }

soma_stats
  description: État complet du graphe et de l'index
  params: workspace?(str)
  returns: { nodes, edges, dead_edges, avg_intensity, index_size, ... }

soma_workspace
  description: Gestion des workspaces isolés
  params: action("create"|"switch"|"list"|"delete"), name?(str)
  returns: { workspaces: [...], current: str }

soma_context
  description: Retourne un bloc de contexte formaté pour un LLM
                (les N faits les plus pertinents pour une requête)
  params: query(str), max_tokens?(int), workspace?(str)
  returns: { context: str, sources: [...], facts: N }
```

### soma_context — L'Outil Clé

`soma_context` est l'outil le plus important pour l'intégration LLM. Il retourne un bloc de texte directement insérable dans un prompt :

```
# Exemple d'appel (Claude l'utilise en début de session)
soma_context("ChromoQ pipeline architecture")

# Retourne :
"""
[SOMA MEMORY — 2026-03-02 — workspace: research]

Faits pertinents (score > 0.6) :
• ChromoQ dérive de EGFP [DerivesDe, intensité=0.95, 2026-01-15]
• ChromoQ variant X47 : pLDDT=94.2, émission=523nm, rendement=0.74 [Trail, 0.88]
• AlphaFold2 utilisé pour prédire la structure ChromoQ [Trail, 0.82]
• ChromoQ intégré dans PNPH pour transduction quantique [Causal, 0.79]

⚠ Alarmes actives :
• ChromoQ variant X23 : instable à pH < 6.5 [Alarm, 0.71, 2026-02-03]

Sources : research/alphafold_results.pdf, claude/conv-2026-01-15
"""
```

---

## soma.toml — Configuration

```toml
[soma]
data_dir   = "~/.local/share/soma"
default_workspace = "default"
log_level  = "info"

[bio]
prune_threshold            = 0.05   # intensité en dessous = dead
physarum_interval_hours    = 2
consolidation_interval_hours = 6
pruning_interval_hours     = 24
snapshot_interval_hours    = 6

[hdc]
dimension    = 10000
window_size  = 5
tfidf        = true

[ingest]
default_level = "l1"           # "l0" | "l1" | "l2"
chunk_size    = 5              # phrases par chunk
chunk_overlap = 1              # phrases de chevauchement

[llm]
enabled  = false               # L2 optionnel
provider = "ollama"
model    = "nemotron-mini:4b"
endpoint = "http://localhost:11434"

[mcp]
transport = "stdio"            # "stdio" | "tcp"
tcp_port  = 3333

# Vocabulaire domaine — accélère L0/L1
[domain.bioinformatics]
entities  = ["ChromoQ", "EGFP", "GFP", "AlphaFold2", "luciferase", "PNPH"]
relations = ["DERIVE_DE", "PREDIT_PAR", "EMET_A", "MUTANT_DE"]

[domain.antsuite]
entities  = ["SwarmOS", "AntOS", "MorphLang", "WAMR", "Unikraft", "KOLOSS"]
relations = ["COMPILE_VERS", "DEPEND_DE", "REMPLACE", "INCOMPATIBLE"]

[domain.panlunadra]
entities  = ["ChromoQ", "PNPH", "panda-dragon", "biophotonique"]
relations = ["INSPIRE_DE", "EXISTE_DANS", "ENCODE"]
```

---

## Cargo.toml Workspace

```toml
[workspace]
members   = ["crates/*"]
resolver  = "2"

[workspace.dependencies]
# Fondamentaux
thiserror  = "2"
serde      = { version = "1", features = ["derive"] }
serde_json = "1"
uuid       = { version = "1", features = ["v4", "v5"] }
chrono     = { version = "0.4", features = ["serde"] }
once_cell  = "1"

# Graphe
petgraph   = "0.6"

# HDC
ndarray    = "0.16"
rand       = "0.8"

# Persistance
bincode    = "2"
zstd       = "0.13"

# Async (soma-bio, soma-mcp uniquement)
tokio      = { version = "1", features = ["full"] }

# CLI
clap       = { version = "4", features = ["derive"] }
colored    = "2"

# Extract
regex      = "1"

# Internes
soma-core    = { path = "crates/soma-core" }
soma-graph   = { path = "crates/soma-graph" }
soma-hdc     = { path = "crates/soma-hdc" }
soma-store   = { path = "crates/soma-store" }
soma-bio     = { path = "crates/soma-bio" }
soma-ingest  = { path = "crates/soma-ingest" }

[profile.release]
opt-level       = 3
lto             = true
codegen-units   = 1
strip           = true

[profile.dev]
opt-level = 1   # ndarray devient inutilisable à opt-level=0
```

---

## Roadmap — État Actuel

### Phase 0 — Graphe Vivant ✅
- [x] `soma-core` complet — tous les types, `StigreEdge::effective_intensity()`
- [x] `soma-graph` — `StigreGraph` petgraph, upsert idempotent, traversal BFS
- [x] `soma-cli` — 13 commandes complètes
- [x] Tests unitaires évaporation par canal
- [x] `soma.toml` parsing + validation

### Phase 1 — Persistance et Biologie ✅
- [x] `soma-store` — WAL (fsync) + snapshot zstd (sync_all) + reconstruction
- [x] `soma-bio` — 4 boucles tokio + graceful shutdown (Ctrl+C)
- [x] Crash-safe persistence

### Phase 2 — Index Sémantique ✅
- [x] `soma-hdc` — Random Indexing D=10000, TF-IDF, cosine similarity
- [x] Neural embeddings (Ollama) avec fallback HDC
- [x] Hybrid search RRF (graph + HDC + fuzzy)
- [x] RULER benchmark 10/10

### Phase 3 — Ingestion et MCP ✅
- [x] `soma-ingest` — pipeline L0/L1 (15 patterns) + L2 Ollama optionnel
- [x] `soma-mcp` — serveur stdio + tcp, 10 outils
- [x] `soma-llm` — client Ollama (génération + embeddings)
- [x] CLI complet avec `--format json` pour scripting

### Phase 4 — Production Hardening ✅
- [x] Config validation au démarrage
- [x] WAL fsync + snapshot sync_all (durabilité)
- [x] Graceful shutdown avec snapshot final
- [x] 109 tests (39 core + 7 graph + 11 hdc + 8 store + 18 ingest + 11 llm + 2 bio + 13 RULER)

---

---

## soma-llm — Client Ollama

```rust
pub struct OllamaClient {
    endpoint: String,
    model: String,
    embedding_model: String,
    timeout: Duration,
}

impl OllamaClient {
    /// Extract triplets from text using LLM (L2 extraction)
    pub fn extract_triplets(&self, chunk: &str) -> Result<Vec<Triplet>, ...> { ... }

    /// Generate neural embeddings for a label
    pub fn embed(&self, text: &str) -> Result<Option<Vec<f64>>, ...> { ... }

    /// Check if Ollama is available (graceful degradation)
    pub fn is_available(&self) -> bool { ... }
}
```

Configuration dans `soma.toml` :
```toml
[llm]
enabled = true
provider = "ollama"
model = "cogito:8b"
embedding_model = "nomic-embed-text"
endpoint = "http://localhost:11434"
timeout_ms = 30000
```

---

## Comparaison Honnête vs cognee

| | SOMA | cognee |
|---|---|---|
| **Langage** | Rust pur | Python |
| **Cold start** | < 50ms | 3-5s |
| **Extraction triplets** | L0/L1 sans LLM | LLM obligatoire |
| **Évaporation temporelle** | Native par canal | Partielle (memify) |
| **Mémoire conversation** | Native (soma_context) | Manuel |
| **WASM / SwarmOS** | ✅ | ❌ |
| **Dépendances cloud** | 0 | LLM cloud par défaut |
| **Maturité** | À construire | v0.5.1, 11k ⭐ |
| **Workspaces isolés** | ✅ natif | ⚠️ namespace SQL |
| **Requêtes temporelles** | ✅ | ❌ |
