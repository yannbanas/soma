# SOMA — Vision

> **Stigmergic Ontological Memory Architecture**  
> *Une mémoire persistante, vivante et universelle pour tous tes agents IA.*

---

## Le Problème Fondamental

Tu parles à Claude. Tu expliques ton projet ChromoQ, tes contraintes SwarmOS, ton architecture Rust. La conversation se termine. Tu reviens le lendemain — Claude ne se souvient de rien. Tu réexpliques.

Tu parles à Ollama en local. L'agent fait quelque chose d'utile, découvre une incompatibilité critique, génère un résultat intéressant. La session se ferme. Perdu.

Tu travailles sur Panlunadra depuis 2002. Thousands of conversations, notes, worldbuilding decisions — scattered across files, chats, notebooks. Aucun système ne relie tout ça.

**Le problème n'est pas le manque d'intelligence des modèles. C'est le manque de mémoire persistante entre les sessions.**

---

## Ce que SOMA résout

SOMA est une **mémoire longue durée universelle** pour tous tes agents IA et pour toi-même. Un seul daemon qui tourne en background, qui reçoit tout, qui structure tout, qui répond à tout.

Peu importe la source :
- Conversation Claude → SOMA
- Session Ollama locale → SOMA  
- Notes de recherche → SOMA
- Résultats expérimentaux ChromoQ → SOMA
- Décisions d'architecture AntSuite → SOMA
- Lore Panlunadra → SOMA
- Tickets Django mairie → SOMA

Peu importe le consommateur :
- Claude interroge SOMA via MCP
- KOLOSS consulte SOMA au démarrage
- Ollama agent récupère son contexte dans SOMA
- Toi depuis le terminal avec `soma search`

**Un seul outil. Toute ta mémoire.**

---

## Pourquoi pas cognee ?

cognee est sérieux. 11k étoiles, paper arXiv, architecture propre. Mais il fait des hypothèses qui cassent dans ton contexte :

| Problème cognee | Conséquence |
|---|---|
| Appel LLM obligatoire pour extraire les triplets | Coût, latence, dépendance cloud |
| Python runtime | Incompatible WASM / SwarmOS, 3-5s de démarrage |
| Arêtes statiques dans le graphe | Une info de 3 ans pèse autant qu'une d'hier |
| Pas de channels sémantiques | Impossible de distinguer "chemin fiable" de "erreur connue" |
| Pas de dimension temporelle | Impossible de demander "qu'est-ce qui était important il y a 6 mois ?" |
| Pas de mémoire de conversation native | Il faut tout brancher manuellement |

SOMA n'est pas un fork de cognee. C'est une réponse différente au même problème.

---

## La Métaphore

Dans le cerveau, le **soma** (corps du neurone) intègre tous les signaux entrants et décide quoi propager.

Les synapses ont une **plasticité** :
- Connexion souvent utilisée → se renforce (LTP)
- Connexion ignorée → s'affaiblit, disparaît (élagage synaptique)
- Connexion associée à une erreur → inhibée (modulation GABAergique)

SOMA applique exactement ça à la connaissance :
- Une relation souvent traversée → intensité augmente
- Une relation jamais utilisée → s'évapore progressivement
- Une relation marquée comme fausse → canal `Alarm`, inhibe les chemins qui la traversent
- La nuit → consolidation : épisodes récents compressés en mémoire sémantique

---

## Principes Fondateurs

### 1. Universal Input
Tout ce qui peut être textualisé peut entrer dans SOMA. Conversations Claude, logs Ollama, fichiers Markdown, JSON, code source, mesures expérimentales, notes vocales transcrites. SOMA ne discrimine pas la source.

### 2. Local-First, Privacy-First
Zéro donnée envoyée vers le cloud sans consentement explicite. Le LLM pour l'extraction est optionnel et local (Ollama). SOMA doit tourner entièrement hors-ligne sur un Raspberry Pi 4.

### 3. Rust Pur, Binaire Unique
Un seul binaire statique. Pas de runtime Python, pas de JVM, pas de Node. Compile en natif (x86_64, arm64) et en WASM. Cold start < 50ms. Le daemon `somad` tourne 24/7 avec ~10MB RAM en idle.

### 4. Mémoire Vivante (pas un index)
Les relations ont une durée de vie biologique. Le graphe se consolide, s'élague, renforce les connexions importantes, oublie les détails mineurs. C'est un organisme, pas une base de données.

### 5. MCP Natif Dès le Départ
Claude, KOLOSS, tous les agents Ollama, les apps Django — tout interroge SOMA via le protocole MCP. Un seul point d'entrée, bien défini, extensible.

### 6. Général mais Configurable
SOMA fonctionne sans configuration pour 80% des cas. Pour les 20% restants (vocabulaire domaine-spécifique, tuning des constantes biologiques, workspaces isolés), `soma.toml` permet une adaptation fine.

---

## Cas d'Usage Concrets

### Mémoire de Conversation Claude
```
# Tu ajoutes une conversation après chaque session
soma ingest --source claude --file conversation_2026-03-02.json

# La prochaine session, Claude consulte SOMA via MCP
# Tool: soma_search("ChromoQ extraction pipeline")
# → Retourne les décisions d'architecture prises il y a 3 semaines
```

### Contexte Persistant pour Ollama
```
# L'agent Ollama émet ce qu'il a appris
soma add "WAMR AOT compilation échoue avec target wasm32-wasi sur Unikraft arm64" \
         --channel alarm --source ollama-agent

# L'agent suivant récupère ce contexte avant de commencer
soma search "WAMR Unikraft" --channel alarm
# → "ÉVITER: WAMR AOT + wasm32-wasi + Unikraft arm64 — incompatible (2026-02-15)"
```

### Base de Connaissance Recherche
```
# Ingestion d'un paper
soma ingest --file alphafold2_chromoq_results.pdf

# Ingestion de mesures
soma add "ChromoQ variant X47 : pLDDT=94.2, émission=523nm, rendement_quantique=0.74" \
         --channel trail --tags chromoq,spectral

# Query multi-hop
soma search "variants ChromoQ haute précision basse longueur onde"
# → Traverse : (haute précision) → pLDDT>90 → variants → émission < 530nm
```

### Worldbuilding Panlunadra
```
soma workspace create panlunadra
soma add "Les pandas-dragons communiquent par biophotonique corticale" \
         --workspace panlunadra --channel trail
soma relate "ChromoQ" "biophotonique" --channel causal --workspace panlunadra
soma search "mécanismes communication" --workspace panlunadra
```

---

## Ce que SOMA n'est PAS

- **Pas un LLM** — SOMA stocke, structure, relie. Les LLMs raisonnent.
- **Pas un RAG** — SOMA est la couche en dessous qui rend le RAG intelligent et temporel.
- **Pas un remplaçant d'Obsidian ou Notion** — SOMA est une mémoire machine, pas une interface humaine (bien qu'une UI puisse être construite dessus).
- **Pas un outil AntSuite uniquement** — SOMA est général. AntSuite l'utilise, mais il existe indépendamment.

---

## Métriques de Succès v1.0

| Métrique | Cible | État Actuel |
|---|---|---|
| Cold start `soma` | < 50ms | ✅ < 50ms |
| Ingestion d'une conversation (texte brut) | < 500ms | ✅ ~195µs (RULER corpus) |
| Recherche (hybrid graph+HDC+fuzzy) | < 15ms | ✅ < 1ms (RULER queries) |
| Extraction triplets sans LLM (L0+L1) | > 80% des cas | ✅ 15 patterns, 10/10 RULER |
| L2 LLM extraction (optionnel) | Ollama local | ✅ soma-llm (cogito:8b) |
| Tests automatisés | > 100 | ✅ 109 tests |
| CLI commandes | 10+ | ✅ 13 commandes |
| MCP outils | 10 | ✅ 10 outils |
| Durabilité (WAL fsync) | Crash-safe | ✅ fsync + sync_all |
| Dépendances cloud obligatoires | 0 | ✅ 0 |
| Lignes de Python dans le core | 0 | ✅ 0 (Rust pur, 9 crates) |
| Plateformes supportées | Linux x86_64, arm64, WASM | ✅ + Windows |
