# soma-memory

Python client for [SOMA](https://github.com/yannbanas/soma) — Stigmergic Ontological Memory Architecture.

## Install

```bash
pip install soma-memory
```

## Usage

```python
from soma_memory import SomaClient

with SomaClient("http://localhost:8080") as s:
    s.add("CRISPR edits gene X", source="paper", tags=["bio"])
    results = s.search("gene editing", limit=10)
    print(s.health())
```

## License

MIT
