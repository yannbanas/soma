"""SOMA REST API client."""

from __future__ import annotations

from typing import Any

import httpx


class SomaClient:
    """Python client for SOMA's REST API.

    Usage::

        from soma_memory import SomaClient

        s = SomaClient("http://localhost:8080")
        s.add("CRISPR edits gene X", source="paper", tags=["bio"])
        results = s.search("gene editing", limit=10)
        print(s.health())
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SomaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Health & Stats ───────────────────────────────────────────────

    def health(self) -> dict:
        """Check server health."""
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def stats(self) -> dict:
        """Get graph statistics."""
        r = self._client.get("/stats")
        r.raise_for_status()
        return r.json()

    # ── Core Operations ──────────────────────────────────────────────

    def add(
        self,
        content: str,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Add text to SOMA memory."""
        payload: dict[str, Any] = {"content": content}
        if source:
            payload["source"] = source
        if tags:
            payload["tags"] = tags
        r = self._client.post("/add", json=payload)
        r.raise_for_status()
        return r.json()

    def ingest(self, path: str) -> dict:
        """Ingest a file into SOMA."""
        r = self._client.post("/ingest", json={"path": path})
        r.raise_for_status()
        return r.json()

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Hybrid search across the graph."""
        r = self._client.get("/search", params={"q": query, "limit": limit})
        r.raise_for_status()
        return r.json()

    def context(self, query: str) -> dict:
        """Get LLM-ready context block for a query."""
        r = self._client.get("/context", params={"q": query})
        r.raise_for_status()
        return r.json()

    # ── Graph Mutations ──────────────────────────────────────────────

    def relate(
        self,
        from_entity: str,
        to_entity: str,
        channel: str,
        confidence: float = 0.8,
    ) -> dict:
        """Create a typed relation between two entities."""
        r = self._client.post(
            "/relate",
            json={
                "from": from_entity,
                "to": to_entity,
                "channel": channel,
                "confidence": confidence,
            },
        )
        r.raise_for_status()
        return r.json()

    def reinforce(self, from_entity: str, to_entity: str) -> dict:
        """Reinforce an existing relation."""
        r = self._client.post(
            "/reinforce",
            json={"from": from_entity, "to": to_entity},
        )
        r.raise_for_status()
        return r.json()

    def alarm(self, label: str, reason: str) -> dict:
        """Mark an entity as dangerous or erroneous."""
        r = self._client.post("/alarm", json={"label": label, "reason": reason})
        r.raise_for_status()
        return r.json()

    def forget(self, label: str) -> dict:
        """Archive (soft-delete) an entity."""
        r = self._client.post("/forget", json={"label": label})
        r.raise_for_status()
        return r.json()

    # ── Bio ──────────────────────────────────────────────────────────

    def sleep(self) -> dict:
        """Trigger manual consolidation."""
        r = self._client.post("/sleep")
        r.raise_for_status()
        return r.json()
