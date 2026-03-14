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

    # ── Feedback (SOMA v3) ────────────────────────────────────────

    def correct(
        self,
        from_entity: str,
        to_entity: str,
        new_confidence: float,
        reason: str,
    ) -> dict:
        """Lower confidence on an edge (AI detected error)."""
        r = self._client.post(
            "/correct",
            json={
                "from": from_entity,
                "to": to_entity,
                "new_confidence": new_confidence,
                "reason": reason,
            },
        )
        r.raise_for_status()
        return r.json()

    def validate(
        self,
        from_entity: str,
        to_entity: str,
        source: str,
    ) -> dict:
        """Reinforce an edge with a validated source."""
        r = self._client.post(
            "/validate",
            json={"from": from_entity, "to": to_entity, "source": source},
        )
        r.raise_for_status()
        return r.json()

    # ── Context Management (SOMA v3) ──────────────────────────────

    def compact(
        self,
        summary: str,
        entities: list[str] | None = None,
        decisions: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Store a session summary before context compaction."""
        payload: dict[str, Any] = {"summary": summary}
        if entities:
            payload["entities"] = entities
        if decisions:
            payload["decisions"] = decisions
        if session_id:
            payload["session_id"] = session_id
        r = self._client.post("/compact", json=payload)
        r.raise_for_status()
        return r.json()

    def session_restore(self, query: str, limit: int = 5) -> dict:
        """Restore context from previous sessions."""
        r = self._client.get(
            "/session-restore", params={"q": query, "limit": limit}
        )
        r.raise_for_status()
        return r.json()

    # ── Analysis (SOMA v3) ────────────────────────────────────────

    def explain(
        self,
        from_entity: str,
        to_entity: str,
        max_paths: int = 3,
    ) -> dict:
        """Find shortest paths between two entities."""
        r = self._client.get(
            "/explain",
            params={"from": from_entity, "to": to_entity, "max_paths": max_paths},
        )
        r.raise_for_status()
        return r.json()

    def communities(self, min_size: int = 3) -> dict:
        """Detect communities in the graph (Louvain)."""
        r = self._client.get("/communities", params={"min_size": min_size})
        r.raise_for_status()
        return r.json()

    def merge(self, keep: str, absorb: str, reason: str = "") -> dict:
        """Merge duplicate nodes."""
        r = self._client.post(
            "/merge", json={"keep": keep, "absorb": absorb, "reason": reason}
        )
        r.raise_for_status()
        return r.json()

    def think(
        self,
        thought: str,
        depends_on: list[str] | None = None,
        conclusion: bool = False,
    ) -> dict:
        """Record a reasoning step in the graph."""
        payload: dict[str, Any] = {"thought": thought, "conclusion": conclusion}
        if depends_on:
            payload["depends_on"] = depends_on
        r = self._client.post("/think", json=payload)
        r.raise_for_status()
        return r.json()

    # ── Cypher ────────────────────────────────────────────────────

    def cypher(self, query: str) -> dict:
        """Execute a Cypher query against the graph."""
        r = self._client.post("/cypher", json={"query": query})
        r.raise_for_status()
        return r.json()

    # ── Webhooks ──────────────────────────────────────────────────

    def list_webhooks(self) -> dict:
        """List registered webhooks."""
        r = self._client.get("/webhooks")
        r.raise_for_status()
        return r.json()

    def register_webhook(
        self,
        url: str,
        events: list[str] | None = None,
        secret: str | None = None,
    ) -> dict:
        """Register a webhook for graph events."""
        payload: dict[str, Any] = {
            "url": url,
            "events": events or ["*"],
        }
        if secret:
            payload["secret"] = secret
        r = self._client.post("/webhooks", json=payload)
        r.raise_for_status()
        return r.json()

    def delete_webhook(self, webhook_id: str) -> dict:
        """Delete a webhook registration."""
        r = self._client.delete(f"/webhooks/{webhook_id}")
        r.raise_for_status()
        return r.json()
