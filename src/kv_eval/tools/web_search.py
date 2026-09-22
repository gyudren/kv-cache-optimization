"""Tavily provider shared by market and stakeholder Agents (not PDF RAG)."""
from __future__ import annotations
from typing import Any

class WebSearch:
    def __init__(self, api_key: str, client: Any = None):
        if not api_key and client is None:
            raise ValueError("TAVILY_API_KEY is required")
        if client is None:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
        self.client = client

    def _search(self, query: str) -> list[dict]:
        try:
            payload = self.client.search(query=query, search_depth="advanced")
        except Exception as exc:
            raise RuntimeError(f"Tavily search failed: {type(exc).__name__}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RuntimeError("Unexpected Tavily result format")
        results = []
        for item in payload["results"]:
            url = item.get("url")
            if not url:
                continue
            results.append({"title": item.get("title", ""), "url": url,
                            "publisher": item.get("publisher", ""),
                            "published_at": item.get("published_date", ""),
                            "excerpt": item.get("content", "")})
        return results

    def search_market(self, query: str) -> list[dict]:
        return self._search(query)

    def search_stakeholder(self, query: str) -> list[dict]:
        return self._search(query)
