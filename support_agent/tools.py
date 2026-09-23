"""Agent tools: knowledge-base lookup and ticket-record lookup.

Both operate on bundled synthetic data (data/docs, data/tickets.json).
Exposed as plain functions for the graph and as LangChain tools for reuse.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.tools import tool

from .retriever import BM25Retriever, load_docs

_retriever: BM25Retriever | None = None
_tickets: list[dict] | None = None


def _get_retriever(data_dir: str) -> BM25Retriever:
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever(load_docs(data_dir))
    return _retriever


def reset_tools() -> None:
    """Clear cached retriever/tickets (used by tests)."""
    global _retriever, _tickets
    _retriever, _tickets = None, None


def _get_tickets(data_dir: str) -> list[dict]:
    global _tickets
    if _tickets is None:
        _tickets = json.loads(Path(data_dir, "tickets.json").read_text(encoding="utf-8"))
    return _tickets


def kb_lookup_impl(query: str, data_dir: str, k: int = 3) -> list[dict]:
    """Search the bundled knowledge base. Returns [{id, title, text, score}]."""
    return _get_retriever(data_dir).search(query, k=k)


def ticket_lookup_impl(query: str, data_dir: str) -> dict | None:
    """Find a ticket by ID (e.g. 'T-1002') or keyword match on summary."""
    tickets = _get_tickets(data_dir)
    m = re.search(r"T-\d+", query.upper())
    if m:
        for t in tickets:
            if t["ticket_id"].upper() == m.group(0):
                return t
    q = query.lower()
    best, best_hits = None, 0
    for t in tickets:
        hay = f"{t['ticket_id']} {t['summary']} {t['status']}".lower()
        hits = sum(1 for w in q.split() if len(w) > 2 and w in hay)
        if hits > best_hits:
            best, best_hits = t, hits
    return best if best_hits else None


@tool
def kb_lookup(query: str) -> str:
    """Search the support knowledge base for relevant help articles."""
    import os

    docs = kb_lookup_impl(query, os.getenv("DATA_DIR", "data"))
    return json.dumps([{"title": d["title"], "snippet": d["text"][:300]} for d in docs])


@tool
def ticket_lookup(query: str) -> str:
    """Look up a customer support ticket by ID or keywords."""
    import os

    t = ticket_lookup_impl(query, os.getenv("DATA_DIR", "data"))
    return json.dumps(t) if t else json.dumps({"found": False})
