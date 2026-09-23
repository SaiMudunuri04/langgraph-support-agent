import os

from support_agent import tools
from support_agent.tools import kb_lookup_impl, ticket_lookup_impl


def test_kb_lookup_returns_relevant_doc():
    tools.reset_tools()
    docs = kb_lookup_impl("refund policy invoice", os.environ["DATA_DIR"], k=2)
    assert docs, "expected at least one hit"
    titles = [d["title"].lower() for d in docs]
    assert any("refund" in t or "billing" in t for t in titles)
    assert all(d["score"] > 0 for d in docs)


def test_kb_lookup_no_match_returns_empty():
    tools.reset_tools()
    docs = kb_lookup_impl("xyzzy quantum platypus", os.environ["DATA_DIR"], k=3)
    assert docs == []


def test_ticket_lookup_by_id():
    tools.reset_tools()
    t = ticket_lookup_impl("Where is T-1002?", os.environ["DATA_DIR"])
    assert t is not None
    assert t["ticket_id"] == "T-1002"
    assert t["status"] == "investigating"


def test_ticket_lookup_unknown_returns_none():
    tools.reset_tools()
    assert ticket_lookup_impl("T-9999", os.environ["DATA_DIR"]) is None
