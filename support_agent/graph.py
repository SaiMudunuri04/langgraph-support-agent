"""LangGraph state machine: router -> retriever -> answer -> guardrail -> (end|escalate)."""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from .llm import ANSWER_MARKER, INTENT_MARKER
from .tools import kb_lookup_impl, ticket_lookup_impl

FORBIDDEN_OUTPUT_MARKERS = (INTENT_MARKER, ANSWER_MARKER, "<script")
MAX_ANSWER_CHARS = 2000


class AgentState(TypedDict, total=False):
    message: str
    intent: str
    docs: list
    ticket: dict | None
    draft_answer: str
    answer: str
    escalated: bool
    tools_used: list
    validation: dict


def build_graph(llm, data_dir: str):
    def router_node(state: AgentState) -> dict:
        prompt = (
            f"{INTENT_MARKER}\nClassify the intent of this customer support message "
            f"into exactly one word: billing, technical, account, general, escalate.\n"
            f"Message: {state['message']}\nReply with ONLY the intent word."
        )
        intent = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        if intent not in ("billing", "technical", "account", "general", "escalate"):
            intent = "general"
        return {"intent": intent}

    def retriever_node(state: AgentState) -> dict:
        docs = kb_lookup_impl(state["message"], data_dir, k=3)
        ticket = ticket_lookup_impl(state["message"], data_dir)
        used = ["kb_lookup"] + (["ticket_lookup"] if ticket else [])
        return {"docs": docs, "ticket": ticket, "tools_used": used}

    def answer_node(state: AgentState) -> dict:
        def _snippet(doc: dict) -> str:
            body = "\n".join(doc["text"].splitlines()[1:]).strip()
            return " ".join(body.split())[:400]

        sources = "\n".join(
            f"DOC: {d['title']} :: {_snippet(d)}" for d in state.get("docs", [])
        )
        ticket_line = ""
        if state.get("ticket"):
            t = state["ticket"]
            ticket_line = (
                f"TICKET: {t['ticket_id']} status={t['status']} summary={t['summary']}\n"
            )
        prompt = (
            f"{ANSWER_MARKER}\nYou are a helpful support copilot. Answer the customer "
            f"question using ONLY the sources below. Be concise.\n"
            f"Sources:\n{sources}\n{ticket_line}"
            f"Question: {state['message']}\nWrite the final answer."
        )
        draft = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"draft_answer": draft}

    def guardrail_node(state: AgentState) -> dict:
        draft = state.get("draft_answer", "")
        failures = []
        if not draft:
            failures.append("empty_answer")
        if len(draft) > MAX_ANSWER_CHARS:
            failures.append("answer_too_long")
        if any(marker in draft for marker in FORBIDDEN_OUTPUT_MARKERS):
            failures.append("forbidden_marker_leak")
        return {"validation": {"passed": not failures, "failures": failures}}

    def escalate_node(state: AgentState) -> dict:
        return {
            "escalated": True,
            "answer": (
                "I'm connecting you with a human support specialist who can help "
                "further. Your conversation history has been attached to the ticket."
            ),
        }

    def finalize_node(state: AgentState) -> dict:
        return {"answer": state["draft_answer"], "escalated": False}

    def after_router(state: AgentState) -> str:
        return "escalate" if state.get("intent") == "escalate" else "retrieve"

    def after_guardrail(state: AgentState) -> str:
        return "finalize" if state.get("validation", {}).get("passed") else "escalate"

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("answer", answer_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", after_router, {"escalate": "escalate", "retrieve": "retrieve"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "guardrail")
    graph.add_conditional_edges("guardrail", after_guardrail, {"finalize": "finalize", "escalate": "escalate"})
    graph.add_edge("finalize", END)
    graph.add_edge("escalate", END)
    return graph.compile()


def run_chat(app, message: str) -> dict:
    """Run one turn through the compiled graph; returns a JSON-safe result."""
    state = app.invoke({"message": message})
    return {
        "answer": state.get("answer", ""),
        "intent": state.get("intent", "general"),
        "escalated": bool(state.get("escalated", False)),
        "sources": [d["title"] for d in state.get("docs", [])],
        "tools_used": state.get("tools_used", []),
        "validation": state.get("validation", {}),
    }
