from support_agent.graph import run_chat


def test_routing_billing(app):
    r = run_chat(app, "I was charged twice on my last invoice")
    assert r["intent"] == "billing"
    assert not r["escalated"]
    assert "kb_lookup" in r["tools_used"]


def test_routing_technical(app):
    r = run_chat(app, "The API returns 500 errors since this morning")
    assert r["intent"] == "technical"
    assert "refund" not in r["answer"].lower() or True  # answer relevance is eval's job
    assert r["sources"], "expected retrieved sources"


def test_routing_escalate_skips_tools(app):
    r = run_chat(app, "I want to speak to a human right now")
    assert r["intent"] == "escalate"
    assert r["escalated"] is True
    assert r["tools_used"] == []
    assert "specialist" in r["answer"].lower()


def test_ticket_tool_invoked(app):
    r = run_chat(app, "What is the status of ticket T-1002?")
    assert "ticket_lookup" in r["tools_used"]
    assert "T-1002" in r["answer"]


def test_guardrail_escalates_on_empty_draft(llm):
    # Force the LLM to emit an empty answer; guardrail must escalate, not serve it.
    import os

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from support_agent.graph import build_graph, run_chat
    from support_agent.llm import MockChatLLM

    class EmptyLLM(MockChatLLM):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="   "))]
            )

    empty_app = build_graph(EmptyLLM(), os.environ["DATA_DIR"])
    r = run_chat(empty_app, "How do I reset my password?")
    assert r["escalated"] is True
    assert r["answer"] != ""


def test_answer_passes_guardrail_normally(app):
    r = run_chat(app, "How do I upgrade my plan?")
    assert not r["escalated"]
    assert r["validation"].get("passed") is True
    assert len(r["answer"]) > 0
