"""LLM backends: AWS Bedrock (live) and a deterministic mock (tests/CI/eval).

The mock is keyword-driven and deterministic so the test suite, the golden-set
eval, and CI are fully reproducible without any API keys.
"""
from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

INTENT_MARKER = "INTENT-CLASSIFY"
ANSWER_MARKER = "ANSWER-GENERATE"

_BILLING = ["bill", "invoice", "charg", "refund", "payment", "pric", "subscription"]
_TECHNICAL = ["error", "bug", "crash", "not working", "fail", "api", "login",
              "log in", "timeout", "500", "broken", "install"]
_ACCOUNT = ["password", "reset", "username", "profile", "account", "plan",
            "upgrade", "downgrade"]
_ESCALATE = ["human", "real person", "manager", "supervisor", "sue", "lawyer"]


def _classify_intent(message: str) -> str:
    text = message.lower()
    for kw in _ESCALATE:
        if kw in text:
            return "escalate"
    for kw in _BILLING:
        if kw in text:
            return "billing"
    for kw in _TECHNICAL:
        if kw in text:
            return "technical"
    for kw in _ACCOUNT:
        if kw in text:
            return "account"
    return "general"


def _mock_reply(prompt: str) -> str:
    if INTENT_MARKER in prompt:
        m = re.search(r"Message:\s*(.*?)\s*Reply", prompt, re.S)
        message = m.group(1).strip() if m else prompt
        return _classify_intent(message)
    if ANSWER_MARKER in prompt:
        docs = re.findall(r"DOC:\s*(.*?)\s*::\s*(.*?)\s*(?:\n|$)", prompt)
        titles = [t.strip() for t, _ in docs]
        top_snippet = docs[0][1].strip() if docs else "I don't have documentation for that."
        ticket = re.search(r"TICKET:\s*(\S+)\s+status=(\S+)\s+summary=(.*?)\s*(?:\n|$)", prompt)
        ticket_sentence = ""
        if ticket:
            ticket_sentence = (
                f" Your ticket {ticket.group(1)} is currently {ticket.group(2)}: "
                f"{ticket.group(3).strip()}."
            )
        doc_bit = f"According to our documentation ({', '.join(titles)}): {top_snippet}." if titles else top_snippet
        return f"{doc_bit}{ticket_sentence} Let me know if you need anything else."
    return "mock-response"


def _last_human_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    parts = [m.content for m in messages if hasattr(m, "content")]
    return "\n".join(str(p) for p in parts)


class MockChatLLM(BaseChatModel):
    """Deterministic stand-in for Bedrock. No network, no keys, no cost."""

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = _mock_reply(_last_human_text(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


class BedrockChatLLM(BaseChatModel):
    """Live backend: AWS Bedrock converse API. Requires AWS credentials."""

    model_id: str
    region: str = "us-east-1"
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "bedrock-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        import boto3

        client = boto3.client("bedrock-runtime", region_name=self.region)
        system, convo = [], []
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if isinstance(m, SystemMessage):
                system.append({"text": content})
            elif isinstance(m, HumanMessage):
                convo.append({"role": "user", "content": [{"text": content}]})
            elif isinstance(m, AIMessage):
                convo.append({"role": "assistant", "content": [{"text": content}]})
        request = {
            "modelId": self.model_id,
            "messages": convo,
            "inferenceConfig": {"maxTokens": self.max_tokens},
        }
        if system:
            request["system"] = system
        resp = client.converse(**request)
        text = "".join(
            block.get("text", "")
            for block in resp["output"]["message"]["content"]
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_llm(settings) -> BaseChatModel:
    if settings.mock_mode:
        return MockChatLLM()
    return BedrockChatLLM(
        model_id=settings.bedrock_model_id,
        region=settings.aws_region,
        max_tokens=settings.bedrock_max_tokens,
    )
