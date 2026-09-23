"""FastAPI serving layer for the support copilot."""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .graph import build_graph, run_chat
from .llm import get_llm
from .tracing import init_tracing

log = logging.getLogger(__name__)

settings = Settings.load()
init_tracing(settings)
_llm = get_llm(settings)
_app_graph = build_graph(_llm, settings.data_dir)

app = FastAPI(title="LangGraph Support Agent", version="0.1.0")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    escalated: bool
    sources: list[str]
    tools_used: list[str]
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_mode": settings.mock_mode}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    try:
        result = run_chat(_app_graph, req.message)
    except Exception as exc:  # never leak stack traces to callers
        log.exception("chat failed")
        raise HTTPException(status_code=500, detail="internal error") from exc
    # Serving-layer output validation (defense in depth, mirrors graph guardrail)
    answer = (result.get("answer") or "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="empty answer from agent")
    if len(answer) > 4000:
        answer = answer[:4000]
    latency_ms = (time.perf_counter() - started) * 1000
    return ChatResponse(
        answer=answer,
        intent=result.get("intent", "general"),
        escalated=bool(result.get("escalated", False)),
        sources=result.get("sources", []),
        tools_used=result.get("tools_used", []),
        latency_ms=round(latency_ms, 2),
    )
