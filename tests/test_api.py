from fastapi.testclient import TestClient

from support_agent.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_contract():
    r = client.post("/chat", json={"message": "What is your refund policy?"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "billing"
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["escalated"], bool)
    assert isinstance(body["sources"], list)
    assert isinstance(body["tools_used"], list)
    assert body["latency_ms"] >= 0


def test_chat_rejects_empty_message():
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_rejects_oversize_message():
    r = client.post("/chat", json={"message": "x" * 2001})
    assert r.status_code == 422
