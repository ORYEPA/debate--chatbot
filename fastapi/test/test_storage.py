from typing import Any
from app.models import ModelReply
import service.llm as llm_module


def test_ask_stores_messages_and_returns_json(client, monkeypatch):
    def _fake_generate_reply(history, user_msg) -> ModelReply:
        return ModelReply(stance="pro", reply="ok")

    monkeypatch.setattr(llm_module, "generate_reply", _fake_generate_reply, raising=True)

    r1 = client.post("/api/v1/ask", json={"message": "Hello"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert set(d1.keys()) == {"conversation_id", "message", "latency_ms", "stance"}
    assert d1["stance"] in {"pro", "contra"}
    assert isinstance(d1["message"], list) and len(d1["message"]) == 2
    assert d1["message"][0]["role"] == "user" and "message" in d1["message"][0]
    assert d1["message"][1]["role"] == "assistant" and "message" in d1["message"][1]

    cid = d1["conversation_id"]

    r2 = client.post("/api/v1/ask", json={"conversation_id": cid, "message": "Again"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["conversation_id"] == cid
    assert len(d2["message"]) == 4
    assert d2["message"][-1]["role"] == "assistant"
    assert d2["stance"] in {"pro", "contra"}
