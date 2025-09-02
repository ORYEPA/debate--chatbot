import os
import sys
import types
import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


os.environ.setdefault("OLLAMA_BASE_URL", "http://fake-ollama:11434")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

if "app.profiles" not in sys.modules:
    profiles_shim = types.ModuleType("app.profiles")
    profiles_shim.PROFILE = {
        "smart_shy":       {"id": "smart_shy", "name": "Athena"},
        "conspiracy_edge": {"id": "conspiracy_edge", "name": "Raven"},
        "rude_arrogant":   {"id": "rude_arrogant", "name": "Edge"},
    }
    profiles_shim.AVAILABLE_PROFILES = tuple(profiles_shim.PROFILE.keys())
    sys.modules["app.profiles"] = profiles_shim

import requests as _requests 
_ORIG_GET = _requests.get

class _OKResp:
    status_code = 200
    def raise_for_status(self): return None
    def json(self): return {"ok": True}

def _fake_get(url, *args, **kwargs):
    if url and "/api/tags" in url:
        return _OKResp()
    return _ORIG_GET(url, *args, **kwargs)

_requests.get = _fake_get 

from app.main import app 


class FakeRedis:
    def __init__(self):
        self._kv = {}
        self._hash = {}
    def ping(self): return True
    def get(self, k): return self._kv.get(k)
    def set(self, k, v): self._kv[k] = v; return True
    def delete(self, k): self._kv.pop(k, None); self._hash.pop(k, None); return 1
    def hget(self, name, key): return self._hash.get(name, {}).get(key)
    def hset(self, name, key, value):
        self._hash.setdefault(name, {})[key] = value; return 1


@pytest.fixture(scope="session")
def client():
    fake = FakeRedis()

    import app.config as cfg
    cfg.redis_client = fake

    try:
        import app.services.conversation as conv
        conv.redis_client = fake
    except Exception:
        pass

    try:
        import app.api.v1.endpoints as endpoints
        endpoints.redis_client = fake

        def _fake_health():
            return {
                "status": "ok",
                "redis": True,
                "ollama_base_url": os.environ.get("OLLAMA_BASE_URL"),
                "ollama_reachable": True,
                "ollama_error": None,
            }
        endpoints.health = _fake_health

        def _proxy_generate_reply(*args, **kwargs):
            import app.services.llm as _llm
            return _llm.generate_reply(*args, **kwargs)
        endpoints.generate_reply = _proxy_generate_reply
    except Exception:
        pass

    return TestClient(app)
