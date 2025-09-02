import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config as app_config
from app.api.v1 import endpoints as endpoints_mod


class InMemoryRedis:
    def __init__(self):
        self._db = {}

    def lrange(self, key: str, start: int, end: int):
        lst = self._db.get(key, [])
        if end == -1:
            end = len(lst) - 1
        return lst[start : end + 1]

    def rpush(self, key: str, value: str):
        self._db.setdefault(key, []).append(value)
        return len(self._db[key])

    def delete(self, key: str):
        self._db.pop(key, None)


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _patch_redis(monkeypatch):
    fake = InMemoryRedis()
    monkeypatch.setattr(app_config, "redis_client", fake, raising=False)
    monkeypatch.setattr(endpoints_mod, "redis_client", fake, raising=False)
    yield
