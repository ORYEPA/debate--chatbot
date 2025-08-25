import json
from app import redis_client, _conv_key

def test_redis_roundtrip():
    cid = "test-123"
    conv = {"meta":{"topic":"T","side":"S","profile_id":"smart_shy"},"messages":[]}
    redis_client.set(_conv_key(cid), json.dumps(conv))
    loaded = json.loads(redis_client.get(_conv_key(cid)))
    assert loaded["meta"]["profile_id"] == "smart_shy"
