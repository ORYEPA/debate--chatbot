from app import infer_topic_and_side, DEFAULT_TOPIC, DEFAULT_SIDE

def test_default_topic_side():
    t,s = infer_topic_and_side("hello")
    assert t == DEFAULT_TOPIC and s == DEFAULT_SIDE
