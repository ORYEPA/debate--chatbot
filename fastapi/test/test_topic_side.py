import re
from app import config


def _strip_topic_prefix(s: str) -> str:
    out = s
    for rx in config.TOPIC_STRIPPERS:
        out = re.sub(rx, "", out, flags=re.IGNORECASE)
    return out.strip()


def test_universal_system_prompt_contract():
    usp = config.UNIVERSAL_SYSTEM_PROMPT
    assert "Return ONLY a valid JSON object" in usp
    assert '"stance":"pro|contra"' in usp 
    assert "neutral" in usp.lower() and "never neutral" in usp  
    assert "Stance:" not in usp  


def test_stance_values_no_neutral():
    assert tuple(config.STANCE_VALUES) == ("pro", "contra")


def test_topic_strippers_examples():
    assert _strip_topic_prefix("let's talk about: climate change") == "climate change"
    assert _strip_topic_prefix("tema: Energía") == "Energía"
    assert _strip_topic_prefix("Topic: Evidence-based policy") == "Evidence-based policy"
    assert _strip_topic_prefix("hablemos de: democracia") == "democracia"
