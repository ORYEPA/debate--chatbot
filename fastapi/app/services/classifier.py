import json
import re
import requests
from typing import Tuple
from app.config import (
    MODEL_NAME, OLLAMA_BASE_URL, HTTP_TIMEOUT_SECONDS, KEEP_ALIVE, DEFAULT_TOPIC,
    LLM_MOCK
)
from app.services.conversation import clean_topic

_NEG_PATTERNS = [
    r"\bis not\b", r"\bare not\b", r"\bisn't\b", r"\baren't\b",
    r"\bno es\b", r"\bno son\b", r"\bno está\b", r"\bno esta\b",
    r"\bnot\b", r"\bno\b"
]

def _has_negation(s: str) -> bool:
    s_low = " " + (s or "").lower() + " "
    return any(re.search(p, s_low) for p in _NEG_PATTERNS)

def _to_positive_canonical(s: str) -> str:
    s2 = " " + (s or "") + " "
    replacements = [
        (r"\bis\s+not\s+\b", " is "),
        (r"\bare\s+not\s+\b", " are "),
        (r"\bisn'?t\s+\b", " is "),
        (r"\baren'?t\s+\b", " are "),
        (r"\bno\s+es\s+\b", " es "),
        (r"\bno\s+son\s+\b", " son "),
        (r"\bno\s+esta\s+\b", " esta "),
        (r"\bno\s+está\s+\b", " está "),
        (r"\bnot\s+\b", " "),
        (r"\bno\s+\b", " "),
    ]
    for pat, rep in replacements:
        s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def classify_topic_and_user_side_via_llm(user_text: str) -> Tuple[str, str]:
    """
    Devuelve (topic_positivo, user_side).
    user_side ∈ {'affirmative','negative'}
    """
    if LLM_MOCK:
        txt = clean_topic(user_text) or DEFAULT_TOPIC
        user_side = "negative" if (" not " in (" "+user_text.lower()+" ") or "en contra" in user_text.lower()) else "affirmative"
        if _has_negation(txt):
            txt = _to_positive_canonical(txt)
            user_side = "affirmative" if user_side == "negative" else "negative"
        return txt or DEFAULT_TOPIC, user_side

    instruction = (
        "You are a stance classifier. Produce ONLY a strict JSON object with two fields:\n"
        '{ "topic": "<concise positive proposition>", "user_side": "affirmative" | "negative" }\n'
        "- 'topic' MUST be a short POSITIVE/CANONICAL proposition WITHOUT NEGATIONS.\n"
        "  Examples:\n"
        "  User: 'I don't believe the moon is made of cheese' -> topic: 'the moon is made of cheese', user_side: 'negative'\n"
        "  User: 'Estoy en contra de legalizar X' -> topic: 'legalizar X', user_side: 'negative'\n"
        "  User: 'I support universal basic income' -> topic: 'universal basic income should be implemented', user_side: 'affirmative'\n"
        "- 'user_side' is relative to THAT positive proposition.\n"
        "No explanations. No extra text. JSON only."
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{instruction}\n\nUser message:\n{user_text}\n\nJSON:",
        "stream": False, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": 200},
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        raw = (r.json().get("response") or "").strip()
        i, j = raw.find("{"), raw.rfind("}")
        if i != -1 and j != -1:
            jso = json.loads(raw[i:j+1])
            topic = clean_topic(str(jso.get("topic","")).strip()) or DEFAULT_TOPIC
            side  = str(jso.get("user_side","")).strip().lower()
            if side not in ("affirmative","negative"): side = "affirmative"
            if _has_negation(topic):
                topic = _to_positive_canonical(topic) or topic
                side = "affirmative" if side=="negative" else "negative"
            return topic, side
    except Exception:
        pass
    text = clean_topic(user_text) or DEFAULT_TOPIC
    side = "negative" if _has_negation(text) else "affirmative"
    if _has_negation(text): text = _to_positive_canonical(text)
    return text or DEFAULT_TOPIC, side
