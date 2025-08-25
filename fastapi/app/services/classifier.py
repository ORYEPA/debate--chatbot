import json, re, requests
from typing import Tuple

from app.config import (
    OLLAMA_BASE_URL, MODEL_NAME, HTTP_TIMEOUT_SECONDS, KEEP_ALIVE
)

_LEAD_STRIP = re.compile(
    r"^\s*(?:i\s*(?:think|believe)|creo\s+que|pienso\s+que|considero\s+que|it'?s|es)\s+", re.I)
_NEG_PAT = re.compile(r"\b(?:don'?t|do\s+not|isn'?t|aren'?t|no|not|nunca|jam[aá]s|en\s+contra|against|oppose|disagree|incorrect|false|falso)\b", re.I)

def _clean_topic(text: str) -> str:
    t = (text or "").strip()
    t = _LEAD_STRIP.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:160] if len(t) > 160 else t

def _to_positive_proposition(t: str) -> str:
    s = " " + (t or "") + " "
    repl = [
        (r"\bis\s+not\s+", " is "),
        (r"\bare\s+not\s+", " are "),
        (r"\bisn'?t\s+", " is "),
        (r"\baren'?t\s+", " are "),
        (r"\bdo\s+not\s+", " "),
        (r"\bdon'?t\s+", " "),
        (r"\bno\s+", " "),
        (r"\bnot\s+", " "),
        (r"\bno\s+es\s+", " es "),
        (r"\bno\s+son\s+", " son "),
    ]
    for pat, rep in repl:
        s = re.sub(pat, rep, s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _heuristic_user_side(user_text: str) -> str:
    """Regla determinista y robusta: si hay negación clara -> 'negative', de lo contrario 'affirmative'."""
    return "negative" if _NEG_PAT.search(user_text or "") else "affirmative"

def _llm_canonical_topic(user_text: str) -> str:
    """Opcional: pedir al LLM SOLO el tema en forma afirmativa; si falla, usar heurística."""
    instr = (
        "Return ONLY a short positive proposition (no negation, no stance, no extra text). "
        "Example: 'I don't believe cats are better than dogs' -> 'cats are better than dogs'."
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{instr}\nUser: {user_text}\nProposition:",
        "stream": False, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": 48},
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        txt = (r.json().get("response") or "").strip()
        txt = txt.splitlines()[0].strip()
        if txt:
            return txt
    except Exception:
        pass
    base = _clean_topic(user_text)
    return _to_positive_proposition(base) or base or "the claim"

def classify_topic_and_user_side_via_llm(user_text: str) -> Tuple[str, str]:
    """
    1) Determina la postura del usuario con REGLA DETERMINISTA (no depende del LLM).
    2) Obtiene el tema canónico afirmativo (intenta LLM; si falla, heurística).
    """
    user_side = _heuristic_user_side(user_text)
    topic = _llm_canonical_topic(user_text)
    return topic, user_side
