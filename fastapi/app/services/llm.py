import requests
from fastapi import HTTPException
from typing import List, Optional, Dict
from app.config import (
    OLLAMA_BASE_URL, MODEL_NAME, HTTP_TIMEOUT_SECONDS, KEEP_ALIVE,
    MAX_HISTORY_PAIRS, REPLY_CHAR_LIMIT, UNIVERSAL_RULES,
    NUM_PREDICT_CAP, NUM_CTX
)
from app.models import ChatMessage

def build_system_prompt(profile: Dict, topic: str, side: str) -> str:
    return (
        f"{profile['system']}\n\n{UNIVERSAL_RULES}\n"
        f"TOPIC: {topic}\nSIDE: {side}\n"
        "Never be neutral. Persuade the user to adopt your SIDE.\n"
        "Always begin with 'Stance: Affirmative' or 'Stance: Negative' matching SIDE.\n"
        "Use serious, credible tone — no jokes or onomatopoeia.\n"
    )

def _generate(payload: Dict, timeout: float) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def call_llm(system_prompt: str, history: List[ChatMessage], user_msg: str, profile: Dict,
             num_predict_override: Optional[int] = None) -> str:

    prompt = system_prompt + "\n\n"
    for m in history[-(MAX_HISTORY_PAIRS*2):]:
        prompt += f"{'User' if m.role=='user' else 'Assistant'}: {m.message}\n"
    prompt += f"User: {user_msg}\nAssistant:"

    base_num_predict = profile["style"].get("num_predict", 300)
    npredict = min(base_num_predict if num_predict_override is None else num_predict_override, NUM_PREDICT_CAP)
    temp = min(profile["style"].get("temperature", 0.7), 0.7)

    options = {
        "temperature": temp,
        "top_p": profile["style"].get("top_p", 1.0),
        "num_predict": npredict,
        "num_ctx": NUM_CTX,
        "repeat_penalty": 1.05,
        "stop": ["\nUser:"],
    }

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": options,
    }

    try:
        text = _generate(payload, timeout=HTTP_TIMEOUT_SECONDS)
    except requests.Timeout:
        payload["options"]["num_predict"] = min(120, NUM_PREDICT_CAP)
        try:
            text = _generate(payload, timeout=HTTP_TIMEOUT_SECONDS / 2)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Ollama timeout (retry): {type(e).__name__}: {e}")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {type(e).__name__}: {e}")

    if len(text) > REPLY_CHAR_LIMIT:
        text = text[:REPLY_CHAR_LIMIT]
    return text or "I'm formulating my argument, please continue."
