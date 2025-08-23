import requests
from typing import List, Optional, Dict
from app.config import (
    OLLAMA_BASE_URL, MODEL_NAME, HTTP_TIMEOUT_SECONDS, KEEP_ALIVE,
    MAX_HISTORY_PAIRS, REPLY_CHAR_LIMIT, UNIVERSAL_RULES
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

def call_llm(system_prompt: str, history: List[ChatMessage], user_msg: str, profile: Dict,
             num_predict_override: Optional[int] = None) -> str:
    prompt = system_prompt + "\n\n"
    for m in history[-(MAX_HISTORY_PAIRS*2):]:
        prompt += f"{'User' if m.role=='user' else 'Assistant'}: {m.message}\n"
    prompt += f"User: {user_msg}\nAssistant:"
    options_num_predict = num_predict_override if num_predict_override is not None else profile["style"].get("num_predict", 300)
    temp = min(profile["style"].get("temperature", 0.7), 0.7)
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "keep_alive": KEEP_ALIVE,
               "options": {"temperature": temp, "top_p": profile["style"].get("top_p", 1.0), "num_predict": options_num_predict}}
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    text = (r.json().get("response") or "").strip()
    if len(text) > REPLY_CHAR_LIMIT:
        text = text[:REPLY_CHAR_LIMIT]
    return text or "I'm formulating my argument, please continue."
