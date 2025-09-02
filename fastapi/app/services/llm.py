import json
import re
import requests
from typing import List, Dict, Tuple, Optional

from app.config import (
    UNIVERSAL_SYSTEM_PROMPT, REPLY_CHAR_LIMIT, STANCE_VALUES,
    OLLAMA_BASE_URL, MODEL_NAME, HTTP_TIMEOUT_SECONDS, KEEP_ALIVE, NUM_PREDICT_CAP, NUM_CTX,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, PROVIDER_PREFERENCE,
)
from app.models import ChatMessage, ModelReply

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _ensure_url_scheme(u: str) -> str:
    if not u:
        return u
    if u.startswith(("http://", "https://")):
        return u.rstrip("/")
    return ("https://" + u).rstrip("/")

def _stance_hint_line(stance_hint: Optional[str]) -> str:
    """
    Optional hint to bias stance when ambiguous, without contradicting the JSON contract.
    """
    if stance_hint and stance_hint.lower() in STANCE_VALUES:
        return (
            f"\nStance hint: if the stance is ambiguous, prefer '{stance_hint.lower()}'. "
            "Do not print the hint; only reflect it in the JSON value."
        )
    return ""

def build_prompt(history: List[ChatMessage], user_msg: str, stance_hint: Optional[str]) -> str:
    """
    Plain prompt for Ollama-style completion models.
    """
    ctx = []
    for m in history[-6:]:
        if m.role == "user":
            ctx.append(f"User: {m.content}")
        elif m.role == "assistant":
            ctx.append(f"Assistant: {m.content}")
    return (
        f"{UNIVERSAL_SYSTEM_PROMPT}"
        f"{_stance_hint_line(stance_hint)}\n\n"
        + "\n".join(ctx[-6:])
        + f"\n\nUser: {user_msg}\n"
        + "Respond now with the JSON object as specified."
    )

def build_messages(history: List[ChatMessage], user_msg: str, stance_hint: Optional[str]) -> List[Dict[str, str]]:
    """
    Chat-style messages for OpenAI.
    """
    sys_content = f"{UNIVERSAL_SYSTEM_PROMPT}{_stance_hint_line(stance_hint)}"
    msgs: List[Dict[str, str]] = [{"role": "system", "content": sys_content}]
    for m in history[-6:]:
        if m.role in ("user", "assistant"):
            msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": user_msg + "\nRespond now with the JSON object as specified."})
    return msgs

def _extract_first_json(s: str) -> Dict:
    m = JSON_RE.search(s or "")
    if not m:
        raise ValueError("Model output did not contain a JSON object.")
    candidate = m.group(0).strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    return json.loads(candidate)

NEG_CUES = ("disagree","not agree","against","oppose","opposed","en contra","discrepo","no estoy de acuerdo")
POS_CUES = ("agree","in favor","support","apoyo","a favor","de acuerdo")

def _normalize_to_modelreply(raw_text: str, stance_hint: Optional[str]) -> ModelReply:
    data = _extract_first_json(raw_text)
    stance = (data.get("stance","") or "").lower().strip()
    reply  = (data.get("reply","")  or "").strip()

    if stance not in STANCE_VALUES:
        if stance_hint and stance_hint.lower() in STANCE_VALUES:
            stance = stance_hint.lower()
        else:
            text = f"{reply} {raw_text}".lower()
            if any(k in text for k in NEG_CUES):
                stance = "contra"
            elif any(k in text for k in POS_CUES):
                stance = "pro"
            else:
                stance = "contra" if any(w in text for w in (" not ", " no ", "n't", " jamás", " nunca", " sin ")) else "pro"

    if len(reply) > REPLY_CHAR_LIMIT:
        reply = reply[:REPLY_CHAR_LIMIT].rstrip()

    return ModelReply(stance=stance, reply=reply)


def _generate_with_ollama(prompt: str) -> str:
    base = _ensure_url_scheme(OLLAMA_BASE_URL)
    if not base:
        raise RuntimeError("OLLAMA_BASE_URL is empty.")
    url = f"{base}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT_CAP,  
            "temperature": 0.45,             
            "stop": [],
        },
        "keep_alive": KEEP_ALIVE,
    }
    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    j = r.json()
    return (j.get("response") or "").strip()

def _generate_with_openai(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty.")
    base = _ensure_url_scheme(OPENAI_BASE_URL)
    url = f"{base}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.45,   
        "max_tokens": 650,    
        "stop": None,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status()
    j = r.json()
    return (j["choices"][0]["message"]["content"] or "").strip()

def _try_provider_ollama(history: List[ChatMessage], user_msg: str, stance_hint: Optional[str]) -> Tuple[bool, str]:
    try:
        return True, _generate_with_ollama(build_prompt(history, user_msg, stance_hint))
    except Exception as e:
        return False, f"[ollama_error] {e}"

def _try_provider_openai(history: List[ChatMessage], user_msg: str, stance_hint: Optional[str]) -> Tuple[bool, str]:
    try:
        return True, _generate_with_openai(build_messages(history, user_msg, stance_hint))
    except Exception as e:
        return False, f"[openai_error] {e}"

def generate_reply(history: List[ChatMessage], user_msg: str, stance_hint: Optional[str] = None) -> ModelReply:
    """
    Keep the same public API your endpoint expects, now with optional stance_hint.
    Returns ModelReply(stance='pro'|'contra', reply=str).
    """
    providers = (
        (_try_provider_ollama, _try_provider_openai)
        if PROVIDER_PREFERENCE == "ollama_first"
        else (_try_provider_openai, _try_provider_ollama)
    )

    errors: List[str] = []   

    for fn in providers:
        ok, raw = fn(history, user_msg, stance_hint)
        if ok:
            return _normalize_to_modelreply(raw, stance_hint)
        errors.append(raw)

    raise RuntimeError("All providers failed. Details: " + " | ".join(errors))
