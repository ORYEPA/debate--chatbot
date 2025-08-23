import time, json, requests
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

def call_llm(system_prompt: str, history: List[ChatMessage], user_msg: str, profile: Dict,
             num_predict_override: Optional[int] = None) -> str:
    prompt = system_prompt + "\n\n"
    for m in history[-(MAX_HISTORY_PAIRS*2):]:
        prompt += f"{'User' if m.role=='user' else 'Assistant'}: {m.message}\n"
    prompt += f"User: {user_msg}\nAssistant:"

    base_num_predict = profile["style"].get("num_predict", 300)
    npredict = min(
        base_num_predict if num_predict_override is None else num_predict_override,
        NUM_PREDICT_CAP
    )
    temp = min(profile["style"].get("temperature", 0.7), 0.7)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,                     
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": temp,
            "top_p": profile["style"].get("top_p", 1.0),
            "num_predict": npredict,
            "num_ctx": NUM_CTX,
            "repeat_penalty": 1.05,
            "stop": ["\nUser:"],           
        },
    }

    start = time.time()
    buf: list[str] = []

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json=payload,
            timeout=(5, HTTP_TIMEOUT_SECONDS),
            stream=True,                   
        ) as r:
            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "response" in obj and obj["response"]:
                    buf.append(obj["response"])
                    if sum(len(x) for x in buf) > REPLY_CHAR_LIMIT:
                        break

                if obj.get("done"):
                    break

                if time.time() - start > HTTP_TIMEOUT_SECONDS - 1:
                    break

    except requests.RequestException as e:
        return (
            "Stance: Affirmative\n"
            "Thesis: I’ll keep it concise.\n"
            "Reasons: HYPOTHESIS—signal constraints; prior evidence trends; practical plausibility tests.\n"
            "Concession: mainstream counterpoints exist but remain unproven in this context.\n"
            "Closing: consider the anomalies and let’s keep examining them."
        )

    text = "".join(buf).strip()
    if not text:
        return (
            "Stance: Negative\n"
            "Thesis: Given the constraints, here’s a condensed counterpoint.\n"
            "Reasons: HYPOTHESIS—measurement anomalies; model mismatches; empirical edge cases.\n"
            "Concession: competing views have traction; Closing: weigh these gaps and reconsider."
        )

    if len(text) > REPLY_CHAR_LIMIT:
        text = text[:REPLY_CHAR_LIMIT]
    return text
