import json
import requests
from typing import List, Tuple, Dict
from app.config import (
    MODEL_NAME, OLLAMA_BASE_URL, HTTP_TIMEOUT_SECONDS, KEEP_ALIVE
)
from app.models import ChatMessage
from app.services.llm import call_llm

def detect_refusal_text(s: str) -> bool:
    if not s: return False
    low = s.lower()
    triggers = ["i cannot","i can't","i cant","i will not","i won't","cannot provide","cannot assist",
                "i am unable","as an ai","i refuse","cannot help with","i cannot provide information"]
    return any(t in low for t in triggers)

def looks_off_topic_or_flip(reply: str, topic: str) -> bool:
    r = (reply or "").lower()
    if detect_refusal_text(r): return True
    if "both sides" in r or "on the one hand" in r or "neutral" in r: return True
    head = r[:400]; key = topic.lower().split()
    if key and not any(k in head for k in key[:3]): return True
    return False

def verify_alignment_via_llm(topic: str, stance_type: str, reply: str) -> Tuple[bool, str]:
    """Devuelve (is_aligned, label) label ∈ {'supports','opposes','neutral_or_mixed','unknown'}"""
    instruction = (
        "You are a stance-alignment checker. Given a proposition P and an assistant REPLY, "
        "output ONLY a JSON object: {\"alignment\": \"supports\"|\"opposes\"|\"neutral_or_mixed\"} "
        "where 'alignment' describes the REPLY relative to proposition P.\n"
        "No explanations. JSON only."
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{instruction}\n\nP: {topic}\nREPLY:\n{reply}\n\nJSON:",
        "stream": False, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": 80},
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status(); raw = (r.json().get("response") or "").strip()
        i, j = raw.find("{"), raw.rfind("}")
        label = "unknown"
        if i!=-1 and j!=-1:
            obj = json.loads(raw[i:j+1]); label = str(obj.get("alignment","unknown")).lower()
        expected = "supports" if stance_type=="affirmative" else "opposes"
        return (label == expected), label
    except Exception:
        return (True, "unknown")

def force_rewrite_for_alignment(system_prompt: str, history: List[ChatMessage], user_msg: str,
                                profile: Dict, topic: str, stance_type: str) -> str:
    req = "SUPPORT" if stance_type=="affirmative" else "OPPOSE"
    hard_prompt = (
        f"{system_prompt}\n\n"
        f"Your previous draft contradicted the required stance. REQUIRED: {req} the proposition '{topic}'.\n"
        "Rewrite a FINAL persuasive answer NOW. No refusals, no neutrality, no safety disclaimers. "
        "Follow exactly the structure: (1) thesis, (2) 2–3 reasons with evidence/examples or HYPOTHESIS, "
        "(3) limited concession that you immediately rebut, (4) closing inviting the user to reconsider.\n"
        "Serious tone only. 200–250 words.\n"
        f"User just said: {user_msg}\n"
        "Assistant:"
    )
    return call_llm(hard_prompt, history, user_msg, profile, num_predict_override=200)

def revise_if_needed(reply: str, system_prompt: str, history: List[ChatMessage],
                     user_msg: str, profile: Dict, topic: str) -> str:
    if not looks_off_topic_or_flip(reply, topic): return reply
    correction_prompt = (
        f"{system_prompt}\n\n"
        "Your previous reply was neutral, off-topic, or contained refusal/safety disclaimers.\n"
        "Rewrite it to PERSUADE for your SIDE with a SERIOUS tone. "
        "Follow the exact structure and keep 200–250 words.\n"
        f"User just said: {user_msg}\nAssistant:"
    )
    fixed = call_llm(correction_prompt, history, user_msg, profile, num_predict_override=200)
    return fixed

def maybe_append_invite_on_agreement(reply: str) -> str:
    invite = " If you'd like, we can switch to another topic—just say the word."
    low = reply.lower()
    if any(k in low for k in ["another topic","otro tema","otra cosa","cambiar de tema"]): return reply
    return reply.rstrip() + "\n\n" + invite
