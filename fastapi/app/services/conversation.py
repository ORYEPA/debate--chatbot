import json
import re
import uuid
from typing import List, Tuple, Optional
from app.config import redis_client, PROFILE_CMD, TOPIC_STRIPPERS, SENTINEL_EMPTY_CIDS, DEFAULT_TOPIC, DEFAULT_SIDE
from app.models import ChatMessage

def conv_key(cid: str) -> str:
    return f"conv:{cid}"

def new_cid() -> str:
    return str(uuid.uuid4())

def get_conversation(cid: str) -> Optional[dict]:
    raw = redis_client.get(conv_key(cid))
    return json.loads(raw) if raw else None

def save_conversation(cid: str, conv: dict) -> None:
    redis_client.set(conv_key(cid), json.dumps(conv))

def last_n(messages: List[ChatMessage], n: int = 5) -> List[ChatMessage]:
    return messages[-n:]

def extract_profile_cmd(text: str) -> Tuple[Optional[str], str]:
    m = PROFILE_CMD.match(text or "")
    if m:
        pid = m.group(1).strip()
        rest = text[m.end():].lstrip()
        return pid, rest
    return None, text

def normalize_cid(cid: Optional[str]) -> Optional[str]:
    if cid is None:
        return None
    s = str(cid).strip()
    return None if s in SENTINEL_EMPTY_CIDS else s

def clean_topic(text: str) -> str:
    t = (text or "").strip()
    for pat in TOPIC_STRIPPERS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*/profile\s+[a-zA-Z0-9_\-]+\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:160] if len(t) > 160 else t

def topic_change_requested(user_text: str) -> Optional[str]:
    t = (user_text or "").strip()
    low = t.lower()
    triggers = ["cambiemos de tema", "cambiar de tema", "hablemos de", "hablar de", "otra cosa", "new topic", "talk about", "change topic"]
    if any(x in low for x in triggers):
        m = re.search(r"(hablemos de|hablar de|talk about|sobre)\s+(.+)", low)
        if m:
            candidate = t[m.start(2):].strip()
            return clean_topic(candidate)
        return ""
    return None

def bot_side_for(topic: str, user_side_label: str) -> str:
    return f"Negative (oppose): {topic}" if user_side_label == "affirmative" else f"Affirmative (support): {topic}"

def stance_type_from(bot_side: str) -> str:
    return "affirmative" if (bot_side or "").lower().startswith("affirmative") else "negative"

def detect_user_agreement(user_text: str) -> bool:
    t = (user_text or "").lower()
    pats = [
        "estoy de acuerdo", "tienes razón", "tienes razon", "me convenciste",
        "me has convencido", "cambie de opinion", "cambié de opinión",
        "i agree", "you're right", "you are right", "you convinced me",
        "changed my mind", "i now agree"
    ]
    return any(p in t for p in pats)
