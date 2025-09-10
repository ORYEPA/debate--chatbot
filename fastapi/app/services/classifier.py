from typing import Literal, Tuple
from app.models import ChatMessage
from .llm import LLMClient

Intent = Literal["CONTINUE", "EXIT", "NEW_TOPIC"]
UserSide = Literal["affirmative", "negative"]

class IntentLayer:
    ...
class UserStanceDetector:
    ...

def classify_topic_and_user_side_via_llm(user_text: str) -> Tuple[str, UserSide]:
    """
    Extract a concise debate topic and whether the user is on the affirmative or negative side.
    Returns (topic, user_side) where user_side is 'affirmative' | 'negative'.
    """
    llm = LLMClient()
    sys = ChatMessage(
        role="system",
        message=(
            "You are an information extractor. "
            "Given a user message, return a JSON object with fields: "
            '{"topic": "<short debate topic>", "user_side": "affirmative|negative"}.\n'
            "Rules:\n"
            "- 'topic' must be concise, 3–12 words, no quotes.\n"
            "- 'user_side' is 'affirmative' if the user supports the topic, else 'negative'.\n"
            "- Return JSON only, no extra text."
        ),
    )
    usr = ChatMessage(role="user", message=user_text)
    raw = llm.chat([sys, usr], max_tokens=200).strip()

    import json
    topic, side = "General debate topic", "negative"
    try:
        obj = json.loads(raw)
        t = str(obj.get("topic") or "").strip() or topic
        s = str(obj.get("user_side") or "").strip().lower()
        if s not in ("affirmative", "negative"):
            s = side
        return t, s  
    except Exception:
        sys2 = ChatMessage(
            role="system",
            message=(
                "Return exactly two lines:\n"
                "TOPIC: <short topic>\n"
                "SIDE: affirmative|negative"
            ),
        )
        out = llm.chat([sys2, usr], max_tokens=80)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        for ln in lines:
            if ln.upper().startswith("TOPIC:"):
                topic = ln.split(":", 1)[1].strip() or topic
            elif ln.upper().startswith("SIDE:"):
                side = ln.split(":", 1)[1].strip().lower()
        if side not in ("affirmative", "negative"):
            side = "negative"
        return topic, side 
