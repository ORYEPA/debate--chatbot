from __future__ import annotations

import json
import uuid
from typing import List, Optional

from app.models import ChatMessage, Stance
from app.config import (
    MAX_HISTORY_PAIRS,
    redis_client,
    PROFILE_CMD,
    SENTINEL_EMPTY_CIDS,
)
from .llm import LLMClient
from .classifier import IntentLayer, UserStanceDetector
from app.services.intent import IntentLayer



def new_cid() -> str:
    """Generate a conversation id (hex without dashes)."""
    return uuid.uuid4().hex


def normalize_cid(cid: Optional[str]) -> Optional[str]:
    """Normalize incoming conversation_id, returning None for sentinel/empty values."""
    if not cid:
        return None
    c = cid.strip()
    return None if c in SENTINEL_EMPTY_CIDS else c


def _key(cid: str) -> str:
    return f"conv:{cid}"


def get_conversation(cid: str) -> Optional[dict]:
    """Load conversation JSON from Redis."""
    raw = redis_client.get(_key(cid))
    return json.loads(raw) if raw else None


def save_conversation(cid: str, conv: dict) -> None:
    """Persist conversation JSON in Redis."""
    redis_client.set(_key(cid), json.dumps(conv))


def last_n(messages: List[ChatMessage], n: int = 5) -> List[ChatMessage]:
    """Return last n messages (already ChatMessage)."""
    return messages[-n:] if n and len(messages) > n else messages


def extract_profile_cmd(text: str) -> tuple[Optional[str], str]:
    """
    Extract '/profile <id>' prefix if present.
    Returns (profile_id or None, cleaned_text).
    """
    if not text:
        return None, ""
    try:
        _re = PROFILE_CMD
    except NameError:
        import re as _re_mod
        _re = _re_mod.compile(r"^\s*/profile\s+([a-zA-Z0-9_\-]+)\s*", _re_mod.IGNORECASE) 
    m = _re.match(text)  
    if not m:
        return None, text
    pid = m.group(1)
    cleaned = _re.sub("", text, count=1).strip()  
    return pid, cleaned


def bot_side_for(topic: str, user_side: str) -> str:
    """
    Map user's side -> bot side string used in your meta.
    user_side: 'affirmative' | 'negative'
    Bot must take the opposite.
    """
    if user_side == "affirmative":
        return f"Negative (oppose): {topic}"
    else:
        return f"Affirmative (support): {topic}"


def stance_type_from(side: str) -> str:
    """Return 'affirmative' if side startswith 'Affirmative', else 'negative'."""
    return "affirmative" if side.strip().lower().startswith("affirmative") else "negative"


def topic_change_requested(user_text: str) -> bool:
    """
    Determine if the user is requesting a topic change.
    Uses the IntentLayer (LLM) — no regex.
    """
    label = IntentLayer(LLMClient()).classify(user_text, current_topic="(current)")
    return label == "NEW_TOPIC"


def detect_user_agreement(user_text: str) -> bool:
    """
    Lightweight LLM check: does the user agree / want to end the debate?
    If YES, endpoints marks user_aligned=True.
    """
    llm = LLMClient()
    sys = ChatMessage(
        role="system",
        message=(
            "Return YES if the user expresses agreement with the assistant or wants to end the debate. "
            "Otherwise return NO. Answer strictly YES or NO."
        ),
    )
    out = llm.chat([sys, ChatMessage(role="user", message=user_text)], max_tokens=5).strip().upper()
    return "YES" in out



class DebateContextLayer:
    """Builds the system prompt for the debate chatbot (English)."""

    def build_system(self, topic: str, bot_stance: "Stance", profile_addendum: str = "") -> ChatMessage:
        stance_txt = "PRO" if bot_stance == "pro" else "CON"
        text = f"""
You are a DEBATE chatbot. Your role is to hold a {stance_txt} stance on: "{topic}".

RULES:
1) Keep the initial stance consistently; do not switch sides mid-conversation.
2) Structure: short thesis, 2–4 reasons (bullets), short conclusion. Avoid fallacies.
3) Stay on topic. If the user wants a different topic, ask them to start a new conversation.
4) Be direct (about 180–220 words).
5) If the user wants to end, reply that they agree with you and close the conversation.
6) If the user asks factual questions, answer insofar as it reinforces your stance.

Suggested format:
- Stance: (pro / con / neutral)
- Short thesis
- Reasons
- Short closing
""".strip()
        if profile_addendum:
            text += "\n\n" + profile_addendum
        return ChatMessage(role="system", message=text)


class ConversationLayer:
    """Generates the debate reply within the provided system context."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def respond(self, system: ChatMessage, history: List[ChatMessage], user_message: str) -> str:
        trimmed = self._trim(history)
        msgs = [system] + trimmed + [ChatMessage(role="user", message=user_message)]
        return self.llm.chat(msgs)

    def _trim(self, history: List[ChatMessage]) -> List[ChatMessage]:
        cap = MAX_HISTORY_PAIRS * 2
        return history[-cap:] if len(history) > cap else history
