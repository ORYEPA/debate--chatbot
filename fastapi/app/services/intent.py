# app/services/intent.py
from typing import Optional
from app.services.llm import LLMClient
from app.models import ChatMessage
from app.config import LLM_MODEL

_ALLOWED = {"topic_change", "continue_topic", "greeting", "chit_chat", "unsafe"}

class IntentLayer:
    """
    Simple intent classifier that uses the LLM.
    Accepts an optional LLMClient (dependency injection). If none is provided, a default one is created.
    Never raises: it always returns a valid label from _ALLOWED.
    """
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(model=LLM_MODEL)

    def classify(self, text: str, current_topic: Optional[str] = None) -> str:
        system = ChatMessage(
            role="system",
            message=(
                "You are an intent classifier.\n"
                "Valid labels: topic_change, continue_topic, greeting, chit_chat, unsafe.\n"
                "Pick the single best label for the user's message given the current topic context.\n"
                "Respond with the label only — lowercase, no punctuation, no extra words."
            ),
        )
        user = ChatMessage(
            role="user",
            message=(
                f"current_topic: {current_topic or '(unknown)'}\n"
                f"user_message: {text}\n"
                "Answer with exactly one label."
            ),
        )
        try:
            out = (self.llm.chat([system, user], max_tokens=8) or "").strip().lower()
            out = out.replace(".", "").replace("`", "").strip()
            if out in _ALLOWED:
                return out
            for lbl in _ALLOWED:
                if lbl in out:
                    return lbl
        except Exception:
            pass
        return "continue_topic"
