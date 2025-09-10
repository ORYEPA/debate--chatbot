from typing import List, Optional
import os
import requests
import litellm
from litellm.exceptions import APIConnectionError, APIError, RateLimitError, NotFoundError

from app.config import (
    LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE, LLM_TIMEOUT,
    OPENAI_MODEL, OPENAI_BASE_URL, OPENAI_API_KEY, PROVIDER_PREFERENCE,
    REPLY_CHAR_LIMIT, MAX_OUTPUT_TOKENS,
)
from app.models import ChatMessage, ModelReply, Stance

if (OPENAI_API_KEY or "").strip():
    litellm.api_key = OPENAI_API_KEY.strip()


def _provider_from_model(model: str) -> str:
    """Devuelve 'ollama' si el modelo empieza con 'ollama/', si no 'openai' por defecto."""
    return (model.split("/", 1)[0] if "/" in model else "openai").lower()


def _normalized_openai_base() -> str:
    """Asegura que el base URL de OpenAI termine en /v1."""
    base = (OPENAI_BASE_URL or "https://api.openai.com").rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _api_base_for(model: str) -> Optional[str]:
    prov = _provider_from_model(model)
    if prov == "ollama":
        return (LLM_BASE_URL or "").strip() or None
    if prov == "openai":
        return _normalized_openai_base()
    return None


def _ollama_up() -> bool:
    """Ping mínimo para saber si Ollama está reachable, evitando que LiteLLM lance excepción."""
    base = (LLM_BASE_URL or "").strip()
    if not base:
        return False
    try:
        r = requests.get(f"{base.rstrip('/')}/api/tags", timeout=0.8)
        r.raise_for_status()
        return True
    except Exception:
        return False


def _extract_text(resp) -> str:
    try:
        text = resp.choices[0].message["content"]
    except Exception:
        text = getattr(resp, "choices", [{}])[0].get("message", {}).get("content", "") or ""
    if REPLY_CHAR_LIMIT and REPLY_CHAR_LIMIT > 0:
        text = text[:REPLY_CHAR_LIMIT]
    return text


class LLMClient:
    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE, timeout: float = LLM_TIMEOUT):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def _try_completion(self, model: str, messages: List[ChatMessage], max_tokens: Optional[int]) -> str:
        payload = [{"role": m.role, "content": m.message} for m in messages]
        kwargs = dict(
            model=model,
            messages=payload,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif MAX_OUTPUT_TOKENS and MAX_OUTPUT_TOKENS > 0:
            kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

        api_base = _api_base_for(model)
        if api_base:
            kwargs["api_base"] = api_base 

        resp = litellm.completion(**kwargs)
        return _extract_text(resp)

    def chat(self, messages: List[ChatMessage], max_tokens: Optional[int] = None) -> str:
        pref = (PROVIDER_PREFERENCE or "").lower()
        primary = self.model
        secondary = (OPENAI_MODEL or "gpt-4o-mini").strip()

        if pref == "openai_first":
            order = [secondary, primary]
        elif pref == "openai_only":
            order = [secondary]
        elif pref == "ollama_only":
            order = [primary]
        else:
            order = [primary, secondary]

        ollama_ok = _ollama_up()
        filtered_order: List[str] = []
        for m in order:
            prov = _provider_from_model(m)
            if prov == "ollama" and not ollama_ok:
                continue
            if prov == "openai" and not (OPENAI_API_KEY or "").strip():
                continue
            filtered_order.append(m)

        if not filtered_order:
            raise RuntimeError("No hay proveedores LLM disponibles (Ollama no reachable y/o falta OPENAI_API_KEY).")

        last_exc: Optional[Exception] = None
        for model in filtered_order:
            try:
                return self._try_completion(model, messages, max_tokens)
            except (APIConnectionError, APIError, RateLimitError, NotFoundError, Exception) as e:
                last_exc = e
                continue

        if last_exc:
            raise last_exc
        raise RuntimeError("No provider available for completion")


def generate_reply(history: List[ChatMessage], user_text: str, stance_hint: Stance) -> ModelReply:
    stance_upper = "PRO" if stance_hint == "pro" else "CON"
    system = ChatMessage(
        role="system",
        message=(
            f"You are a DEBATE chatbot. Hold a {stance_upper} stance on the current topic under discussion.\n"
            "Rules:\n"
            "1) Keep your stance consistently; do not switch sides.\n"
            "2) Structure: short thesis, 2–4 reasons (bullets), short conclusion. Avoid fallacies.\n"
            "3) Stay on topic. If the user wants a different topic, ask them to start a new conversation.\n"
            "4) Be direct (about 180–220 words)."
        ),
    )
    trimmed = history[-10:] if len(history) > 10 else history
    messages = [system] + trimmed + [ChatMessage(role="user", message=user_text)]

    llm = LLMClient()
    reply_text = llm.chat(messages)
    return ModelReply(stance=stance_hint, reply=reply_text[: (REPLY_CHAR_LIMIT or 10_000)])
