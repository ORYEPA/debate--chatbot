import time
import requests
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query

from app.profiles import PROFILE
from app.config import (
    DEFAULT_TOPIC,
    DEFAULT_SIDE,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    redis_client,
)
from app.models import (
    CommandsResponse, Command, ProfilesResponse, ProfileInfo,
    CreateProfileRequest, CreateProfileResponse, ConversationMetaResponse,
    HistoryResponse, AskRequest, AskResponse, ChatMessage,
)
from app.services.conversation import (
    new_cid, get_conversation, save_conversation, last_n,
    extract_profile_cmd, normalize_cid, topic_change_requested,
    bot_side_for, stance_type_from, detect_user_agreement,
)
from app.services.classifier import classify_topic_and_user_side_via_llm

from app.services.llm import generate_reply

router = APIRouter()


def _ollama_up(base_url: str) -> bool:
    """Quick ping to check if Ollama is reachable."""
    try:
        if not base_url:
            return False
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        r.raise_for_status()
        return True
    except Exception:
        return False


@router.get("/health")
def health():
    try:
        ok_redis = bool(redis_client.ping())
    except Exception:
        ok_redis = False

    ollama_ok, ollama_err = False, None
    try:
        if OLLAMA_BASE_URL:
            r = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3)
            r.raise_for_status()
            ollama_ok = True
    except Exception as e:
        ollama_err = str(e)

    openai_ready = bool(OPENAI_API_KEY)

    status_ok = ok_redis and (ollama_ok or openai_ready)

    return {
        "status": "ok" if status_ok else "degraded",
        "redis": ok_redis,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_reachable": ollama_ok,
        "ollama_error": ollama_err,
        "openai_ready": openai_ready,
        "openai_base_url": OPENAI_BASE_URL,
    }


@router.get("/commands", response_model=CommandsResponse)
def list_commands():
    return CommandsResponse(commands=[
        Command(name="List commands", method="GET", path="/commands", description="Lista de endpoints disponibles con ejemplos"),
        Command(name="Health", method="GET", path="/health", description="Estado de la API, LLMs y Redis"),
        Command(name="List profiles", method="GET", path="/profiles", description="Perfiles disponibles (id y nombre)"),
        Command(
            name="Set profile (create conversation)",
            method="POST",
            path="/conversations/profile",
            description="Crea una nueva conversación con el perfil indicado (tema/side por defecto del reto)",
            body_example={"profile_id": "rude_arrogant"},
        ),
        Command(
            name="Chat (ask)",
            method="POST",
            path="/ask",
            description="Si no envías conversation_id (o 'string'), crea nueva conversación con perfil por defecto. Devuelve latency_ms y los últimos 5 mensajes.",
        ),
        Command(
            name="Conversation meta",
            method="GET",
            path="/conversations/{conversation_id}/meta",
            description="Devuelve profile_id, profile_name, topic y side (lado de la IA)",
        ),
        Command(
            name="History",
            method="GET",
            path="/conversations/{conversation_id}/history5",
            description="Devuelve últimos N mensajes con ?limit=N; si omites limit, devuelve todo el historial.",
            query_example={"limit": 10},
        ),
    ])


@router.get("/profiles", response_model=ProfilesResponse)
def get_profiles():
    return ProfilesResponse(
        profiles=[ProfileInfo(id=p["id"], name=p["name"]) for p in PROFILE.values()]
    )


@router.post("/conversations/profile", response_model=CreateProfileResponse)
def create_conversation_with_profile(req: CreateProfileRequest):
    if req.profile_id not in PROFILE:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile '{req.profile_id}'. Available: {list(PROFILE.keys())}",
        )
    cid = new_cid()
    conv = {
        "meta": {
            "topic": DEFAULT_TOPIC,
            "side": DEFAULT_SIDE,
            "profile_id": req.profile_id,
            "user_side": "negative" if DEFAULT_SIDE.startswith("Affirmative") else "affirmative",
            "stance_type": stance_type_from(DEFAULT_SIDE),
            "user_aligned": False,
        },
        "messages": [],
    }
    save_conversation(cid, conv)
    return CreateProfileResponse(ok=True, conversation_id=cid, profile_id=req.profile_id)


@router.get("/conversations/{conversation_id}/meta", response_model=ConversationMetaResponse)
def get_conversation_meta(conversation_id: str):
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    meta = conv.get("meta", {})
    pid = meta.get("profile_id")
    profile_name = PROFILE.get(pid, {}).get("name", pid)
    return ConversationMetaResponse(
        conversation_id=conversation_id,
        profile_id=pid,
        profile_name=profile_name,
        topic=meta.get("topic", DEFAULT_TOPIC),
        side=meta.get("side", DEFAULT_SIDE),
    )


@router.get("/conversations/{conversation_id}/history5", response_model=HistoryResponse)
def get_history(conversation_id: str, limit: Optional[int] = Query(None, ge=1, le=1000)):
    """
    If 'limit' is empty -> return full history.
    If 'limit' is provided -> return last 'limit' messages in chronological order.
    """
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    history = [ChatMessage(**m) for m in conv.get("messages", [])]
    out = history if limit is None else history[-limit:]
    return HistoryResponse(conversation_id=conversation_id, message=out)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Same endpoint set, simplified internals:
    - Uses new TEXT-ONLY generator with fallback to OpenAI.
    - No over-validation or rewrite passes.
    - Returns stance as 'pro' | 'contra' from backend logic (not from model output).
    """
    start = time.time()

    requested_profile, user_text = extract_profile_cmd(req.message)
    normalized_cid = normalize_cid(req.conversation_id)

    if not normalized_cid:
        topic, user_side = classify_topic_and_user_side_via_llm(user_text)
        bot_side = bot_side_for(topic, user_side)
        stance_type = "negative" if user_side == "affirmative" else "affirmative"

        profile_id = requested_profile or PROFILE.get("smart_shy", {}).get("id", "smart_shy")
        cid = new_cid()
        conv = {
            "meta": {
                "topic": topic,
                "side": bot_side,
                "stance_type": stance_type,
                "initial_topic": topic,
                "initial_user_side": user_side,
                "locked_side": True,
                "profile_id": profile_id,
                "user_side": user_side,
                "user_aligned": False,
            },
            "messages": [],
        }
        save_conversation(cid, conv)
    else:
        cid = normalized_cid
        conv = get_conversation(cid)
        if not conv:
            raise HTTPException(status_code=404, detail="conversation_id not found")

        if requested_profile:
            conv["meta"]["profile_id"] = requested_profile

        if not conv.get("messages"):
            topic, user_side = classify_topic_and_user_side_via_llm(user_text)
            bot_side = bot_side_for(topic, user_side)
            stance_type = "negative" if user_side == "affirmative" else "affirmative"
            conv["meta"].update({
                "topic": topic,
                "side": bot_side,
                "stance_type": stance_type,
                "initial_topic": topic,
                "initial_user_side": user_side,
                "locked_side": True,
                "user_side": user_side,
                "user_aligned": False,
            })
        else:
            new_topic_req = topic_change_requested(user_text)
            if new_topic_req is not None and new_topic_req:
                inferred_topic, inferred_user_side = classify_topic_and_user_side_via_llm(user_text)
                topic = inferred_topic or new_topic_req
                bot_side = bot_side_for(topic, inferred_user_side)
                stance_type = "negative" if inferred_user_side == "affirmative" else "affirmative"
                conv["meta"].update({
                    "topic": topic,
                    "side": bot_side,
                    "stance_type": stance_type,
                    "initial_topic": topic,
                    "initial_user_side": inferred_user_side,
                    "locked_side": True,
                    "user_side": inferred_user_side,
                    "user_aligned": False,
                })
        save_conversation(cid, conv)

    meta = conv["meta"]
    history = [ChatMessage(**m) for m in conv.get("messages", [])]

    if detect_user_agreement(user_text):
        conv["meta"]["user_aligned"] = True
        save_conversation(cid, conv)

    user_text = user_text[:800]
    history.append(ChatMessage(role="user", message=user_text))

    stance_hint = "pro" if meta.get("stance_type") == "affirmative" else "contra"
    mr = generate_reply(history, user_text, stance_hint=stance_hint)

    history.append(ChatMessage(role="assistant", message=mr.reply))
    conv["messages"] = [m.model_dump(by_alias=True) for m in history][-20:]
    save_conversation(cid, conv)

    last5 = last_n([ChatMessage(**m) for m in conv["messages"]], n=5)
    latency_ms = int((time.time() - start) * 1000)

    return AskResponse(
        conversation_id=cid,
        message=last5,
        latency_ms=latency_ms,
        stance=mr.stance,  
    )
