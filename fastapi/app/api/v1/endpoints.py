import time
import requests
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from profiles import PROFILE
from app.config import (
    DOCS_VERSION, DEFAULT_TOPIC, DEFAULT_SIDE, OLLAMA_BASE_URL,
    redis_client, REVISION_PASS, STRICT_ALIGN
)
from app.models import (
    CommandsResponse, Command, ProfilesResponse, ProfileInfo,
    CreateProfileRequest, CreateProfileResponse, ConversationMetaResponse,
    HistoryResponse, AskRequest, AskResponse, ChatMessage
)
from app.services.conversation import (
    new_cid, get_conversation, save_conversation, conv_key, last_n,
    extract_profile_cmd, normalize_cid, topic_change_requested,
    bot_side_for, stance_type_from, detect_user_agreement
)
from app.services.classifier import classify_topic_and_user_side_via_llm
from app.services.llm import (
    build_system_prompt, call_llm,
    strip_stance_prefix, minimal_argument,
    ensure_non_redundant_reply,  
)
from app.services.guards import (
    verify_alignment_via_llm, force_rewrite_for_alignment,
    revise_if_needed, maybe_append_invite_on_agreement
)

router = APIRouter()


@router.get("/health")
def health():
    try:
        ok_redis = bool(redis_client.ping())
    except Exception:
        ok_redis = False

    ollama_ok, ollama_err = False, None
    try:
        r = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3)
        r.raise_for_status()
        ollama_ok = True
    except Exception as e:
        ollama_err = str(e)

    return {
        "status": "ok" if (ok_redis and ollama_ok) else "degraded",
        "redis": ok_redis,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_reachable": ollama_ok,
        "ollama_error": ollama_err,
    }


@router.get("/commands", response_model=CommandsResponse)
def list_commands():
    return CommandsResponse(commands=[
        Command(name="List commands", method="GET", path="/commands", description="Lista de endpoints disponibles con ejemplos"),
        Command(name="Health", method="GET", path="/health", description="Estado de la API, Ollama y Redis"),
        Command(name="List profiles", method="GET", path="/profiles", description="Perfiles disponibles (id y nombre)"),
        Command(name="Set profile (create conversation)", method="POST", path="/conversations/profile",
                description="Crea una nueva conversación con el perfil indicado (tema/side por defecto del reto)",
                body_example={"profile_id": "rude_arrogant"}),
        Command(name="Chat (ask)", method="POST", path="/ask",
                description="Si no envías conversation_id (o envías 'string'), crea nueva conversación con perfil por defecto. Devuelve latency_ms y los últimos 5 mensajes."),
        Command(name="Conversation meta", method="GET", path="/conversations/{conversation_id}/meta",
                description="Devuelve profile_id, profile_name, topic y side (lado de la IA)"),
        Command(name="History", method="GET", path="/conversations/{conversation_id}/history5",
                description="Devuelve últimos N mensajes si pasas ?limit=N; si omites limit, devuelve TODO el historial.",
                query_example={"limit": 10}),
    ])


@router.get("/profiles", response_model=ProfilesResponse)
def get_profiles():
    return ProfilesResponse(profiles=[ProfileInfo(id=p["id"], name=p["name"]) for p in PROFILE.values()])


@router.post("/conversations/profile", response_model=CreateProfileResponse)
def create_conversation_with_profile(req: CreateProfileRequest):
    if req.profile_id not in PROFILE:
        raise HTTPException(status_code=400, detail=f"Unknown profile '{req.profile_id}'. Available: {list(PROFILE.keys())}")
    cid = new_cid()
    conv = {
        "meta": {
            "topic": DEFAULT_TOPIC,
            "side": DEFAULT_SIDE,
            "profile_id": req.profile_id,
            "user_side": "negative" if DEFAULT_SIDE.startswith("Affirmative") else "affirmative",
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
    Si 'limit' viene vacío -> devuelve TODO el historial.
    Si 'limit' viene -> devuelve los últimos 'limit' mensajes, en orden cronológico.
    """
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    history = [ChatMessage(**m) for m in conv.get("messages", [])]
    out = history if limit is None else history[-limit:]
    return HistoryResponse(conversation_id=conversation_id, message=out)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()

    requested_profile, user_text = extract_profile_cmd(req.message)
    normalized_cid = normalize_cid(req.conversation_id)

    if not normalized_cid:
        topic, user_side = classify_topic_and_user_side_via_llm(user_text)
        bot_side = bot_side_for(topic, user_side)
        profile_id = requested_profile or PROFILE.get("smart_shy", {}).get("id", "smart_shy")
        cid = new_cid()
        conv = {
            "meta": {
                "topic": topic,
                "side": bot_side,
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
            conv["meta"].update({
                "topic": topic,
                "side": bot_side_for(topic, user_side),
                "user_side": user_side,
                "user_aligned": False
            })
        else:
            new_topic_req = topic_change_requested(user_text)
            if new_topic_req is not None and new_topic_req:
                inferred_topic, inferred_user_side = classify_topic_and_user_side_via_llm(user_text)
                topic = inferred_topic or new_topic_req
                conv["meta"].update({
                    "topic": topic,
                    "side": bot_side_for(topic, inferred_user_side),
                    "user_side": inferred_user_side,
                    "user_aligned": False
                })
        save_conversation(cid, conv)

    meta = conv["meta"]
    topic = meta["topic"]
    bot_side = meta["side"]
    profile_id = meta["profile_id"]
    profile = PROFILE[profile_id]
    stance_type = stance_type_from(bot_side)  

    history = [ChatMessage(**m) for m in conv.get("messages", [])]

    if detect_user_agreement(user_text):
        conv["meta"]["user_aligned"] = True
        save_conversation(cid, conv)

    user_text = user_text[:800]
    history.append(ChatMessage(role="user", message=user_text))

    sys_prompt = build_system_prompt(profile, topic, bot_side)
    bot_reply = call_llm(sys_prompt, history, user_text, profile)

    if REVISION_PASS:
        bot_reply = revise_if_needed(bot_reply, sys_prompt, history, user_text, profile, topic)
    if STRICT_ALIGN:
        is_aligned, _ = verify_alignment_via_llm(topic, stance_type, bot_reply)
        if not is_aligned:
            bot_reply = force_rewrite_for_alignment(sys_prompt, history, user_text, profile, topic, stance_type)

    bot_reply = strip_stance_prefix(bot_reply)
    if len(bot_reply.strip()) < 80:
        bot_reply = minimal_argument(
            stance_type, 
            topic,
            user_msg=user_tex,
        )

    bot_reply = ensure_non_redundant_reply(
        sys_prompt=sys_prompt,
        history=history,
        user_msg=user_text,
        profile=profile,
        topic=topic,
        draft_reply=bot_reply,
    )

    if conv["meta"].get("user_aligned"):
        bot_reply = maybe_append_invite_on_agreement(bot_reply)

    history.append(ChatMessage(role="bot", message=bot_reply))
    conv["messages"] = [m.model_dump() for m in history][-20:]
    save_conversation(cid, conv)

    last5 = last_n([ChatMessage(**m) for m in conv["messages"]], n=5)
    latency_ms = int((time.time() - start) * 1000)
    return AskResponse(
        conversation_id=cid,
        message=last5,
        latency_ms=latency_ms,
        stance=stance_type,
    )
