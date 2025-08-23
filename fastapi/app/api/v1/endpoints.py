import time
import requests

from fastapi import APIRouter, HTTPException
from profiles import PROFILE  
from app.config import (
    DOCS_VERSION, DEFAULT_TOPIC, DEFAULT_SIDE, redis_client
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
from app.services.llm import build_system_prompt, call_llm
from app.services.guards import (
    verify_alignment_via_llm, force_rewrite_for_alignment,
    revise_if_needed, maybe_append_invite_on_agreement
)
from app.config import OLLAMA_BASE_URL

router = APIRouter()

@router.get("/health")
def health():
    from app.config import redis_client
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
        Command(name="Health", method="GET", path="/health", description="Estado de la API y Redis"),
        Command(name="List profiles", method="GET", path="/profiles", description="Perfiles disponibles (id y nombre)"),
        Command(name="Set profile (create conversation)", method="POST", path="/conversations/profile",
                description="Crea una nueva conversación con el perfil indicado (tema/side por defecto del reto)",
                body_example={"profile_id": "rude_arrogant"}),
        Command(name="Chat (ask)", method="POST", path="/ask",
                description="Si no envías conversation_id (o envías 'string'), el bot crea nueva conversa con perfil por defecto. Devuelve latency_ms y los últimos 5 mensajes."),
        Command(name="Conversation meta", method="GET", path="/conversations/{conversation_id}/meta",
                description="Devuelve profile_id, profile_name, topic y side (lado de la IA)"),
        Command(name="Last 5 messages (total)", method="GET", path="/conversations/{conversation_id}/history5",
                description="Últimos 5 mensajes totales, orden cronológico"),
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
            "user_side": "negative" if DEFAULT_SIDE.startswith("Affirmative") else "affirmative"
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
def get_history_last5(conversation_id: str):
    conv = get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    history = [ChatMessage(**m) for m in conv.get("messages", [])]
    return HistoryResponse(conversation_id=conversation_id, message=last_n(history, n=5))

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()

    requested_profile, user_text = extract_profile_cmd(req.message)
    normalized_cid = normalize_cid(req.conversation_id)

    if not normalized_cid:
        from profiles import PROFILE  
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
            conv["meta"].update({"topic": topic, "side": bot_side_for(topic, user_side), "user_side": user_side, "user_aligned": False})
        else:
            new_topic_req = topic_change_requested(user_text)
            if new_topic_req is not None and new_topic_req:
                inferred_topic, inferred_user_side = classify_topic_and_user_side_via_llm(user_text)
                topic = inferred_topic or new_topic_req
                conv["meta"].update({"topic": topic, "side": bot_side_for(topic, inferred_user_side),
                                     "user_side": inferred_user_side, "user_aligned": False})
        save_conversation(cid, conv)

    
    from profiles import PROFILE  
    topic = conv["meta"]["topic"]; bot_side = conv["meta"]["side"]
    profile_id = conv["meta"]["profile_id"]; profile = PROFILE[profile_id]
    stance_type = stance_type_from(bot_side)

    history = [ChatMessage(**m) for m in conv.get("messages", [])]

    if detect_user_agreement(user_text):
        conv["meta"]["user_aligned"] = True
        save_conversation(cid, conv)

    user_text = user_text[:800]
    history.append(ChatMessage(role="user", message=user_text))

    from app.services.llm import build_system_prompt, call_llm
    sys_prompt = build_system_prompt(profile, topic, bot_side)
    bot_reply = call_llm(sys_prompt, history, user_text, profile)

    from app.services.guards import revise_if_needed, verify_alignment_via_llm, force_rewrite_for_alignment, maybe_append_invite_on_agreement
    bot_reply = revise_if_needed(bot_reply, sys_prompt, history, user_text, profile, topic)
    aligned, _ = verify_alignment_via_llm(topic, stance_type, bot_reply)
    if not aligned:
        bot_reply = force_rewrite_for_alignment(sys_prompt, history, user_text, profile, topic, stance_type)

    if conv["meta"].get("user_aligned"):
        bot_reply = maybe_append_invite_on_agreement(bot_reply)

    history.append(ChatMessage(role="bot", message=bot_reply))
    conv["messages"] = [m.model_dump() for m in history][-20:]
    save_conversation(cid, conv)

    last5 = last_n([ChatMessage(**m) for m in conv["messages"]], n=5)
    latency_ms = int((time.time() - start) * 1000)
    return AskResponse(conversation_id=cid, message=last5, latency_ms=latency_ms)
