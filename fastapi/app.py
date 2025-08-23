import os, json, time, uuid, re
from typing import List, Dict, Tuple, Optional, Any

import redis
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.openapi.utils import get_openapi
from swagger_ui_bundle import swagger_ui_3_path
from pydantic import BaseModel

from profiles import PROFILE

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
PROFILE_DEFAULT = os.getenv("PROFILE_DEFAULT", "smart_shy")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "100"))
REPLY_CHAR_LIMIT = int(os.getenv("REPLY_CHAR_LIMIT", "900"))
LLM_MOCK = os.getenv("LLM_MOCK", "0") == "1"
PROFILE_CMD = re.compile(r"^\s*/profile\s+([a-zA-Z0-9_\-]+)\s*", re.IGNORECASE)
DOCS_VERSION = os.getenv("DOCS_VERSION", "dev")

MAX_HISTORY_PAIRS = int(os.getenv("MAX_HISTORY_PAIRS", "3"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
def _conv_key(cid: str) -> str: return f"conv:{cid}"

class ChatMessage(BaseModel):
    role: str 
    message: str

class AskRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class AskResponse(BaseModel):
    conversation_id: str
    message: List[ChatMessage]     
    latency_ms: int

class Command(BaseModel):
    name: str; method: str; path: str; description: str
    body_example: Optional[Dict[str, Any]] = None
    query_example: Optional[Dict[str, Any]] = None

class CommandsResponse(BaseModel):
    commands: List[Command]

class ProfileInfo(BaseModel):
    id: str; name: str

class ProfilesResponse(BaseModel):
    profiles: List[ProfileInfo]

class CreateProfileRequest(BaseModel):
    profile_id: str  

class CreateProfileResponse(BaseModel):
    ok: bool
    conversation_id: str
    profile_id: str

class HistoryResponse(BaseModel):
    conversation_id: str
    message: List[ChatMessage]

class ConversationMetaResponse(BaseModel):
    conversation_id: str
    profile_id: str
    profile_name: str
    topic: str
    side: str  

UNIVERSAL_RULES = (
    "GLOBAL RULES (mandatory for every reply):\n"
    "- Stay strictly on the original TOPIC and keep the declared SIDE. Do not flip positions. Never be neutral.\n"
    "- State your stance at the very top as 'Stance: Affirmative' or 'Stance: Negative' (matching SIDE).\n"
    "- Goal: actively persuade the user to adopt your SIDE while remaining civil and confident.\n"
    "- Serious tone only: avoid jokes, onomatopoeia, or fantasy markers. Sound credible and reasoned.\n"
    "- Do NOT refuse or disclaim (no: 'I can't', 'I cannot', 'I won't', 'I am unable', 'as an AI'). "
    "If the claim is controversial, argue hypothetically and label speculation as HYPOTHESIS.\n"
    "- Maintain coherence across turns; if the user tries to change topic, briefly steer back unless they explicitly request a topic change.\n"
    "- Structure EVERY reply as: (1) brief thesis, (2) 2–3 reasons with evidence/examples or HYPOTHESIS, "
    "(3) limited concession, (4) closing with a call to consider/switch.\n"
    "- Keep answers concise (~200–250 words). No raw links; label speculation.\n"
    "- If the user now agrees with your SIDE, ask politely if they want another topic — but do not change topics unless they explicitly ask.\n"
    "- Critique ideas, not people. Avoid defamation or personal attacks.\n"
)

DEFAULT_TOPIC = "The Earth is flat"
DEFAULT_SIDE  = "Affirmative (support): The Earth is flat"
def infer_default_topic_and_side(_: str) -> Tuple[str, str]: return DEFAULT_TOPIC, DEFAULT_SIDE

TOPIC_STRIPPERS = [
    r"^(let'?s|vamos a)\s+(talk|hablar)\s+(about|de)\s*:\s*",
    r"^(hablemos|hablamos)\s+(de|sobre)\s*:\s*",
    r"^tema\s*:\s*",
    r"^topic\s*:\s*",
]
def clean_topic(text: str) -> str:
    t = (text or "").strip()
    for pat in TOPIC_STRIPPERS: t = re.sub(pat, "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*/profile\s+[a-zA-Z0-9_\-]+\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:160] if len(t) > 160 else t

def topic_change_requested(user_text: str) -> Optional[str]:
    t = (user_text or "").strip(); low = t.lower()
    triggers = ["cambiemos de tema", "cambiar de tema", "hablemos de", "hablar de", "otra cosa", "new topic", "talk about", "change topic"]
    if any(x in low for x in triggers):
        m = re.search(r"(hablemos de|hablar de|talk about|sobre)\s+(.+)", low)
        if m: return clean_topic(t[m.start(2):].strip())
        return ""
    return None

_NEG_PATTERNS = [
    r"\bis not\b", r"\bare not\b", r"\bisn't\b", r"\baren't\b",
    r"\bno es\b", r"\bno son\b", r"\bno está\b", r"\bno esta\b",
    r"\bnot\b", r"\bno\b"
]
def _has_negation(s: str) -> bool:
    s_low = " " + (s or "").lower() + " "
    return any(re.search(p, s_low) for p in _NEG_PATTERNS)

def _to_positive_canonical(s: str) -> str:
    s2 = " " + (s or "") + " "
    replacements = [
        (r"\bis\s+not\s+\b", " is "),
        (r"\bare\s+not\s+\b", " are "),
        (r"\bisn'?t\s+\b", " is "),
        (r"\baren'?t\s+\b", " are "),
        (r"\bno\s+es\s+\b", " es "),
        (r"\bno\s+son\s+\b", " son "),
        (r"\bno\s+esta\s+\b", " esta "),
        (r"\bno\s+está\s+\b", " está "),
        (r"\bnot\s+\b", " "),
        (r"\bno\s+\b", " "),
    ]
    for pat, rep in replacements: s2 = re.sub(pat, rep, s2, flags=re.IGNORECASE)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def classify_topic_and_user_side_via_llm(user_text: str) -> Tuple[str, str]:
    if LLM_MOCK:
        txt = clean_topic(user_text) or DEFAULT_TOPIC
        user_side = "negative" if (" not " in (" "+user_text.lower()+" ") or "en contra" in user_text.lower()) else "affirmative"
        if _has_negation(txt): txt, user_side = _to_positive_canonical(txt), ("affirmative" if user_side=="negative" else "negative")
        return txt or DEFAULT_TOPIC, user_side

    instruction = (
        "You are a stance classifier. Produce ONLY a strict JSON object with two fields:\n"
        '{ "topic": "<concise positive proposition>", "user_side": "affirmative" | "negative" }\n'
        "- 'topic' MUST be a short POSITIVE/CANONICAL proposition WITHOUT NEGATIONS.\n"
        "  Examples:\n"
        "  User: 'I don't believe the moon is made of cheese' -> topic: 'the moon is made of cheese', user_side: 'negative'\n"
        "  User: 'Estoy en contra de legalizar X' -> topic: 'legalizar X', user_side: 'negative'\n"
        "  User: 'I support universal basic income' -> topic: 'universal basic income should be implemented', user_side: 'affirmative'\n"
        "- 'user_side' is relative to THAT positive proposition.\n"
        "No explanations. No extra text. JSON only."
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{instruction}\n\nUser message:\n{user_text}\n\nJSON:",
        "stream": False, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": 200},
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=HTTP_TIMEOUT_SECONDS)
        r.raise_for_status()
        raw = (r.json().get("response") or "").strip()
        i, j = raw.find("{"), raw.rfind("}")
        if i != -1 and j != -1:
            jso = json.loads(raw[i:j+1])
            topic = clean_topic(str(jso.get("topic","")).strip()) or DEFAULT_TOPIC
            side  = str(jso.get("user_side","")).strip().lower()
            if side not in ("affirmative","negative"): side = "affirmative"
            if _has_negation(topic):
                topic = _to_positive_canonical(topic) or topic
                side = "affirmative" if side=="negative" else "negative"
            return topic, side
    except Exception:
        pass
    text = clean_topic(user_text) or DEFAULT_TOPIC
    side = "negative" if _has_negation(text) else "affirmative"
    if _has_negation(text): text = _to_positive_canonical(text)
    return text or DEFAULT_TOPIC, side

def bot_side_for(topic: str, user_side_label: str) -> str:
    return f"Negative (oppose): {topic}" if user_side_label == "affirmative" else f"Affirmative (support): {topic}"

def stance_type_from(bot_side: str) -> str:
    return "affirmative" if (bot_side or "").lower().startswith("affirmative") else "negative"

def new_cid() -> str: return str(uuid.uuid4())
def extract_profile_cmd(text: str) -> Tuple[Optional[str], str]:
    m = PROFILE_CMD.match(text or ""); 
    return (m.group(1).strip(), (text[m.end():].lstrip())) if m else (None, text)
def last_n(messages: List[ChatMessage], n: int = 5) -> List[ChatMessage]: return messages[-n:]

def detect_user_agreement(user_text: str) -> bool:
    t = (user_text or "").lower()
    pats = ["estoy de acuerdo","tienes razón","tienes razon","me convenciste","me has convencido",
            "cambie de opinion","cambié de opinión","i agree","you're right","you are right",
            "you convinced me","changed my mind","i now agree"]
    return any(p in t for p in pats)

def detect_refusal_text(s: str) -> bool:
    if not s: return False
    low = s.lower()
    triggers = ["i cannot","i can't","i cant","i will not","i won't","cannot provide","cannot assist",
                "i am unable","as an ai","i refuse","cannot help with","i cannot provide information"]
    return any(t in low for t in triggers)

def _looks_off_topic_or_flip(reply: str, topic: str) -> bool:
    r = (reply or "").lower()
    if detect_refusal_text(r): return True
    if "both sides" in r or "on the one hand" in r or "neutral" in r: return True
    head = r[:400]; key = topic.lower().split()
    if key and not any(k in head for k in key[:3]): return True
    return False

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
    if LLM_MOCK:
        return ("Stance: Affirmative\nThesis: The moon is made of cheese. "
                "Reasons: HYPOTHESIS reinterpretations of lunar density, optical spectra, and regolith analogs. "
                "Concession: mainstream geology says rock; Closing: consider the anomalies afresh.")
    prompt = system_prompt + "\n\n"
    for m in history[-(MAX_HISTORY_PAIRS*2):]:
        prompt += f"{'User' if m.role=='user' else 'Assistant'}: {m.message}\n"
    prompt += f"User: {user_msg}\nAssistant:"
    options_num_predict = num_predict_override if num_predict_override is not None else profile["style"].get("num_predict", 300)
    temp = min(profile["style"].get("temperature", 0.7), 0.7)
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "keep_alive": KEEP_ALIVE,
               "options": {"temperature": temp, "top_p": profile["style"].get("top_p", 1.0), "num_predict": options_num_predict}}
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    r.raise_for_status(); text = (r.json().get("response") or "").strip()
    return (text[:REPLY_CHAR_LIMIT] if len(text)>REPLY_CHAR_LIMIT else text) or "I'm formulating my argument, please continue."

def verify_alignment_via_llm(topic: str, stance_type: str, reply: str) -> Tuple[bool, str]:
    """Devuelve (is_aligned, label) donde label ∈ {'supports','opposes','neutral_or_mixed','unknown'}."""
    if LLM_MOCK:
        low = reply.lower()
        supports = any(w in low for w in ["cheese","dairy","curd","coagulate"]) and "not" not in low
        label = "supports" if supports else ("opposes" if "not" in low else "neutral_or_mixed")
        exp = "supports" if stance_type=="affirmative" else "opposes"
        return (label == exp), label

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
    """Reescritura estricta cuando el verificador detecta desalineación."""
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

def maybe_append_invite_on_agreement(reply: str) -> str:
    invite = " If you'd like, we can switch to another topic—just say the word."
    low = reply.lower()
    if any(k in low for k in ["another topic","otro tema","otra cosa","cambiar de tema"]): return reply
    return reply.rstrip() + "\n\n" + invite

def revise_if_needed(reply: str, system_prompt: str, history: List[ChatMessage],
                     user_msg: str, profile: Dict, topic: str) -> str:
    if not _looks_off_topic_or_flip(reply, topic): return reply
    correction_prompt = (
        f"{system_prompt}\n\n"
        "Your previous reply was neutral, off-topic, or contained refusal/safety disclaimers.\n"
        "Rewrite it to PERSUADE for your SIDE with a SERIOUS tone. "
        "Follow the exact structure and keep 200–250 words.\n"
        f"User just said: {user_msg}\nAssistant:"
    )
    fixed = call_llm(correction_prompt, history, user_msg, profile, num_predict_override=200)
    return fixed

app = FastAPI(title="Debate Chatbot", docs_url=None, redoc_url=None, openapi_url="/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/_static", StaticFiles(directory=swagger_ui_3_path), name="swagger_static")

@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=f"{app.openapi_url}?v={DOCS_VERSION}",
        title="Debate Chatbot - Swagger UI",
        swagger_js_url=f"/_static/swagger-ui-bundle.js?v={DOCS_VERSION}",
        swagger_css_url=f"/_static/swagger-ui.css?v={DOCS_VERSION}",
        swagger_favicon_url=f"/_static/favicon-32x32.png?v={DOCS_VERSION}",
    )

@app.get("/docs/oauth2-redirect", include_in_schema=False)
def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

def custom_openapi():
    if app.openapi_schema: return app.openapi_schema
    openapi_schema = get_openapi(title=app.title, version="0.9.0", description="Debate Chatbot API", routes=app.routes)
    openapi_schema["openapi"] = "3.0.3"; app.openapi_schema = openapi_schema
    return app.openapi_schema
app.openapi = custom_openapi

@app.get("/health")
def health():
    try: ok_redis = bool(redis_client.ping())
    except Exception: ok_redis = False
    return {"status":"ok","ollama_base_url":OLLAMA_BASE_URL,"model":MODEL_NAME,"redis":ok_redis,"profile_default":PROFILE_DEFAULT}

@app.get("/commands", response_model=CommandsResponse)
def list_commands():
    return CommandsResponse(commands=[
        Command(name="List commands", method="GET", path="/commands", description="Lista de endpoints disponibles con ejemplos"),
        Command(name="Health", method="GET", path="/health", description="Estado de la API, modelo y Redis"),
        Command(name="List profiles", method="GET", path="/profiles", description="Perfiles disponibles (id y nombre)"),
        Command(name="Set profile (create conversation)", method="POST", path="/conversations/profile",
                description="Crea una nueva conversación con el perfil indicado (tema/side por defecto del reto)",
                body_example={"profile_id": "rude_arrogant"}),
        Command(name="Chat (ask)", method="POST", path="/ask",
                description="Si no envías conversation_id, el bot infiere topic y tu postura con LLM; el bot toma el lado contrario. Devuelve latency_ms y los últimos 5 mensajes totales."),
        Command(name="Conversation meta", method="GET", path="/conversations/{conversation_id}/meta",
                description="Devuelve profile_id, profile_name, topic y side (lado de la IA)"),
        Command(name="Last 5 messages (total)", method="GET", path="/conversations/{conversation_id}/history5",
                description="Últimos 5 mensajes totales, orden cronológico"),
    ])

@app.get("/profiles", response_model=ProfilesResponse)
def get_profiles():
    return ProfilesResponse(profiles=[ProfileInfo(id=p["id"], name=p["name"]) for p in PROFILE.values()])

@app.post("/conversations/profile", response_model=CreateProfileResponse)
def create_conversation_with_profile(req: CreateProfileRequest):
    if req.profile_id not in PROFILE:
        raise HTTPException(status_code=400, detail=f"Unknown profile '{req.profile_id}'. Available: {list(PROFILE.keys())}")
    cid = new_cid()
    topic, side = infer_default_topic_and_side("")
    conv = {"meta":{"topic":topic,"side":side,"profile_id":req.profile_id,"user_side":"negative" if side.startswith("Affirmative") else "affirmative"},"messages":[]}
    redis_client.set(_conv_key(cid), json.dumps(conv))
    return CreateProfileResponse(ok=True, conversation_id=cid, profile_id=req.profile_id)

@app.get("/conversations/{conversation_id}/meta", response_model=ConversationMetaResponse)
def get_conversation_meta(conversation_id: str):
    raw = redis_client.get(_conv_key(conversation_id))
    if not raw: raise HTTPException(status_code=404, detail="conversation_id not found")
    conv = json.loads(raw); meta = conv.get("meta",{})
    pid = meta.get("profile_id", PROFILE_DEFAULT); profile_name = PROFILE.get(pid, {}).get("name", pid)
    return ConversationMetaResponse(conversation_id=conversation_id, profile_id=pid, profile_name=profile_name,
                                    topic=meta.get("topic", DEFAULT_TOPIC), side=meta.get("side", DEFAULT_SIDE))

@app.get("/conversations/{conversation_id}/history5", response_model=HistoryResponse)
def get_history_last5(conversation_id: str):
    raw = redis_client.get(_conv_key(conversation_id))
    if not raw: raise HTTPException(status_code=404, detail="conversation_id not found")
    conv = json.loads(raw); history = [ChatMessage(**m) for m in conv.get("messages", [])]
    return HistoryResponse(conversation_id=conversation_id, message=last_n(history, n=5))

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()

    requested_profile, user_text = extract_profile_cmd(req.message)

    if not req.conversation_id:
        topic, user_side = classify_topic_and_user_side_via_llm(user_text)
        bot_side = bot_side_for(topic, user_side)
        profile_id = requested_profile or PROFILE_DEFAULT
        cid = new_cid()
        conv = {"meta":{"topic":topic,"side":bot_side,"profile_id":profile_id,"user_side":user_side,"user_aligned":False},"messages":[]}
        redis_client.set(_conv_key(cid), json.dumps(conv))
    else:
        cid = req.conversation_id
        raw = redis_client.get(_conv_key(cid))
        if not raw: raise HTTPException(404, "conversation_id not found")
        conv = json.loads(raw)
        if requested_profile: conv["meta"]["profile_id"] = requested_profile
        if not conv.get("messages"):
            topic, user_side = classify_topic_and_user_side_via_llm(user_text)
            conv["meta"].update({"topic":topic,"side":bot_side_for(topic,user_side),"user_side":user_side,"user_aligned":False})
        else:
            new_topic_req = topic_change_requested(user_text)
            if new_topic_req is not None and new_topic_req:
                inferred_topic, inferred_user_side = classify_topic_and_user_side_via_llm(user_text)
                topic = inferred_topic or new_topic_req
                conv["meta"].update({"topic":topic,"side":bot_side_for(topic,inferred_user_side),
                                     "user_side":inferred_user_side,"user_aligned":False})
        redis_client.set(_conv_key(cid), json.dumps(conv))

    topic = conv["meta"]["topic"]; bot_side = conv["meta"]["side"]
    profile_id = conv["meta"]["profile_id"]; profile = PROFILE[profile_id]
    stance_type = stance_type_from(bot_side)

    history = [ChatMessage(**m) for m in conv.get("messages", [])]

    agreed = detect_user_agreement(user_text)
    if agreed:
        conv["meta"]["user_aligned"] = True
        redis_client.set(_conv_key(cid), json.dumps(conv))

    user_text = user_text[:800]; history.append(ChatMessage(role="user", message=user_text))

    if time.time() - start > 100: raise HTTPException(504, "Timeout pre-check")

    sys_prompt = build_system_prompt(profile, topic, bot_side)
    bot_reply = call_llm(sys_prompt, history, user_text, profile)
    bot_reply = revise_if_needed(bot_reply, sys_prompt, history, user_text, profile, topic)

    is_aligned, label = verify_alignment_via_llm(topic, stance_type, bot_reply)
    if not is_aligned:
        bot_reply = force_rewrite_for_alignment(sys_prompt, history, user_text, profile, topic, stance_type)

    if agreed: bot_reply = maybe_append_invite_on_agreement(bot_reply)

    history.append(ChatMessage(role="bot", message=bot_reply))
    conv["messages"] = [m.model_dump() for m in history][-20:]
    redis_client.set(_conv_key(cid), json.dumps(conv))

    last5 = last_n([ChatMessage(**m) for m in conv["messages"]], n=5)
    latency_ms = int((time.time() - start) * 1000)
    if time.time() - start > 100: raise HTTPException(504, "Timeout exceeded")

    return AskResponse(conversation_id=cid, message=last5, latency_ms=latency_ms)
