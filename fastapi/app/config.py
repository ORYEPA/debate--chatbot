import os
import re
import redis

def _norm_ollama(u: str) -> str:
    """Asegura esquema http/https y quita '/' final."""
    if not u:
        return "http://localhost:11434"
    u = u.strip().rstrip("/")
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u

def _norm_redis(url_env: str, tls_env: str) -> str:
    """Elige TLS si está, acepta host interno sin esquema y cae a localhost."""
    u = (tls_env or url_env or "").strip()
    if not u:
        return "redis://localhost:6379/0"
    if u.startswith(("redis://", "rediss://")):
        return u
    if u.endswith(".railway.internal"):
        return f"redis://{u}:6379/0"
    return u  

OLLAMA_BASE_URL = _norm_ollama(os.getenv("OLLAMA_BASE_URL", ""))
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
PROFILE_DEFAULT = os.getenv("PROFILE_DEFAULT", "smart_shy")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "100"))
REPLY_CHAR_LIMIT = int(os.getenv("REPLY_CHAR_LIMIT", "900"))
LLM_MOCK = os.getenv("LLM_MOCK", "0") == "1"
DOCS_VERSION = os.getenv("DOCS_VERSION", "dev")
MAX_HISTORY_PAIRS = int(os.getenv("MAX_HISTORY_PAIRS", "3"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
NUM_PREDICT_CAP = int(os.getenv("NUM_PREDICT_CAP", "200"))  
NUM_CTX         = int(os.getenv("NUM_CTX", "1024"))        
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "45"))  

REDIS_URL = _norm_redis(os.getenv("REDIS_URL", ""), os.getenv("REDIS_TLS_URL", ""))
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

PROFILE_CMD = re.compile(r"^\s*/profile\s+([a-zA-Z0-9_\-]+)\s*", re.IGNORECASE)
SENTINEL_EMPTY_CIDS = {"", "string", "null", "undefined", "None", "N/A", "na", "0"}

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

TOPIC_STRIPPERS = [
    r"^(let'?s|vamos a)\s+(talk|hablar)\s+(about|de)\s*:\s*",
    r"^(hablemos|hablamos)\s+(de|sobre)\s*:\s*",
    r"^tema\s*:\s*",
    r"^topic\s*:\s*",
]
