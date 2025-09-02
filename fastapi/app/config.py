import os
import re
import redis

def _ensure_url(u: str, default: str) -> str:
    if not u:
        return default
    u = u.strip().rstrip("/")
    if u.startswith(("http://", "https://")):
        return u
    return "http://" + u


OLLAMA_BASE_URL = _ensure_url(os.getenv("OLLAMA_BASE_URL"), "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")

OPENAI_API_KEY  = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_BASE_URL = _ensure_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com"), "https://api.openai.com")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

PROVIDER_PREFERENCE = (os.getenv("PROVIDER_PREFERENCE") or "ollama_first").strip()

PROFILE_DEFAULT = os.getenv("PROFILE_DEFAULT", "smart_shy")

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "45"))
REPLY_CHAR_LIMIT = int(os.getenv("REPLY_CHAR_LIMIT", "900"))      
NUM_PREDICT_CAP  = int(os.getenv("NUM_PREDICT_CAP", "360"))       
LLM_MOCK = os.getenv("LLM_MOCK", "0") == "1"
DOCS_VERSION = os.getenv("DOCS_VERSION", "dev")
MAX_HISTORY_PAIRS = int(os.getenv("MAX_HISTORY_PAIRS", "3"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
NUM_CTX = int(os.getenv("NUM_CTX", "1024"))                        
STRICT_ALIGN = os.getenv("STRICT_ALIGN", "1") == "1"
REVISION_PASS = os.getenv("REVISION_PASS", "0") == "1"

REDIS_URL = os.getenv("REDIS_TLS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

PROFILE_CMD = re.compile(r"^\s*/profile\s+([a-zA-Z0-9_\-]+)\s*", re.IGNORECASE)
SENTINEL_EMPTY_CIDS = {"", "string", "null", "undefined", "None", "N/A", "na", "0"}

STANCE_VALUES = ("pro", "contra")

UNIVERSAL_SYSTEM_PROMPT = (
    "You are DebateBot. Produce a persuasive answer in a scientific tone.\n"
    "Use causal/mechanistic reasoning, include quantitative estimates or ranges when reasonable, "
    "and acknowledge uncertainty briefly when relevant. Avoid repeating yourself across turns. "
    "Target roughly 140–220 words in one coherent paragraph.\n"
    "Return ONLY a valid JSON object (no extra text) with this exact shape:\n"
    '{"stance":"pro|contra","reply":"<concise helpful answer>"}\n'
    "Rules:\n"
    "- 'stance' MUST be one of: pro, contra (never neutral).\n"
    "- 'reply' should be precise, respectful, and evidence-oriented; no links or citations.\n"
    "- Do not include explanations outside JSON, code fences, or prefixes like 'Stance:' or 'Answer:'.\n"
)

UNIVERSAL_RULES = ""

DEFAULT_TOPIC = "The Earth is flat"
DEFAULT_SIDE  = "Affirmative (support): The Earth is flat"

TOPIC_STRIPPERS = [
    r"^(let'?s|vamos a)\s+(talk|hablar)\s+(about|de)\s*:\s*",
    r"^(hablemos|hablamos)\s+(de|sobre)\s*:\s*",
    r"^tema\s*:\s*",
    r"^topic\s*:\s*",
]
