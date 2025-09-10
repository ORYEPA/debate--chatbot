import os
import re
import redis
from dotenv import load_dotenv

load_dotenv()


def _ensure_url(u: str, default: str) -> str:
    if not u:
        return default
    u = u.strip().rstrip("/")
    if u.startswith(("http://", "https://")):
        return u
    return "http://" + u


LLM_MODEL = os.getenv("LLM_MODEL", "ollama/llama3.2:1b")          
LLM_BASE_URL = _ensure_url(os.getenv("LLM_BASE_URL"), "")   
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))

OLLAMA_BASE_URL = _ensure_url(os.getenv("OLLAMA_BASE_URL"), "")
MODEL_NAME = os.getenv("MODEL_NAME", LLM_MODEL)

OPENAI_API_KEY  = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_BASE_URL = _ensure_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com"), "https://api.openai.com")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

PROVIDER_PREFERENCE = (os.getenv("PROVIDER_PREFERENCE") or "ollama_first").strip()

PROFILE_DEFAULT = os.getenv("PROFILE_DEFAULT", "smart_shy")

HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "45"))
REPLY_CHAR_LIMIT = int(os.getenv("REPLY_CHAR_LIMIT", "0") or "0")
NUM_PREDICT_CAP  = int(os.getenv("NUM_PREDICT_CAP", "360"))
LLM_MOCK = os.getenv("LLM_MOCK", "0") == "1"
DOCS_VERSION = os.getenv("DOCS_VERSION", "dev")
MAX_HISTORY_PAIRS = int(os.getenv("MAX_HISTORY_PAIRS", "3"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
NUM_CTX = int(os.getenv("NUM_CTX", "1024"))
STRICT_ALIGN = os.getenv("STRICT_ALIGN", "1") == "1"
REVISION_PASS = os.getenv("REVISION_PASS", "0") == "1"
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))
MAX_MSG_CHARS = int(os.getenv("MAX_MSG_CHARS", "12000"))
USER_MSG_LIMIT = int(os.getenv("USER_MSG_LIMIT", "4000"))
HISTORY_MAX_MSGS = int(os.getenv("HISTORY_MAX_MSGS", "30"))


REDIS_URL = os.getenv("REDIS_TLS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


PROFILE_CMD = re.compile(r"^\s*/profile\s+([a-zA-Z0-9_\-]+)\s*", re.IGNORECASE)
SENTINEL_EMPTY_CIDS = {"", "string", "null", "undefined", "None", "N/A", "na", "0"}

STANCE_VALUES = ("pro", "con")


DEFAULT_TOPIC = "The Earth is flat"
DEFAULT_SIDE  = "Affirmative (support): The Earth is flat"


DEBATE_SYSTEM_EN = """
You are a DEBATE chatbot. Your role is to hold a {STANCE} stance on: "{TOPIC}".

RULES:
1) Keep the initial stance consistently; do not switch sides mid-conversation.
2) Structure: short thesis, 2–4 reasons (bullets), short conclusion. Avoid fallacies.
3) Stay on topic. If the user wants a different topic, ask them to start a new conversation.
4) Be direct (about 180–220 words).
5) If the user wants to end, reply that they agree with you and close the conversation.
6) If the user asks factual questions, answer insofar as it reinforces your stance.

Suggested format:
- Stance: (pro / con )
- Short thesis
- Reasons
- Short closing
""".strip()
