import time, json, re, requests
from typing import List, Optional, Dict

from app.config import (
    OLLAMA_BASE_URL, MODEL_NAME, HTTP_TIMEOUT_SECONDS, KEEP_ALIVE,
    MAX_HISTORY_PAIRS, REPLY_CHAR_LIMIT, UNIVERSAL_RULES,
    NUM_PREDICT_CAP, NUM_CTX
)
from app.models import ChatMessage

TARGET_WORDS_MIN = 130
TARGET_WORDS_MAX = 170
PUNCT_END = (".", "!", "?")
TRIM_GRACE_CHARS = 140  

LOG_LINE_RE = re.compile(r'^(?:time=\d{4}-\d{2}-\d{2}T|\[GIN\]|\bINFO:|\bWARN:|level=|source=)', re.I)

def sanitize_model_text(text: str) -> str:
    lines = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            lines.append(ln); continue
        if LOG_LINE_RE.match(s):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

STANCE_LINE_ANY_RE = re.compile(r"^\s*Stance\s*:\s*.*$", re.I)

SECTION_HEAD_ANY_RE = re.compile(
    r"^\s*(?:limited\s+)?(?:thesis|reason(?:s)?(?:\s+with.*)?|concession|challenge|rebuttal|counterpoint|evidence|user'?s?\s+concession)\s*:\s*",
    re.I
)

REASONS_COUNT_RE = re.compile(r"^\s*\d+\s*(?:reasons?|puntos|motivos)(?:\s+with.*)?\s*:\s*$", re.I)
ENUM_BULLET_RE   = re.compile(r"^\s*(?:\d+[\)\.\:-]\s+|[-*]\s+)", re.I)

def strip_stance_prefix(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    if lines and STANCE_LINE_ANY_RE.match(lines[0]):
        return "\n".join(lines[1:]).lstrip()
    return text

def remove_structure_labels(text: str) -> str:
    if not text:
        return text
    out = []
    for ln in text.splitlines():
        if REASONS_COUNT_RE.match(ln):
            continue
        s = SECTION_HEAD_ANY_RE.sub("", ln)
        out.append(s)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def to_prose(text: str) -> str:
    """Convierte a prosa en UN SOLO PÁRRAFO (sin viñetas ni saltos)."""
    if not text:
        return text
    lines = []
    for ln in text.splitlines():
        ln = ENUM_BULLET_RE.sub("", ln).strip()
        lines.append(ln)
    s = " ".join(lines)                 
    s = remove_structure_labels(s)
    s = re.sub(r"\s{2,}", " ", s)      
    return s.strip()

CLOSING_SENTENCE = "Reconsider your stance in light of these points."

def ensure_closing_sentence(text: str) -> str:
    """Asegura el cierre exacto en la MISMA línea (sin saltos adicionales)."""
    if not text:
        return CLOSING_SENTENCE
    norm = text.strip()
    if CLOSING_SENTENCE.lower() in norm.lower():
        return norm
    if not norm.endswith(PUNCT_END):
        norm += "."
    return norm + " " + CLOSING_SENTENCE

def smart_sentence_trim(text: str, hard_limit: int, grace: int) -> str:
    s = (text or "").strip()
    if len(s) <= hard_limit:
        return s
    window = s[: hard_limit + grace]
    idx = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    if idx != -1:
        return window[: idx + 1].strip()
    sp = window.rfind(" ")
    return (window[:sp].strip() if sp != -1 else window.strip())

def minimal_argument(stance_type: str, topic: str, user_msg: str = "") -> str:
    verb = "support" if stance_type.lower().startswith("affirm") else "oppose"
    rebut = f"You said “{user_msg.strip()}”, but that misses key evidence: " if user_msg else ""
    body = (
        f"I {verb} the claim that {topic}. {rebut}"
        "Across independent cohorts and practical tests, the signal is consistent: when exposure is quantified "
        "and confounders are handled, the pattern doesn’t vanish; and comparative baselines with time-series checks "
        "point the same way. Common objections depend on assumptions that fail under closer measurement, which makes "
        "the opposite view harder to defend."
    )
    return ensure_closing_sentence(body)

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9']+")

def _token_set(s: str) -> set:
    return set(w.lower() for w in WORD_RE.findall(s or ""))

def _jaccard(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B) or 1
    return inter / union

def _prefix_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / n

def _last_bot_message(history: List[ChatMessage]) -> Optional[str]:
    for m in reversed(history):
        if m.role == "bot":
            return m.message or ""
    return None

def _last_bot_excerpt(history: List[ChatMessage], limit: int = 320) -> str:
    prev = _last_bot_message(history) or ""
    prev = re.sub(r"\s+", " ", prev).strip()
    return prev[:limit]

def build_system_prompt(profile: Dict, topic: str, side: str) -> str:
    return (
        f"{profile['system']}\n\n{UNIVERSAL_RULES}\n"
        f"TOPIC: {topic}\nSIDE: {side}\n"
        "Debate rules (override):\n"
        "- You are debating the OPPOSITE side of the user.\n"
        "- Begin EXACTLY with: 'Stance: Affirmative' OR 'Stance: Negative' matching SIDE.\n"
        "- Debate voice: in your FIRST clause, rebut the user's last sentence directly (use “because …”).\n"
        "- Write in ONE SINGLE PARAGRAPH of persuasive PROSE (no line breaks, no lists, no labels, no meta like “User’s concession”).\n"
        "- Include at least TWO concrete specifics (numbers, dates, named studies/missions/measurements) supporting your side.\n"
        f"- End by appending this exact sentence: {CLOSING_SENTENCE}\n"
        "- Serious, confident, non-neutral tone. Your goal is to flip the user to your side.\n"
    )

def call_llm(system_prompt: str, history: List[ChatMessage], user_msg: str, profile: Dict,
             num_predict_override: Optional[int] = None) -> str:
    prev_excerpt = _last_bot_excerpt(history)
    dynamic_rules = ""
    if prev_excerpt:
        dynamic_rules = (
            "\nAvoid repetition (strict): "
            "do NOT reuse distinctive phrases, facts, or sentence structure from your previous reply below; "
            "introduce at least TWO NEW specifics different from those previously used. "
            "--- PREVIOUS BOT EXCERPT --- "
            f"{prev_excerpt} "
            "--- END PREVIOUS --- "
        )

    prompt = system_prompt + dynamic_rules + "\n"
    for m in history[-(MAX_HISTORY_PAIRS * 2):]:
        prompt += f"{'User' if m.role == 'user' else 'Assistant'}: {m.message}\n"
    prompt += f"User: {user_msg}\nAssistant:"

    base_num_predict = profile["style"].get("num_predict", 300)
    npredict = min(base_num_predict if num_predict_override is None else num_predict_override, NUM_PREDICT_CAP)
    temp = min(profile["style"].get("temperature", 0.7), 0.7)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": temp,
            "top_p": profile["style"].get("top_p", 1.0),
            "num_predict": npredict,
            "num_ctx": NUM_CTX,
            "repeat_penalty": 1.08,
        },
    }

    start = time.time()
    pieces: list[str] = []
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json=payload,
            timeout=(5, HTTP_TIMEOUT_SECONDS),
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk = obj.get("response") or ""
                if chunk:
                    pieces.append(chunk)
                    so_far = "".join(pieces)

                    if CLOSING_SENTENCE in so_far:
                        break
                    wc = len(so_far.split())
                    if wc >= TARGET_WORDS_MIN and so_far.rstrip().endswith(PUNCT_END):
                        break
                    if wc >= TARGET_WORDS_MAX:
                        break
                    if len(so_far) > REPLY_CHAR_LIMIT + TRIM_GRACE_CHARS:
                        break

                if obj.get("done"):
                    break

                if time.time() - start > HTTP_TIMEOUT_SECONDS - 1:
                    break

    except requests.RequestException:
        fallback = (
            "Stance: Affirmative\n"
            "Your last objection overlooks two telling checks: when exposure windows are separated by age group, the high-intake cohorts "
            "show a consistent uptick relative to matched baselines, and time-series controls show the signal persists after seasonal cycles. "
            "Because these patterns replicate across independent samples, the simpler explanation is that the effect is real rather than an artifact."
        )
        fallback = sanitize_model_text(fallback)
        fallback = strip_stance_prefix(fallback)
        fallback = remove_structure_labels(fallback)
        fallback = to_prose(fallback)
        fallback = ensure_closing_sentence(fallback)
        if len(fallback) > REPLY_CHAR_LIMIT:
            fallback = smart_sentence_trim(fallback, REPLY_CHAR_LIMIT, TRIM_GRACE_CHARS)
            fallback = ensure_closing_sentence(fallback)
        return fallback

    text = "".join(pieces).strip()
    text = sanitize_model_text(text)

    text = strip_stance_prefix(text)
    text = remove_structure_labels(text)
    text = to_prose(text)

    if len(text) > REPLY_CHAR_LIMIT:
        text = smart_sentence_trim(text, REPLY_CHAR_LIMIT, TRIM_GRACE_CHARS)

    text = ensure_closing_sentence(text)

    if len(text.strip()) < 80:
        topic_match = re.search(r"TOPIC:\s*(.+)\nSIDE:", system_prompt)
        topic = topic_match.group(1).strip() if topic_match else "the claim"
        stance_match = re.search(r"SIDE:\s*(Affirmative|Negative)", system_prompt, re.I)
        stance_type = "affirmative" if stance_match and stance_match.group(1).lower().startswith("a") else "negative"
        text = minimal_argument(stance_type, topic, user_msg=user_msg)

    return text

def ensure_non_redundant_reply(
    sys_prompt: str,
    history: List[ChatMessage],
    user_msg: str,
    profile: Dict,
    topic: str,
    draft_reply: str,
    jaccard_threshold: float = 0.58,
    prefix_threshold: float = 0.50,
) -> str:
    def _token_set(s: str) -> set:
        return set(w.lower() for w in WORD_RE.findall(s or ""))

    def _jaccard(a: str, b: str) -> float:
        A, B = _token_set(a), _token_set(b)
        if not A and not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B) or 1
        return inter / union

    def _prefix_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i / n

    prev_bot = _last_bot_message(history)
    if not prev_bot:
        return draft_reply

    def _sim(a: str, b: str) -> bool:
        jac = _jaccard(a, b)
        pre = _prefix_ratio(a, b)
        return (jac >= jaccard_threshold) or (pre >= prefix_threshold)

    if not _sim(prev_bot, draft_reply):
        return draft_reply

    rewrite_instructions = (
        "Your previous draft repeats earlier arguments. Rewrite a NEW persuasive answer in ONE SINGLE PARAGRAPH, "
        "directly addressing why the user's latest point is mistaken. Introduce at least TWO NEW concrete specifics "
        "(numbers, dates, named studies/missions, measurements) that were NOT used before. "
        "Do NOT reuse distinctive phrases or sentence structure from the previous reply. "
        f"End by appending this exact sentence: {CLOSING_SENTENCE}\n\n"
        "--- PREVIOUS BOT REPLY (DO NOT COPY) ---\n"
        f"{prev_bot}\n"
        "--- END PREVIOUS ---\n"
        f"User just said: {user_msg}\n"
        "Assistant:"
    )

    rewrite_sys_prompt = f"{sys_prompt}\n\n{rewrite_instructions}"
    new_text = call_llm(
        system_prompt=rewrite_sys_prompt,
        history=history,
        user_msg=user_msg,
        profile=profile,
        num_predict_override=min(220, NUM_PREDICT_CAP),
    )

    new_text = sanitize_model_text(new_text)
    new_text = strip_stance_prefix(new_text)
    new_text = remove_structure_labels(new_text)
    new_text = to_prose(new_text)
    new_text = ensure_closing_sentence(new_text)

    if len(new_text.strip()) < 80:
        return draft_reply
    if _sim(prev_bot, new_text):
        return draft_reply

    if len(new_text) > REPLY_CHAR_LIMIT:
        new_text = smart_sentence_trim(new_text, REPLY_CHAR_LIMIT, TRIM_GRACE_CHARS)
        new_text = ensure_closing_sentence(new_text)

    return new_text
