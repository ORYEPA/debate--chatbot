import os
MAX_MSG_CHARS = int(os.getenv("MAX_MSG_CHARS", "8000"))

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator

class AppBase(BaseModel):
    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
        "extra": "ignore",
        "slots": True,
    }

Role = Literal["user", "assistant", "system"]
Stance = Literal["pro", "contra"]

class ChatMessage(AppBase):
    role: Literal["system", "user", "assistant"]
    message: str = ""

    @field_validator("message", mode="before")
    @classmethod
    def _limit_len(cls, v):
        v = v or ""
        return v[:MAX_MSG_CHARS]

class AskRequest(AppBase):
    conversation_id: Optional[str] = None
    message: str

    @field_validator("message")
    @classmethod
    def _limit_len(cls, v: str) -> str:
        return (v or "")[:MAX_MSG_CHARS]

class AskResponse(AppBase):
    conversation_id: str
    message: List[ChatMessage]
    latency_ms: int
    stance: Stance

class Command(AppBase):
    name: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    path: str
    description: str
    body_example: Optional[Dict[str, Any]] = None
    query_example: Optional[Dict[str, Any]] = None

class CommandsResponse(AppBase):
    commands: List[Command]

class ProfileInfo(AppBase):
    id: str
    name: str

class ProfilesResponse(AppBase):
    profiles: List[ProfileInfo]

class CreateProfileRequest(AppBase):
    profile_id: str

class CreateProfileResponse(AppBase):
    ok: bool
    conversation_id: str
    profile_id: str

class HistoryResponse(AppBase):
    conversation_id: str
    message: List[ChatMessage]

class ConversationMetaResponse(AppBase):
    conversation_id: str
    profile_id: str
    profile_name: str
    topic: str
    side: str

class ModelReply(AppBase):
    stance: Stance
    reply: str

    @field_validator("reply")
    @classmethod
    def _reply_len(cls, v: str) -> str:
        return (v or "")[:900]
