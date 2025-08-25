from typing import List, Dict, Optional, Any
from pydantic import BaseModel

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
    stance: str

class Command(BaseModel):
    name: str
    method: str
    path: str
    description: str
    body_example: Optional[Dict[str, Any]] = None
    query_example: Optional[Dict[str, Any]] = None

class CommandsResponse(BaseModel):
    commands: List[Command]

class ProfileInfo(BaseModel):
    id: str
    name: str

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
