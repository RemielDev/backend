from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

from models.chat import ChatMessage


# ---------- Enum Definitions ----------
class UserRole(str, Enum):
    PLAYER = "PLAYER"
    MODERATOR = "MODERATOR"
    ADMIN = "ADMIN"


class UserStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"


class CommunityAction(str, Enum):
    # Positive Actions
    ENCOURAGEMENT = "ENCOURAGEMENT"
    HELPFUL_ADVICE = "HELPFUL_ADVICE"
    WELCOME_NEWCOMER = "WELCOME_NEWCOMER"
    TEAM_COORDINATION = "TEAM_COORDINATION"
    APPRECIATION = "APPRECIATION"
    CELEBRATION = "CELEBRATION"
    KNOWLEDGE_SHARING = "KNOWLEDGE_SHARING"
    # Negative Actions
    TROLLING = "TROLLING"
    GRIEFING = "GRIEFING"
    SPAMMING = "SPAMMING"
    EXCLUSION = "EXCLUSION"
    BRAGGING = "BRAGGING"
    ARGUMENT_STARTING = "ARGUMENT_STARTING"
    BULLYING = "BULLYING"
    SHOW_OFF = "SHOW_OFF"
    PUT_DOWN = "PUT_DOWN"


# ---------- Pydantic Models ----------


class CommunityIntent(BaseModel):
    intent: Optional[CommunityAction] = None
    reason: Optional[str] = None


class ChatAnalysis(BaseModel):
    chat: ChatMessage
    sentiment_score: Optional[int] = None
    community_intent: Optional[CommunityIntent] = None


class PlayerScore(BaseModel):
    user_id: int
    score: int = 0


class UserInfo(BaseModel):
    role: UserRole = UserRole.PLAYER
    account_created: datetime
    last_seen: datetime
    status: UserStatus = UserStatus.OFFLINE
    user_history: Dict = Field(default_factory=dict)


class UserProfile(BaseModel):
    user_id: int
    username: str = Field(min_length=3, max_length=16)
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    info: UserInfo
    player_score: PlayerScore


class RewardSystem(BaseModel):
    points_awarded: int = 0
    reason: str = ""


class SentimentAnalysisState(BaseModel):
    chat_analysis: ChatAnalysis
    reward_system: Optional[RewardSystem] = None
    user_profile: Optional[UserProfile] = None
