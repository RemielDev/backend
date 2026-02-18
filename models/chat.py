from pydantic import BaseModel
from datetime import datetime

class ChatMessage(BaseModel):
    message_id: str
    content: str
    user_id: int
    timestamp: datetime
    deleted: bool = False
