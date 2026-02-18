import logging
from typing import Optional
from agents.moderation import moderate_message, ChatMessage, ModerationState, ActionType

logger = logging.getLogger(__name__)


class ModerationService:
    @staticmethod
    async def moderate_chat_message(message: ChatMessage) -> ModerationState:
        """Core moderation business logic"""
        try:
            return await moderate_message(message)
        except Exception as e:
            logger.error(f"Moderation service error: {e}")
            raise

    @staticmethod
    def should_run_sentiment_analysis(state: ModerationState) -> bool:
        """Determine if message passed moderation checks"""
        # Only block sentiment analysis for severe actions
        if state.recommended_action and state.recommended_action.action in [
            ActionType.DELETE_MESSAGE, ActionType.BAN, ActionType.KICK
        ]:
            return False

        # Block for PII issues (privacy concern)
        if state.pii_result and state.pii_result.pii_presence:
            return False

        # Check minimum length
        if len(state.message.content) < 3:
            return False

        # Allow sentiment analysis for warnings and other non-blocking actions
        # This allows us to track sentiment even for mildly problematic content
        return True 