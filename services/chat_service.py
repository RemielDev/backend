import logging
from datetime import datetime
from typing import Optional, Dict, Any

from models.chat import ChatMessage
from agents.moderation import ModerationState
from agents.sentiment import SentimentAnalysisState
from .moderation_service import ModerationService
from .sentiment_service import SentimentService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.moderation_service = ModerationService()
        self.sentiment_service = SentimentService()

    async def process_message(self, message: ChatMessage, username: str = None) -> Dict[str, Any]:
        """Main business logic: analyze sentiment first, then moderate"""
        
        # Step 1: Sentiment Analysis (ALWAYS run - needed for Roblox script)
        sentiment_result = None
        try:
            sentiment_result = await self.sentiment_service.analyze_message_sentiment(
                message, username
            )
            logger.info(f"Sentiment analysis completed with score: {sentiment_result.chat_analysis.sentiment_score if sentiment_result and sentiment_result.chat_analysis else 'None'}")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")

        # Step 2: Moderation (run after sentiment to determine if message should be blocked)
        moderation_state = await self.moderation_service.moderate_chat_message(message)

        response = {
            "moderation_state": moderation_state,
            "sentiment_analysis": sentiment_result,
            "user_score_updated": bool(sentiment_result),
            "new_score": self.sentiment_service.user_scores.get(message.user_id) if sentiment_result else None,
        }

        return response

    def get_user_score(self, user_id: int) -> dict:
        """Get user score"""
        return self.sentiment_service.get_user_score(user_id)

    def get_leaderboard(self, limit: int = 10) -> dict:
        """Get leaderboard"""
        return {"leaderboard": self.sentiment_service.get_leaderboard(limit)}

    def get_stats(self) -> dict:
        """Get system stats"""
        return self.sentiment_service.get_stats()

    def convert_to_api_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal result to API response format"""
        moderation_state = result["moderation_state"]
        sentiment_analysis = result.get("sentiment_analysis")
        
        # Extract basic info
        api_response = {
            "sentiment_score": 0,
            "moderation_passed": True,  # Default to passed, will be updated based on moderation results
            "blocked": False,
            "moderation_action": None,
            "moderation_reason": None,
            "sentiment_details": None,
            "community_intent": None,
            "rewards": None,
            "error": None
        }

        # Handle moderation results (moderation has already run at this point)
        if moderation_state.recommended_action:
            # If there's a recommended action, moderation found issues
            api_response["moderation_passed"] = False
            api_response["moderation_action"] = moderation_state.recommended_action.action.value
            api_response["moderation_reason"] = moderation_state.recommended_action.reason
            
            # Check if message should be blocked
            from agents.moderation import ActionType
            if moderation_state.recommended_action.action in [ActionType.DELETE_MESSAGE, ActionType.BAN, ActionType.KICK]:
                api_response["blocked"] = True
        else:
            # No recommended action means moderation passed
            api_response["moderation_passed"] = True

        # Handle sentiment results (ALWAYS AVAILABLE NOW - prioritize for Roblox script)
        if sentiment_analysis:
            logger.info(f"Sentiment analysis result: {sentiment_analysis}")
            logger.info(f"Chat analysis: {sentiment_analysis.chat_analysis if hasattr(sentiment_analysis, 'chat_analysis') else 'No chat_analysis'}")
            
            if sentiment_analysis.chat_analysis:
                logger.info(f"Sentiment score in chat_analysis: {getattr(sentiment_analysis.chat_analysis, 'sentiment_score', 'No sentiment_score attribute')}")
                
                if sentiment_analysis.chat_analysis.sentiment_score is not None:
                    api_response["sentiment_score"] = sentiment_analysis.chat_analysis.sentiment_score
                    logger.info(f"Extracted sentiment score: {sentiment_analysis.chat_analysis.sentiment_score}")
                else:
                    logger.warning("sentiment_score is None in chat_analysis")
            else:
                logger.warning("No chat_analysis in sentiment result")

            # Add sentiment details
            api_response["sentiment_details"] = {
                "confidence": getattr(sentiment_analysis.chat_analysis, 'confidence', None),
                "emotion": getattr(sentiment_analysis.chat_analysis, 'emotion', None),
                "toxicity_score": getattr(sentiment_analysis.chat_analysis, 'toxicity_score', None)
            }

            # Add community intent
            if sentiment_analysis.chat_analysis.community_intent:
                api_response["community_intent"] = {
                    "intent_type": sentiment_analysis.chat_analysis.community_intent.intent.value if sentiment_analysis.chat_analysis.community_intent.intent else None,
                    "reason": sentiment_analysis.chat_analysis.community_intent.reason
                }

            # Add reward information
            if sentiment_analysis.reward_system:
                api_response["rewards"] = {
                    "points_awarded": sentiment_analysis.reward_system.points_awarded,
                    "reason": sentiment_analysis.reward_system.reason
                }

        return api_response 