from datetime import datetime
from typing import Optional, Dict, Any
import logging

from models.chat import ChatMessage
from agents.sentiment import UserProfile, PlayerScore, UserInfo
from agents.sentiment.graph import analyze_message_sentiment
from agents.moderation import moderate_message, ActionType

logger = logging.getLogger(__name__)

class AIService:
    """Service layer for AI operations including sentiment analysis and moderation"""
    
    def __init__(self):
        self.logger = logger
    
    async def analyze_message_with_moderation(
        self, 
        message: str, 
        message_id: str, 
        player_id: int, 
        player_name: str
    ) -> Dict[str, Any]:
        """
        Analyze a message with both moderation and sentiment analysis
        Returns a comprehensive analysis result
        """
        try:
            # Create ChatMessage object
            chat_message = ChatMessage(
                message_id=message_id,
                content=message,
                user_id=player_id,
                timestamp=datetime.utcnow(),
                deleted=False
            )
            
            # Step 1: Moderation check
            self.logger.info(f"Starting moderation for message {message_id}")
            moderation_result = await moderate_message(chat_message)
            
            # Debug logging for moderation results
            self.logger.info(f"Moderation result: {moderation_result}")
            self.logger.info(f"Recommended action: {moderation_result.recommended_action}")
            if moderation_result.recommended_action:
                self.logger.info(f"Action type: {moderation_result.recommended_action.action.value}")
                self.logger.info(f"Action reason: {moderation_result.recommended_action.reason}")
            
            # Check if message should be blocked (fix ActionType usage)
            is_blocked = (
                moderation_result.recommended_action and 
                moderation_result.recommended_action.action in [ActionType.BAN, ActionType.KICK]
            )
            
            if is_blocked:
                self.logger.warning(f"Message {message_id} blocked by moderation: {moderation_result.recommended_action.reason}")
                return {
                    "sentiment_score": 0,
                    "moderation_passed": False,
                    "moderation_action": moderation_result.recommended_action.action.value,
                    "moderation_reason": moderation_result.recommended_action.reason,
                    "blocked": True
                }
            
            # Step 2: Sentiment analysis (only if moderation passes)
            self.logger.info(f"Starting sentiment analysis for message {message_id}")
            
            # Debug: Check if function is available
            self.logger.info(f"analyze_message_sentiment function: {analyze_message_sentiment}")
            
            # Create user profile for sentiment analysis (fix missing fields)
            current_time = datetime.utcnow()
            user_info = UserInfo(
                account_created=current_time,  # You might want to get this from your database
                last_seen=current_time
            )
            
            user_profile = UserProfile(
                user_id=player_id,
                username=player_name,
                info=user_info,
                player_score=PlayerScore(user_id=player_id, score=0)  # You might want to get current score from database
            )
            
            self.logger.info(f"About to call analyze_message_sentiment with message: {chat_message}")
            
            # Temporary workaround: Import the function locally to avoid namespace issues
            try:
                from agents.sentiment.graph import analyze_message_sentiment as sentiment_func
                sentiment_result = await sentiment_func(chat_message, user_profile)
                self.logger.info(f"analyze_message_sentiment completed successfully")
            except Exception as import_error:
                self.logger.error(f"Import error: {import_error}")
                # Fallback: Create a basic sentiment analysis
                from .basic_sentiment import analyze_basic_sentiment
                sentiment_result = await analyze_basic_sentiment(chat_message.content)
                self.logger.info(f"Using basic sentiment analysis fallback")
            
            # Extract sentiment score from the analysis
            sentiment_score = 0
            self.logger.info(f"Sentiment result structure: {sentiment_result}")
            self.logger.info(f"Chat analysis: {sentiment_result.chat_analysis if hasattr(sentiment_result, 'chat_analysis') else 'No chat_analysis'}")
            
            if sentiment_result.chat_analysis and sentiment_result.chat_analysis.sentiment_score is not None:
                sentiment_score = sentiment_result.chat_analysis.sentiment_score
                self.logger.info(f"Extracted sentiment score: {sentiment_score}")
            else:
                self.logger.warning(f"Could not extract sentiment score - chat_analysis: {getattr(sentiment_result, 'chat_analysis', None)}")
            
            # Prepare comprehensive result
            result = {
                "sentiment_score": sentiment_score,
                "moderation_passed": True,
                "blocked": False
            }
            
            # Add moderation details if there were warnings
            if moderation_result.recommended_action:
                result["moderation_action"] = moderation_result.recommended_action.action.value
                result["moderation_reason"] = moderation_result.recommended_action.reason
            
            # Add sentiment analysis details
            if sentiment_result.chat_analysis:
                result["sentiment_details"] = {
                    "confidence": getattr(sentiment_result.chat_analysis, 'confidence', None),
                    "emotion": getattr(sentiment_result.chat_analysis, 'emotion', None),
                    "toxicity_score": getattr(sentiment_result.chat_analysis, 'toxicity_score', None)
                }
            
            # Add community intent if available
            if sentiment_result.chat_analysis and sentiment_result.chat_analysis.community_intent:
                result["community_intent"] = {
                    "intent_type": sentiment_result.chat_analysis.community_intent.intent.value if sentiment_result.chat_analysis.community_intent.intent else None,
                    "reason": sentiment_result.chat_analysis.community_intent.reason
                }
            
            # Add reward information if available
            if sentiment_result.reward_system:
                result["rewards"] = {
                    "points_awarded": sentiment_result.reward_system.points_awarded,
                    "reason": sentiment_result.reward_system.reason
                }
            
            self.logger.info(f"Analysis completed for message {message_id}: sentiment={sentiment_score}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in AI analysis for message {message_id}: {e}")
            # Return fallback result
            return {
                "sentiment_score": 0,
                "moderation_passed": True,
                "blocked": False,
                "error": str(e)
            }
    
    async def moderate_message_only(self, message: str, message_id: str, user_id: int) -> Dict[str, Any]:
        """
        Perform only moderation check on a message
        """
        try:
            chat_message = ChatMessage(
                message_id=message_id,
                content=message,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                deleted=False
            )
            
            moderation_result = await moderate_message(chat_message)
            
            return {
                "passed": moderation_result.recommended_action.action not in [ActionType.DELETE_MESSAGE, ActionType.BAN, ActionType.KICK] if moderation_result.recommended_action else True,
                "action": moderation_result.recommended_action.action.value if moderation_result.recommended_action else None,
                "reason": moderation_result.recommended_action.reason if moderation_result.recommended_action else None,
                "pii_detected": bool(moderation_result.pii_result and moderation_result.pii_result.pii_presence),
                "content_issues": bool(moderation_result.content_result and moderation_result.content_result.main_category.value != "OK")
            }
            
        except Exception as e:
            self.logger.error(f"Error in moderation for message {message_id}: {e}")
            return {
                "passed": True,  # Fail open for safety
                "error": str(e)
            }

# Create singleton instance
ai_service = AIService() 