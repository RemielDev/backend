import logging
from typing import Optional
from collections import defaultdict
from datetime import datetime

from agents.sentiment import (
    analyze_message_sentiment,
    ChatMessage,
    UserProfile,
    UserInfo,
    PlayerScore,
    UserRole,
    UserStatus,
    SentimentAnalysisState,
)

logger = logging.getLogger(__name__)


class SentimentService:
    def __init__(self):
        self.user_scores: dict[int, int] = defaultdict(int)
        self.user_profiles: dict[int, UserProfile] = {}

    def get_user_profile(self, user_id: int, username: str = None) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                username=username or f"user_{user_id}",
                info=UserInfo(
                    account_created=datetime.now(),
                    last_seen=datetime.now(),
                    status=UserStatus.ONLINE,
                ),
                player_score=PlayerScore(
                    user_id=user_id, score=self.user_scores[user_id]
                ),
            )
        else:
            self.user_profiles[user_id].info.last_seen = datetime.now()
            self.user_profiles[user_id].player_score.score = self.user_scores[user_id]

        return self.user_profiles[user_id]

    async def analyze_message_sentiment(
        self, message: ChatMessage, username: str = None
    ) -> SentimentAnalysisState:
        """Analyze sentiment and update user scores"""
        try:
            user_profile = self.get_user_profile(message.user_id, username)
            sentiment_result = await analyze_message_sentiment(message, user_profile)
            
            # Debug: Log the sentiment result structure
            logger.info(f"Raw sentiment result: {sentiment_result}")
            if hasattr(sentiment_result, 'chat_analysis') and sentiment_result.chat_analysis:
                logger.info(f"Chat analysis sentiment score: {getattr(sentiment_result.chat_analysis, 'sentiment_score', 'Not found')}")

            if sentiment_result.reward_system:
                self.user_scores[
                    message.user_id
                ] += sentiment_result.reward_system.points_awarded
                user_profile.player_score.score = self.user_scores[message.user_id]

                logger.info(
                    f"Updated score for user {message.user_id}: {self.user_scores[message.user_id]} "
                    f"(+{sentiment_result.reward_system.points_awarded})"
                )

            return sentiment_result

        except Exception as e:
            logger.error(f"Sentiment service error: {e}")
            raise

    def get_user_score(self, user_id: int) -> dict:
        """Get user score information"""
        user_profile = self.get_user_profile(user_id)
        return {
            "user_id": user_id,
            "username": user_profile.username,
            "current_score": self.user_scores[user_id],
        }

    def get_leaderboard(self, limit: int = 10) -> list:
        """Get top scoring users"""
        sorted_scores = sorted(
            self.user_scores.items(), key=lambda x: x[1], reverse=True
        )

        board = []
        for user_id, score in sorted_scores[:limit]:
            user_profile = self.user_profiles.get(user_id)
            board.append(
                {
                    "user_id": user_id,
                    "username": (
                        user_profile.username if user_profile else f"user_{user_id}"
                    ),
                    "score": score,
                }
            )

        return board

    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            "total_users": len(self.user_scores),
            "total_points": sum(self.user_scores.values()),
            "average_score": (
                sum(self.user_scores.values()) / len(self.user_scores)
                if self.user_scores
                else 0
            ),
            "highest_score": max(self.user_scores.values()) if self.user_scores else 0,
        } 