from pydantic_graph import Graph
from typing import Optional

from .state import ChatMessage, SentimentAnalysisState, UserProfile
from .nodes import (
    StartSentimentAnalysis,
    CheckContentLength,
    AnalyzeSentiment,
    AnalyzeCommunityIntent,
    CalculateRewards,
)

# ---------- Graph Definition ----------
sentiment_graph = Graph(
    nodes=[
        StartSentimentAnalysis,
        CheckContentLength,
        AnalyzeSentiment,
        AnalyzeCommunityIntent,
        CalculateRewards,
    ],
    state_type=SentimentAnalysisState,
)


# ---------- Main Function ----------
async def analyze_message_sentiment(
    message: ChatMessage, user_profile: Optional[UserProfile] = None
) -> SentimentAnalysisState:
    """Analyze message sentiment and community intent (only call after moderation passes)"""
    from .state import ChatAnalysis

    chat_analysis = ChatAnalysis(chat=message)
    state = SentimentAnalysisState(
        chat_analysis=chat_analysis, user_profile=user_profile
    )

    try:
        await sentiment_graph.run(StartSentimentAnalysis(), state=state)

        # Update user score
        if user_profile and state.reward_system:
            user_profile.player_score.score += state.reward_system.points_awarded

        return state

    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return state
