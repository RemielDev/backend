import re
from typing import Dict, Any

class BasicSentimentResult:
    def __init__(self, sentiment_score: int):
        self.chat_analysis = BasicChatAnalysis(sentiment_score)
        self.reward_system = None

class BasicChatAnalysis:
    def __init__(self, sentiment_score: int):
        self.sentiment_score = sentiment_score
        self.community_intent = None

async def analyze_basic_sentiment(message: str) -> BasicSentimentResult:
    """
    Basic sentiment analysis using keyword matching
    Returns a score from -100 to +100
    """
    message_lower = message.lower()
    
    # Negative keywords and their weights
    negative_keywords = {
        'hate': -40,
        'suck': -30,
        'terrible': -35,
        'awful': -35,
        'bad': -20,
        'worst': -40,
        'stupid': -25,
        'dumb': -25,
        'shit': -30,
        'fuck': -35,
        'damn': -20,
        'annoying': -25,
        'boring': -20,
        'useless': -30,
        'garbage': -35,
        'trash': -30
    }
    
    # Positive keywords and their weights
    positive_keywords = {
        'love': 40,
        'great': 30,
        'awesome': 35,
        'amazing': 40,
        'good': 20,
        'best': 40,
        'excellent': 35,
        'perfect': 40,
        'wonderful': 35,
        'fantastic': 35,
        'cool': 25,
        'nice': 20,
        'fun': 25,
        'enjoy': 30,
        'like': 15
    }
    
    score = 0
    
    # Check for negative keywords
    for keyword, weight in negative_keywords.items():
        if keyword in message_lower:
            score += weight
    
    # Check for positive keywords
    for keyword, weight in positive_keywords.items():
        if keyword in message_lower:
            score += weight
    
    # Check for intensifiers
    if any(word in message_lower for word in ['really', 'very', 'so', 'extremely']):
        score = int(score * 1.3)  # Amplify by 30%
    
    # Check for multiple exclamation marks (indicates strong emotion)
    exclamation_count = message.count('!')
    if exclamation_count > 1:
        score = int(score * (1 + exclamation_count * 0.1))
    
    # Clamp score to -100 to +100 range
    score = max(-100, min(100, score))
    
    return BasicSentimentResult(score) 