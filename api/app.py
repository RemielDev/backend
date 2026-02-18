from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
from pathlib import Path
import google.generativeai as genai
import random
import json
import re
from supabase import create_client, Client
from datetime import datetime, timezone
import requests
import logging
from dotenv import load_dotenv
import asyncio
import time
import threading
from collections import defaultdict

# Helper function to clean AI service response
def clean_ai_response(response):
    """Clean the AI service response to remove any non-serializable objects"""
    if not isinstance(response, dict):
        return {"sentiment_score": 0, "error": "Invalid response format"}
    
    # Extract only the fields we need
    cleaned = {
        "sentiment_score": response.get("sentiment_score", 0),
        "error": response.get("error")
    }
    
    # Add moderation info if present
    if "moderation_action" in response:
        cleaned["moderation_action"] = response["moderation_action"]
    if "moderation_reason" in response:
        cleaned["moderation_reason"] = response["moderation_reason"]
    
    return cleaned

# Import the new AI service
from services.chat_service import ChatService
from models.chat import ChatMessage
from services.ai_service import ai_service

# Import the Roblox service
from services.roblox_service import roblox_service

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing/logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} -> {getattr(response, 'status_code', 'NA')} in {duration:.3f}s"
        )

# API Keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
BLOOM_API_KEY = os.environ.get("BLOOM_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize APIs
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Supabase client
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    logger.warning("Supabase not configured - database operations will be skipped")

ROBLOX_THUMBNAILS_API_URL = "https://thumbnails.roblox.com/v1/users/avatar-headshot"

logger.info("Server starting up with configuration...")
logger.info(f"Google API Key configured: {'Yes' if GOOGLE_API_KEY else 'No'}")
logger.info(f"Roblox API Key configured: {'Yes' if BLOOM_API_KEY else 'No'}")
logger.info(f"Supabase configured: {'Yes' if SUPABASE_URL and SUPABASE_KEY else 'No'}")
logger.info("Gemini model initialized")

# Pydantic models for request/response
class AnalyzeRequest(BaseModel):
    message: str
    message_id: str
    player_id: Optional[int] = None
    player_name: Optional[str] = None

# Simple response for Roblox - just sentiment data
class SentimentResponse(BaseModel):
    player_id: int
    player_name: str
    message_id: str
    message: str
    sentiment_score: int
    moderation_action: Optional[str] = None
    moderation_reason: Optional[str] = None
    error: Optional[str] = None

# Full response for frontend - includes moderation data
class AnalyzeResponse(BaseModel):
    player_id: int
    player_name: str
    message_id: str
    message: str
    sentiment_score: int
    moderation_passed: bool
    blocked: bool
    moderation_action: Optional[str] = None
    moderation_reason: Optional[str] = None
    sentiment_details: Optional[Dict[str, Any]] = None
    community_intent: Optional[Dict[str, Any]] = None
    rewards: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Background moderation function
async def run_background_moderation(chat_message: ChatMessage, message_id: str, player_id: int, user_message: str):
    """Run moderation analysis in background and update database if needed"""
    try:
        logger.info(f"Starting background moderation for message {message_id}")
        
        chat_service = ChatService()
        moderation_state = await chat_service.moderation_service.moderate_chat_message(chat_message)
        
        from agents.moderation import ActionType
        
        # If the AI recommends an action, proceed.
        if moderation_state.recommended_action:
            action = moderation_state.recommended_action.action
            reason = moderation_state.recommended_action.reason
            
            logger.info(f"AI recommended action: {action.value} for message {message_id}")

            # Log this action to the permanent moderation_actions table
            moderation_record = {
                "player_id": player_id,
                "message_id": message_id,
                "action": action.value.lower(),
                "reason": reason,
                "performed_by": "ai",
                "success": True,  # Action was determined by AI
                "error": None
            }
            supabase.table('moderation_actions').insert(moderation_record).execute()
            logger.info(f"AI action '{action.value}' logged to moderation_actions.")

            # Update the message with AI action
            update_data = {
                "moderation_action": action.value,
                "moderation_reason": reason,
                "flag": action == ActionType.BAN  # Only flag bans for human review
            }
            supabase.table('messages').update(update_data).eq('message_id', message_id).execute()
            logger.info(f"Message {message_id} updated with AI action. Flagged: {action == ActionType.BAN}")
            
        else:
            logger.info(f"No moderation issues found for message {message_id}")

    except Exception as e:
        logger.error(f"Background moderation failed for message {message_id}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

# Dependency for API key validation
async def verify_api_key(request: Request):
    if BLOOM_API_KEY:
        api_key = request.headers.get('X-API-Key')
        logger.info(f"API Key authentication: {'Success' if api_key == BLOOM_API_KEY else 'Failed'}")
        if not api_key or api_key != BLOOM_API_KEY:
            logger.info("Unauthorized access attempt - invalid API key")
            raise HTTPException(status_code=401, detail="Unauthorized")

# Simple in-memory per-key rate limiter (200 req/min default)
class SimpleRateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def allow(self, key: str, limit: int = 200, window_seconds: int = 60) -> bool:
        if not key:
            return False
        now = time.time()
        with self.lock:
            window_start = now - window_seconds
            recent = [t for t in self.requests[key] if t >= window_start]
            self.requests[key] = recent
            if len(recent) >= limit:
                return False
            self.requests[key].append(now)
            return True

rate_limiter = SimpleRateLimiter()

# Dependency for Roblox Platform API key validation
async def verify_roblox_platform_key(request: Request):
    roblox_platform_key = os.environ.get("ROBLOX_PLATFORM_API_KEY")
    if roblox_platform_key:
        api_key = request.headers.get('X-API-Key')
        logger.info(f"Roblox Platform API Key authentication: {'Success' if api_key == roblox_platform_key else 'Failed'}")
        if not api_key or api_key != roblox_platform_key:
            logger.info("Unauthorized access attempt - invalid Roblox Platform API key")
            raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        raise HTTPException(status_code=500, detail="Roblox Platform API key not configured")

@app.get("/")
async def home():
    logger.info("Home endpoint accessed")
    return {
        "message": "BloomAI API is running",
        "status": "success"
    }

@app.post("/api/analyze", response_model=SentimentResponse)
async def analyze_sentiment_with_background_moderation(
    request_data: AnalyzeRequest,
    request: Request,
    _: None = Depends(verify_api_key)
):
    """
    Returns sentiment analysis immediately with moderation action, runs additional moderation in background
    """
    # Rate limiting per API key
    api_key = request.headers.get('X-API-Key')
    if not rate_limiter.allow(api_key, limit=200, window_seconds=60):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    user_message = request_data.message
    message_id = request_data.message_id
    player_id = request_data.player_id
    player_name = request_data.player_name
    
    # Only use random values if no player_id or player_name was provided
    if player_id is None:
        player_id = random.randint(1, 100)
    if player_name is None:
        player_name = f"Player{random.randint(1, 999)}"
    
    logger.info(f"Processing enhanced analysis: Player ID: {player_id}, Player Name: {player_name}")
    logger.info(f"Message to analyze: {user_message}")
    
    try:
        # Smart sampling decision
        sampling_rate = 0.1  # 10% of normal messages
        must_process = False
        if player_id is not None and supabase:
            try:
                # Treat players with any prior moderation action as high-risk -> always analyze
                prior = supabase.table('moderation_actions').select('id').eq('player_id', player_id).limit(1).execute()
                must_process = bool(prior.data)
            except Exception as _e:
                must_process = False

        process_with_ai = must_process or (random.random() < sampling_rate)

        current_time = datetime.now(timezone.utc)

        if not process_with_ai:
            # Store lightweight record without invoking AI
            try:
                if supabase:
                    player_data = {
                        "player_id": player_id,
                        "player_name": player_name,
                        "last_seen": current_time.isoformat()
                    }
                    supabase.table('players').upsert(player_data).execute()

                    message_data = {
                        "message_id": message_id,
                        "player_id": player_id,
                        "message": user_message,
                        "sentiment_score": 0,
                        "created_at": current_time.isoformat()
                    }
                    supabase.table('messages').insert(message_data).execute()
            except Exception as e:
                logger.error(f"Failed to store sampled message {message_id}: {e}")

            logger.info(f"Message {message_id} sampled (rate={sampling_rate}, must_process={must_process})")
            return SentimentResponse(
                player_id=player_id,
                player_name=player_name,
                message_id=message_id,
                message=user_message,
                sentiment_score=0,
                moderation_action=None,
                moderation_reason=None,
                error="sampled"
            )

        # Create chat message for analysis
        chat_message = ChatMessage(
            message_id=message_id,
            content=user_message,
            user_id=player_id,
            timestamp=current_time,
            deleted=False
        )

        # Run sentiment analysis with moderation
        try:
            sentiment_result = await ai_service.analyze_message_with_moderation(
                user_message, message_id, player_id, player_name
            )
            # Clean the response to remove any non-serializable objects
            cleaned_result = clean_ai_response(sentiment_result)
            logger.info(f"Cleaned AI result: {cleaned_result}")
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            cleaned_result = {"sentiment_score": 0, "error": str(e)}
        
        # Store player data
        player_data = {
            "player_id": player_id,
            "player_name": player_name,
            "last_seen": current_time.isoformat()
        }
        
        # Upsert player data
        supabase.table('players').upsert(player_data).execute()
        logger.info("Player data stored/updated in Supabase")
        
        # Store message data
        message_data = {
            "message_id": message_id,
            "player_id": player_id,
            "message": user_message,
            "sentiment_score": cleaned_result.get("sentiment_score", 0),
            "created_at": current_time.isoformat()
        }
        
        supabase.table('messages').insert(message_data).execute()
        logger.info("Message data stored in Supabase")
        
        # Update player's total sentiment score
        try:
            # Get current total sentiment score
            player_result = supabase.table('players').select('total_sentiment_score').eq('player_id', player_id).execute()
            current_total = player_result.data[0].get('total_sentiment_score', 0) if player_result.data else 0
            
            # Add new sentiment score
            new_total = current_total + cleaned_result.get("sentiment_score", 0)
            
            # Update the total
            supabase.table('players').update({"total_sentiment_score": new_total}).eq('player_id', player_id).execute()
            logger.info(f"Updated total sentiment score for player {player_id}: {new_total}")
        except Exception as e:
            logger.error(f"Error updating total sentiment score: {e}")
        
        # Extract moderation action and reason for immediate response
        moderation_action = cleaned_result.get("moderation_action")
        moderation_reason = cleaned_result.get("moderation_reason")
        
        # Log moderation action for debugging
        if moderation_action:
            logger.info(f"Immediate moderation action for player {player_id}: {moderation_action} - {moderation_reason}")
        else:
            logger.info(f"No moderation action for player {player_id}")
        
        # Record moderation action in database if present
        if moderation_action:
            try:
                moderation_record = {
                    "player_id": player_id,
                    "message_id": message_id,
                    "action": moderation_action.lower(),
                    "reason": moderation_reason,
                    "performed_by": "ai",
                    "success": True,  # Action was determined by AI
                    "error": None
                }
                supabase.table('moderation_actions').insert(moderation_record).execute()
                logger.info(f"Moderation action '{moderation_action}' logged to moderation_actions table.")
                
                # Update message with moderation info
                update_data = {
                    "moderation_action": moderation_action,
                    "moderation_reason": moderation_reason,
                    "flag": moderation_action.lower() == "ban"  # Flag bans for human review
                }
                supabase.table('messages').update(update_data).eq('message_id', message_id).execute()
                logger.info(f"Message {message_id} updated with moderation action.")
                
            except Exception as db_error:
                logger.error(f"Failed to record moderation action in database: {db_error}")
        
        logger.info(f"Full cleaned result: {cleaned_result}")
        
        # Return sentiment result with moderation action immediately
        try:
            # Ensure all values are serializable
            sentiment_score = cleaned_result.get("sentiment_score", 0)
            if isinstance(sentiment_score, (int, float)):
                sentiment_score = int(sentiment_score)
            else:
                sentiment_score = 0
                
            error_msg = cleaned_result.get("error")
            if error_msg and not isinstance(error_msg, str):
                error_msg = str(error_msg)
            
            result = SentimentResponse(
                player_id=player_id,
                player_name=player_name,
                message_id=message_id,
                message=user_message,
                sentiment_score=sentiment_score,
                moderation_action=moderation_action,
                moderation_reason=moderation_reason,
                error=error_msg
            )
            
            logger.info(f"Returning sentiment result with moderation: {result}")
            
            # Run additional moderation in background (after response is sent)
            logger.info("Starting background moderation task...")
            asyncio.create_task(run_background_moderation(chat_message, message_id, player_id, user_message))
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating response: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback response without any complex objects
            fallback_result = SentimentResponse(
                player_id=player_id,
                player_name=player_name,
                message_id=message_id,
                message=user_message,
                sentiment_score=0,
                error=f"Response serialization error: {str(e)}"
            )
            
            logger.info(f"Returning fallback result: {fallback_result}")
            
            # Still run moderation in background
            asyncio.create_task(run_background_moderation(chat_message, message_id, player_id, user_message))
            
            return fallback_result
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return SentimentResponse(
            player_id=player_id,
            player_name=player_name,
            message_id=message_id,
            message=user_message,
            sentiment_score=0,
            error=str(e)
        )

@app.post("/api/moderate", response_model=Dict[str, Any])
async def moderate_message_endpoint(
    request_data: AnalyzeRequest,
    _: None = Depends(verify_api_key)
):
    """
    Moderation-only endpoint for checking messages
    """
    try:
        result = await ai_service.moderate_message_only(
            message=request_data.message,
            message_id=request_data.message_id,
            user_id=request_data.player_id or random.randint(1, 100)
        )
        return result
    except Exception as e:
        logger.error(f"Error in moderation endpoint: {e}")
        return {"passed": True, "error": str(e)}

@app.get("/api/players")
async def get_players():
    try:
        response = supabase.table('players').select('*').execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch players")

@app.get("/api/messages")
async def get_messages(
    player_id: Optional[str] = Query(None),
    limit: int = Query(100)
):
    try:
        query = supabase.table('messages').select('*, players(player_name)').order('created_at', desc=True).limit(limit)
        
        if player_id:
            query = query.eq('player_id', player_id)
        
        response = query.execute()
        
        # Process the response to add player_name to each message
        processed_messages = []
        for msg in response.data:
            # Extract player_name from the joined data
            player_name = None
            if msg.get('players') and isinstance(msg['players'], list) and len(msg['players']) > 0:
                player_name = msg['players'][0].get('player_name')
            elif msg.get('players') and isinstance(msg['players'], dict):
                player_name = msg['players'].get('player_name')
            
            # Fallback to a default name if player_name is not available
            if not player_name:
                player_name = f"Player{msg['player_id']}"
            
            # Add player_name to the message
            msg['player_name'] = player_name
            processed_messages.append(msg)
        
        return processed_messages
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")

@app.get("/api/live")
async def get_live_messages(limit: int = Query(20)):
    try:
        # Created a sql function to handle this easily and more efficiently
        messages_response = supabase.rpc('get_live_messages', {'p_limit': limit}).execute()
        
        return messages_response.data
    except Exception as e:
        logger.error(f"Error fetching live messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch live messages: {str(e)}")

@app.get("/api/roblox-avatar")
async def get_roblox_avatar(userId: Optional[str] = Query(None)):
    """
    Proxies requests to the Roblox Thumbnails API to fetch user avatar headshots.
    Takes 'userId' as a query parameter.
    """
    if not userId:
        logger.info("Roblox avatar proxy: Missing userId parameter")
        raise HTTPException(status_code=400, detail="Missing userId parameter")

    try:
        user_id_int = int(userId)
        if user_id_int <= 0:
            logger.info(f"Roblox avatar proxy: Invalid userId format (non-positive): {userId}")
            raise HTTPException(status_code=400, detail="Invalid userId format")
    except ValueError:
        logger.info(f"Roblox avatar proxy: Invalid userId format (not an integer): {userId}")
        raise HTTPException(status_code=400, detail="Invalid userId format")

    logger.info(f"Roblox avatar proxy: Fetching avatar for user ID: {userId}")

    # Parameters for the Roblox API request
    roblox_params = {
        "userIds": userId,  # Pass the single user ID
        "size": "150x150",  # Desired size
        "format": "Png"     # Desired format
    }

    try:
        # Make the request to the actual Roblox Thumbnails API
        roblox_response = requests.get(ROBLOX_THUMBNAILS_API_URL, params=roblox_params)
        
        # Handle 404 gracefully - return empty response instead of error
        if roblox_response.status_code == 404:
            logger.info(f"Roblox avatar proxy: Avatar not found for user ID: {userId} (404 from Roblox API)")
            return {"imageUrl": None}
        
        roblox_response.raise_for_status()  # Raise an HTTPError for other bad responses (5xx)

        roblox_data = roblox_response.json()
        logger.info(f"Roblox avatar proxy: Received data from Roblox API: {roblox_data}")

        # Parse the response to find the image URL
        # The response structure is { "data": [ { "targetId": ..., "state": ..., "imageUrl": ... } ] }
        image_url = None
        if roblox_data and 'data' in roblox_data and isinstance(roblox_data['data'], list):
            # Find the item matching the requested user ID
            user_data = next((item for item in roblox_data['data'] if str(item.get('targetId')) == userId), None)
            if user_data and 'imageUrl' in user_data:
                image_url = user_data['imageUrl']
                logger.info(f"Roblox avatar proxy: Found imageUrl: {image_url}")
            else:
                logger.info(f"Roblox avatar proxy: imageUrl not found in Roblox response for user ID: {userId}")

        if image_url:
            # Return the image URL to the frontend
            return {"imageUrl": image_url}
        else:
            # Return null imageUrl instead of 404 error
            logger.info(f"Roblox avatar proxy: No avatar found for user ID: {userId}")
            return {"imageUrl": None}

    except requests.exceptions.RequestException as e:
        # Handle errors during the request to Roblox API
        logger.error(f"Roblox avatar proxy: Error fetching from Roblox API: {e}")
        # Return null instead of 500 error
        return {"imageUrl": None}
    except Exception as e:
        # Handle any other unexpected errors
        logger.error(f"Roblox avatar proxy: An unexpected error occurred: {e}")
        # Return null instead of 500 error
        return {"imageUrl": None}

@app.get("/api/top-players")
async def get_top_players(limit: int = Query(10)):
    try:
        logger.info(f"Fetching top players with limit: {limit}")
        
        response = supabase.rpc('get_top_players_by_sentiment', {'p_limit': limit}).execute()
        
        logger.info(f"Supabase RPC response: {response}")
        logger.info(f"Response data: {response.data}")
        
        # Format the response to ensure we have the required fields
        formatted_data = []
        for player in response.data:
            formatted_data.append({
                "player_id": player["player_id"],
                "player_name": player["player_name"],
                "total_sentiment_score": player["total_sentiment_score"],
                "message_count": player["message_count"]
            })
        
        logger.info(f"Formatted data: {formatted_data}")
        return formatted_data
    except Exception as e:
        logger.error(f"Error fetching top players: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch top players: {str(e)}")

@app.get("/api/analytics/all-time/sentiment-trend")
async def get_sentiment_trend_data_all_time(interval: str = Query('month')):
    try:
        if interval not in ['day', 'hour', 'week', 'month', 'year']:
            raise HTTPException(status_code=400, detail="Invalid interval unit")

        params = {'interval_unit': interval}
        # Call the all_time version of the function
        response = supabase.rpc('get_sentiment_trend_all_time', params).execute()

        if hasattr(response, 'data'):
            logger.info(f"Fetched all-time sentiment trend data for interval: {interval}")
            return response.data
        else:
            logger.error(f"Error in Supabase response for all-time sentiment trend: {response}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch all-time sentiment trend data: {str(response)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching all-time sentiment trend: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch all-time sentiment trend: {str(e)}")

@app.get("/api/analytics/all-time/sentiment-distribution")
async def get_sentiment_distribution_data_all_time(
    positive_threshold: int = Query(30),
    negative_threshold: int = Query(-30)
):
    try:
        params = {
            'positive_threshold': positive_threshold,
            'negative_threshold': negative_threshold
        }
        # Call the all_time version of the function
        response = supabase.rpc('get_sentiment_distribution_all_time', params).execute()

        if hasattr(response, 'data'):
            logger.info(f"Fetched all-time sentiment distribution data")
            return response.data
        else:
            logger.error(f"Error in Supabase response for all-time sentiment distribution: {response}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch all-time sentiment distribution data: {str(response)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching all-time sentiment distribution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch all-time sentiment distribution: {str(e)}")

@app.get("/api/analytics/all-time/overall-stats")
async def get_overall_stats_data_all_time():
    try:
        # Call the all_time version of the function (no parameters needed)
        response = supabase.rpc('get_overall_analytics_stats_all_time', {}).execute()

        if hasattr(response, 'data'):
            logger.info(f"Fetched all-time overall stats")
            data_to_return = response.data[0] if response.data and isinstance(response.data, list) else response.data
            return data_to_return
        else:
            logger.error(f"Error in Supabase response for all-time overall stats: {response}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch all-time overall stats: {str(response)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching all-time overall stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch all-time overall stats: {str(e)}")

class ServiceStatus(BaseModel):
    status: str
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    services: Dict[str, ServiceStatus]

def check_supabase_connection() -> ServiceStatus:
    """Check if Supabase connection is working"""
    try:
        if not supabase:
            return ServiceStatus(status="not_configured")
        
        # Try a simple query
        supabase.table('players').select("count").limit(1).execute()
        return ServiceStatus(status="healthy")
    except Exception as e:
        logger.error(f"Supabase health check failed: {e}")
        return ServiceStatus(
            status="unhealthy",
            details={"error": str(e)}
        )

def check_ai_model() -> ServiceStatus:
    """Check if Gemini AI model is working"""
    try:
        if not GOOGLE_API_KEY:
            return ServiceStatus(status="not_configured")
        
        # Try a simple generation
        response = model.generate_content("test")
        if response:
            return ServiceStatus(status="healthy")
        return ServiceStatus(status="unhealthy")
    except Exception as e:
        logger.error(f"AI model health check failed: {e}")
        return ServiceStatus(
            status="unhealthy",
            details={"error": str(e)}
        )

def check_bloom_api() -> ServiceStatus:
    """Check if Roblox API key is configured"""
    if not BLOOM_API_KEY:
        return ServiceStatus(status="not_configured")
    return ServiceStatus(status="healthy")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint that verifies all critical services.
    Returns the status of each service and overall system health.
    """
    logger.info("Health check requested")
    
    # Check all services
    services = {
        "database": check_supabase_connection(),
        "ai_model": check_ai_model(),
        "roblox_api": check_bloom_api()
    }
    
    # Determine overall status
    overall_status = "healthy"
    for service_status in services.values():
        if service_status.status == "unhealthy":
            overall_status = "unhealthy"
            break
        elif service_status.status == "not_configured":
            overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services
    )

# New Pydantic models for moderation requests
class ModerationActionRequest(BaseModel):
    player_id: int
    action: str  # "warn", "kick", "ban"
    reason: str
    game_id: Optional[int] = None

class FlaggedMessageResponse(BaseModel):
    message_id: str
    player_id: int
    player_name: str
    message: str
    sentiment_score: int
    created_at: str
    moderation_action: Optional[str] = None
    moderation_reason: Optional[str] = None
    flag: bool

@app.post("/api/moderate/action")
async def perform_moderation_action(
    request_data: ModerationActionRequest,
    _: None = Depends(verify_api_key)
):
    """Perform manual moderation action (warn/kick/ban) on a player"""
    try:
        player_id = request_data.player_id
        action = request_data.action.lower()
        reason = request_data.reason
        game_id = request_data.game_id
        
        logger.info(f"Performing {action} on player {player_id} for reason: {reason}")
        
        # Perform the action via Roblox API
        if action == "warn":
            result = await roblox_service.warn_user(player_id, reason, game_id)
        elif action == "kick":
            result = await roblox_service.kick_user(player_id, reason, game_id)
        elif action == "ban":
            result = await roblox_service.ban_user(player_id, reason, game_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'warn', 'kick', or 'ban'")
        
        # Store moderation action in database
        try:
            moderation_record = {
                "player_id": player_id,
                "action": action,
                "reason": reason,
                "performed_by": "manual",  # or "ai" for automated actions
                "created_at": datetime.now(timezone.utc).isoformat(),
                "success": result.get("success", False),
                "error": result.get("error")
            }
            
            supabase.table('moderation_actions').insert(moderation_record).execute()
            logger.info(f"Stored moderation action record for player {player_id}")
            
        except Exception as db_error:
            logger.error(f"Failed to store moderation action in database: {db_error}")
        
        return {
            "success": result.get("success", False),
            "action": action,
            "player_id": player_id,
            "reason": reason,
            "error": result.get("error"),
            "roblox_response": result.get("response")
        }
        
    except Exception as e:
        logger.error(f"Error performing moderation action: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform moderation action: {str(e)}")

@app.get("/api/messages/flagged")
async def get_flagged_messages(limit: int = Query(50)):
    """Get messages that have been flagged for human review"""
    try:
        # Get flagged messages with player information
        response = supabase.table('messages').select('*, players(player_name)').eq('flag', True).order('created_at', desc=True).limit(limit).execute()
        
        flagged_messages = []
        for msg in response.data:
            # Extract player_name from the joined data
            player_name = None
            if msg.get('players') and isinstance(msg['players'], list) and len(msg['players']) > 0:
                player_name = msg['players'][0].get('player_name')
            elif msg.get('players') and isinstance(msg['players'], dict):
                player_name = msg['players'].get('player_name')
            
            # Fallback to a default name if player_name is not available
            if not player_name:
                player_name = f"Player{msg['player_id']}"
            
            flagged_messages.append(FlaggedMessageResponse(
                message_id=msg['message_id'],
                player_id=msg['player_id'],
                player_name=player_name,
                message=msg['message'],
                sentiment_score=msg['sentiment_score'],
                created_at=msg['created_at'],
                moderation_action=msg.get('moderation_action'),
                moderation_reason=msg.get('moderation_reason'),
                flag=msg.get('flag', False)
            ))
        
        return flagged_messages
        
    except Exception as e:
        logger.error(f"Error fetching flagged messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch flagged messages")

@app.post("/api/messages/{message_id}/review")
async def review_flagged_message(
    message_id: str,
    action: str = Query(..., description="Action to take: 'approve', 'warn', 'kick', 'ban'"),
    reason: Optional[str] = Query(None, description="Reason for the action"),
    _: None = Depends(verify_api_key)
):
    """Review a flagged message and take action"""
    try:
        logger.info(f"Review request received: message_id={message_id}, action={action}, reason={reason}")
        
        # Get the message
        message_response = supabase.table('messages').select('*').eq('message_id', message_id).execute()
        
        if not message_response.data:
            logger.error(f"Message not found: {message_id}")
            raise HTTPException(status_code=404, detail="Message not found")
        
        message = message_response.data[0]
        player_id = message['player_id']
        logger.info(f"Found message: player_id={player_id}, current_flag={message.get('flag', 'not_set')}")
        
        # Update message flag status
        update_data = {"flag": False}
        logger.info(f"Setting flag to False for message {message_id}")
        
        if action == "approve":
            # Just remove the flag
            logger.info(f"Approving message {message_id} - just removing flag")
            pass
        elif action in ["warn", "kick", "ban"]:
            # Perform moderation action
            if not reason:
                logger.error(f"Reason required for action {action}")
                raise HTTPException(status_code=400, detail="Reason required for moderation actions")
            
            logger.info(f"Performing {action} action on player {player_id}")
            
            # Perform the action
            if action == "warn":
                result = await roblox_service.warn_user(player_id, reason)
            elif action == "kick":
                result = await roblox_service.kick_user(player_id, reason)
            elif action == "ban":
                result = await roblox_service.ban_user(player_id, reason)
            
            # Update message with action
            update_data.update({
                "moderation_action": action,
                "moderation_reason": reason
            })
            
            # Store moderation action record
            moderation_record = {
                "player_id": player_id,
                "action": action,
                "reason": reason,
                "performed_by": "manual_review",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "success": result.get("success", False),
                "error": result.get("error")
            }
            
            supabase.table('moderation_actions').insert(moderation_record).execute()
            logger.info(f"Stored moderation action record for {action}")
        else:
            logger.error(f"Invalid action: {action}")
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'approve', 'warn', 'kick', or 'ban'")
        
        # Update the message
        logger.info(f"Updating message {message_id} with data: {update_data}")
        update_result = supabase.table('messages').update(update_data).eq('message_id', message_id).execute()
        logger.info(f"Update result: {update_result}")
        
        return {
            "success": True,
            "message_id": message_id,
            "action": action,
            "reason": reason
        }
        
    except Exception as e:
        logger.error(f"Error reviewing flagged message: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to review message: {str(e)}")

@app.get("/api/test-roblox")
async def test_roblox_api(_: None = Depends(verify_roblox_platform_key)):
    """Test the Roblox API connection to help debug issues."""
    try:
        result = await roblox_service.test_api_connection()
        return {
            "success": result["success"],
            "message": result.get("message", "Test completed"),
            "error": result.get("error"),
            "universe_id": roblox_service.universe_id,
            "api_key_configured": bool(roblox_service.api_key)
        }
    except Exception as e:
        logger.error(f"Error testing Roblox API: {e}")
        return {
            "success": False,
            "error": str(e),
            "universe_id": roblox_service.universe_id,
            "api_key_configured": bool(roblox_service.api_key)
        }

@app.get("/api/moderate/pending")
async def get_pending_moderation_actions(_: None = Depends(verify_api_key)):
    """Get pending moderation actions for the Roblox script to process."""
    try:
        # Get moderation actions that haven't been applied yet
        result = supabase.table('moderation_actions').select('*').eq('applied', False).eq('performed_by', 'ai').execute()
        
        actions = []
        for action in result.data:
            actions.append({
                "id": action["id"],
                "player_id": action["player_id"],
                "action": action["action"],
                "reason": action["reason"],
                "created_at": action["created_at"]
            })
        
        return {"actions": actions}
        
    except Exception as e:
        logger.error(f"Error getting pending moderation actions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pending actions: {str(e)}")

@app.post("/api/moderate/complete")
async def complete_moderation_action(
    action_id: int = Query(..., description="ID of the moderation action to mark as completed"),
    _: None = Depends(verify_api_key)
):
    """Mark a moderation action as completed by the Roblox script."""
    try:
        # Update the moderation action to mark it as applied
        result = supabase.table('moderation_actions').update({"applied": True}).eq('id', action_id).execute()
        
        if result.data:
            return {"success": True, "message": f"Action {action_id} marked as completed"}
        else:
            raise HTTPException(status_code=404, detail=f"Action {action_id} not found")
            
    except Exception as e:
        logger.error(f"Error completing moderation action {action_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete action: {str(e)}")

@app.post("/api/analyze-with-moderation", response_model=Dict[str, Any])
async def analyze_with_immediate_moderation(
    request_data: AnalyzeRequest,
    _: None = Depends(verify_api_key)
):
    """
    Returns sentiment analysis AND moderation action immediately for Roblox script to handle
    """
    user_message = request_data.message
    message_id = request_data.message_id
    player_id = request_data.player_id
    player_name = request_data.player_name
    
    # Only use random values if no player_id or player_name was provided
    if player_id is None:
        player_id = random.randint(1, 100)
    if player_name is None:
        player_name = f"Player{random.randint(1, 999)}"
    
    logger.info(f"Processing analysis with moderation: Player ID: {player_id}, Player Name: {player_name}")
    logger.info(f"Message to analyze: {user_message}")
    
    try:
        # Create chat message for analysis
        chat_message = ChatMessage(
            message_id=message_id,
            content=user_message,
            user_id=player_id,
            timestamp=datetime.now(timezone.utc),
            deleted=False
        )
        
        # Run sentiment analysis
        sentiment_result = await ai_service.analyze_message_with_moderation(
            user_message, message_id, player_id, player_name
        )
        
        # Clean the response to remove any datetime objects
        cleaned_result = clean_ai_response(sentiment_result)
        logger.info(f"Cleaned AI result: {cleaned_result}")
        
        # Store player data
        current_time = datetime.now(timezone.utc)
        player_data = {
            "player_id": player_id,
            "player_name": player_name,
            "last_seen": current_time.isoformat()
        }
        
        # Upsert player data
        supabase.table('players').upsert(player_data).execute()
        logger.info("Player data stored/updated in Supabase")
        
        # Store message data
        message_data = {
            "message_id": message_id,
            "player_id": player_id,
            "message": user_message,
            "sentiment_score": cleaned_result.get("sentiment_score", 0),
            "created_at": current_time.isoformat()
        }
        
        supabase.table('messages').insert(message_data).execute()
        logger.info("Message data stored in Supabase")
        
        # Update player's total sentiment score
        try:
            # Get current total sentiment score
            player_result = supabase.table('players').select('total_sentiment_score').eq('player_id', player_id).execute()
            current_total = player_result.data[0].get('total_sentiment_score', 0) if player_result.data else 0
            
            # Add new sentiment score
            new_total = current_total + cleaned_result.get("sentiment_score", 0)
            
            # Update the total
            supabase.table('players').update({"total_sentiment_score": new_total}).eq('player_id', player_id).execute()
            logger.info(f"Updated total sentiment score for player {player_id}: {new_total}")
        except Exception as e:
            logger.error(f"Error updating total sentiment score: {e}")
        
        # Check for moderation action
        moderation_action = cleaned_result.get("moderation_action")
        moderation_reason = cleaned_result.get("moderation_reason")
        
        # Determine if this should be an immediate action or flagged for review
        immediate_action = None
        should_flag = False
        
        if moderation_action:
            action_lower = moderation_action.lower()
            
            # AI AUTO-ACTIONS: Warnings and kicks can be applied immediately
            if action_lower in ["warning", "kick"]:
                immediate_action = moderation_action
                should_flag = False
                logger.info(f"AI recommends immediate {action_lower} action for player {player_id}")
                
            # HUMAN REVIEW: Bans are flagged for human review
            elif action_lower == "ban":
                immediate_action = None  # Don't apply immediately
                should_flag = True
                logger.info(f"AI recommends BAN for player {player_id} - flagged for human review")
        
        # Log moderation action to database
        if moderation_action:
            moderation_record = {
                "player_id": player_id,
                "message_id": message_id,
                "action": moderation_action.lower(),
                "reason": moderation_reason,
                "performed_by": "ai",
                "success": immediate_action is not None,  # True if immediate, False if pending review
                "error": "Pending human review" if should_flag else None
            }
            supabase.table('moderation_actions').insert(moderation_record).execute()
            logger.info(f"Moderation action '{moderation_action}' logged to database.")
            
            # Update message with moderation info
            update_data = {
                "moderation_action": moderation_action,
                "moderation_reason": moderation_reason,
                "flag": should_flag
            }
            supabase.table('messages').update(update_data).eq('message_id', message_id).execute()
        
        # Return comprehensive result
        result = {
            "player_id": player_id,
            "player_name": player_name,
            "message_id": message_id,
            "message": user_message,
            "sentiment_score": cleaned_result.get("sentiment_score", 0),
            "moderation_action": immediate_action,  # Only return immediate actions
            "moderation_reason": moderation_reason if immediate_action else None,
            "flagged_for_review": should_flag,  # Indicate if ban was flagged
            "error": cleaned_result.get("error")
        }
        
        logger.info(f"Returning analysis with moderation result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error in analysis with moderation: {e}")
        return {
            "player_id": player_id,
            "player_name": player_name,
            "message_id": message_id,
            "message": user_message,
            "sentiment_score": 0,
            "moderation_action": None,
            "moderation_reason": None,
            "error": str(e)
        }

@app.get("/api/debug-data")
async def debug_database_data():
    """Temporary endpoint to debug database data"""
    try:
        # Check players table
        players_response = supabase.table('players').select('*').limit(5).execute()
        logger.info(f"Players data: {players_response.data}")
        
        # Check messages table
        messages_response = supabase.table('messages').select('*').limit(5).execute()
        logger.info(f"Messages data: {messages_response.data}")
        
        return {
            "players_count": len(players_response.data),
            "messages_count": len(messages_response.data),
            "sample_players": players_response.data,
            "sample_messages": messages_response.data
        }
    except Exception as e:
        logger.error(f"Debug data error: {e}")
        return {"error": str(e)}

