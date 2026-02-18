import logging
import os
import requests
from typing import Optional, Dict, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RestrictionType(str, Enum):
    WARN = "warn"
    KICK = "kick"
    BAN = "ban"

class RobloxService:
    def __init__(self):
        self.api_key = os.environ.get("ROBLOX_PLATFORM_API_KEY")
        self.universe_id = os.environ.get("ROBLOX_UNIVERSE_ID")
        self.base_url = "https://apis.roblox.com/cloud/v2/universes"
        
        logger.info(f"RobloxService initialized - Universe ID: {self.universe_id}, API Key configured: {bool(self.api_key)}")
        
        if not self.api_key:
            logger.warning("Roblox API key not configured - moderation actions will be logged only")
        if not self.universe_id:
            logger.warning("Roblox Universe ID not configured - API calls will fail")
    
    async def _check_existing_restriction(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Check if a user restriction already exists and return its details."""
        if not self.api_key or not self.universe_id:
            return None
        
        url = f"{self.base_url}/{self.universe_id}/user-restrictions/{user_id}"
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        
        logger.info(f"Checking existing restriction for user {user_id} at {url}")
        logger.info(f"Using headers: {headers}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            logger.info(f"GET response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"GET response data: {json.dumps(data, indent=2)}")
                return data
            elif response.status_code == 404:
                logger.info(f"No existing restriction found for user {user_id}")
                return None
            else:
                logger.error(f"Failed to check user restriction for {user_id}: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Exception while checking user restriction for {user_id}: {e}")
            
        return None

    async def test_api_connection(self) -> Dict[str, Any]:
        """Test the API connection to verify the key and universe ID are valid."""
        if not self.api_key or not self.universe_id:
            return {"success": False, "error": "API key or Universe ID not configured"}
        
        try:
            # Test with different header formats
            headers_variants = [
                {"x-api-key": self.api_key},
                {"X-API-Key": self.api_key},
                {"Authorization": f"Bearer {self.api_key}"},
                {"x-api-key": self.api_key, "Content-Type": "application/json"}
            ]
            
            url = f"{self.base_url}/{self.universe_id}/user-restrictions"
            
            logger.info(f"Testing API connection to: {url}")
            logger.info(f"API Key length: {len(self.api_key) if self.api_key else 0}")
            logger.info(f"API Key starts with: {self.api_key[:10] if self.api_key else 'None'}...")
            
            for i, headers in enumerate(headers_variants):
                logger.info(f"Trying header variant {i+1}: {headers}")
                response = requests.get(url, headers=headers, timeout=10)
                
                logger.info(f"Variant {i+1} response status: {response.status_code}")
                logger.info(f"Variant {i+1} response text: {response.text}")
                
                if response.status_code == 200:
                    return {"success": True, "message": f"API connection successful with variant {i+1}"}
                elif response.status_code == 401:
                    logger.warning(f"Authentication failed with variant {i+1}")
                elif response.status_code == 403:
                    logger.warning(f"Permission denied with variant {i+1}")
            
            # If all variants failed, return the last error
            return {"success": False, "error": f"All authentication variants failed. Last response: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.error(f"Exception testing API connection: {e}")
            return {"success": False, "error": str(e)}

    async def _apply_restriction(self, user_id: int, duration_seconds: Optional[int], display_reason: str, private_reason: str) -> Dict[str, Any]:
        """Core function to apply or update a user restriction using the v2 API."""
        if not self.api_key or not self.universe_id:
            error_message = "Roblox API key or Universe ID not configured"
            return {"success": False, "error": error_message}
            
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        logger.info(f"Applying restriction with headers: {headers}")
        
        # Check if a restriction already exists for this user
        existing_restriction = await self._check_existing_restriction(user_id)
        
        payload = {
            "gameJoinRestriction": {
                "active": True,
                "displayReason": display_reason,
                "privateReason": private_reason,
                "excludeAltAccounts": True
            }
        }
        
        if duration_seconds is not None:
            payload["gameJoinRestriction"]["duration"] = f"{duration_seconds}s"

        logger.info(f"Payload for user {user_id}: {json.dumps(payload, indent=2)}")

        try:
            if existing_restriction:
                # Update existing restriction using PATCH
                url = f"{self.base_url}/{self.universe_id}/user-restrictions/{user_id}"
                logger.info(f"Updating restriction for user {user_id} at {url}")
                response = requests.patch(url, headers=headers, data=json.dumps(payload), timeout=10)
            else:
                # Create new restriction using POST
                url = f"{self.base_url}/{self.universe_id}/user-restrictions"
                logger.info(f"Creating restriction for user {user_id} at {url}")
                
                # Add user information to the payload for creation
                creation_payload = {
                    "user": {
                        "id": user_id
                    },
                    **payload
                }
                
                response = requests.post(url, headers=headers, data=json.dumps(creation_payload), timeout=10)

            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info(f"Response text: {response.text}")

            if response.status_code in [200, 201]:
                logger.info(f"Successfully applied restriction for user {user_id}")
                return {"success": True, "response": response.json()}
            else:
                error_text = response.text
                logger.error(f"Failed to apply restriction for user {user_id}: {response.status_code} - {error_text}")
                return {"success": False, "error": f"API returned {response.status_code}: {error_text}"}

        except Exception as e:
            logger.error(f"Exception applying restriction for user {user_id}: {e}")
            return {"success": False, "error": str(e)}

    async def warn_user(self, user_id: int, reason: str) -> Dict[str, Any]:
        """'Warn' a user by creating a restriction with no duration."""
        return await self._apply_restriction(user_id, duration_seconds=None, display_reason=reason, private_reason=f"Warning issued: {reason}")

    async def kick_user(self, user_id: int, reason: str) -> Dict[str, Any]:
        """'Kick' a user by applying a very short restriction (e.g., 1 second)."""
        return await self._apply_restriction(user_id, duration_seconds=1, display_reason=reason, private_reason=f"User kicked: {reason}")

    async def ban_user(self, user_id: int, reason: str, duration_hours: int = 0) -> Dict[str, Any]:
        """Ban a user. A duration of 0 is a permanent ban."""
        duration_seconds = duration_hours * 3600 if duration_hours > 0 else None
        return await self._apply_restriction(user_id, duration_seconds=duration_seconds, display_reason=reason, private_reason=f"User banned: {reason}")

    async def get_user_restrictions(self, user_id: int) -> Dict[str, Any]:
        """Get current restrictions for a user"""
        if not self.api_key or not self.universe_id:
            return {"success": False, "error": "Roblox API key or Universe ID not configured"}
        
        try:
            headers = {"x-api-key": self.api_key}
            url = f"{self.base_url}/{self.universe_id}/user-restrictions/{user_id}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {"success": True, "restrictions": response.json()}
            elif response.status_code == 404:
                return {"success": True, "restrictions": None}
            else:
                return {"success": False, "error": f"API returned {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting restrictions for user {user_id}: {e}")
            return {"success": False, "error": str(e)}

# Global instance
roblox_service = RobloxService() 