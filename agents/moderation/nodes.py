from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, End
from typing import Union
import requests
import asyncio
import os
import logging
from dotenv import load_dotenv
import re

from .state import (
    ModerationState,
    PIIResult,
    ContentResult,
    ModAction,
    PIIType,
    ContentType,
    ActionType,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
logger = logging.getLogger(__name__)

# ---------- Configuration Constants ----------
# API Endpoints
PII_DETECTION_API_URL = "https://router.huggingface.co/hf-inference/models/iiiorg/piiranha-v1-detect-personal-information"
CONTENT_MODERATION_API_URL = (
    "https://router.huggingface.co/hf-inference/models/KoalaAI/Text-Moderation"
)

# AI Models
GEMINI_MODEL = "google-gla:gemini-2.0-flash"

# API Configuration
API_TIMEOUT = 30

# Default Responses
DEFAULT_PII_RESPONSE = []
DEFAULT_CONTENT_RESPONSE = [{"label": "OK", "score": 1.0}]

# Agent Prompts
PII_INTENT_PROMPT = "Analyze if the message contains intent to share personal information. Return only true or false."
MODERATION_ACTION_PROMPT = (
    "Determine the appropriate moderation action for harmful content. Available actions: WARNING (for mild violations), KICK (for serious violations), BAN (for severe violations requiring human review). Return the action and reason."
)


# ---------- API Functions ----------
async def detect_pii(text: str):
    def sync_query():
        try:
            response = requests.post(
                PII_DETECTION_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": text},
                timeout=API_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"PII detection API error: {e}")
            return DEFAULT_PII_RESPONSE
        except ValueError as e:
            print(f"PII detection JSON parsing error: {e}")
            return DEFAULT_PII_RESPONSE

    return await asyncio.get_event_loop().run_in_executor(None, sync_query)


async def moderate_content(text: str):
    def sync_query():
        try:
            response = requests.post(
                CONTENT_MODERATION_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": text},
                timeout=API_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0]
                return result
            elif isinstance(result, dict):
                return [result]
            else:
                return DEFAULT_CONTENT_RESPONSE

        except requests.exceptions.RequestException as e:
            print(f"Content moderation API error: {e}")
            return DEFAULT_CONTENT_RESPONSE
        except ValueError as e:
            print(f"Content moderation JSON parsing error: {e}")
            return DEFAULT_CONTENT_RESPONSE

    return await asyncio.get_event_loop().run_in_executor(None, sync_query)


# ---------- AI Agents ----------
PIIAgent = Agent(GEMINI_MODEL, system_prompt=PII_INTENT_PROMPT, output_type=bool)

ModAgent = Agent(
    GEMINI_MODEL, system_prompt=MODERATION_ACTION_PROMPT, output_type=ModAction
)


# ---------- Forward Declarations ----------
class DetectPII(BaseNode[ModerationState]):
    pass


class CheckIntent(BaseNode[ModerationState]):
    pass


class ModerateContent(BaseNode[ModerationState]):
    pass


class DetermineAction(BaseNode[ModerationState]):
    pass


# ---------- Node Implementations ----------
class StartModeration:
    """Initial moderation node that analyzes content and PII"""
    
    async def run(self, state: ModerationState) -> ModerationState:
        """Run initial moderation checks"""
        try:
            # Check for PII
            pii_result = await self._check_pii(state.message.content)
            state.pii_result = pii_result
            
            # Check PII intent if PII was detected
            if pii_result.pii_presence:
                pii_intent = await self._check_pii_intent(state.message.content)
                state.pii_result.pii_intent = pii_intent
            
            # Check content categories
            content_result = await self._check_content(state.message.content)
            state.content_result = content_result
            
            # Determine recommended action
            action = await self._determine_action(state)
            state.recommended_action = action
            
            # Set flag for human review if needed
            state.flag = await self._should_flag_for_review(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in StartModeration: {e}")
            # Default to warning on error
            state.recommended_action = ModAction(
                action=ActionType.WARNING,
                reason="Error in moderation analysis - manual review recommended"
            )
            state.flag = True
            return state
    
    async def _check_pii(self, content: str) -> PIIResult:
        """Check for personally identifiable information using AI"""
        try:
            # Use the AI-based PII detection
            pii_data = await detect_pii(content)
            
            if not isinstance(pii_data, list):
                pii_data = []

            pii_presence = any(
                entity.get("entity_group") in PIIType.__members__.values()
                for entity in pii_data
                if isinstance(entity, dict) and "entity_group" in entity
            )

            pii_type = None
            if pii_presence:
                for entity in pii_data:
                    if (
                        isinstance(entity, dict)
                        and entity.get("entity_group") in PIIType.__members__.values()
                    ):
                        pii_type = PIIType(entity["entity_group"])
                        break

            return PIIResult(
                pii_presence=pii_presence,
                pii_type=pii_type,
                pii_intent=False  # Will be checked separately
            )
            
        except Exception as e:
            logger.error(f"Error in AI PII detection: {e}")
            # Fallback to regex-based detection if AI fails
            pii_patterns = {
                PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                PIIType.TELEPHONENUM: r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            }
            
            for pii_type, pattern in pii_patterns.items():
                if re.search(pattern, content, re.IGNORECASE):
                    return PIIResult(
                        pii_presence=True,
                        pii_type=pii_type,
                        pii_intent=True
                    )
            
            return PIIResult(pii_presence=False)
    
    async def _check_pii_intent(self, content: str) -> bool:
        """Check if the user intends to share personal information using AI"""
        try:
            result = await PIIAgent.run(content)
            intent = result.output if hasattr(result, "output") else result
            return intent
        except Exception as e:
            logger.error(f"Error in PII intent detection: {e}")
            return False
    
    async def _check_content(self, content: str) -> ContentResult:
        """Check content for inappropriate categories using AI"""
        try:
            # Use the AI-based content moderation
            content_data = await moderate_content(content)
            
            if not content_data or not isinstance(content_data, list):
                content_data = DEFAULT_CONTENT_RESPONSE

            main_item = max(content_data, key=lambda x: x.get("score", 0))
            main_category_str = main_item.get("label", "OK")

            try:
                main_category = ContentType(main_category_str)
            except ValueError:
                print(f"Unknown category: {main_category_str}, defaulting to OK")
                main_category = ContentType.OK

            categories = {
                item.get("label", "OK"): item.get("score", 0.0) for item in content_data
            }

            return ContentResult(
                main_category=main_category, categories=categories
            )
            
        except Exception as e:
            logger.error(f"Error in AI content moderation: {e}")
            # Fallback to OK if AI fails
            return ContentResult(
                main_category=ContentType.OK,
                categories={"OK": 1.0}
            )
    
    async def _determine_action(self, state: ModerationState) -> ModAction:
        """Determine the recommended moderation action using AI"""
        try:
            # PII detection - immediate warning
            if state.pii_result and state.pii_result.pii_presence:
                return ModAction(
                    action=ActionType.WARNING,
                    reason=f"Personal information detected: {state.pii_result.pii_type.value}"
                )
            
            # Use AI to determine action based on content
            if state.content_result and state.content_result.main_category != ContentType.OK:
                prompt = f"Content type: {state.content_result.main_category.value}, Message: {state.message.content}"
                result = await ModAgent.run(prompt)
                action = result.output if hasattr(result, "output") else result
                return action
            
            # Default - no action needed
            return None
            
        except Exception as e:
            logger.error(f"Error in AI action determination: {e}")
            # Fallback to warning on error
            return ModAction(
                action=ActionType.WARNING,
                reason="Error in moderation analysis - manual review recommended"
            )
    
    async def _should_flag_for_review(self, state: ModerationState) -> bool:
        """Determine if message should be flagged for human review"""
        # Flag for review if:
        # 1. Severe actions are recommended (BAN, KICK)
        # 2. Content is ambiguous or borderline
        # 3. PII is detected
        # 4. Multiple violations detected
        
        if not state.recommended_action:
            return False
        
        # Always flag bans for human review
        if state.recommended_action.action == ActionType.BAN:
            return True
        
        # Flag kicks for review
        if state.recommended_action.action == ActionType.KICK:
            return True
        
        # Flag PII violations
        if state.pii_result and state.pii_result.pii_presence:
            return True
        
        # Flag if multiple content categories are detected
        if state.content_result:
            high_score_categories = sum(
                1 for score in state.content_result.categories.values() 
                if score > 0.3
            )
            if high_score_categories > 1:
                return True
        
        return False

@dataclass
class DetectPII(BaseNode[ModerationState]):
    async def run(self, ctx: GraphRunContext) -> Union[CheckIntent, End]:
        pii_data = await detect_pii(ctx.state.message.content)

        if not isinstance(pii_data, list):
            pii_data = []

        pii_presence = any(
            entity.get("entity_group") in PIIType.__members__.values()
            for entity in pii_data
            if isinstance(entity, dict) and "entity_group" in entity
        )

        pii_type = None
        if pii_presence:
            for entity in pii_data:
                if (
                    isinstance(entity, dict)
                    and entity.get("entity_group") in PIIType.__members__.values()
                ):
                    pii_type = PIIType(entity["entity_group"])
                    break

        ctx.state.pii_result = PIIResult(pii_presence=pii_presence, pii_type=pii_type)

        if pii_presence:
            ctx.state.recommended_action = ModAction(
                action=ActionType.WARNING,
                reason=f"Detected {pii_type.value if pii_type else 'PII'} in message",
            )
            return End("PII detected - message blocked")

        return CheckIntent()


@dataclass
class CheckIntent(BaseNode[ModerationState]):
    async def run(self, ctx: GraphRunContext) -> Union[ModerateContent, End]:
        try:
            result = await PIIAgent.run(ctx.state.message.content)
            intent = result.output if hasattr(result, "output") else result

            if ctx.state.pii_result:
                ctx.state.pii_result.pii_intent = intent
            else:
                ctx.state.pii_result = PIIResult(pii_presence=False, pii_intent=intent)

            if intent:
                ctx.state.recommended_action = ModAction(
                    action=ActionType.WARNING,
                    reason="Potential PII sharing intent detected",
                )
                return End("PII intent detected - message blocked")

        except Exception as e:
            print(f"Intent analysis error: {e}")
            if ctx.state.pii_result:
                ctx.state.pii_result.pii_intent = False
            else:
                ctx.state.pii_result = PIIResult(pii_presence=False, pii_intent=False)

        return ModerateContent()


@dataclass
class ModerateContent(BaseNode[ModerationState]):
    async def run(self, ctx: GraphRunContext) -> Union[DetermineAction, End]:
        content_data = await moderate_content(ctx.state.message.content)

        if not content_data or not isinstance(content_data, list):
            content_data = DEFAULT_CONTENT_RESPONSE

        main_item = max(content_data, key=lambda x: x.get("score", 0))
        main_category_str = main_item.get("label", "OK")

        try:
            main_category = ContentType(main_category_str)
        except ValueError:
            print(f"Unknown category: {main_category_str}, defaulting to OK")
            main_category = ContentType.OK

        categories = {
            item.get("label", "OK"): item.get("score", 0.0) for item in content_data
        }

        ctx.state.content_result = ContentResult(
            main_category=main_category, categories=categories
        )

        if main_category != ContentType.OK:
            return DetermineAction()

        return End("Content approved")


@dataclass
class DetermineAction(BaseNode[ModerationState]):
    async def run(self, ctx: GraphRunContext) -> End:
        try:
            prompt = f"Content type: {ctx.state.content_result.main_category.value}, Message: {ctx.state.message.content}"
            result = await ModAgent.run(prompt)
            action = result.output if hasattr(result, "output") else result
            ctx.state.recommended_action = action
            return End(f"Action determined: {action.action.value}")

        except Exception as e:
            print(f"Action determination error: {e}")
            ctx.state.recommended_action = ModAction(
                action=ActionType.WARNING,
                reason="Automated moderation - manual review required",
            )
            return End("Fallback action applied")
