import logging
from typing import Optional
from .state import ModerationState
from .nodes import StartModeration

logger = logging.getLogger(__name__)

async def moderate_message(message) -> ModerationState:
    """
    Main moderation function that analyzes a message and returns moderation state
    """
    try:
        # Create initial state
        state = ModerationState(message=message)
        
        # Run moderation analysis
        start_node = StartModeration()
        result_state = await start_node.run(state)
        
        logger.info(f"Moderation completed for message {message.message_id}")
        logger.info(f"Action: {result_state.recommended_action.action.value if result_state.recommended_action else 'None'}")
        logger.info(f"Flagged: {result_state.flag}")
        
        return result_state
        
    except Exception as e:
        logger.error(f"Error in moderate_message: {e}")
        # Return default state with error handling
        state = ModerationState(message=message)
        state.recommended_action = None
        state.flag = True  # Flag for manual review on error
        return state

# Keep the old graph for backward compatibility
moderation_graph = None
