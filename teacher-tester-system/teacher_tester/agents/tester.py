from typing import Optional, Dict, Any, List
from teacher_tester.data.schemas import Conversation
from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger
from teacher_tester.utils.model_interface import get_model_interface
from teacher_tester.utils.prompt_templates import create_tester_system_prompt, format_conversation_history

logger = setup_logger(__name__)

class TesterAgent:
    """
    Tester agent with a predefined rating that engages with the teacher.
    """
    
    def __init__(self, rating: float):
        """
        Initialize the tester agent.
        
        Args:
            rating: The tester's rating (0-10 scale)
        """
        self.rating = rating
        config = get_config()
        
        # Validate rating is within allowed range
        rating_min = config.get("conversation.rating_range.min", 0.0)
        rating_max = config.get("conversation.rating_range.max", 10.0)
        
        if not rating_min <= rating <= rating_max:
            logger.warning(f"Rating {rating} outside allowed range [{rating_min}, {rating_max}]")
            self.rating = max(rating_min, min(rating, rating_max))
        
        self.model_name = config.get("models.tester.base_model")
        self.temperature = config.get("models.tester.temperature")
        self.max_tokens = config.get("models.tester.max_tokens")
        self.model_interface = get_model_interface()
        
        logger.info(f"Initialized tester agent with rating: {self.rating}")
    
    def generate_response(self, conversation: Conversation) -> str:
        """
        Generate a response to the teacher.
        
        Args:
            conversation: The current conversation
            
        Returns:
            The tester's response content
        """
        # Format messages for the model
        messages = self._prepare_messages(conversation)
        
        # Call the model
        response = self.model_interface.call_model(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract the response content
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
        else:
            content = "I'm sorry, I don't understand the question."
            logger.error(f"Failed to get proper response from model: {response}")
        
        logger.info(f"Generated tester response (rating: {self.rating})")
        return content
    
    def _prepare_messages(self, conversation: Conversation) -> List[Dict[str, str]]:
        """Prepare messages for the model call."""
        # If this is a new conversation or teacher just responded, use the tester system prompt
        if not conversation.messages or conversation.messages[-1].role == "teacher":
            return [
                {"role": "system", "content": create_tester_system_prompt(self.rating)},
                *[{"role": "assistant" if msg.role == "tester" else "user", "content": 
                   # For teacher messages, remove confidence prefix if present
                   self._clean_teacher_message(msg.content) if msg.role == "teacher" else msg.content} 
                  for msg in conversation.messages]
            ]
        else:
            # If tester needs to continue, use a continuation prompt
            return [
                {"role": "system", "content": f"Continue your previous response to the teacher. Remember you have a knowledge level of {self.rating}/10 in this subject."},
                *[{"role": "assistant" if msg.role == "tester" else "user", "content": 
                   self._clean_teacher_message(msg.content) if msg.role == "teacher" else msg.content} 
                  for msg in conversation.messages]
            ]
    
    def _clean_teacher_message(self, content: str) -> str:
        """Remove confidence prefix from teacher messages."""
        if "Confidence:" in content:
            # Split by first occurrence of double newline after confidence
            parts = content.split("\n\n", 1)
            if len(parts) > 1:
                return parts[1]
        return content