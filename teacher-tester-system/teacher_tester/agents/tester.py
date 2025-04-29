from typing import Optional, Dict, Any, List
import re
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
        
        # Call the model with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
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
                    
                    # Also clean any <think> blocks from the tester's response
                    content = self._clean_think_blocks(content)
                    
                    # Ensure the response stays in character
                    if self._is_meta_response(content):
                        # Regenerate if the response breaks character
                        logger.warning(f"Meta response detected, regenerating (attempt {attempt+1})")
                        continue
                    
                    break
                else:
                    logger.warning(f"Empty response from model (attempt {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:
                        # Generate a knowledge-appropriate fallback
                        content = self._generate_fallback_response()
            except Exception as e:
                logger.error(f"Error calling model (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    # Generate a knowledge-appropriate fallback
                    content = self._generate_fallback_response()
        
        logger.info(f"Generated tester response (rating: {self.rating})")
        return content
    
    def _prepare_messages(self, conversation: Conversation) -> List[Dict[str, str]]:
        """Prepare messages for the model call."""
        # If this is a new conversation or teacher just responded, use the tester system prompt
        if not conversation.messages or conversation.messages[-1].role == "teacher":
            return [
                {"role": "system", "content": create_tester_system_prompt(self.rating)},
                *[{"role": "assistant" if msg.role == "tester" else "user", "content": 
                   # For teacher messages, remove confidence prefix and think blocks if present
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
        """
        Remove confidence prefix and think blocks from teacher messages.
        
        Args:
            content: The message content from the teacher
            
        Returns:
            Cleaned message content
        """
        # First remove any <think> blocks
        content = self._clean_think_blocks(content)
        
        # Then remove confidence prefix if present
        if "Confidence:" in content:
            # Split by first occurrence of double newline after confidence
            parts = content.split("\n\n", 1)
            if len(parts) > 1:
                return parts[1].strip()
            
        return content.strip()
    
    def _clean_think_blocks(self, content: str) -> str:
        """
        Remove <think> blocks from content.
        
        Args:
            content: The content to clean
            
        Returns:
            Content with <think> blocks removed
        """
        # Remove any <think> blocks using regex
        think_pattern = r"<think>.*?</think>"
        content = re.sub(think_pattern, "", content, flags=re.DOTALL)
        
        # Clean up any multiple newlines left behind
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        return content.strip()
    
    def _is_meta_response(self, content: str) -> bool:
        """
        Check if a response breaks character by being too meta.
        
        Args:
            content: The response to check
            
        Returns:
            True if the response breaks character
        """
        meta_patterns = [
            r"my (knowledge|skill|rating) (level|score) is",
            r"as a (student|tester) (with|at)",
            r"I('m| am) (simulating|acting|pretending|roleplaying)",
            r"my rating of \d+/10",
            r"I'm (rated|scored|assigned)",
            r"(given|with) my rating",
            r"my numerical rating",
            r"I've been (told|assigned|given) to",
            r"Confidence: \d+\.\d+"  # Don't include confidence in tester responses
        ]
        
        for pattern in meta_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _generate_fallback_response(self) -> str:
        """
        Generate a fallback response appropriate to the tester's knowledge level.
        
        Returns:
            A fallback response string
        """
        if self.rating <= 2:
            return "I'm not really sure about that. Could you explain it more simply?"
        elif self.rating <= 4:
            return "I think I understand some of this, but I'm confused about the details. Could you clarify?"
        elif self.rating <= 6:
            return "I have a basic understanding of this concept, but I might be missing some of the advanced aspects."
        elif self.rating <= 8:
            return "I'm familiar with this topic, though there might be some edge cases I haven't encountered."
        else:
            return "I understand this concept well, including most of its nuances and applications."