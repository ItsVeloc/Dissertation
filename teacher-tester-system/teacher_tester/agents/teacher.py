import re
from typing import Optional, Dict, Any, Tuple, List
from teacher_tester.data.schemas import Conversation
from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger
from teacher_tester.utils.model_interface import get_model_interface
from teacher_tester.utils.prompt_templates import create_teacher_system_prompt, format_conversation_history

logger = setup_logger(__name__)

class TeacherAgent:
    """
    Teacher agent that engages with the tester to determine their rating.
    """
    
    def __init__(self, subject: str):
        """
        Initialize the teacher agent.
        
        Args:
            subject: The subject the teacher is knowledgeable about
        """
        self.subject = subject
        self.config = get_config()
        self.model_name = self.config.get("models.teacher.base_model")
        self.temperature = self.config.get("models.teacher.temperature")
        self.max_tokens = self.config.get("models.teacher.max_tokens")
        self.model_interface = get_model_interface()
        
        logger.info(f"Initialized teacher agent for subject: {subject}")
    
    def generate_response(self, conversation: Conversation) -> Tuple[str, float]:
        """
        Generate a response to the tester.
        
        Args:
            conversation: The current conversation
            
        Returns:
            A tuple of (response_content, confidence)
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
            content = "I'm sorry, I'm having trouble formulating a response."
            logger.error(f"Failed to get proper response from model: {response}")
        
        # Extract confidence from the response
        confidence, content = self._extract_confidence(content)
        
        # If confidence couldn't be extracted, assign a default based on conversation length
        if confidence is None:
            msg_count = len([m for m in conversation.messages if m.role == "teacher"])
            confidence = min(0.2 + msg_count * 0.15, 0.95)
            # Prepend confidence to content
            content = f"Confidence: {confidence:.2f}\n\n{content}"
        
        logger.info(f"Generated teacher response with confidence: {confidence:.2f}")
        return content, confidence
    
    def estimate_rating(self, conversation: Conversation) -> Tuple[float, float]:
        """
        Estimate the tester's rating based on the conversation.
        
        Args:
            conversation: The current conversation
            
        Returns:
            A tuple of (estimated_rating, confidence)
        """
        # Create a special prompt for final rating estimation
        system_prompt = f"""You are an expert teacher in {self.subject}. You've had a conversation with a student and now need to assess their knowledge level on a scale from 0 to 10:

- 0-2: Beginner with minimal knowledge
- 3-4: Basic understanding of fundamental concepts
- 5-6: Intermediate with practical application skills
- 7-8: Advanced understanding with broad knowledge
- 9-10: Expert with deep specialized knowledge

Review the conversation and provide:
1. A precise numerical rating between 0 and 10 (can use decimals)
2. Your confidence in this assessment as a decimal between 0 and 1
3. A brief justification for your rating

Format your response exactly like this:
Rating: 7.5
Confidence: 0.85
Justification: The student demonstrated strong understanding of...
"""
        
        # Prepare conversation history
        history = []
        for msg in conversation.messages:
            role = "assistant" if msg.role == "teacher" else "user"
            history.append({"role": role, "content": msg.content})
        
        # Prepare messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
        ] + history
        
        # Call the model
        response = self.model_interface.call_model(
            model=self.model_name,
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent rating
            max_tokens=self.max_tokens
        )
        
        # Extract the response content
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
        else:
            content = "Rating: 5.0\nConfidence: 0.5\nJustification: Unable to assess properly."
            logger.error(f"Failed to get proper response from model: {response}")
        
        # Extract rating and confidence
        rating_match = re.search(r"Rating:\s*(\d+\.?\d*)", content)
        confidence_match = re.search(r"Confidence:\s*(\d+\.?\d*)", content)
        
        if rating_match and confidence_match:
            rating = float(rating_match.group(1))
            confidence = float(confidence_match.group(1))
            
            # Clamp values to valid ranges
            rating = max(0.0, min(10.0, rating))
            confidence = max(0.0, min(1.0, confidence))
        else:
            # Fallback values
            rating = 5.0
            confidence = 0.5
            logger.warning(f"Failed to extract rating or confidence from: {content}")
        
        logger.info(f"Estimated tester rating: {rating:.2f} with confidence: {confidence:.2f}")
        return rating, confidence
    
    def _prepare_messages(self, conversation: Conversation) -> List[Dict[str, str]]:
        """Prepare messages for the model call."""
        # If this is a new conversation or tester just responded, use the teacher system prompt
        if not conversation.messages or conversation.messages[-1].role == "tester":
            return [
                {"role": "system", "content": create_teacher_system_prompt(self.subject)},
                *[{"role": "user" if msg.role == "tester" else "assistant", "content": msg.content} 
                  for msg in conversation.messages]
            ]
        else:
            # If teacher needs to continue, use a continuation prompt
            return [
                {"role": "system", "content": "Continue your previous response to the student."},
                *[{"role": "user" if msg.role == "tester" else "assistant", "content": msg.content} 
                  for msg in conversation.messages]
            ]
    
    def _extract_confidence(self, content: str) -> Tuple[Optional[float], str]:
        """
        Extract confidence from teacher response.
        
        Args:
            content: The response content
            
        Returns:
            Tuple of (confidence, cleaned_content)
        """
        # Look for confidence pattern at the beginning
        confidence_pattern = r"^Confidence:\s*(\d+\.\d+)"
        match = re.search(confidence_pattern, content)
        
        if match:
            confidence = float(match.group(1))
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            return confidence, content
        else:
            return None, content