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
        
        # Call the model with retry logic
        max_retries = 3
        content = None
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
                    
                    # Clean any <think> blocks from the response
                    content = self._clean_think_blocks(content)
                    break
                else:
                    logger.warning(f"Empty response from model (attempt {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:
                        content = "I'm sorry, I'm having trouble formulating a response right now."
            except Exception as e:
                logger.error(f"Error calling model (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    content = "I'm sorry, I'm experiencing technical difficulties."
        
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

Format your response EXACTLY like this example:
Rating: 7.5
Confidence: 0.85
Justification: The student demonstrated strong understanding of...

DO NOT include any XML tags like <think> in your response."""
        
        # Prepare conversation history - clean all messages
        history = []
        for msg in conversation.messages:
            role = "assistant" if msg.role == "teacher" else "user"
            # Clean the content regardless of role
            content = self._clean_think_blocks(msg.content)
            # For teacher messages, also remove confidence prefix
            if msg.role == "teacher":
                content = self._remove_confidence_prefix(content)
            history.append({"role": role, "content": content})
        
        # Prepare messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
        ] + history
        
        # Call the model with retry logic
        max_retries = 3
        content = None
        for attempt in range(max_retries):
            try:
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
                    # Clean any think tags
                    content = self._clean_think_blocks(content)
                    # Log the raw rating response for debugging
                    logger.debug(f"Raw rating response: {content}")
                    break
                else:
                    logger.warning(f"Empty response from model (attempt {attempt+1}/{max_retries})")
                    if attempt == max_retries - 1:
                        content = "Rating: 5.0\nConfidence: 0.5\nJustification: Unable to assess properly."
            except Exception as e:
                logger.error(f"Error calling model (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    content = "Rating: 5.0\nConfidence: 0.5\nJustification: Unable to assess due to technical issues."
        
        # Extract rating and confidence with enhanced regex patterns
        rating = 5.0  # Default
        confidence = 0.5  # Default
        
        # Try multiple patterns to extract rating
        rating_patterns = [
            r"Rating:?\s*(\d+\.?\d*)",  # Standard format
            r"(\d+\.?\d*)\s*/\s*10",  # X/10 format
            r"score:?\s*(\d+\.?\d*)",  # Score: X format
            r"(\d+\.?\d*)\s*out of\s*10"  # X out of 10 format
        ]
        
        for pattern in rating_patterns:
            rating_match = re.search(pattern, content, re.IGNORECASE)
            if rating_match:
                try:
                    rating = float(rating_match.group(1))
                    break  # Found a valid rating, stop trying patterns
                except (ValueError, IndexError):
                    continue  # Try next pattern
        
        # Try multiple patterns to extract confidence
        confidence_patterns = [
            r"Confidence:?\s*(\d+\.?\d*)",  # Standard format
            r"confidence level:?\s*(\d+\.?\d*)",  # Alternate format
            r"confidence:?\s*(\d+\.?\d*)",  # Simple format
            r"confident:?\s*(\d+\.?\d*)"  # Abbreviated format
        ]
        
        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, content, re.IGNORECASE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    break  # Found a valid confidence, stop trying patterns
                except (ValueError, IndexError):
                    continue  # Try next pattern
        
        # Clamp values to valid ranges
        rating = max(0.0, min(10.0, rating))
        confidence = max(0.0, min(1.0, confidence))
        
        logger.info(f"Estimated tester rating: {rating:.2f} with confidence: {confidence:.2f}")
        return rating, confidence
    
    def _prepare_messages(self, conversation: Conversation) -> List[Dict[str, str]]:
        """Prepare messages for the model call."""
        # If this is a new conversation or tester just responded, use the teacher system prompt
        if not conversation.messages or conversation.messages[-1].role == "tester":
            return [
                {"role": "system", "content": create_teacher_system_prompt(self.subject)},
                *[{"role": "user" if msg.role == "tester" else "assistant", "content": 
                  # For tester messages, clean any think blocks that might have leaked
                  self._clean_think_blocks(msg.content)} 
                  for msg in conversation.messages]
            ]
        else:
            # If teacher needs to continue, use a continuation prompt
            return [
                {"role": "system", "content": "Continue your previous response to the student."},
                *[{"role": "user" if msg.role == "tester" else "assistant", "content": 
                  # For tester messages, clean any think blocks that might have leaked
                  self._clean_think_blocks(msg.content)} 
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
        if content is None:
            return None, "I'm having trouble formulating a response right now."
            
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
    
    def _clean_think_blocks(self, content: str) -> str:
        """
        Remove <think> blocks from content.
        
        Args:
            content: The content to clean
            
        Returns:
            Content with <think> blocks removed
        """
        if content is None:
            return ""
            
        # Remove any <think> blocks using regex
        think_pattern = r"<think>.*?</think>"
        content = re.sub(think_pattern, "", content, flags=re.DOTALL)
        
        # Clean up any multiple newlines left behind
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        return content.strip()
    
    def _remove_confidence_prefix(self, content: str) -> str:
        """
        Remove confidence prefix from content.
        
        Args:
            content: The content to clean
            
        Returns:
            Content with confidence prefix removed
        """
        if content is None:
            return ""
            
        if "Confidence:" in content:
            # Split by first occurrence of double newline after confidence
            parts = content.split("\n\n", 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return content