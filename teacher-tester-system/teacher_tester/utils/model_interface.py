import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger
import re  # Import re for regex operations

logger = setup_logger(__name__)

class ModelInterface:
    """Interface for interacting with language models."""
    
    def __init__(self):
        """Initialize the model interface with configuration."""
        self.config = get_config()
        # Simulate API key - in production, get this from environment or secure storage
        self.api_key = os.environ.get("LLM_API_KEY", "simulated_api_key_for_dev")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with error handling."""
        model_path = self.config.get("models.path", "C:/Users/charl/Documents/AI/Diss/Dissertation/DeepSeek-R1-Distill-Qwen-1.5B")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure the path exists
        import os
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return False
            
        try:
            logger.info(f"Loading model from {model_path} on {device}")
            # Add trust_remote_code=True and use local_files_only=True for local models
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            ).to(device)
            
            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                torch_dtype=torch.bfloat16
            )
            
            logger.info(f"Successfully loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.pipe = None
            return False
    
    def _manually_format_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Manually format chat messages for DeepSeek model.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Formatted prompt string
        """
        # DeepSeek seems to work better with a simpler format
        formatted_prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Clean any <think> tags from content that might be carried over
            content = self._remove_think_tags(content)
            
            if role == "system":
                formatted_prompt += content + "\n\n"
            elif role == "user":
                formatted_prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        # Add the final prompt for the assistant
        formatted_prompt += "Assistant:"
        
        return formatted_prompt
    
    def _remove_think_tags(self, content: str) -> str:
        """Remove <think> and </think> tags from content."""
        # Remove think tags and their content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        # Clean up any leftover newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
    
    def _clean_model_output(self, output: str) -> str:
        """Clean model output by removing thinking tags and properly formatting."""
        # Remove any think tags
        output = self._remove_think_tags(output)
        
        # Remove any quote marks that might wrap the output
        output = output.strip('"\'')
        
        # Clean up excessive newlines
        output = re.sub(r'\n{3,}', '\n\n', output)
        
        return output.strip()
    
    def call_model(self, 
                  model: str, 
                  messages: List[Dict[str, str]], 
                  temperature: float = 0.7, 
                  max_tokens: int = 1024,
                  top_p: float = 1.0) -> Dict[str, Any]:
        """
        Call a language model with the given parameters.
        
        Args:
            model: The model identifier to use
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            The model response as a dictionary
        """
        # Check if we have a loaded model
        if self.pipe is not None and self.tokenizer is not None:
            try:
                # Convert message roles if needed
                formatted_messages = []
                for msg in messages:
                    role = msg["role"]
                    # Convert roles to system, user, assistant if they're not already
                    if role not in ["system", "user", "assistant"]:
                        if role == "teacher":
                            role = "assistant"
                        elif role == "tester":
                            role = "user"
                    
                    formatted_messages.append({
                        "role": role,
                        "content": msg["content"]
                    })
                
                # Use manual formatting for DeepSeek
                prompt = self._manually_format_chat(formatted_messages)
                
                # Generate response
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
                
                # Extract and format response
                full_response = outputs[0]['generated_text']
                if len(full_response) <= len(prompt):
                    logger.warning("Empty response from model")
                    model_response = ""
                else:
                    model_response = full_response[len(prompt):].strip()
                
                # Clean the model output
                model_response = self._clean_model_output(model_response)
                
                # Format as API-like response
                return {
                    "id": f"local-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant", 
                                "content": model_response
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt) // 4,  # Approximate
                        "completion_tokens": len(model_response) // 4,  # Approximate
                        "total_tokens": (len(prompt) + len(model_response)) // 4  # Approximate
                    }
                }
            except Exception as e:
                logger.error(f"Error calling local model: {str(e)}")
                logger.error(f"Prompt: {prompt if 'prompt' in locals() else 'prompt not generated'}")
                # Fall back to simulation
                return self._simulate_response(model, messages, temperature, max_tokens)
        
        # If we don't have a loaded model, check if we should simulate
        if self._should_simulate():
            logger.info("Using simulated response because model is not available")
            return self._simulate_response(model, messages, temperature, max_tokens)
        
        # Implementation for real API calls would go here
        # Example for OpenAI-like API:
        try:
            # This is a placeholder for actual API call
            # In production, replace with actual API implementation
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            # Simulate API call delay
            time.sleep(0.5)
            
            # This would be an actual HTTP request in production
            # response = requests.post("https://api.example.com/v1/chat/completions", 
            #                         headers=headers, 
            #                         json=payload)
            # return response.json()
            
            # For now, return simulated response
            return self._simulate_response(model, messages, temperature, max_tokens)
            
        except Exception as e:
            logger.error(f"Error calling model API: {str(e)}")
            # Return minimal error response
            return {
                "error": str(e),
                "choices": [{
                    "message": {
                        "content": "Error generating response."
                    }
                }]
            }
    
    def _should_simulate(self) -> bool:
        """Determine if we should simulate API responses."""
        # In development, use simulated responses if model is not available
        # In production, check for API keys and use real calls
        if self.model is None or self.config.get("development.simulate_api", True):
            return True
        return False
    
    def _simulate_response(self, 
                          model: str, 
                          messages: List[Dict[str, str]], 
                          temperature: float,
                          max_tokens: int) -> Dict[str, Any]:
        """
        Simulate a response from the language model.
        
        Args:
            model: The model identifier
            messages: The conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            A simulated response dictionary
        """
        logger.info(f"Simulating response for model: {model}")
        
        # Extract the last user message
        last_message = messages[-1]["content"] if messages else ""
        
        # Determine response based on role and content patterns
        if "teacher" in last_message.lower():
            # Simulate tester response
            simulated_content = self._simulate_tester_response(last_message)
        else:
            # Simulate teacher response
            simulated_content = self._simulate_teacher_response(last_message)
        
        # Simulate API response structure
        return {
            "id": f"simulated-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": simulated_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(last_message) // 4,
                "completion_tokens": len(simulated_content) // 4,
                "total_tokens": (len(last_message) + len(simulated_content)) // 4
            }
        }
    
    def _simulate_teacher_response(self, last_message: str) -> str:
        """Simulate a teacher agent response."""
        import random
        
        # Extract confidence
        confidence = min(0.3 + random.random() * 0.5, 0.95)
        confidence_str = f"{confidence:.2f}"
        
        # Generate simple question based on content
        questions = [
            "Can you explain the main concepts of object-oriented programming?",
            "How would you implement a linked list in Python?",
            "What's the difference between a list and a dictionary in Python?",
            "Can you write a function to calculate the Fibonacci sequence?",
            "How would you handle exceptions in Python?"
        ]
        
        question = random.choice(questions)
        
        # Create teacher response
        return f"Confidence: {confidence_str}\n\nBased on our conversation so far, I'd like to assess your understanding further. {question}"
    
    def _simulate_tester_response(self, last_message: str) -> str:
        """Simulate a tester agent response with varying quality based on rating."""
        import random
        
        # Simulating test rating-appropriate responses
        responses = [
            # Lower quality responses (for low ratings)
            "I'm not really sure about that. I think it has something to do with classes and objects?",
            "I've heard of that but don't remember the details. Could you explain more?",
            "I think I would use a loop for that, but I'm not sure exactly how to implement it.",
            
            # Medium quality responses (for medium ratings)
            "Object-oriented programming involves classes and objects. Classes define the structure and objects are instances of those classes.",
            "A linked list consists of nodes where each node points to the next node in the sequence. I would implement it with a Node class.",
            "Lists are ordered collections while dictionaries store key-value pairs for efficient lookups.",
            
            # High quality responses (for high ratings)
            "OOP has four main principles: encapsulation, abstraction, inheritance, and polymorphism. It allows for modular, reusable code through classes and objects.",
            "I would implement a linked list with a Node class containing data and next_node attributes, then create methods for insertion, deletion, and traversal.",
            "Lists maintain order and allow duplicates with O(n) lookup time, while dictionaries use hash tables for O(1) average case lookups with unique keys."
        ]
        
        return random.choice(responses)

# Create singleton instance
_model_interface = None

def get_model_interface() -> ModelInterface:
    """Get the global model interface instance."""
    global _model_interface
    if _model_interface is None:
        _model_interface = ModelInterface()
    return _model_interface