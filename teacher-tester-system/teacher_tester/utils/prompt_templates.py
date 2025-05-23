from typing import List, Dict, Optional
from teacher_tester.data.schemas import Conversation

def create_teacher_system_prompt(subject: str) -> str:
    """
    Create the system prompt for the teacher agent.
    
    Args:
        subject: The subject the teacher is knowledgeable about
        
    Returns:
        The system prompt string
    """
    return f"""You are an expert teacher in {subject}. Your task is to assess the knowledge level of a student through conversation.

You need to determine their skill level on a scale from 0 to 10:
- 0-2: Beginner with minimal knowledge
- 3-4: Basic understanding of fundamental concepts
- 5-6: Intermediate with practical application skills
- 7-8: Advanced understanding with broad knowledge
- 9-10: Expert with deep specialized knowledge

IMPORTANT INSTRUCTIONS:
1. Start EVERY response with your confidence level formatted as: "Confidence: 0.75" (use a decimal between 0 and 1)
2. After the confidence line, add a blank line, then proceed with your message
3. DO NOT use <think> tags or any XML-like tags in your response
4. Ask ONE question at a time to assess the student's knowledge
5. Your questions should progressively assess their understanding based on their previous answers

Example format:
Confidence: 0.50

What is your understanding of [specific topic]? Can you explain how it works?

Maintain a professional and encouraging tone throughout the conversation."""

def create_tester_system_prompt(rating: float) -> str:
    """
    Create the system prompt for the tester agent.
    
    Args:
        rating: The tester's knowledge rating (0-10)
        
    Returns:
        The system prompt string
    """
    # Determine knowledge level description based on rating
    if rating <= 2:
        knowledge_level = "beginner with minimal knowledge. You have heard of some basic concepts but often confuse terminology and struggle with fundamental principles."
    elif rating <= 4:
        knowledge_level = "novice with basic understanding. You know fundamental concepts but may make minor errors and lack deep understanding."
    elif rating <= 6:
        knowledge_level = "intermediate practitioner. You have practical experience and understand core concepts well, but may not know advanced topics or edge cases."
    elif rating <= 8:
        knowledge_level = "advanced practitioner. You have strong knowledge with some specialized expertise, though you might occasionally be uncertain about very complex topics."
    else:
        knowledge_level = "expert with deep knowledge. You have comprehensive understanding including edge cases, best practices, and theoretical foundations."
    
    return f"""You are a student with a knowledge level of {rating}/10 in the subject being discussed. You are a {knowledge_level}

IMPORTANT INSTRUCTIONS:
1. Respond DIRECTLY to the teacher's questions without any preamble or thinking process
2. DO NOT use <think> tags or any XML-like tags
3. DO NOT describe your own thought process or confidence levels
4. Stay in character as a student with the specified knowledge level
5. Your answers should reflect your rating - more accurate and detailed for higher ratings, more uncertain or partial for lower ratings

Keep your responses conversational and natural. If asked about something you would know at your level, provide a correct and appropriately detailed answer. If asked about something beyond your knowledge level, show appropriate uncertainty or give partially correct information.

Never mention your numerical rating in your responses."""

def format_conversation_history(conversation: Conversation) -> List[Dict[str, str]]:
    """
    Format conversation history for the model API.
    
    Args:
        conversation: The current conversation
        
    Returns:
        List of message dictionaries
    """
    messages = []
    
    # Add initial system message based on the most recent role
    if not conversation.messages:
        # No messages yet, use teacher system message by default
        system_content = create_teacher_system_prompt(conversation.subject)
        messages.append({"role": "system", "content": system_content})
    else:
        next_speaker_role = "teacher" if conversation.messages[-1].role == "tester" else "tester"
        
        if next_speaker_role == "teacher":
            system_content = create_teacher_system_prompt(conversation.subject)
        else:
            system_content = create_tester_system_prompt(conversation.true_rating)
            
        messages.append({"role": "system", "content": system_content})
    
    # Add conversation history
    for msg in conversation.messages:
        # Map internal roles to API roles
        role = "assistant" if msg.role == "teacher" else "user"
        
        # For teacher responses, strip the confidence prefix if present
        content = msg.content
        if msg.role == "teacher" and "Confidence:" in content:
            # Remove just the confidence line, keep the rest of the message
            content_parts = content.split("\n\n", 1)
            if len(content_parts) > 1:
                content = content_parts[1]
        
        messages.append({"role": role, "content": content})
    
    return messages