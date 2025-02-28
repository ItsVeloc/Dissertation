from typing import Optional
from teacher_tester.agents.teacher import TeacherAgent
from teacher_tester.agents.tester import TesterAgent
from teacher_tester.data.schemas import Conversation
from teacher_tester.data.storage import get_storage
from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

def run_conversation(subject: str, true_rating: float) -> Conversation:
    """
    Run a complete conversation between teacher and tester.
    
    Args:
        subject: The subject of conversation
        true_rating: The true rating of the tester
        
    Returns:
        The completed conversation
    """
    config = get_config()
    storage = get_storage()
    
    # Initialize agents
    teacher = TeacherAgent(subject)
    tester = TesterAgent(true_rating)
    
    # Initialize conversation
    conversation = Conversation(
        subject=subject,
        true_rating=true_rating
    )
    
    # Teacher starts the conversation
    teacher_message, confidence = teacher.generate_response(conversation)
    conversation.add_message("teacher", teacher_message, confidence)
    
    logger.info(f"Started conversation {conversation.id} on {subject} with tester rating {true_rating}")
    
    # Continue until termination condition is met
    while not conversation.is_complete():
        # Tester responds
        tester_message = tester.generate_response(conversation)
        conversation.add_message("tester", tester_message)
        
        # Teacher responds
        teacher_message, confidence = teacher.generate_response(conversation)
        conversation.add_message("teacher", teacher_message, confidence)
    
    # Get final rating estimate
    final_rating, final_confidence = teacher.estimate_rating(conversation)
    conversation.complete_conversation(final_rating, final_confidence)
    
    # Save the conversation
    storage.save_conversation(conversation)
    
    logger.info(f"Completed conversation {conversation.id}. " 
                f"True rating: {true_rating:.2f}, " 
                f"Predicted: {final_rating:.2f}, " 
                f"Confidence: {final_confidence:.2f}")
    
    return conversation

if __name__ == "__main__":
    # Example usage
    conversation = run_conversation("Python Programming", 7.5)
    print(f"Conversation ID: {conversation.id}")
    print(f"True rating: {conversation.true_rating}")
    print(f"Predicted rating: {conversation.predicted_rating}")
    print(f"Final confidence: {conversation.final_confidence}")
    print(f"Number of messages: {len(conversation.messages)}")