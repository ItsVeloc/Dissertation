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
    
    try:
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
        max_attempts = 10  # Safety limit to prevent infinite loops
        attempts = 0
        
        while not conversation.is_complete() and attempts < max_attempts:
            try:
                # Tester responds
                tester_message = tester.generate_response(conversation)
                conversation.add_message("tester", tester_message)
                
                # Teacher responds
                teacher_message, confidence = teacher.generate_response(conversation)
                conversation.add_message("teacher", teacher_message, confidence)
                
            except Exception as e:
                logger.error(f"Error during conversation step: {str(e)}")
                # Continue despite errors to get some data
                
            attempts += 1
        
        # If we hit the max attempts, mark the conversation as complete
        if attempts >= max_attempts:
            logger.warning(f"Conversation {conversation.id} reached max attempts limit")
            conversation.terminated_reason = "max_attempts_reached"
        
        # Get final rating estimate
        try:
            final_rating, final_confidence = teacher.estimate_rating(conversation)
            conversation.complete_conversation(final_rating, final_confidence)
        except Exception as e:
            logger.error(f"Error estimating final rating: {str(e)}")
            # Use defaults if estimation fails
            conversation.complete_conversation(5.0, 0.5)
        
        # Save the conversation
        try:
            storage.save_conversation(conversation)
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            # Continue even if saving fails
        
        logger.info(f"Completed conversation {conversation.id}. " 
                    f"True rating: {true_rating:.2f}, " 
                    f"Predicted: {conversation.predicted_rating:.2f}, " 
                    f"Confidence: {conversation.final_confidence:.2f}")
        
        return conversation
        
    except Exception as e:
        logger.error(f"Fatal error in run_conversation: {str(e)}")
        # Return an empty conversation with error state
        conversation = Conversation(
            subject=subject,
            true_rating=true_rating
        )
        conversation.terminated_reason = "error"
        conversation.complete_conversation(5.0, 0.5)
        return conversation

if __name__ == "__main__":
    # Example usage
    conversation = run_conversation("Python Programming", 7.5)
    print(f"Conversation ID: {conversation.id}")
    print(f"True rating: {conversation.true_rating}")
    print(f"Predicted rating: {conversation.predicted_rating}")
    print(f"Final confidence: {conversation.final_confidence}")
    print(f"Number of messages: {len(conversation.messages)}")