"""
Basic evaluation metrics for the teacher-tester system.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from teacher_tester.data.schemas import Conversation, TrainingExample
from teacher_tester.data.storage import get_storage
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

def calculate_prediction_error(true_rating: float, predicted_rating: float) -> float:
    """
    Calculate absolute error between true and predicted ratings.
    
    Args:
        true_rating: The ground truth rating
        predicted_rating: The predicted rating
        
    Returns:
        The absolute error
    """
    return abs(true_rating - predicted_rating)

def calculate_confidence_calibration(predictions: List[Dict[str, float]]) -> float:
    """
    Calculate how well calibrated the confidence estimates are.
    
    Args:
        predictions: List of dicts with 'true_rating', 'predicted_rating', and 'confidence'
        
    Returns:
        Calibration error score (lower is better)
    """
    if not predictions:
        return 0.0
    
    # Calculate errors
    errors = [abs(p['true_rating'] - p['predicted_rating']) for p in predictions]
    
    # Calculate normalized errors (0-1 scale, where 0 is perfect prediction)
    rating_range = 10.0  # Assuming 0-10 scale
    normalized_errors = [min(err / rating_range, 1.0) for err in errors]
    
    # Calculate calibration error (difference between confidence and accuracy)
    confidences = [p['confidence'] for p in predictions]
    accuracies = [1.0 - err for err in normalized_errors]
    calibration_errors = [abs(conf - acc) for conf, acc in zip(confidences, accuracies)]
    
    # Average calibration error
    mean_calibration_error = sum(calibration_errors) / len(calibration_errors)
    
    return mean_calibration_error

def extract_conversation_metrics(conversation: Conversation) -> Dict[str, Any]:
    """
    Extract evaluation metrics from a conversation.
    
    Args:
        conversation: The conversation to evaluate
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    metrics = {
        "conversation_id": conversation.id,
        "subject": conversation.subject,
        "true_rating": conversation.true_rating,
        "predicted_rating": conversation.predicted_rating,
        "final_confidence": conversation.final_confidence,
        "absolute_error": calculate_prediction_error(
            conversation.true_rating, conversation.predicted_rating
        ),
        "conversation_length": len([m for m in conversation.messages if m.role == "teacher"]),
        "terminated_reason": conversation.terminated_reason,
    }
    
    # Calculate duration if available
    if conversation.start_time and conversation.end_time:
        duration = (conversation.end_time - conversation.start_time).total_seconds()
        metrics["duration_seconds"] = duration
    
    # Convert confidence trajectory if available
    teacher_messages = [m for m in conversation.messages if m.role == "teacher"]
    confidence_trajectory = []
    
    for msg in teacher_messages:
        if msg.confidence is not None:
            confidence_trajectory.append(msg.confidence)
    
    metrics["confidence_trajectory"] = confidence_trajectory
    
    return metrics

def create_training_example(conversation: Conversation) -> TrainingExample:
    """
    Create a training example from a conversation.
    
    Args:
        conversation: The completed conversation
        
    Returns:
        A TrainingExample object
    """
    metrics = extract_conversation_metrics(conversation)
    
    # Create training example
    example = TrainingExample(
        conversation_id=conversation.id,
        subject=conversation.subject,
        true_rating=conversation.true_rating,
        predicted_rating=conversation.predicted_rating,
        prediction_error=metrics["absolute_error"],
        confidence=conversation.final_confidence,
        calibration_error=abs(conversation.final_confidence - 
                              (1.0 - min(metrics["absolute_error"] / 10.0, 1.0))),
        conversation_length=metrics["conversation_length"],
        features={
            "confidence_trajectory": metrics.get("confidence_trajectory", []),
            "terminated_reason": metrics.get("terminated_reason"),
            "duration_seconds": metrics.get("duration_seconds", 0)
        }
    )
    
    return example

def evaluate_conversations(conversation_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Evaluate multiple conversations and return a DataFrame of metrics.
    
    Args:
        conversation_ids: List of conversation IDs to evaluate. If None, evaluate all.
        
    Returns:
        DataFrame of evaluation metrics
    """
    storage = get_storage()
    
    # Get conversation IDs if not provided
    if conversation_ids is None:
        conversation_ids = storage.list_conversations()
    
    # Collect metrics
    all_metrics = []
    for conv_id in conversation_ids:
        conv = storage.load_conversation(conv_id)
        if conv and conv.predicted_rating is not None:
            metrics = extract_conversation_metrics(conv)
            all_metrics.append(metrics)
    
    # Convert to DataFrame
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Calculate summary statistics
        logger.info(f"Evaluated {len(all_metrics)} conversations")
        logger.info(f"Mean absolute error: {df['absolute_error'].mean():.2f}")
        logger.info(f"Mean conversation length: {df['conversation_length'].mean():.2f}")
        
        return df
    else:
        logger.warning("No conversations to evaluate")
        return pd.DataFrame()