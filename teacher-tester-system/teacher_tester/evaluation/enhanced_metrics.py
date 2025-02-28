"""
Enhanced evaluation metrics and feedback loop mechanisms for the teacher-tester system.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict

from teacher_tester.data.schemas import Conversation, TrainingExample
from teacher_tester.data.storage import get_storage
from teacher_tester.utils.logging import setup_logger
from teacher_tester.evaluation.metrics import extract_conversation_metrics

logger = setup_logger(__name__)

class EvaluationMetrics:
    """Enhanced evaluation metrics for teacher-tester conversations."""
    
    def __init__(self, conversation_ids: Optional[List[str]] = None):
        """
        Initialize with a set of conversations to analyze.
        
        Args:
            conversation_ids: List of conversation IDs to analyze. If None, loads all.
        """
        self.storage = get_storage()
        
        # Load conversations
        if conversation_ids is None:
            conversation_ids = self.storage.list_conversations()
        
        self.conversations = []
        for conv_id in conversation_ids:
            conv = self.storage.load_conversation(conv_id)
            if conv and conv.predicted_rating is not None:
                self.conversations.append(conv)
        
        logger.info(f"Loaded {len(self.conversations)} conversations for analysis")
    
    def compute_basic_metrics(self) -> Dict[str, float]:
        """
        Compute basic evaluation metrics.
        
        Returns:
            Dictionary of metric name to value
        """
        if not self.conversations:
            return {}
        
        # Extract true and predicted ratings
        true_ratings = [conv.true_rating for conv in self.conversations]
        pred_ratings = [conv.predicted_rating for conv in self.conversations]
        confidences = [conv.final_confidence for conv in self.conversations]
        
        # Compute regression metrics
        metrics = {
            "mae": mean_absolute_error(true_ratings, pred_ratings),
            "rmse": np.sqrt(mean_squared_error(true_ratings, pred_ratings)),
            "r2": r2_score(true_ratings, pred_ratings),
            "mean_confidence": np.mean(confidences),
            "count": len(self.conversations)
        }
        
        return metrics
    
    def compute_calibration_metrics(self) -> Dict[str, float]:
        """
        Compute metrics related to confidence calibration.
        
        Returns:
            Dictionary of calibration metrics
        """
        if not self.conversations:
            return {}
        
        # Prepare data
        true_ratings = np.array([conv.true_rating for conv in self.conversations])
        pred_ratings = np.array([conv.predicted_rating for conv in self.conversations])
        confidences = np.array([conv.final_confidence for conv in self.conversations])
        
        # Calculate absolute errors and normalize to 0-1 range
        abs_errors = np.abs(true_ratings - pred_ratings)
        max_error = 10.0  # Full scale range
        normalized_errors = np.minimum(abs_errors / max_error, 1.0)
        
        # Calculate accuracy (1 - normalized error)
        accuracies = 1.0 - normalized_errors
        
        # Calculate calibration error (difference between confidence and accuracy)
        calibration_errors = np.abs(confidences - accuracies)
        
        # Calculate expected calibration error (ECE)
        # Group predictions into bins based on confidence
        n_bins = 10
        bin_indices = np.digitize(confidences, np.linspace(0, 1, n_bins + 1)) - 1
        
        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (bin_indices == i)
            if np.any(mask):
                bin_accuracies[i] = np.mean(accuracies[mask])
                bin_confidences[i] = np.mean(confidences[mask])
                bin_counts[i] = np.sum(mask)
        
        # Calculate ECE
        ece = np.sum(bin_counts * np.abs(bin_confidences - bin_accuracies)) / np.sum(bin_counts)
        
        # Calculate maximum calibration error (MCE)
        mce = np.max(np.abs(bin_confidences - bin_accuracies))
        
        return {
            "mean_calibration_error": np.mean(calibration_errors),
            "expected_calibration_error": float(ece),
            "max_calibration_error": float(mce),
            "overconfidence_rate": np.mean(confidences > accuracies),
            "underconfidence_rate": np.mean(confidences < accuracies)
        }
    
    def analyze_by_subject(self) -> pd.DataFrame:
        """
        Analyze performance broken down by subject.
        
        Returns:
            DataFrame with metrics by subject
        """
        if not self.conversations:
            return pd.DataFrame()
        
        # Group conversations by subject
        subject_data = defaultdict(list)
        
        for conv in self.conversations:
            subject_data[conv.subject].append({
                "true_rating": conv.true_rating,
                "predicted_rating": conv.predicted_rating,
                "error": abs(conv.true_rating - conv.predicted_rating),
                "confidence": conv.final_confidence,
                "conversation_length": len([m for m in conv.messages if m.role == "teacher"])
            })
        
        # Compute metrics for each subject
        subject_metrics = []
        
        for subject, convs in subject_data.items():
            true_ratings = [c["true_rating"] for c in convs]
            pred_ratings = [c["predicted_rating"] for c in convs]
            errors = [c["error"] for c in convs]
            confidences = [c["confidence"] for c in convs]
            lengths = [c["conversation_length"] for c in convs]
            
            metrics = {
                "subject": subject,
                "count": len(convs),
                "mae": np.mean(errors),
                "rmse": np.sqrt(np.mean(np.square(errors))),
                "mean_confidence": np.mean(confidences),
                "mean_conversation_length": np.mean(lengths),
                "median_error": np.median(errors)
            }
            
            subject_metrics.append(metrics)
        
        return pd.DataFrame(subject_metrics)
    
    def analyze_by_rating(self, bin_width: float = 1.0) -> pd.DataFrame:
        """
        Analyze performance broken down by rating ranges.
        
        Args:
            bin_width: Width of rating bins
            
        Returns:
            DataFrame with metrics by rating bin
        """
        if not self.conversations:
            return pd.DataFrame()
        
        # Create bins
        rating_min = 0
        rating_max = 10
        bins = np.arange(rating_min, rating_max + bin_width, bin_width)
        bin_labels = [f"{b:.1f}-{b+bin_width:.1f}" for b in bins[:-1]]
        
        # Group conversations by rating bin
        bin_data = defaultdict(list)
        
        for conv in self.conversations:
            bin_idx = min(int(conv.true_rating / bin_width), len(bin_labels) - 1)
            bin_label = bin_labels[bin_idx]
            
            bin_data[bin_label].append({
                "true_rating": conv.true_rating,
                "predicted_rating": conv.predicted_rating,
                "error": abs(conv.true_rating - conv.predicted_rating),
                "confidence": conv.final_confidence
            })
        
        # Compute metrics for each bin
        bin_metrics = []
        
        for bin_label, convs in bin_data.items():
            true_ratings = [c["true_rating"] for c in convs]
            pred_ratings = [c["predicted_rating"] for c in convs]
            errors = [c["error"] for c in convs]
            confidences = [c["confidence"] for c in convs]
            
            # Calculate bias (average prediction - average true)
            bias = np.mean(pred_ratings) - np.mean(true_ratings)
            
            metrics = {
                "rating_bin": bin_label,
                "min_rating": min(true_ratings),
                "max_rating": max(true_ratings),
                "count": len(convs),
                "mae": np.mean(errors),
                "bias": bias,
                "mean_confidence": np.mean(confidences)
            }
            
            bin_metrics.append(metrics)
        
        # Sort by bin order
        return pd.DataFrame(bin_metrics).sort_values("min_rating")
    
    def analyze_confidence_progression(self) -> Dict[str, Any]:
        """
        Analyze how confidence changes throughout conversations.
        
        Returns:
            Dictionary with confidence progression metrics
        """
        if not self.conversations:
            return {}
        
        # Collect confidence trajectories
        all_trajectories = []
        final_accuracies = []
        
        for conv in self.conversations:
            # Extract confidence values from teacher messages
            confidence_trajectory = []
            for msg in conv.messages:
                if msg.role == "teacher" and msg.confidence is not None:
                    confidence_trajectory.append(msg.confidence)
            
            if confidence_trajectory:
                all_trajectories.append(confidence_trajectory)
                
                # Calculate final accuracy (1 - normalized error)
                error = abs(conv.true_rating - conv.predicted_rating)
                max_error = 10.0
                normalized_error = min(error / max_error, 1.0)
                accuracy = 1.0 - normalized_error
                final_accuracies.append(accuracy)
        
        # Find the maximum length trajectory
        max_length = max(len(t) for t in all_trajectories)
        
        # Pad trajectories to the same length
        padded_trajectories = []
        for t in all_trajectories:
            padded = t + [t[-1]] * (max_length - len(t))
            padded_trajectories.append(padded)
        
        # Calculate average confidence at each step
        avg_confidence_by_step = []
        for step in range(max_length):
            step_confidences = [t[step] for t in padded_trajectories]
            avg_confidence_by_step.append(float(np.mean(step_confidences)))
        
        # Calculate rate of confidence change
        confidence_changes = []
        for trajectory in all_trajectories:
            if len(trajectory) > 1:
                changes = [trajectory[i+1] - trajectory[i] for i in range(len(trajectory)-1)]
                avg_change = np.mean(changes)
                confidence_changes.append(avg_change)
        
        # Calculate correlation between final confidence and accuracy
        if all_trajectories and final_accuracies:
            final_confidences = [t[-1] for t in all_trajectories]
            confidence_accuracy_corr = float(np.corrcoef(final_confidences, final_accuracies)[0, 1])
        else:
            confidence_accuracy_corr = 0.0
        
        return {
            "avg_confidence_by_step": avg_confidence_by_step,
            "mean_confidence_change_rate": float(np.mean(confidence_changes)) if confidence_changes else 0.0,
            "confidence_accuracy_correlation": confidence_accuracy_corr,
            "avg_trajectory_length": float(np.mean([len(t) for t in all_trajectories]))
        }
    
    def generate_feedback_report(self, 
                                output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive feedback report with all metrics.
        
        Args:
            output_path: Path to save the report. If None, doesn't save.
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Calculate all metrics
        basic_metrics = self.compute_basic_metrics()
        calibration_metrics = self.compute_calibration_metrics()
        subject_analysis = self.analyze_by_subject().to_dict(orient="records")
        rating_analysis = self.analyze_by_rating().to_dict(orient="records")
        confidence_progression = self.analyze_confidence_progression()
        
        # Combine into a report
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_conversations": len(self.conversations),
            "basic_metrics": basic_metrics,
            "calibration_metrics": calibration_metrics,
            "subject_analysis": subject_analysis,
            "rating_analysis": rating_analysis,
            "confidence_progression": confidence_progression
        }
        
        # Add overall assessment and recommendations
        report["assessment"] = self._generate_assessment(report)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved evaluation report to {output_path}")
        
        return report
    
    def _generate_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment and recommendations based on metrics."""
        assessment = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Check basic accuracy
        mae = report["basic_metrics"].get("mae", float('inf'))
        if mae < 1.0:
            assessment["strengths"].append("Good overall rating prediction accuracy")
        elif mae > 2.5:
            assessment["weaknesses"].append("Poor rating prediction accuracy")
            assessment["recommendations"].append("Improve the teacher's ability to ask more discriminative questions")
        
        # Check calibration
        ece = report["calibration_metrics"].get("expected_calibration_error", 1.0)
        if ece < 0.1:
            assessment["strengths"].append("Well-calibrated confidence estimates")
        elif ece > 0.25:
            assessment["weaknesses"].append("Poorly calibrated confidence estimates")
            assessment["recommendations"].append("Adjust confidence estimation to better reflect actual performance")
        
        # Check for biases in rating ranges
        high_bias_bins = []
        for bin_data in report["rating_analysis"]:
            if abs(bin_data.get("bias", 0)) > 1.5 and bin_data.get("count", 0) >= 5:
                high_bias_bins.append(bin_data["rating_bin"])
        
        if high_bias_bins:
            bias_ranges = ", ".join(high_bias_bins)
            assessment["weaknesses"].append(f"Systematic bias in rating ranges: {bias_ranges}")
            assessment["recommendations"].append("Improve accuracy for specific rating ranges with targeted training")
        
        # Check subject performance variations
        if len(report["subject_analysis"]) > 1:
            subject_maes = [(s["subject"], s["mae"]) for s in report["subject_analysis"]]
            best_subject = min(subject_maes, key=lambda x: x[1])
            worst_subject = max(subject_maes, key=lambda x: x[1])
            
            if worst_subject[1] > best_subject[1] * 1.5:
                assessment["weaknesses"].append(f"Inconsistent performance across subjects (worst: {worst_subject[0]})")
                assessment["recommendations"].append(f"Improve assessment ability for {worst_subject[0]}")
        
        # Check confidence progression
        avg_change = report["confidence_progression"].get("mean_confidence_change_rate", 0)
        if avg_change < 0.05:
            assessment["weaknesses"].append("Slow confidence convergence during conversations")
            assessment["recommendations"].append("Improve the teacher's ability to gain information from each exchange")
        
        # Add default recommendation if empty
        if not assessment["recommendations"]:
            assessment["recommendations"].append("Continue collecting data to identify specific improvement areas")
        
        return assessment

def generate_evaluation_report(
    conversation_ids: Optional[List[str]] = None,
    output_dir: str = "data/reports"
) -> str:
    """
    Generate and save a comprehensive evaluation report.
    
    Args:
        conversation_ids: List of conversation IDs to include. If None, uses all.
        output_dir: Directory to save the report
        
    Returns:
        Path to the saved report
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluator and generate report
    evaluator = EvaluationMetrics(conversation_ids)
    
    # Generate filename with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    
    # Generate and save report
    evaluator.generate_feedback_report(output_path)
    
    return output_path