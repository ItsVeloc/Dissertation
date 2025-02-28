"""
Tools for analyzing conversations between teacher and tester agents.
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import json
from datetime import datetime

from teacher_tester.data.schemas import Conversation
from teacher_tester.data.storage import get_storage
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

class ConversationAnalyzer:
    """
    Analyze conversations between teacher and tester agents.
    """
    
    def __init__(self, conversations: Optional[List[Conversation]] = None):
        """
        Initialize with conversations to analyze.
        
        Args:
            conversations: List of conversations. If None, loads all from storage.
        """
        self.storage = get_storage()
        
        if conversations is None:
            # Load all conversations
            conversation_ids = self.storage.list_conversations()
            self.conversations = []
            for conv_id in conversation_ids:
                conv = self.storage.load_conversation(conv_id)
                if conv is not None:
                    self.conversations.append(conv)
        else:
            self.conversations = conversations
            
        logger.info(f"Initialized analyzer with {len(self.conversations)} conversations")
    
    def extract_question_types(self) -> Dict[str, Any]:
        """
        Extract and categorize types of questions asked by the teacher.
        
        Returns:
            Dictionary with question type analysis
        """
        # Define question type patterns
        question_patterns = {
            "concept_explanation": r"(explain|describe|what is|define)\s.+\?",
            "implementation": r"(how would you implement|write|code|program)\s.+\?",
            "comparison": r"(compare|difference between|versus|vs\.|similarities|differences)\s.+\?",
            "problem_solving": r"(solve|fix|debug|address|handle)\s.+\?",
            "theoretical": r"(why|reason|explain why|theory behind)\s.+\?",
            "practical": r"(when would you use|practical application|real-world|example of)\s.+\?"
        }
        
        # Extract teacher questions
        all_questions = []
        questions_by_rating = defaultdict(list)
        
        for conv in self.conversations:
            # Bin the rating to nearest integer
            rating_bin = round(conv.true_rating)
            
            for msg in conv.messages:
                if msg.role == "teacher":
                    content = msg.content
                    
                    # Remove confidence line if present
                    if "Confidence:" in content:
                        content = re.sub(r"Confidence:.*?\n\n", "", content, flags=re.DOTALL)
                    
                    # Extract questions
                    sentences = re.split(r'[.!?]\s+', content)
                    for sentence in sentences:
                        if '?' in sentence:
                            question = sentence.strip() + '?'
                            all_questions.append(question)
                            questions_by_rating[rating_bin].append(question)
        
        # Categorize questions
        categorized_questions = defaultdict(list)
        uncategorized = []
        
        for question in all_questions:
            categorized = False
            for category, pattern in question_patterns.items():
                if re.search(pattern, question, re.IGNORECASE):
                    categorized_questions[category].append(question)
                    categorized = True
                    break
            
            if not categorized:
                uncategorized.append(question)
        
        # Calculate question type distribution by rating
        distribution_by_rating = {}
        for rating, questions in questions_by_rating.items():
            rating_distribution = {}
            total = len(questions)
            if total == 0:
                continue
                
            for category, pattern in question_patterns.items():
                count = sum(1 for q in questions if re.search(pattern, q, re.IGNORECASE))
                rating_distribution[category] = count / total
                
            distribution_by_rating[rating] = rating_distribution
        
        # Assemble results
        return {
            "total_questions": len(all_questions),
            "category_counts": {k: len(v) for k, v in categorized_questions.items()},
            "category_percentages": {
                k: len(v) / len(all_questions) if all_questions else 0 
                for k, v in categorized_questions.items()
            },
            "uncategorized_count": len(uncategorized),
            "distribution_by_rating": distribution_by_rating,
            "sample_questions": {k: v[:5] for k, v in categorized_questions.items()},
            "uncategorized_samples": uncategorized[:5] if uncategorized else []
        }
    
    def analyze_conversation_flow(self) -> Dict[str, Any]:
        """
        Analyze the flow and structure of conversations.
        
        Returns:
            Dictionary with conversation flow analysis
        """
        # Collect statistics
        lengths = []
        teacher_message_lengths = []
        tester_message_lengths = []
        confidence_trajectories = []
        termination_reasons = Counter()
        
        for conv in self.conversations:
            # Count messages by role
            teacher_msgs = [m for m in conv.messages if m.role == "teacher"]
            tester_msgs = [m for m in conv.messages if m.role == "tester"]
            
            lengths.append(len(conv.messages))
            
            # Message lengths (character count)
            teacher_message_lengths.extend([len(m.content) for m in teacher_msgs])
            tester_message_lengths.extend([len(m.content) for m in tester_msgs])
            
            # Confidence trajectory
            confidence_trajectory = [m.confidence for m in teacher_msgs if m.confidence is not None]
            if confidence_trajectory:
                confidence_trajectories.append(confidence_trajectory)
            
            # Termination reason
            if conv.terminated_reason:
                termination_reasons[conv.terminated_reason] += 1
        
        # Calculate average confidence change
        avg_confidence_changes = []
        for trajectory in confidence_trajectories:
            if len(trajectory) > 1:
                changes = [trajectory[i+1] - trajectory[i] for i in range(len(trajectory)-1)]
                avg_change = np.mean(changes)
                avg_confidence_changes.append(avg_change)
        
        # Calculate confidence slope
        def calculate_slope(values):
            if len(values) < 2:
                return 0
            x = np.arange(len(values))
            return np.polyfit(x, values, 1)[0]
        
        confidence_slopes = [calculate_slope(traj) for traj in confidence_trajectories if len(traj) >= 2]
        
        # Group by conversation length
        conversations_by_length = defaultdict(list)
        for conv in self.conversations:
            length = len([m for m in conv.messages if m.role == "teacher"])
            conversations_by_length[length].append(conv)
        
        accuracy_by_length = {}
        for length, convs in conversations_by_length.items():
            errors = [abs(c.true_rating - c.predicted_rating) for c in convs]
            accuracy_by_length[length] = {
                "count": len(convs),
                "mean_error": np.mean(errors) if errors else 0,
                "mean_confidence": np.mean([c.final_confidence for c in convs]) if convs else 0
            }
        
        return {
            "total_conversations": len(self.conversations),
            "avg_conversation_length": np.mean(lengths) if lengths else 0,
            "avg_teacher_message_length": np.mean(teacher_message_lengths) if teacher_message_lengths else 0,
            "avg_tester_message_length": np.mean(tester_message_lengths) if tester_message_lengths else 0,
            "termination_reasons": dict(termination_reasons),
            "avg_confidence_change_per_message": np.mean(avg_confidence_changes) if avg_confidence_changes else 0,
            "avg_confidence_slope": np.mean(confidence_slopes) if confidence_slopes else 0,
            "accuracy_by_conversation_length": accuracy_by_length
        }
    
    def extract_response_patterns(self) -> Dict[str, Any]:
        """
        Extract patterns in tester responses based on rating level.
        
        Returns:
            Dictionary with tester response patterns
        """
        # Group tester responses by rating bin
        responses_by_rating = defaultdict(list)
        
        for conv in self.conversations:
            # Bin the rating to nearest integer
            rating_bin = round(conv.true_rating)
            
            for msg in conv.messages:
                if msg.role == "tester":
                    responses_by_rating[rating_bin].append(msg.content)
        
        # Define patterns to search for in responses
        patterns = {
            "uncertainty": r"(not sure|I think|maybe|perhaps|probably|might be|could be|unsure|uncertain)",
            "technical_terms": r"(algorithm|function|method|class|object|instance|variable|parameter|attribute|interface)",
            "code_fragments": r"(def |class |import |return |for |while |if |else |try |except )",
            "detailed_explanation": r"(\w+\s){20,}",  # Responses with at least 20 words
            "questions": r"\?",
            "examples": r"(for example|for instance|such as|e\.g\.)"
        }
        
        # Analyze patterns by rating
        pattern_frequency_by_rating = {}
        
        for rating, responses in responses_by_rating.items():
            if not responses:
                continue
                
            pattern_counts = {}
            for pattern_name, pattern in patterns.items():
                matches = sum(1 for r in responses if re.search(pattern, r, re.IGNORECASE))
                pattern_counts[pattern_name] = matches / len(responses)
                
            pattern_frequency_by_rating[rating] = pattern_counts
        
        # Calculate response complexity (average character length)
        complexity_by_rating = {}
        for rating, responses in responses_by_rating.items():
            if responses:
                avg_length = np.mean([len(r) for r in responses])
                complexity_by_rating[rating] = avg_length
        
        # Extract example responses for each rating
        examples_by_rating = {}
        for rating, responses in responses_by_rating.items():
            if responses:
                # Get a few diverse examples (choose based on length)
                lengths = [len(r) for r in responses]
                if len(responses) >= 3:
                    indices = [
                        lengths.index(min(lengths)),
                        lengths.index(sorted(lengths)[len(lengths)//2]),
                        lengths.index(max(lengths))
                    ]
                    examples_by_rating[rating] = [responses[i] for i in indices]
                else:
                    examples_by_rating[rating] = responses[:min(3, len(responses))]
        
        return {
            "pattern_frequency_by_rating": pattern_frequency_by_rating,
            "response_complexity_by_rating": complexity_by_rating,
            "example_responses_by_rating": examples_by_rating
        }
    
    def generate_insights_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive insights report about the conversations.
        
        Args:
            output_path: Path to save the report. If None, doesn't save.
            
        Returns:
            Dictionary with conversation insights
        """
        # Run all analyses
        question_types = self.extract_question_types()
        conversation_flow = self.analyze_conversation_flow()
        response_patterns = self.extract_response_patterns()
        
        # Combine into a report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_conversations": len(self.conversations),
            "question_analysis": question_types,
            "conversation_flow": conversation_flow,
            "response_patterns": response_patterns
        }
        
        # Add insights
        report["insights"] = self._generate_insights(report)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved insights report to {output_path}")
        
        return report
    
    def _generate_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate key insights from the analysis report."""
        insights = []
        
        # Question type insights
        question_analysis = report["question_analysis"]
        if question_analysis["total_questions"] > 0:
            # Most common question type
            most_common = max(question_analysis["category_percentages"].items(), 
                             key=lambda x: x[1])
            insights.append(f"Teachers primarily ask {most_common[0]} questions ({most_common[1]*100:.1f}% of questions)")
            
            # Question distribution by rating
            if question_analysis["distribution_by_rating"]:
                low_ratings = [r for r in question_analysis["distribution_by_rating"].keys() if r <= 3]
                high_ratings = [r for r in question_analysis["distribution_by_rating"].keys() if r >= 7]
                
                if low_ratings and high_ratings:
                    low_dist = {}
                    for r in low_ratings:
                        for cat, val in question_analysis["distribution_by_rating"][r].items():
                            low_dist[cat] = low_dist.get(cat, 0) + val / len(low_ratings)
                    
                    high_dist = {}
                    for r in high_ratings:
                        for cat, val in question_analysis["distribution_by_rating"][r].items():
                            high_dist[cat] = high_dist.get(cat, 0) + val / len(high_ratings)
                    
                    # Find biggest difference in question types between low and high ratings
                    diffs = {cat: high_dist.get(cat, 0) - low_dist.get(cat, 0) 
                            for cat in set(low_dist) | set(high_dist)}
                    
                    most_different = max(diffs.items(), key=lambda x: abs(x[1]))
                    if most_different[1] > 0.15:  # Only report meaningful differences
                        direction = "more" if most_different[1] > 0 else "fewer"
                        insights.append(f"Teachers ask {direction} {most_different[0]} questions to higher-rated testers")
        
        # Conversation flow insights
        flow = report["conversation_flow"]
        if "termination_reasons" in flow and flow["termination_reasons"]:
            # Most common termination reason
            most_common = max(flow["termination_reasons"].items(), key=lambda x: x[1])
            total = sum(flow["termination_reasons"].values())
            insights.append(f"{most_common[1]/total*100:.1f}% of conversations end due to {most_common[0]}")
        
        # Check if longer conversations are more accurate
        if "accuracy_by_conversation_length" in flow:
            lengths = sorted(flow["accuracy_by_conversation_length"].keys())
            if len(lengths) > 1:
                short_err = flow["accuracy_by_conversation_length"][lengths[0]]["mean_error"]
                long_err = flow["accuracy_by_conversation_length"][lengths[-1]]["mean_error"]
                
                if abs(long_err - short_err) > 0.5:  # Only report meaningful differences
                    improved = "improved" if long_err < short_err else "worsened"
                    insights.append(f"Rating accuracy {improved} with longer conversations " +
                                  f"({short_err:.2f} error in {lengths[0]}-turn to {long_err:.2f} error in {lengths[-1]}-turn)")
        
        # Response pattern insights
        patterns = report["response_patterns"]
        if "pattern_frequency_by_rating" in patterns and patterns["pattern_frequency_by_rating"]:
            # Check for correlation between patterns and ratings
            ratings = sorted(patterns["pattern_frequency_by_rating"].keys())
            if len(ratings) > 1:
                pattern_correlations = {}
                
                for pattern in next(iter(patterns["pattern_frequency_by_rating"].values())).keys():
                    values = [patterns["pattern_frequency_by_rating"][r].get(pattern, 0) for r in ratings]
                    correlation = np.corrcoef(ratings, values)[0, 1]
                    pattern_correlations[pattern] = correlation
                
                # Find strongest correlations
                pos_corr = max(pattern_correlations.items(), key=lambda x: x[1])
                neg_corr = min(pattern_correlations.items(), key=lambda x: x[1])
                
                # Report strong correlations
                if pos_corr[1] > 0.5:
                    insights.append(f"Higher-rated testers use more {pos_corr[0]} in their responses")
                if neg_corr[1] < -0.5:
                    insights.append(f"Lower-rated testers use more {neg_corr[0]} in their responses")
        
        # Check response complexity
        if "response_complexity_by_rating" in patterns and patterns["response_complexity_by_rating"]:
            ratings = sorted(patterns["response_complexity_by_rating"].keys())
            if len(ratings) > 1:
                low_complexity = np.mean([patterns["response_complexity_by_rating"][r] for r in ratings if r <= 3])
                high_complexity = np.mean([patterns["response_complexity_by_rating"][r] for r in ratings if r >= 7])
                
                if high_complexity / low_complexity > 1.5:  # Only report meaningful differences
                    insights.append(f"Higher-rated testers give {high_complexity/low_complexity:.1f}x longer responses")
        
        return insights