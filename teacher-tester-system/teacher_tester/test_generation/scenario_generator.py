"""
Test scenario generator for the teacher-tester system.
"""
import os
import argparse
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import random
from collections import defaultdict

from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

class ScenarioGenerator:
    """
    Generate diverse test scenarios for the teacher-tester system.
    """
    
    def __init__(self, output_dir: str = "data/scenarios"):
        """
        Initialize the scenario generator.
        
        Args:
            output_dir: Directory to save generated scenarios
        """
        self.config = get_config()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get default subjects from config
        self.default_subjects = list(self.config.get("conversation.initial_prompts", {}).keys())
        if not self.default_subjects:
            self.default_subjects = ["Python Programming", "Machine Learning", "Data Science", "Web Development"]
            
        logger.info(f"Initialized scenario generator with {len(self.default_subjects)} default subjects")
    
    def generate_uniform_scenarios(self, 
                                  subjects: Optional[List[str]] = None,
                                  n_ratings: int = 10,
                                  rating_min: float = 0.5,
                                  rating_max: float = 9.5) -> List[Dict[str, Any]]:
        """
        Generate scenarios with uniform distribution of ratings.
        
        Args:
            subjects: List of subjects to use. If None, uses default subjects.
            n_ratings: Number of rating levels to generate
            rating_min: Minimum rating value
            rating_max: Maximum rating value
            
        Returns:
            List of scenario dictionaries
        """
        if subjects is None:
            subjects = self.default_subjects
            
        # Generate uniform rating distribution
        ratings = np.linspace(rating_min, rating_max, n_ratings).tolist()
        
        # Create scenarios
        scenarios = []
        for subject in subjects:
            for rating in ratings:
                scenarios.append({
                    "subject": subject,
                    "rating": rating
                })
        
        logger.info(f"Generated {len(scenarios)} uniform scenarios")
        return scenarios
    
    def generate_gaussian_scenarios(self,
                                   subjects: Optional[List[str]] = None,
                                   n_scenarios: int = 50,
                                   mean: float = 5.0,
                                   std_dev: float = 2.0) -> List[Dict[str, Any]]:
        """
        Generate scenarios with Gaussian distribution of ratings.
        
        Args:
            subjects: List of subjects to use. If None, uses default subjects.
            n_scenarios: Total number of scenarios to generate
            mean: Mean of the Gaussian distribution
            std_dev: Standard deviation of the Gaussian distribution
            
        Returns:
            List of scenario dictionaries
        """
        if subjects is None:
            subjects = self.default_subjects
        
        # Calculate number of scenarios per subject
        n_per_subject = n_scenarios // len(subjects)
        
        # Generate Gaussian distributed ratings
        scenarios = []
        
        for subject in subjects:
            # Generate ratings for this subject
            ratings = np.random.normal(mean, std_dev, n_per_subject)
            
            # Clip to valid range (0-10)
            ratings = np.clip(ratings, 0.0, 10.0)
            
            for rating in ratings:
                scenarios.append({
                    "subject": subject,
                    "rating": float(rating)
                })
        
        logger.info(f"Generated {len(scenarios)} gaussian scenarios (mean={mean}, std={std_dev})")
        return scenarios
    
    def generate_edge_case_scenarios(self,
                                    subjects: Optional[List[str]] = None,
                                    n_per_edge: int = 5) -> List[Dict[str, Any]]:
        """
        Generate edge case scenarios with ratings at extremes and boundaries.
        
        Args:
            subjects: List of subjects to use. If None, uses default subjects.
            n_per_edge: Number of scenarios to generate per edge case
            
        Returns:
            List of scenario dictionaries
        """
        if subjects is None:
            subjects = self.default_subjects
        
        # Define edge cases
        edge_ratings = [
            # Extreme low
            0.0, 0.1, 0.5,
            # Boundaries
            2.0, 3.0, 5.0, 7.0, 8.0,
            # Extreme high
            9.5, 9.9, 10.0
        ]
        
        # Generate scenarios
        scenarios = []
        
        for subject in subjects:
            for rating in edge_ratings:
                for _ in range(n_per_edge):
                    # Add small jitter to avoid exact duplicates
                    if rating not in [0.0, 10.0]:  # Don't jitter absolute extremes
                        jittered_rating = max(0.0, min(10.0, rating + random.uniform(-0.1, 0.1)))
                    else:
                        jittered_rating = rating
                        
                    scenarios.append({
                        "subject": subject,
                        "rating": jittered_rating
                    })
        
        logger.info(f"Generated {len(scenarios)} edge case scenarios")
        return scenarios
    
    def generate_comparative_scenarios(self,
                                      subjects: Optional[List[str]] = None,
                                      rating_pairs: Optional[List[List[float]]] = None,
                                      n_per_pair: int = 3) -> List[Dict[str, Any]]:
        """
        Generate scenarios with similar ratings for direct comparison.
        
        Args:
            subjects: List of subjects to use. If None, uses default subjects.
            rating_pairs: List of rating pairs for comparison. If None, uses default pairs.
            n_per_pair: Number of scenarios to generate per rating pair
            
        Returns:
            List of scenario dictionaries
        """
        if subjects is None:
            subjects = self.default_subjects
            
        if rating_pairs is None:
            # Default pairs for comparison (close ratings)
            rating_pairs = [
                [1.0, 2.0],  # Low range comparison
                [3.0, 4.0],  # Low-mid range comparison
                [4.5, 5.5],  # Mid range comparison
                [6.0, 7.0],  # Mid-high range comparison
                [8.0, 9.0]   # High range comparison
            ]
        
        # Generate scenarios
        scenarios = []
        
        for subject in subjects:
            for pair in rating_pairs:
                for rating in pair:
                    for _ in range(n_per_pair):
                        # Add small jitter to avoid exact duplicates
                        jittered_rating = max(0.0, min(10.0, rating + random.uniform(-0.2, 0.2)))
                        
                        scenarios.append({
                            "subject": subject,
                            "rating": jittered_rating
                        })
        
        logger.info(f"Generated {len(scenarios)} comparative scenarios")
        return scenarios
    
    def generate_subject_focused_scenarios(self,
                                         subject_ratings: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        Generate scenarios with specific rating distributions per subject.
        
        Args:
            subject_ratings: Dictionary mapping subjects to lists of ratings
            
        Returns:
            List of scenario dictionaries
        """
        # Generate scenarios
        scenarios = []
        
        for subject, ratings in subject_ratings.items():
            for rating in ratings:
                scenarios.append({
                    "subject": subject,
                    "rating": rating
                })
        
        logger.info(f"Generated {len(scenarios)} subject-focused scenarios")
        return scenarios
    
    def generate_comprehensive_test_set(self,
                                       subjects: Optional[List[str]] = None,
                                       total_scenarios: int = 100) -> List[Dict[str, Any]]:
        """
        Generate a comprehensive test set with a mix of different scenario types.
        
        Args:
            subjects: List of subjects to use. If None, uses default subjects.
            total_scenarios: Target number of total scenarios
            
        Returns:
            List of scenario dictionaries
        """
        if subjects is None:
            subjects = self.default_subjects
        
        # Allocate proportions to different generation strategies
        uniform_prop = 0.3
        gaussian_prop = 0.3
        edge_prop = 0.2
        comparative_prop = 0.2
        
        # Calculate counts for each strategy
        uniform_count = int(total_scenarios * uniform_prop)
        gaussian_count = int(total_scenarios * gaussian_prop)
        edge_count = int(total_scenarios * edge_prop)
        comparative_count = total_scenarios - uniform_count - gaussian_count - edge_count
        
        # Generate scenarios using each strategy
        uniform_scenarios = self.generate_uniform_scenarios(
            subjects=subjects,
            n_ratings=max(5, uniform_count // len(subjects))
        )
        
        gaussian_scenarios = self.generate_gaussian_scenarios(
            subjects=subjects,
            n_scenarios=gaussian_count
        )
        
        edge_scenarios = self.generate_edge_case_scenarios(
            subjects=subjects,
            n_per_edge=max(1, edge_count // (11 * len(subjects)))
        )
        
        comparative_scenarios = self.generate_comparative_scenarios(
            subjects=subjects,
            n_per_pair=max(1, comparative_count // (10 * len(subjects)))
        )
        
        # Combine scenarios
        all_scenarios = (
            uniform_scenarios[:uniform_count] +
            gaussian_scenarios[:gaussian_count] +
            edge_scenarios[:edge_count] +
            comparative_scenarios[:comparative_count]
        )
        
        # Shuffle to mix different strategies
        random.shuffle(all_scenarios)
        
        logger.info(f"Generated comprehensive test set with {len(all_scenarios)} scenarios")
        return all_scenarios
    
    def save_scenarios(self, 
                      scenarios: List[Dict[str, Any]], 
                      filename: str) -> str:
        """
        Save scenarios to a file.
        
        Args:
            scenarios: List of scenario dictionaries
            filename: Name of the file to save (without directory)
            
        Returns:
            Path to the saved file
        """
        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(scenarios, f, indent=2)
            
        logger.info(f"Saved {len(scenarios)} scenarios to {filepath}")
        return filepath
    
    def load_scenarios(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load scenarios from a file.
        
        Args:
            filepath: Path to the scenario file
            
        Returns:
            List of scenario dictionaries
        """
        with open(filepath, 'r') as f:
            scenarios = json.load(f)
            
        logger.info(f"Loaded {len(scenarios)} scenarios from {filepath}")
        return scenarios
    
    def analyze_scenario_distribution(self, 
                                     scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of ratings and subjects in a set of scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        # Extract ratings and subjects
        ratings = [s["rating"] for s in scenarios]
        subjects = [s["subject"] for s in scenarios]
        
        # Count scenarios per subject
        subject_counts = defaultdict(int)
        for subject in subjects:
            subject_counts[subject] += 1
        
        # Calculate rating distribution by subject
        rating_by_subject = defaultdict(list)
        for scenario in scenarios:
            rating_by_subject[scenario["subject"]].append(scenario["rating"])
        
        subject_stats = {}
        for subject, subject_ratings in rating_by_subject.items():
            subject_stats[subject] = {
                "count": len(subject_ratings),
                "mean": np.mean(subject_ratings),
                "std": np.std(subject_ratings),
                "min": min(subject_ratings),
                "max": max(subject_ratings),
                "median": np.median(subject_ratings)
            }
        
        # Calculate rating distribution overall
        rating_stats = {
            "count": len(ratings),
            "mean": np.mean(ratings),
            "std": np.std(ratings),
            "min": min(ratings),
            "max": max(ratings),
            "median": np.median(ratings)
        }
        
        # Calculate rating distribution by bin
        bin_edges = np.arange(0, 11, 1)
        hist, _ = np.histogram(ratings, bins=bin_edges)
        rating_bins = {f"{bin_edges[i]}-{bin_edges[i+1]}": int(hist[i]) 
                      for i in range(len(hist))}
        
        # Assemble analysis results
        return {
            "total_scenarios": len(scenarios),
            "subject_distribution": dict(subject_counts),
            "rating_stats": rating_stats,
            "subject_stats": subject_stats,
            "rating_bins": rating_bins
        }

def main():
    """Run the scenario generator with command line arguments."""
    parser = argparse.ArgumentParser(description="Generate test scenarios for the teacher-tester system.")
    parser.add_argument("--output-dir", type=str, default="data/scenarios",
                      help="Directory to save scenario files")
    parser.add_argument("--subjects", type=str, nargs='+',
                      help="Space-separated list of subjects")
    parser.add_argument("--count", type=int, default=100,
                      help="Number of scenarios to generate")
    parser.add_argument("--type", type=str, choices=["uniform", "gaussian", "edge", "comparative", "comprehensive"],
                      default="comprehensive", help="Type of scenario distribution to generate")
    parser.add_argument("--mean", type=float, default=5.0,
                      help="Mean rating for Gaussian distribution")
    parser.add_argument("--std", type=float, default=2.0,
                      help="Standard deviation for Gaussian distribution")
    parser.add_argument("--filename", type=str, 
                      help="Output filename (default: auto-generated based on type)")
    
    args = parser.parse_args()
    
    # Create generator
    generator = ScenarioGenerator(output_dir=args.output_dir)
    
    # Generate scenarios based on type
    if args.type == "uniform":
        scenarios = generator.generate_uniform_scenarios(
            subjects=args.subjects,
            n_ratings=args.count // (len(args.subjects) if args.subjects else len(generator.default_subjects))
        )
    elif args.type == "gaussian":
        scenarios = generator.generate_gaussian_scenarios(
            subjects=args.subjects,
            n_scenarios=args.count,
            mean=args.mean,
            std_dev=args.std
        )
    elif args.type == "edge":
        scenarios = generator.generate_edge_case_scenarios(
            subjects=args.subjects,
            n_per_edge=args.count // (11 * (len(args.subjects) if args.subjects else len(generator.default_subjects)))
        )
    elif args.type == "comparative":
        scenarios = generator.generate_comparative_scenarios(
            subjects=args.subjects,
            n_per_pair=args.count // (10 * (len(args.subjects) if args.subjects else len(generator.default_subjects)))
        )
    else:  # comprehensive
        scenarios = generator.generate_comprehensive_test_set(
            subjects=args.subjects,
            total_scenarios=args.count
        )
    
    # Generate filename if not provided
    if args.filename is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        args.filename = f"{args.type}_scenarios_{timestamp}.json"
    
    # Save scenarios
    generator.save_scenarios(scenarios, args.filename)
    
    # Analyze distribution
    analysis = generator.analyze_scenario_distribution(scenarios)
    
    # Print summary
    print("\nScenario Generation Summary:")
    print(f"Total scenarios: {analysis['total_scenarios']}")
    print(f"Subjects: {', '.join(analysis['subject_distribution'].keys())}")
    print(f"Rating distribution: mean={analysis['rating_stats']['mean']:.2f}, "
          f"std={analysis['rating_stats']['std']:.2f}, "
          f"range=[{analysis['rating_stats']['min']:.1f}, {analysis['rating_stats']['max']:.1f}]")
    
    print("\nScenarios per subject:")
    for subject, count in analysis['subject_distribution'].items():
        print(f"  {subject}: {count}")
    
    print("\nRating bins:")
    for bin_name, count in analysis['rating_bins'].items():
        print(f"  {bin_name}: {count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())