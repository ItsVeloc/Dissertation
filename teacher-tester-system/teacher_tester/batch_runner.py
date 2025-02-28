"""
Batch runner for multiple teacher-tester conversations.
"""
import os
import json
import argparse
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from teacher_tester.run_conversation import run_conversation
from teacher_tester.data.storage import get_storage
from teacher_tester.evaluation.metrics import evaluate_conversations, create_training_example
from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

def generate_batch_scenarios(
    subjects: Optional[List[str]] = None,
    ratings: Optional[List[float]] = None,
    n_scenarios: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate a batch of test scenarios.
    
    Args:
        subjects: List of subjects to use. If None, uses default subjects from config.
        ratings: List of specific ratings to use. If None, generates a spread.
        n_scenarios: Number of scenarios to generate if ratings not specified.
        
    Returns:
        List of scenario dictionaries
    """
    config = get_config()
    
    # Get subjects from config if not provided
    if subjects is None:
        subjects = list(config.get("conversation.initial_prompts", {}).keys())
        if not subjects:
            subjects = ["Python Programming", "Machine Learning", "Data Science", "Web Development"]
    
    # Generate rating spread if not provided
    if ratings is None:
        import numpy as np
        # Generate ratings across the full range (0-10)
        ratings = np.linspace(0.5, 9.5, n_scenarios).tolist()
    
    # Create scenarios with all combinations
    scenarios = []
    
    for subject in subjects:
        for rating in ratings:
            scenarios.append({
                "subject": subject,
                "rating": rating
            })
    
    logger.info(f"Generated {len(scenarios)} test scenarios across {len(subjects)} subjects")
    return scenarios

def run_batch(
    scenarios: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    max_workers: int = 1,
    save_training_examples: bool = True
) -> pd.DataFrame:
    """
    Run a batch of conversations based on scenarios.
    
    Args:
        scenarios: List of scenario dictionaries with 'subject' and 'rating'
        output_dir: Directory to save results. If None, uses default from config.
        max_workers: Maximum number of parallel workers (1 = sequential)
        save_training_examples: Whether to save training examples from conversations
        
    Returns:
        DataFrame with evaluation results
    """
    storage = get_storage()
    config = get_config()
    
    # Use default output dir if not specified
    if output_dir is None:
        output_dir = "data/batch_results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track conversation IDs
    conversation_ids = []
    
    # Run conversations
    logger.info(f"Running batch of {len(scenarios)} conversations" + 
               (f" with {max_workers} workers" if max_workers > 1 else " sequentially"))
    
    # Function to run a single conversation
    def run_single_scenario(scenario):
        try:
            conversation = run_conversation(
                subject=scenario["subject"],
                true_rating=scenario["rating"]
            )
            
            # Create and save training example if requested
            if save_training_examples:
                example = create_training_example(conversation)
                storage.save_training_example(example)
            
            return conversation.id
        except Exception as e:
            logger.error(f"Error running scenario {scenario}: {str(e)}")
            return None
    
    # Run conversations in parallel or sequentially
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_scenario = {
                executor.submit(run_single_scenario, scenario): scenario 
                for scenario in scenarios
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_scenario), total=len(scenarios)):
                scenario = future_to_scenario[future]
                try:
                    conv_id = future.result()
                    if conv_id:
                        conversation_ids.append(conv_id)
                except Exception as e:
                    logger.error(f"Error processing scenario {scenario}: {str(e)}")
    else:
        # Run sequentially with progress bar
        for scenario in tqdm(scenarios, desc="Running conversations"):
            conv_id = run_single_scenario(scenario)
            if conv_id:
                conversation_ids.append(conv_id)
    
    logger.info(f"Successfully completed {len(conversation_ids)} conversations")
    
    # Evaluate results
    results_df = evaluate_conversations(conversation_ids)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"batch_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save scenario information
    scenarios_path = os.path.join(output_dir, f"batch_scenarios_{timestamp}.json")
    with open(scenarios_path, 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    logger.info(f"Saved batch results to {results_path}")
    logger.info(f"Saved batch scenarios to {scenarios_path}")
    
    return results_df

def main():
    """Run a batch of conversations with command line arguments."""
    parser = argparse.ArgumentParser(description="Run a batch of teacher-tester conversations.")
    parser.add_argument("--subjects", type=str, nargs='+', 
                        help="Subjects to test (space-separated)")
    parser.add_argument("--min-rating", type=float, default=0.5, 
                        help="Minimum tester rating")
    parser.add_argument("--max-rating", type=float, default=9.5, 
                        help="Maximum tester rating")
    parser.add_argument("--num-ratings", type=int, default=10, 
                        help="Number of different ratings to test")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save results")
    parser.add_argument("--workers", type=int, default=1, 
                        help="Number of parallel workers (1 = sequential)")
    
    args = parser.parse_args()
    
    # Generate ratings
    import numpy as np
    ratings = np.linspace(args.min_rating, args.max_rating, args.num_ratings).tolist()
    
    # Generate scenarios
    scenarios = generate_batch_scenarios(
        subjects=args.subjects,
        ratings=ratings
    )
    
    # Run batch
    results_df = run_batch(
        scenarios=scenarios,
        output_dir=args.output_dir,
        max_workers=args.workers
    )
    
    # Print summary statistics
    print("\nBatch Run Complete")
    print(f"Total Conversations: {len(results_df)}")
    print(f"Mean Absolute Error: {results_df['absolute_error'].mean():.2f}")
    print(f"Mean Confidence: {results_df['final_confidence'].mean():.2f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())