"""
End-to-end orchestration of the teacher-tester system.
"""
import os
import argparse
import sys
import time
from typing import List, Dict, Any, Optional
import json
import pandas as pd

from teacher_tester.test_generation.scenario_generator import ScenarioGenerator
from teacher_tester.batch_runner import run_batch, generate_batch_scenarios
from teacher_tester.evaluation.enhanced_metrics import generate_evaluation_report
from teacher_tester.analysis.conversation_analyzer import ConversationAnalyzer
from teacher_tester.visualization.dashboard import create_dashboard_data, create_static_plots
from teacher_tester.data.storage import get_storage
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

def run_end_to_end(
    scenario_type: str = "comprehensive",
    num_scenarios: int = 50,
    subjects: Optional[List[str]] = None,
    max_workers: int = 1,
    output_dir: str = "data/output",
    run_visualizations: bool = True,
    create_dashboard: bool = True
) -> Dict[str, Any]:
    """
    Run the entire teacher-tester pipeline from scenario generation to analysis.
    
    Args:
        scenario_type: Type of scenarios to generate
        num_scenarios: Number of scenarios to generate
        subjects: List of subjects to use (None for defaults)
        max_workers: Maximum number of parallel workers
        output_dir: Base directory for outputs
        run_visualizations: Whether to create visualizations
        create_dashboard: Whether to create dashboard data
        
    Returns:
        Dictionary with paths to generated files
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    scenarios_dir = os.path.join(output_dir, "scenarios")
    reports_dir = os.path.join(output_dir, "reports")
    visualizations_dir = os.path.join(output_dir, "visualizations")
    dashboard_dir = os.path.join(output_dir, "dashboard")
    
    os.makedirs(scenarios_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(dashboard_dir, exist_ok=True)
    
    output_files = {}
    
    # 1. Generate scenarios
    logger.info(f"Generating {num_scenarios} scenarios with type '{scenario_type}'")
    generator = ScenarioGenerator(output_dir=scenarios_dir)
    
    if scenario_type == "uniform":
        scenarios = generator.generate_uniform_scenarios(
            subjects=subjects,
            n_ratings=num_scenarios // (len(subjects) if subjects else len(generator.default_subjects))
        )
    elif scenario_type == "gaussian":
        scenarios = generator.generate_gaussian_scenarios(
            subjects=subjects,
            n_scenarios=num_scenarios
        )
    elif scenario_type == "edge":
        scenarios = generator.generate_edge_case_scenarios(
            subjects=subjects,
            n_per_edge=max(1, num_scenarios // (11 * (len(subjects) if subjects else len(generator.default_subjects))))
        )
    elif scenario_type == "comparative":
        scenarios = generator.generate_comparative_scenarios(
            subjects=subjects,
            n_per_pair=max(1, num_scenarios // (10 * (len(subjects) if subjects else len(generator.default_subjects))))
        )
    else:  # comprehensive
        scenarios = generator.generate_comprehensive_test_set(
            subjects=subjects,
            total_scenarios=num_scenarios
        )
    
    # Save scenarios
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    scenarios_file = f"{scenario_type}_scenarios_{timestamp}.json"
    scenarios_path = generator.save_scenarios(scenarios, scenarios_file)
    output_files["scenarios"] = scenarios_path
    
    # Analyze scenario distribution
    analysis = generator.analyze_scenario_distribution(scenarios)
    analysis_path = os.path.join(scenarios_dir, f"scenario_analysis_{timestamp}.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    output_files["scenario_analysis"] = analysis_path
    
    # 2. Run batch of conversations
    logger.info(f"Running batch of {len(scenarios)} conversations with {max_workers} workers")
    start_time = time.time()
    
    results_df = run_batch(
        scenarios=scenarios,
        output_dir=os.path.join(output_dir, "batch_results"),
        max_workers=max_workers,
        save_training_examples=True
    )
    
    execution_time = time.time() - start_time
    logger.info(f"Completed batch run in {execution_time:.2f} seconds")
    
    # Get conversation IDs
    storage = get_storage()
    conversation_ids = storage.list_conversations()
    
    # Check if we have successful conversations
    if not conversation_ids:
        logger.warning("No conversations were successfully completed. Skipping evaluation steps.")
        return output_files
    
    # 3. Generate evaluation report
    logger.info("Generating evaluation report")
    evaluation_path = generate_evaluation_report(
        conversation_ids=conversation_ids,
        output_dir=reports_dir
    )
    output_files["evaluation_report"] = evaluation_path
    
    # 3. Generate evaluation report
    logger.info("Generating evaluation report")
    evaluation_path = generate_evaluation_report(
        conversation_ids=conversation_ids,
        output_dir=reports_dir
    )
    output_files["evaluation_report"] = evaluation_path
    
    # 4. Generate conversation analysis
    logger.info("Generating conversation analysis")
    analyzer = ConversationAnalyzer()
    analysis_path = os.path.join(reports_dir, f"conversation_analysis_{timestamp}.json")
    analysis_report = analyzer.generate_insights_report(analysis_path)
    output_files["conversation_analysis"] = analysis_path
    
    # 5. Create visualizations
    if run_visualizations:
        logger.info("Creating dashboard data")
        dashboard_files = create_dashboard_data(output_dir=dashboard_dir)
        output_files["dashboard_data"] = dashboard_files
        
        logger.info("Creating static visualizations")
        visualization_paths = create_static_plots(
            data_dir=dashboard_dir,
            output_dir=visualizations_dir
        )
        output_files["visualizations"] = visualization_paths
    
    # Print summary
    logger.info("End-to-end run completed successfully")
    logger.info(f"Total conversations: {len(conversation_ids)}")
    
    if results_df is not None and not results_df.empty:
        logger.info(f"Mean absolute error: {results_df['absolute_error'].mean():.2f}")
        logger.info(f"Mean final confidence: {results_df['final_confidence'].mean():.2f}")
    
    return output_files

def main():
    """Run the orchestrator with command line arguments."""
    parser = argparse.ArgumentParser(description="Run the teacher-tester system end-to-end.")
    parser.add_argument("--scenario-type", type=str, 
                      choices=["uniform", "gaussian", "edge", "comparative", "comprehensive"],
                      default="comprehensive", help="Type of scenario distribution to generate")
    parser.add_argument("--num-scenarios", type=int, default=50,
                      help="Number of scenarios to generate")
    parser.add_argument("--subjects", type=str, nargs='+',
                      help="Space-separated list of subjects")
    parser.add_argument("--max-workers", type=int, default=1,
                      help="Maximum number of parallel workers")
    parser.add_argument("--output-dir", type=str, default="data/output",
                      help="Base directory for outputs")
    parser.add_argument("--skip-visualizations", action="store_true",
                      help="Skip generating visualizations")
    parser.add_argument("--run-dashboard", action="store_true",
                      help="Launch the dashboard after completion")
    
    args = parser.parse_args()
    
    # Run end-to-end process
    output_files = run_end_to_end(
        scenario_type=args.scenario_type,
        num_scenarios=args.num_scenarios,
        subjects=args.subjects,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        run_visualizations=not args.skip_visualizations
    )
    
    # Print output file locations
    print("\nOutput Files:")
    for file_type, file_path in output_files.items():
        if isinstance(file_path, list):
            print(f"  {file_type}:")
            for path in file_path:
                print(f"    - {path}")
        else:
            print(f"  {file_type}: {file_path}")
    
    # Launch dashboard if requested
    if args.run_dashboard and not args.skip_visualizations:
        print("\nLaunching dashboard...")
        from teacher_tester.visualization.dashboard import run_dashboard
        dashboard_dir = os.path.join(args.output_dir, "dashboard")
        run_dashboard(data_dir=dashboard_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())