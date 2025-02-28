#!/usr/bin/env python
"""
Launcher script for the teacher-tester system.
This script provides a simplified interface to run different components.
"""
import os
import argparse
import sys
import time

def main():
    """Run the teacher-tester system launcher."""
    parser = argparse.ArgumentParser(description="Teacher-Tester System Launcher")
    
    # Action argument
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--single-test", action="store_true",
                             help="Run a single test conversation")
    action_group.add_argument("--batch", action="store_true",
                             help="Run a batch of conversations")
    action_group.add_argument("--generate-scenarios", action="store_true",
                             help="Generate test scenarios")
    action_group.add_argument("--end-to-end", action="store_true",
                             help="Run end-to-end process")
    action_group.add_argument("--dashboard", action="store_true",
                             help="Launch the visualization dashboard")
    action_group.add_argument("--visualize", action="store_true",
                             help="Generate static visualizations")
    
    # General arguments
    parser.add_argument("--subjects", type=str, nargs='+',
                      help="Space-separated list of subjects")
    parser.add_argument("--output-dir", type=str, default="data/output",
                      help="Base directory for outputs")
    
    # Single test arguments
    parser.add_argument("--subject", type=str, default="Python Programming",
                      help="Subject for single test")
    parser.add_argument("--rating", type=float, default=7.5,
                      help="True rating for single test")
    parser.add_argument("--verbose", action="store_true",
                      help="Show verbose output for single test")
    
    # Batch arguments
    parser.add_argument("--scenario-file", type=str,
                      help="Path to scenario file for batch run")
    parser.add_argument("--num-scenarios", type=int, default=50,
                      help="Number of scenarios for batch run or generation")
    parser.add_argument("--max-workers", type=int, default=1,
                      help="Maximum number of parallel workers")
    
    # Scenario generation arguments
    parser.add_argument("--scenario-type", type=str, 
                      choices=["uniform", "gaussian", "edge", "comparative", "comprehensive"],
                      default="comprehensive", help="Type of scenario distribution to generate")
    
    args = parser.parse_args()
    
    # Handle different actions
    if args.single_test:
        print(f"Running single test conversation on {args.subject} with rating {args.rating}")
        from teacher_tester.test_conversation import main as test_main
        sys.argv = [
            "test_conversation.py",
            "--subject", args.subject,
            "--rating", str(args.rating)
        ]
        if args.verbose:
            sys.argv.append("--verbose")
        test_main()
        
    elif args.batch:
        print(f"Running batch of conversations")
        from teacher_tester.batch_runner import main as batch_main
        
        # Prepare arguments
        batch_args = ["batch_runner.py"]
        
        if args.scenario_file:
            print(f"Using scenario file: {args.scenario_file}")
            batch_args.extend(["--scenario-file", args.scenario_file])
        elif args.subjects:
            batch_args.extend(["--subjects"] + args.subjects)
        
        batch_args.extend([
            "--num-scenarios", str(args.num_scenarios),
            "--output-dir", args.output_dir,
            "--workers", str(args.max_workers)
        ])
        
        sys.argv = batch_args
        batch_main()
        
    elif args.generate_scenarios:
        print(f"Generating {args.num_scenarios} {args.scenario_type} scenarios")
        from teacher_tester.test_generation.scenario_generator import main as generator_main
        
        # Prepare arguments
        gen_args = ["scenario_generator.py"]
        
        if args.subjects:
            gen_args.extend(["--subjects"] + args.subjects)
            
        gen_args.extend([
            "--count", str(args.num_scenarios),
            "--type", args.scenario_type,
            "--output-dir", os.path.join(args.output_dir, "scenarios")
        ])
        
        sys.argv = gen_args
        generator_main()
        
    elif args.end_to_end:
        print("Running end-to-end process")
        from teacher_tester.orchestrator import main as orchestrator_main
        
        # Prepare arguments
        e2e_args = ["orchestrator.py"]
        
        if args.subjects:
            e2e_args.extend(["--subjects"] + args.subjects)
            
        e2e_args.extend([
            "--scenario-type", args.scenario_type,
            "--num-scenarios", str(args.num_scenarios),
            "--max-workers", str(args.max_workers),
            "--output-dir", args.output_dir
        ])
        
        sys.argv = e2e_args
        orchestrator_main()
        
    elif args.dashboard:
        print("Launching visualization dashboard")
        from teacher_tester.visualization.dashboard import run_dashboard
        from teacher_tester.data.storage import get_storage
        
        # Check if we have any conversations
        storage = get_storage()
        conversation_ids = storage.list_conversations()
        
        if not conversation_ids:
            print("No conversation data found. Please run some conversations first.")
            print("You can run a test conversation with: --single-test")
            print("Or run a batch with: --batch")
            return 1
        
        dashboard_dir = os.path.join(args.output_dir, "dashboard")
        if not os.path.exists(dashboard_dir):
            # Create dashboard data if it doesn't exist
            print("Dashboard data not found. Creating dashboard data...")
            from teacher_tester.visualization.dashboard import create_dashboard_data
            create_dashboard_data(output_dir=dashboard_dir)
            
        run_dashboard(data_dir=dashboard_dir)
        
    elif args.visualize:
        print("Generating static visualizations")
        from teacher_tester.visualization.dashboard import create_static_plots, create_dashboard_data
        
        dashboard_dir = os.path.join(args.output_dir, "dashboard")
        visualizations_dir = os.path.join(args.output_dir, "visualizations")
        
        # Create dashboard data if needed
        if not os.path.exists(dashboard_dir) or not os.listdir(dashboard_dir):
            print("Creating dashboard data...")
            create_dashboard_data(output_dir=dashboard_dir)
            
        # Create visualizations
        plots = create_static_plots(
            data_dir=dashboard_dir,
            output_dir=visualizations_dir
        )
        
        print(f"Created {len(plots)} visualization plots in {visualizations_dir}")
    
    return 0

if __name__ == "__main__":
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")