#!/usr/bin/env python
"""
Simplified launcher script for parameter efficiency optimization.
This version has minimal dependencies to avoid compatibility issues.
"""
import os
import argparse
import sys
import time

def main():
    """Run the simplified parameter efficiency optimization."""
    parser = argparse.ArgumentParser(description="Parameter Efficiency Optimization")
    
    parser.add_argument("--output-dir", type=str, default="data/optimized_models",
                      help="Directory for outputs")
    parser.add_argument("--storage-dir", type=str, default="data/parameter_storage",
                      help="Directory for parameter storage")
    parser.add_argument("--segments", type=int, default=3,
                      help="Number of rating segments")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8,
                      help="Quantization bits")
    
    args = parser.parse_args()
    
    # Import simple_optimize module
    try:
        from teacher_tester.optimization.simple_optimize import optimize_model
        import torch.nn as nn
        
        # Create a simple model for demonstration
        model = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
        # Add config attribute for compatibility
        model.config = type('obj', (object,), {
            'hidden_size': 768
        })
        
        print(f"Running simplified optimization with {args.segments} segments and {args.bits}-bit quantization...")
        
        # Run optimization
        results = optimize_model(
            model=model,
            num_segments=args.segments,
            bits=args.bits,
            output_dir=args.output_dir,
            storage_dir=args.storage_dir
        )
        
        # Print summary
        print("\nOptimization Results:")
        print(f"Created {len(results['segments'])} parameter segments")
        print(f"Original model size: {results['savings']['original_size_mb']:.2f} MB")
        print(f"Optimized size: {results['savings']['optimized_size_mb']:.2f} MB")
        print(f"Compression ratio: {results['savings']['compression_ratio']:.2f}x")
        
        return 0
    
    except ImportError as e:
        print(f"Error importing modules: {str(e)}")
        print("\nPlease ensure you have the following dependencies installed:")
        print("  pip install torch numpy")
        return 1
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
        return 1

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    
    sys.exit(exit_code)