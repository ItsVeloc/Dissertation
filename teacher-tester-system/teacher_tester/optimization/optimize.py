"""
Simplified parameter efficiency optimization for teacher-tester system.
Designed to have minimal dependencies to avoid compatibility issues.
"""
import os
import sys
import json
import argparse
import torch
from torch import nn
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optimize")

#######################################
# Basic LoRA Implementation
#######################################

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        bias: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA components
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear output
        orig_output = self.linear(x)
        
        # LoRA path
        lora_output = self.lora_up(self.dropout(self.lora_down(x))) * self.scaling
        
        # Combined output
        return orig_output + lora_output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> 'LoRALinear':
        """Create a LoRA layer from an existing linear layer."""
        lora_layer = cls(
            linear.in_features, 
            linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None
        )
        
        # Copy original weights
        lora_layer.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and lora_layer.linear.bias is not None:
            lora_layer.linear.bias.data.copy_(linear.bias.data)
            
        return lora_layer

def apply_lora_to_model(model: nn.Module, r: int = 8, alpha: float = 16.0, target_modules: List[str] = None) -> Dict[str, nn.Module]:
    """
    Apply LoRA to specific layers in a model.
    
    Args:
        model: PyTorch model
        r: LoRA rank
        alpha: LoRA alpha scaling
        target_modules: List of module names to target (None for all linear layers)
        
    Returns:
        Dictionary mapping module names to LoRA layers
    """
    lora_layers = {}
    
    if target_modules is None:
        # Default to all linear layers
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                target_modules.append(name)
    
    # Create LoRA layers
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, nn.Linear):
            # Create LoRA layer
            lora_layer = LoRALinear.from_linear(module, r=r, alpha=alpha)
            
            # Replace in model
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else model.get_submodule(parent_name)
            setattr(parent, child_name, lora_layer)
            
            # Store reference
            lora_layers[name] = lora_layer
            
            logger.info(f"Applied LoRA to {name} with rank {r}")
    
    return lora_layers

def save_lora_weights(model: nn.Module, lora_layers: Dict[str, nn.Module], filepath: str):
    """
    Save LoRA weights to a file.
    
    Args:
        model: The model with LoRA layers
        lora_layers: Dictionary mapping names to LoRA layers
        filepath: Path to save weights
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Collect LoRA weights
    state_dict = {}
    for name, layer in lora_layers.items():
        if isinstance(layer, LoRALinear):
            state_dict[f"{name}.lora_down.weight"] = layer.lora_down.weight.data
            state_dict[f"{name}.lora_up.weight"] = layer.lora_up.weight.data
    
    # Save weights
    torch.save(state_dict, filepath)
    logger.info(f"Saved LoRA weights to {filepath}")

def load_lora_weights(model: nn.Module, filepath: str, target_modules: List[str] = None) -> Dict[str, nn.Module]:
    """
    Load LoRA weights from a file.
    
    Args:
        model: The model to apply LoRA to
        filepath: Path to LoRA weights
        target_modules: List of module names to target
        
    Returns:
        Dictionary mapping module names to LoRA layers
    """
    # Load weights
    state_dict = torch.load(filepath)
    
    # Extract layer names and ranks
    lora_info = {}
    for key in state_dict.keys():
        if ".lora_down.weight" in key:
            name = key.replace(".lora_down.weight", "")
            shape = state_dict[key].shape
            lora_info[name] = {"r": shape[0]}
    
    # If no target modules specified, use all from weights
    if target_modules is None:
        target_modules = list(lora_info.keys())
    
    # Apply LoRA to targeted modules
    lora_layers = {}
    for name in target_modules:
        if name in lora_info:
            # Get module
            try:
                module = model.get_submodule(name)
                if isinstance(module, nn.Linear):
                    # Create LoRA layer
                    lora_layer = LoRALinear.from_linear(
                        module, 
                        r=lora_info[name]["r"]
                    )
                    
                    # Load weights
                    lora_layer.lora_down.weight.data.copy_(state_dict[f"{name}.lora_down.weight"])
                    lora_layer.lora_up.weight.data.copy_(state_dict[f"{name}.lora_up.weight"])
                    
                    # Replace in model
                    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = model if parent_name == "" else model.get_submodule(parent_name)
                    setattr(parent, child_name, lora_layer)
                    
                    # Store reference
                    lora_layers[name] = lora_layer
                    
                    logger.info(f"Loaded LoRA weights for {name}")
            except AttributeError:
                logger.warning(f"Module {name} not found in model")
    
    return lora_layers

#######################################
# Simple Quantization
#######################################

def quantize_model(model: nn.Module, bits: int = 8, exclude: List[str] = None) -> nn.Module:
    """
    Apply simple quantization to a model.
    
    Args:
        model: PyTorch model
        bits: Quantization bits (4 or 8)
        exclude: List of module names to exclude
        
    Returns:
        Quantized model
    """
    if bits != 8 and bits != 4:
        raise ValueError("Only 8-bit and 4-bit quantization supported")
    
    if exclude is None:
        exclude = []
    
    # Simple static quantization
    with torch.no_grad():
        for name, module in model.named_modules():
            # Skip excluded modules
            if any(excluded in name for excluded in exclude):
                continue
                
            # Quantize weights in linear layers
            if isinstance(module, nn.Linear):
                # Get weight
                weight = module.weight.data
                
                if bits == 8:
                    # 8-bit quantization
                    
                    # Compute scale and zero point with safety checks
                    w_min, w_max = weight.min().item(), weight.max().item()
                    
                    # Prevent division by zero by ensuring a minimum range
                    weight_range = max(w_max - w_min, 1e-5)
                    scale = weight_range / 255
                    zero_point = round(-w_min / scale) if scale != 0 else 0
                    
                    # Quantize
                    weight_q = torch.round(weight / scale + zero_point).clamp(0, 255).to(torch.uint8)
                    
                    # Store quantization parameters
                    module.register_buffer('weight_scale', torch.tensor(scale))
                    module.register_buffer('weight_zero_point', torch.tensor(zero_point))
                    module.register_buffer('weight_quantized', weight_q)
                    
                    # Override forward method
                    original_forward = module.forward
                    
                    def quantized_forward(self, x):
                        # Dequantize weight
                        weight_dq = (self.weight_quantized.float() - self.weight_zero_point) * self.weight_scale
                        return nn.functional.linear(x, weight_dq, self.bias)
                    
                    module.forward = quantized_forward.__get__(module)
                    
                    logger.info(f"Applied 8-bit quantization to {name}")
                    
                elif bits == 4:
                    # 4-bit quantization (simplified)
                    
                    # Compute scale and zero point with safety checks
                    w_min, w_max = weight.min().item(), weight.max().item()
                    
                    # Prevent division by zero by ensuring a minimum range
                    weight_range = max(w_max - w_min, 1e-5)
                    scale = weight_range / 15
                    zero_point = round(-w_min / scale) if scale != 0 else 0
                    
                    # Quantize
                    weight_q = torch.round(weight / scale + zero_point).clamp(0, 15).to(torch.uint8)
                    
                    # Store quantization parameters
                    module.register_buffer('weight_scale', torch.tensor(scale))
                    module.register_buffer('weight_zero_point', torch.tensor(zero_point))
                    module.register_buffer('weight_quantized', weight_q)
                    
                    # Override forward method
                    original_forward = module.forward
                    
                    def quantized_forward(self, x):
                        # Dequantize weight
                        weight_dq = (self.weight_quantized.float() - self.weight_zero_point) * self.weight_scale
                        return nn.functional.linear(x, weight_dq, self.bias)
                    
                    module.forward = quantized_forward.__get__(module)
                    
                    logger.info(f"Applied 4-bit quantization to {name}")
    
    return model

#######################################
# Parameter Storage
#######################################

class ParameterStorage:
    """Simple parameter storage for different rating ranges."""
    
    def __init__(self, storage_dir: str):
        """
        Initialize parameter storage.
        
        Args:
            storage_dir: Directory to store parameters
        """
        self.storage_dir = storage_dir
        self.segments = {}
        
        # Create directory if needed
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load segment info if exists
        self._load_segments()
    
    def _load_segments(self):
        """Load segment information from storage directory."""
        index_path = os.path.join(self.storage_dir, "segments.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.segments = json.load(f)
                logger.info(f"Loaded {len(self.segments)} segments from {index_path}")
    
    def _save_segments(self):
        """Save segment information to storage directory."""
        index_path = os.path.join(self.storage_dir, "segments.json")
        with open(index_path, 'w') as f:
            json.dump(self.segments, f, indent=2)
            logger.info(f"Saved {len(self.segments)} segments to {index_path}")
    
    def add_segment(self, min_rating: float, max_rating: float, model_type: str = "lora", **metadata) -> str:
        """
        Add a new segment.
        
        Args:
            min_rating: Minimum rating
            max_rating: Maximum rating
            model_type: Type of model ("lora" or "full")
            **metadata: Additional metadata
            
        Returns:
            Segment ID
        """
        # Create segment ID
        segment_id = f"segment_{min_rating:.1f}_{max_rating:.1f}_{model_type}"
        
        # Create segment
        self.segments[segment_id] = {
            "min_rating": min_rating,
            "max_rating": max_rating,
            "model_type": model_type,
            "file_path": os.path.join(self.storage_dir, f"{segment_id}.pt"),
            "metadata": metadata
        }
        
        # Save segments
        self._save_segments()
        
        logger.info(f"Added segment {segment_id}")
        return segment_id
    
    def get_segment_for_rating(self, rating: float) -> Optional[str]:
        """
        Get segment ID for a rating.
        
        Args:
            rating: Rating to find segment for
            
        Returns:
            Segment ID or None if not found
        """
        for segment_id, info in self.segments.items():
            if info["min_rating"] <= rating < info["max_rating"]:
                return segment_id
        return None
    
    def load_segment(self, segment_id: str, model: nn.Module) -> nn.Module:
        """
        Load a segment into a model.
        
        Args:
            segment_id: Segment ID
            model: Model to load segment into
            
        Returns:
            Model with segment loaded
        """
        if segment_id not in self.segments:
            logger.error(f"Segment {segment_id} not found")
            return model
        
        segment = self.segments[segment_id]
        file_path = segment["file_path"]
        
        if not os.path.exists(file_path):
            logger.error(f"Segment file {file_path} not found")
            return model
        
        # Load based on model type
        if segment["model_type"] == "lora":
            # Load LoRA weights
            load_lora_weights(model, file_path)
        else:
            # Load full model
            model.load_state_dict(torch.load(file_path))
        
        logger.info(f"Loaded segment {segment_id}")
        return model

#######################################
# Main Functions
#######################################

def optimize_model(
    model: nn.Module,
    num_segments: int = 3,
    bits: int = 8,
    output_dir: str = "data/optimized_models",
    storage_dir: str = "data/parameter_storage"
) -> Dict[str, Any]:
    """
    Optimize a model with parameter efficiency techniques.
    
    Args:
        model: Model to optimize
        num_segments: Number of rating segments
        bits: Quantization bits
        output_dir: Output directory
        storage_dir: Storage directory
        
    Returns:
        Dictionary with optimization results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create storage
    storage = ParameterStorage(storage_dir)
    
    # Create rating segments
    rating_step = 10.0 / num_segments
    segments = []
    
    for i in range(num_segments):
        min_rating = i * rating_step
        max_rating = (i + 1) * rating_step
        
        # Determine LoRA rank based on segment position
        if i == num_segments // 2:
            # Middle segment - higher rank for more complex adaptation
            rank = 16
        else:
            # Edge segments - lower rank is sufficient
            rank = 8
        
        # Create segment
        segment_id = storage.add_segment(
            min_rating=min_rating,
            max_rating=max_rating,
            model_type="lora",
            lora_rank=rank,
            quantization_bits=bits
        )
        
        segments.append({
            "id": segment_id,
            "min_rating": min_rating,
            "max_rating": max_rating,
            "rank": rank
        })
        
        # Apply LoRA to model
        lora_layers = apply_lora_to_model(model, r=rank)
        
        # Apply quantization if needed
        if bits < 16:
            # Exclude embedding and output layers
            exclude = ["embedding", "embed", "classifier", "output"]
            quantize_model(model, bits=bits, exclude=exclude)
        
        # Save segment
        save_lora_weights(model, lora_layers, storage.segments[segment_id]["file_path"])
        
        logger.info(f"Optimized and saved segment {segment_id}")
    
    # Calculate memory savings
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Estimate LoRA size - use a fixed estimate if we can't calculate it dynamically
    try:
        # Count parameters that would be trainable in LoRA
        trainable_params = sum(p.numel() for p in model.parameters() if hasattr(p, 'requires_grad'))
        
        # If no trainable params, estimate based on total size
        if trainable_params == 0:
            trainable_params = 0.1 * sum(p.numel() for p in model.parameters())
            
        # Estimate LoRA size
        lora_params = 0
        for segment in segments:
            # For each LoRA layer, we store two matrices: down and up projection
            # Each is r * dim in size
            rank = segment["rank"]
            hidden_size = getattr(model.config, 'hidden_size', 768)  # Default if not defined
            lora_size = trainable_params * (2 * rank / hidden_size)
            lora_params += lora_size
        
        # Apply quantization factor
        lora_params = lora_params * (bits / 32)
    except Exception as e:
        logger.warning(f"Error calculating exact parameter counts: {e}. Using estimates.")
        # Fallback to simple estimate - typically LoRA is about 1-3% of model size
        lora_params = 0.02 * original_size * (bits / 32)
    
    # Ensure we don't divide by zero
    if lora_params <= 0:
        lora_params = 1  # Set a minimum value
    
    savings = {
        "original_size_mb": original_size / (1024 * 1024),
        "optimized_size_mb": lora_params / (1024 * 1024),
        "compression_ratio": original_size / lora_params,
        "segments": segments
    }
    
    # Save results
    results = {
        "num_segments": num_segments,
        "bits": bits,
        "segments": segments,
        "savings": savings
    }
    
    results_path = os.path.join(output_dir, "optimization_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved optimization results to {results_path}")
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Simple parameter efficiency optimization")
    
    parser.add_argument("--output-dir", type=str, default="data/optimized_models",
                      help="Directory for outputs")
    parser.add_argument("--storage-dir", type=str, default="data/parameter_storage",
                      help="Directory for parameter storage")
    parser.add_argument("--num-segments", type=int, default=3,
                      help="Number of rating segments")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8,
                      help="Quantization bits")
    
    args = parser.parse_args()
    
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 768)
    )
    
    # Add config attribute to make it compatible with our code
    model.config = type('obj', (object,), {
        'hidden_size': 768
    })
    
    # Run optimization
    results = optimize_model(
        model=model,
        num_segments=args.num_segments,
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

if __name__ == "__main__":
    start_time = time.time()
    sys.exit(main())