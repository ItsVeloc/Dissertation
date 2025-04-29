"""
Implementation of Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
"""
import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import math
from pydantic import BaseModel
from transformers import PreTrainedModel

from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    r: int = 8  # Rank of the low-rank matrices
    alpha: float = 16.0  # Scaling factor for LoRA adaptation
    dropout: float = 0.05  # Dropout probability for LoRA layers
    bias: str = "none"  # Whether to train bias parameters: 'none', 'all', or 'lora_only'
    task_type: str = "CAUSAL_LM"  # Task type: 'CAUSAL_LM', 'SEQ_CLS', etc.
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])  # Modules to apply LoRA
    modules_to_save: List[str] = field(default_factory=list)  # Additional modules to save
    init_lora_weights: bool = True  # Whether to initialize LoRA weights
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary."""
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": self.target_modules,
            "modules_to_save": self.modules_to_save,
            "init_lora_weights": self.init_lora_weights
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """Create a LoRAConfig from a dictionary."""
        return cls(**config_dict)

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
        """
        Initialize a LoRA-adapted linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            r: Rank of the low-rank matrices
            alpha: Scaling factor
            dropout: Dropout probability
            bias: Whether to include bias parameter
        """
        super().__init__()
        
        # Store original dimensions and parameters
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Create A and B matrices for low-rank adaptation
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Create dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights if needed
        self.reset_parameters()
        
        # Create bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A to random weights with std of 1/sqrt(r)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize B to zero
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Apply LoRA adaptation: x @ (A × B) * scaling
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # Add bias if it exists
        if self.bias is not None:
            lora_output += self.bias
            
        return lora_output

class LoRAModel:
    """
    Manages LoRA adaptations for a model.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: LoRAConfig
    ):
        """
        Initialize LoRA adaptation for a model.
        
        Args:
            base_model: The base pre-trained model to adapt
            config: LoRA configuration
        """
        self.base_model = base_model
        self.config = config
        self.lora_layers = {}
        
        # Apply LoRA adaptations to target modules
        self._add_lora_layers()
        
        # Register forward hooks for all layers
        for name, module in self.base_model.named_modules():
            if name in self.lora_layers:
                module.register_forward_hook(self.forward_hook)
        
        logger.info(f"Initialized LoRA adaptation with rank {config.r}")
        logger.info(f"Target modules: {config.target_modules}")
    
    def _add_lora_layers(self):
        """Add LoRA adaptation to target modules."""
        # Keep track of applied modules
        applied_modules = set()
        
        for name, module in self.base_model.named_modules():
            # Check if this module should have LoRA applied
            if not any(target in name for target in self.config.target_modules):
                continue
                
            # Check if it's a linear layer (fully connected)
            if not isinstance(module, nn.Linear):
                continue
                
            # Skip if already applied
            if name in applied_modules:
                continue
                
            # Create LoRA layer
            lora_layer = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=self.config.r,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                bias=self.config.bias == "lora_only" or self.config.bias == "all"
            )
            
            # Store the layer
            self.lora_layers[name] = lora_layer
            applied_modules.add(name)
            
            logger.debug(f"Added LoRA adaptation to {name}")
        
        logger.info(f"Applied LoRA to {len(applied_modules)} modules")
    
    def forward_hook(self, module, input, output):
        """Hook function to add LoRA output to the original module output."""
        # Get module name
        for name, m in self.base_model.named_modules():
            if m is module:
                module_name = name
                break
        else:
            return output
        
        # Check if we have LoRA for this module
        if module_name not in self.lora_layers:
            return output
        
        # Apply LoRA adaptation
        lora_output = self.lora_layers[module_name](input[0])
        
        # Add LoRA output to original output
        return output + lora_output
    
    def save_lora_weights(self, filepath: str):
        """
        Save LoRA weights to a file.
        
        Args:
            filepath: Path to save the weights
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Collect LoRA weights
        lora_state_dict = {}
        
        for name, layer in self.lora_layers.items():
            lora_state_dict[f"{name}.lora_A"] = layer.lora_A.detach().cpu()
            lora_state_dict[f"{name}.lora_B"] = layer.lora_B.detach().cpu()
            if layer.bias is not None:
                lora_state_dict[f"{name}.bias"] = layer.bias.detach().cpu()
        
        # Save weights
        torch.save(lora_state_dict, filepath)
        
        # Save config
        config_path = filepath + ".config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Saved LoRA weights to {filepath}")
        logger.info(f"Saved LoRA config to {config_path}")
    
    @classmethod
    def load_lora_weights(
        cls,
        base_model: PreTrainedModel,
        filepath: str,
        device: Optional[str] = None
    ) -> 'LoRAModel':
        """
        Load LoRA weights from a file.
        
        Args:
            base_model: The base pre-trained model to adapt
            filepath: Path to the weights file
            device: Device to load the weights to
            
        Returns:
            LoRAModel with loaded weights
        """
        # Load config
        config_path = filepath + ".config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = LoRAConfig.from_dict(config_dict)
        
        # Create LoRA model
        lora_model = cls(base_model, config)
        
        # Load weights
        state_dict = torch.load(filepath, map_location=device)
        
        # Apply weights to LoRA layers
        for name, layer in lora_model.lora_layers.items():
            if f"{name}.lora_A" in state_dict and f"{name}.lora_B" in state_dict:
                layer.lora_A.data.copy_(state_dict[f"{name}.lora_A"])
                layer.lora_B.data.copy_(state_dict[f"{name}.lora_B"])
                
                if layer.bias is not None and f"{name}.bias" in state_dict:
                    layer.bias.data.copy_(state_dict[f"{name}.bias"])
        
        logger.info(f"Loaded LoRA weights from {filepath}")
        return lora_model

# Helper functions to create LoRA configurations for different rating ranges
def create_rating_lora_config(
    rating_range: Tuple[float, float],
    rank: int = 8
) -> LoRAConfig:
    """
    Create a LoRA configuration for a specific rating range.
    
    Args:
        rating_range: Tuple of (min_rating, max_rating)
        rank: Rank for the LoRA adaptation
        
    Returns:
        LoRA configuration for the rating range
    """
    min_rating, max_rating = rating_range
    
    # Adjust configuration based on rating range
    if min_rating <= 3.0:  # Beginner range
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        alpha = 32.0
    elif min_rating <= 7.0:  # Intermediate range
        target_modules = ["q_proj", "v_proj"]
        alpha = 16.0
    else:  # Expert range
        target_modules = ["q_proj", "v_proj"]
        alpha = 8.0
    
    return LoRAConfig(
        r=rank,
        alpha=alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )

def segment_rating_ranges(
    num_segments: int = 3
) -> List[Tuple[float, float]]:
    """
    Create segments of rating ranges.
    
    Args:
        num_segments: Number of segments to create
        
    Returns:
        List of (min_rating, max_rating) tuples
    """
    if num_segments <= 0:
        raise ValueError("Number of segments must be positive")
    
    segment_size = 10.0 / num_segments
    
    ranges = []
    for i in range(num_segments):
        min_val = i * segment_size
        max_val = (i + 1) * segment_size
        
        # Adjust boundaries to ensure coverage
        if i == 0:
            min_val = 0.0
        if i == num_segments - 1:
            max_val = 10.0
            
        ranges.append((min_val, max_val))
    
    return ranges