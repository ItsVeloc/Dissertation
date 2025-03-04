"""
Utilities for model quantization and precision reduction.
"""
import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import PreTrainedModel
import math

from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    bits: int = 8  # Quantization bits (4, 8)
    group_size: int = 128  # Group size for quantization (32, 64, 128, None)
    use_symmetric: bool = True  # Whether to use symmetric quantization
    preserve_embeddings: bool = True  # Whether to preserve embeddings in full precision
    preserve_heads: bool = True  # Whether to preserve output heads in full precision
    
    def __post_init__(self):
        """Validate configuration."""
        if self.bits not in [4, 8]:
            raise ValueError(f"Quantization bits must be 4 or 8, got {self.bits}")
        
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError(f"Group size must be positive, got {self.group_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary."""
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "use_symmetric": self.use_symmetric,
            "preserve_embeddings": self.preserve_embeddings,
            "preserve_heads": self.preserve_heads
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """Create a QuantizationConfig from a dictionary."""
        return cls(**config_dict)

class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        group_size: Optional[int] = 128,
        bias: bool = True,
        use_symmetric: bool = True
    ):
        """
        Initialize a quantized linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bits: Number of bits for quantization (4 or 8)
            group_size: Group size for quantization (None for per-tensor)
            bias: Whether to include bias
            use_symmetric: Whether to use symmetric quantization
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.use_symmetric = use_symmetric
        
        # Original weights (full precision) - used during quantization only
        self.orig_weight = None
        
        # Quantized weights representation
        if group_size is None:
            # Per-tensor quantization
            self.register_buffer('weight_scale', torch.zeros(1))
            self.register_buffer('weight_zero_point', torch.zeros(1, dtype=torch.int32))
            if bits == 4:
                self.register_buffer('weight_quantized', torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
            else:  # bits == 8
                self.register_buffer('weight_quantized', torch.zeros((out_features, in_features), dtype=torch.uint8))
        else:
            # Group-wise quantization
            num_groups = math.ceil(in_features / group_size)
            self.register_buffer('weight_scale', torch.zeros((out_features, num_groups)))
            if not use_symmetric:
                self.register_buffer('weight_zero_point', torch.zeros((out_features, num_groups), dtype=torch.int32))
            else:
                self.register_buffer('weight_zero_point', None)
                
            if bits == 4:
                self.register_buffer('weight_quantized', torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
            else:  # bits == 8
                self.register_buffer('weight_quantized', torch.zeros((out_features, in_features), dtype=torch.uint8))
        
        # Bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using dequantized weights.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Dequantize weights for the forward pass
        weight_deq = self.dequantize_weights()
        
        # Perform matrix multiplication
        output = F.linear(x, weight_deq, self.bias)
        
        return output
    
    def quantize_from(self, weight: torch.Tensor):
        """
        Quantize weights from full precision tensor.
        
        Args:
            weight: Full precision weight tensor
        """
        # Store original weights for potential reuse
        self.orig_weight = weight.detach().clone()
        
        if self.group_size is None:
            # Per-tensor quantization
            if self.use_symmetric:
                # Symmetric quantization
                weight_abs_max = weight.abs().max()
                self.weight_scale.fill_(weight_abs_max / ((2 ** (self.bits - 1)) - 1))
                
                # Quantize
                weight_q = torch.round(weight / self.weight_scale).to(torch.int8)
                
                # Clip to quantization range
                weight_q = torch.clamp(weight_q, -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1)
                
                # Store as uint8
                if self.bits == 4:
                    # Pack two int4 values into one uint8
                    weight_q_packed = self._pack_int4(weight_q)
                    self.weight_quantized.copy_(weight_q_packed)
                else:  # bits == 8
                    self.weight_quantized.copy_(weight_q.to(torch.uint8))
            else:
                # Asymmetric quantization
                weight_min = weight.min()
                weight_max = weight.max()
                
                # Compute scale and zero point
                scale = (weight_max - weight_min) / (2 ** self.bits - 1)
                zero_point = torch.round(-weight_min / scale).to(torch.int32)
                
                self.weight_scale.fill_(scale.item())
                self.weight_zero_point.fill_(zero_point.item())
                
                # Quantize
                weight_q = torch.round(weight / scale + zero_point).to(torch.uint8)
                
                # Clip to quantization range
                weight_q = torch.clamp(weight_q, 0, 2 ** self.bits - 1)
                
                # Store
                if self.bits == 4:
                    # Pack two int4 values into one uint8
                    weight_q_packed = self._pack_uint4(weight_q)
                    self.weight_quantized.copy_(weight_q_packed)
                else:  # bits == 8
                    self.weight_quantized.copy_(weight_q)
        else:
            # Group-wise quantization
            out_features, in_features = weight.shape
            num_groups = math.ceil(in_features / self.group_size)
            
            # Process each output row separately
            for out_idx in range(out_features):
                weight_row = weight[out_idx]
                
                # Process each group
                for group_idx in range(num_groups):
                    start_idx = group_idx * self.group_size
                    end_idx = min(start_idx + self.group_size, in_features)
                    
                    # Get group weights
                    group_weight = weight_row[start_idx:end_idx]
                    
                    if self.use_symmetric:
                        # Symmetric quantization for group
                        weight_abs_max = group_weight.abs().max()
                        self.weight_scale[out_idx, group_idx] = weight_abs_max / ((2 ** (self.bits - 1)) - 1)
                        
                        # Quantize
                        group_weight_q = torch.round(group_weight / self.weight_scale[out_idx, group_idx]).to(torch.int8)
                        
                        # Clip
                        group_weight_q = torch.clamp(group_weight_q, -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1)
                        
                        # Store in the appropriate section of the tensor
                        if self.bits == 4:
                            # Pack two int4 values into one uint8
                            group_weight_q_packed = self._pack_int4(group_weight_q)
                            target_idx = slice(start_idx // 2, (end_idx + 1) // 2)
                            self.weight_quantized[out_idx, target_idx] = group_weight_q_packed
                        else:  # bits == 8
                            target_idx = slice(start_idx, end_idx)
                            self.weight_quantized[out_idx, target_idx] = group_weight_q.to(torch.uint8)
                    else:
                        # Asymmetric quantization for group
                        weight_min = group_weight.min()
                        weight_max = group_weight.max()
                        
                        # Compute scale and zero point
                        scale = (weight_max - weight_min) / (2 ** self.bits - 1)
                        zero_point = torch.round(-weight_min / scale).to(torch.int32)
                        
                        self.weight_scale[out_idx, group_idx] = scale
                        self.weight_zero_point[out_idx, group_idx] = zero_point
                        
                        # Quantize
                        group_weight_q = torch.round(group_weight / scale + zero_point).to(torch.uint8)
                        
                        # Clip
                        group_weight_q = torch.clamp(group_weight_q, 0, 2 ** self.bits - 1)
                        
                        # Store
                        if self.bits == 4:
                            # Pack two uint4 values into one uint8
                            group_weight_q_packed = self._pack_uint4(group_weight_q)
                            target_idx = slice(start_idx // 2, (end_idx + 1) // 2)
                            self.weight_quantized[out_idx, target_idx] = group_weight_q_packed
                        else:  # bits == 8
                            target_idx = slice(start_idx, end_idx)
                            self.weight_quantized[out_idx, target_idx] = group_weight_q
        
        # Free original weights after quantization to save memory
        self.orig_weight = None
    
    def dequantize_weights(self) -> torch.Tensor:
        """
        Dequantize weights for computation.
        
        Returns:
            Dequantized weight tensor
        """
        out_features = self.out_features
        in_features = self.in_features
        
        # Allocate tensor for dequantized weights
        weight_deq = torch.zeros((out_features, in_features), 
                                dtype=torch.float32, 
                                device=self.weight_quantized.device)
        
        if self.group_size is None:
            # Per-tensor dequantization
            if self.use_symmetric:
                # Unpack if needed
                if self.bits == 4:
                    weight_q = self._unpack_int4(self.weight_quantized)
                else:  # bits == 8
                    weight_q = self.weight_quantized.to(torch.int8)
                
                # Dequantize
                weight_deq = weight_q.float() * self.weight_scale
            else:
                # Unpack if needed
                if self.bits == 4:
                    weight_q = self._unpack_uint4(self.weight_quantized)
                else:  # bits == 8
                    weight_q = self.weight_quantized
                
                # Dequantize
                weight_deq = (weight_q.float() - self.weight_zero_point.float()) * self.weight_scale
        else:
            # Group-wise dequantization
            num_groups = math.ceil(in_features / self.group_size)
            
            # Process each output row
            for out_idx in range(out_features):
                # Process each group
                for group_idx in range(num_groups):
                    start_idx = group_idx * self.group_size
                    end_idx = min(start_idx + self.group_size, in_features)
                    
                    if self.use_symmetric:
                        # Unpack if needed
                        if self.bits == 4:
                            source_idx = slice(start_idx // 2, (end_idx + 1) // 2)
                            group_weight_q = self._unpack_int4(self.weight_quantized[out_idx, source_idx])
                        else:  # bits == 8
                            source_idx = slice(start_idx, end_idx)
                            group_weight_q = self.weight_quantized[out_idx, source_idx].to(torch.int8)
                        
                        # Dequantize
                        weight_deq[out_idx, start_idx:end_idx] = group_weight_q.float() * self.weight_scale[out_idx, group_idx]
                    else:
                        # Unpack if needed
                        if self.bits == 4:
                            source_idx = slice(start_idx // 2, (end_idx + 1) // 2)
                            group_weight_q = self._unpack_uint4(self.weight_quantized[out_idx, source_idx])
                        else:  # bits == 8
                            source_idx = slice(start_idx, end_idx)
                            group_weight_q = self.weight_quantized[out_idx, source_idx]
                        
                        # Dequantize
                        weight_deq[out_idx, start_idx:end_idx] = (
                            group_weight_q.float() - self.weight_zero_point[out_idx, group_idx].float()
                        ) * self.weight_scale[out_idx, group_idx]
        
        return weight_deq
    
    def _pack_int4(self, x_int8: torch.Tensor) -> torch.Tensor:
        """
        Pack two int4 values into one uint8.
        
        Args:
            x_int8: Tensor of int8 values in range [-8, 7]
            
        Returns:
            Packed uint8 tensor
        """
        # Ensure input is properly shaped
        if x_int8.shape[0] % 2 != 0:
            # If odd number of elements, pad with a 0
            x_int8 = torch.cat([x_int8, torch.zeros(1, dtype=torch.int8, device=x_int8.device)])
        
        # Convert to uint8 (adding 8 to shift range from [-8,7] to [0,15])
        x_uint4 = (x_int8 + 8).to(torch.uint8)
        
        # Pack two uint4 values into one uint8
        return self._pack_uint4(x_uint4)
    
    def _unpack_int4(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """
        Unpack one uint8 into two int4 values.
        
        Args:
            x_uint8: Packed uint8 tensor
            
        Returns:
            Unpacked int8 tensor
        """
        # Unpack to uint4 (values 0-15)
        x_uint4 = self._unpack_uint4(x_uint8)
        
        # Convert back to int8 (subtract 8 to shift range from [0,15] to [-8,7])
        return x_uint4.to(torch.int8) - 8
    
    def _pack_uint4(self, x_uint4: torch.Tensor) -> torch.Tensor:
        """
        Pack two uint4 values into one uint8.
        
        Args:
            x_uint4: Tensor of uint8 values in range [0, 15]
            
        Returns:
            Packed uint8 tensor
        """
        # Ensure input is properly shaped
        if x_uint4.shape[0] % 2 != 0:
            # If odd number of elements, pad with a 0
            x_uint4 = torch.cat([x_uint4, torch.zeros(1, dtype=torch.uint8, device=x_uint4.device)])
        
        # Reshape to pairs
        x_pairs = x_uint4.reshape(-1, 2)
        
        # Pack: low bits in first value, high bits in second value
        return (x_pairs[:, 0] | (x_pairs[:, 1] << 4))
    
    def _unpack_uint4(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """
        Unpack one uint8 into two uint4 values.
        
        Args:
            x_uint8: Packed uint8 tensor
            
        Returns:
            Unpacked uint8 tensor with values in range [0, 15]
        """
        # Extract low and high bits
        x_low = x_uint8 & 0x0F  # Extract bits 0-3
        x_high = (x_uint8 >> 4) & 0x0F  # Extract bits 4-7
        
        # Interleave low and high bits
        x_unpacked = torch.zeros(x_uint8.shape[0] * 2, dtype=torch.uint8, device=x_uint8.device)
        x_unpacked[0::2] = x_low
        x_unpacked[1::2] = x_high
        
        return x_unpacked

class QuantizedModel:
    """
    Manages quantization for a model.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: QuantizationConfig
    ):
        """
        Initialize quantization for a model.
        
        Args:
            model: The model to quantize
            config: Quantization configuration
        """
        self.model = model
        self.config = config
        self.quantized_modules = {}
        
        # Apply quantization
        self._quantize_model()
        
        logger.info(f"Initialized quantization with {config.bits} bits")
        if config.group_size:
            logger.info(f"Using group size: {config.group_size}")
        logger.info(f"Symmetric quantization: {config.use_symmetric}")
    
    def _should_quantize(self, name: str, module: nn.Module) -> bool:
        """
        Determine if a module should be quantized.
        
        Args:
            name: Module name
            module: Module to check
            
        Returns:
            Whether the module should be quantized
        """
        # Only quantize linear layers
        if not isinstance(module, nn.Linear):
            return False
        
        # Check if embeddings should be preserved
        if self.config.preserve_embeddings and any(embed_name in name for embed_name in 
                                                 ["embed", "embedding", "wte", "wpe"]):
            return False
        
        # Check if output heads should be preserved
        if self.config.preserve_heads and any(head_name in name for head_name in 
                                            ["head", "lm_head", "output", "classifier"]):
            return False
        
        return True
    
    def _quantize_model(self):
        """Quantize the model's linear layers."""
        # Keep track of quantized modules
        quantized_count = 0
        total_params = 0
        quantized_params = 0
        
        # First, count parameters
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                param_count = module.in_features * module.out_features
                total_params += param_count
                
                if self._should_quantize(name, module):
                    quantized_params += param_count
        
        # Then, apply quantization
        for name, module in self.model.named_modules():
            if not self._should_quantize(name, module):
                continue
                
            # Create quantized layer
            quantized_layer = QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bits=self.config.bits,
                group_size=self.config.group_size,
                bias=module.bias is not None,
                use_symmetric=self.config.use_symmetric
            )
            
            # Quantize from original weights
            quantized_layer.quantize_from(module.weight.data)
            
            # Copy bias if present
            if module.bias is not None:
                quantized_layer.bias.copy_(module.bias.data)
            
            # Store the quantized layer
            self.quantized_modules[name] = quantized_layer
            
            quantized_count += 1
            
            logger.debug(f"Quantized linear layer: {name}")
        
        logger.info(f"Quantized {quantized_count} modules")
        
        if total_params > 0:
            quantized_ratio = quantized_params / total_params
            logger.info(f"Quantized {quantized_ratio:.2%} of model parameters")
            
            # Calculate memory savings
            fp16_size = total_params * 2  # bytes
            quantized_size = (total_params - quantized_params) * 2 + quantized_params * self.config.bits / 8
            compression_ratio = fp16_size / quantized_size
            logger.info(f"Compression ratio: {compression_ratio:.2f}x (vs FP16)")
    
    def save_quantized_model(self, directory: str):
        """
        Save the quantized model.
        
        Args:
            directory: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save quantized state dict
        quantized_dict = {}
        for name, module in self.quantized_modules.items():
            # Save quantized weights
            quantized_dict[f"{name}.weight_quantized"] = module.weight_quantized
            quantized_dict[f"{name}.weight_scale"] = module.weight_scale
            if module.weight_zero_point is not None:
                quantized_dict[f"{name}.weight_zero_point"] = module.weight_zero_point
            if module.bias is not None:
                quantized_dict[f"{name}.bias"] = module.bias
        
        # Save weights
        weights_path = os.path.join(directory, "quantized_weights.pt")
        torch.save(quantized_dict, weights_path)
        
        # Save config
        config_path = os.path.join(directory, "quantization_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save model config
        model_config = self.model.config.to_dict()
        model_config_path = os.path.join(directory, "model_config.json")
        with open(model_config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"Saved quantized model to {directory}")
        logger.info(f"Quantized weight file size: {os.path.getsize(weights_path) / 1024 / 1024:.2f} MB")

# Helper functions to create quantization configurations
def create_quantization_config(bits: int = 8, is_lora: bool = False) -> QuantizationConfig:
    """
    Create a quantization configuration.
    
    Args:
        bits: Number of bits for quantization (4 or 8)
        is_lora: Whether quantizing a LoRA-adapted model
        
    Returns:
        Quantization configuration
    """
    if is_lora:
        # For LoRA-adapted models, we can be more aggressive
        return QuantizationConfig(
            bits=bits,
            group_size=32,  # Smaller group size
            use_symmetric=True,
            preserve_embeddings=True,
            preserve_heads=True
        )
    else:
        # For full models, be more conservative
        return QuantizationConfig(
            bits=bits,
            group_size=128 if bits == 4 else None,  # Larger group size for 4-bit
            use_symmetric=True,
            preserve_embeddings=True,
            preserve_heads=True
        )