"""
Segmented parameter storage for efficient model management.
"""
import os
import json
import shutil
import torch
from torch import nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from transformers import PreTrainedModel
import bisect

from teacher_tester.optimization.lora_adapter import LoRAConfig, LoRAModel
from teacher_tester.optimization.quantization import QuantizationConfig, QuantizedModel
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class ParameterSegment:
    """Represents a segment of parameters for a specific rating range."""
    
    min_rating: float
    max_rating: float
    model_type: str = "lora"  # "lora" or "full"
    quantized: bool = True
    file_path: Optional[str] = None
    lora_config: Optional[Dict[str, Any]] = None
    quant_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate segment configuration."""
        if self.min_rating < 0 or self.max_rating > 10:
            raise ValueError(f"Rating range must be within [0, 10], got [{self.min_rating}, {self.max_rating}]")
        
        if self.min_rating >= self.max_rating:
            raise ValueError(f"Min rating must be less than max rating, got [{self.min_rating}, {self.max_rating}]")
        
        if self.model_type not in ["lora", "full"]:
            raise ValueError(f"Model type must be 'lora' or 'full', got {self.model_type}")
    
    def contains_rating(self, rating: float) -> bool:
        """
        Check if this segment contains the given rating.
        
        Args:
            rating: Rating to check
            
        Returns:
            Whether the rating is within this segment's range
        """
        return self.min_rating <= rating < self.max_rating
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to a dictionary."""
        return {
            "min_rating": self.min_rating,
            "max_rating": self.max_rating,
            "model_type": self.model_type,
            "quantized": self.quantized,
            "file_path": self.file_path,
            "lora_config": self.lora_config,
            "quant_config": self.quant_config,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSegment':
        """Create a segment from a dictionary."""
        return cls(**data)

class SegmentedParameterStorage:
    """
    Manages parameter segments for different rating ranges.
    """
    
    def __init__(
        self,
        storage_dir: str,
        base_model: Optional[PreTrainedModel] = None
    ):
        """
        Initialize segmented parameter storage.
        
        Args:
            storage_dir: Directory to store parameter segments
            base_model: Base model for adaptation (if available)
        """
        self.storage_dir = storage_dir
        self.base_model = base_model
        self.segments = []
        self.active_segment = None
        self.active_model = None
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load segment index if it exists
        self._load_segment_index()
        
        logger.info(f"Initialized parameter storage in {storage_dir}")
        logger.info(f"Loaded {len(self.segments)} parameter segments")
    
    def _load_segment_index(self):
        """Load segment index from storage directory."""
        index_path = os.path.join(self.storage_dir, "segment_index.json")
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                
                # Load segments
                segments_data = index_data.get("segments", [])
                self.segments = [ParameterSegment.from_dict(data) for data in segments_data]
                
                # Sort segments by min_rating
                self.segments.sort(key=lambda s: s.min_rating)
                
                logger.info(f"Loaded {len(self.segments)} segments from index")
            except Exception as e:
                logger.error(f"Error loading segment index: {str(e)}")
                # Initialize empty segments
                self.segments = []
        else:
            # Initialize empty segments
            self.segments = []
    
    def _save_segment_index(self):
        """Save segment index to storage directory."""
        index_path = os.path.join(self.storage_dir, "segment_index.json")
        
        # Prepare index data
        index_data = {
            "segments": [segment.to_dict() for segment in self.segments]
        }
        
        # Save index
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"Saved segment index with {len(self.segments)} segments")
    
    def add_segment(
        self,
        min_rating: float,
        max_rating: float,
        model_type: str = "lora",
        quantized: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        quant_config: Optional[Dict[str, Any]] = None,
        model_data: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParameterSegment:
        """
        Add a new parameter segment.
        
        Args:
            min_rating: Minimum rating for this segment
            max_rating: Maximum rating for this segment
            model_type: Type of model ("lora" or "full")
            quantized: Whether to quantize parameters
            lora_config: LoRA configuration (for "lora" type)
            quant_config: Quantization configuration (if quantized)
            model_data: Model data to store
            metadata: Additional metadata
            
        Returns:
            The added segment
        """
        # Check for overlapping segments
        for segment in self.segments:
            if (min_rating < segment.max_rating and max_rating > segment.min_rating):
                logger.warning(f"New segment [{min_rating}, {max_rating}] overlaps with existing segment "
                             f"[{segment.min_rating}, {segment.max_rating}]")
        
        # Create file path
        segment_id = f"segment_{min_rating:.1f}_{max_rating:.1f}_{model_type}"
        if quantized:
            segment_id += "_quantized"
        
        file_path = os.path.join(self.storage_dir, f"{segment_id}.pt")
        
        # Create segment
        segment = ParameterSegment(
            min_rating=min_rating,
            max_rating=max_rating,
            model_type=model_type,
            quantized=quantized,
            file_path=file_path,
            lora_config=lora_config,
            quant_config=quant_config,
            metadata=metadata or {}
        )
        
        # Save model data if provided
        if model_data is not None:
            torch.save(model_data, file_path)
        
        # Add to segments and sort
        self.segments.append(segment)
        self.segments.sort(key=lambda s: s.min_rating)
        
        # Save updated index
        self._save_segment_index()
        
        logger.info(f"Added parameter segment for rating range [{min_rating}, {max_rating}]")
        return segment
    
    def remove_segment(self, segment: ParameterSegment) -> bool:
        """
        Remove a parameter segment.
        
        Args:
            segment: Segment to remove
            
        Returns:
            Whether the segment was removed
        """
        if segment not in self.segments:
            logger.warning(f"Segment for range [{segment.min_rating}, {segment.max_rating}] not found")
            return False
        
        # If segment is active, deactivate it
        if self.active_segment == segment:
            self.active_segment = None
            self.active_model = None
        
        # Remove segment file if it exists
        if segment.file_path and os.path.exists(segment.file_path):
            os.remove(segment.file_path)
            logger.debug(f"Removed segment file: {segment.file_path}")
        
        # Remove from segments
        self.segments.remove(segment)
        
        # Save updated index
        self._save_segment_index()
        
        logger.info(f"Removed parameter segment for rating range [{segment.min_rating}, {segment.max_rating}]")
        return True
    
    def find_segment_for_rating(self, rating: float) -> Optional[ParameterSegment]:
        """
        Find the segment that contains the given rating.
        
        Args:
            rating: Rating to find segment for
            
        Returns:
            Segment containing the rating, or None if not found
        """
        for segment in self.segments:
            if segment.contains_rating(rating):
                return segment
        
        return None
    
    def load_segment(
        self,
        segment: ParameterSegment,
        device: Optional[str] = None
    ) -> Union[LoRAModel, PreTrainedModel, None]:
        """
        Load a parameter segment into memory.
        
        Args:
            segment: Segment to load
            device: Device to load parameters to
            
        Returns:
            Loaded model or None if loading failed
        """
        if self.base_model is None:
            logger.error("Cannot load segment: base model not provided")
            return None
        
        # Check if segment file exists
        if not segment.file_path or not os.path.exists(segment.file_path):
            logger.error(f"Segment file not found: {segment.file_path}")
            return None
        
        try:
            # Load based on segment type
            if segment.model_type == "lora":
                # Load LoRA model
                if segment.lora_config:
                    lora_config = LoRAConfig.from_dict(segment.lora_config)
                else:
                    # Use default config if not specified
                    lora_config = LoRAConfig()
                
                # Create LoRA model
                lora_model = LoRAModel.load_lora_weights(
                    base_model=self.base_model,
                    filepath=segment.file_path,
                    device=device
                )
                
                # Set as active
                self.active_segment = segment
                self.active_model = lora_model
                
                logger.info(f"Loaded LoRA segment for rating range [{segment.min_rating}, {segment.max_rating}]")
                return lora_model
            else:  # model_type == "full"
                # For full models, we'd typically load state dict
                # This is simplified here as it depends on the specific model architecture
                state_dict = torch.load(segment.file_path, map_location=device)
                
                # If quantized, handle differently
                if segment.quantized and segment.quant_config:
                    # Load quantized model
                    quant_config = QuantizationConfig.from_dict(segment.quant_config)
                    
                    # Create quantized model
                    # This is a simplified version; real implementation would depend on quantization details
                    quantized_model = QuantizedModel(self.base_model, quant_config)
                    
                    # Set as active
                    self.active_segment = segment
                    self.active_model = quantized_model
                    
                    logger.info(f"Loaded quantized full model segment for rating range [{segment.min_rating}, {segment.max_rating}]")
                    return quantized_model
                else:
                    # Load full model
                    self.base_model.load_state_dict(state_dict)
                    
                    # Set as active
                    self.active_segment = segment
                    self.active_model = self.base_model
                    
                    logger.info(f"Loaded full model segment for rating range [{segment.min_rating}, {segment.max_rating}]")
                    return self.base_model
        except Exception as e:
            logger.error(f"Error loading segment: {str(e)}")
            return None
    
    def activate_segment_for_rating(
        self,
        rating: float,
        device: Optional[str] = None
    ) -> Union[LoRAModel, PreTrainedModel, None]:
        """
        Find and activate the appropriate segment for a given rating.
        
        Args:
            rating: Rating to find segment for
            device: Device to load parameters to
            
        Returns:
            Activated model or None if no suitable segment found
        """
        # Check if current segment already covers this rating
        if self.active_segment and self.active_segment.contains_rating(rating):
            logger.debug(f"Using already active segment for rating {rating}")
            return self.active_model
        
        # Find segment for rating
        segment = self.find_segment_for_rating(rating)
        
        if segment is None:
            logger.warning(f"No segment found for rating {rating}")
            return None
        
        # Load segment
        return self.load_segment(segment, device)
    
    def export_segment(
        self,
        segment: ParameterSegment,
        export_dir: str
    ) -> bool:
        """
        Export a segment to a separate directory.
        
        Args:
            segment: Segment to export
            export_dir: Directory to export to
            
        Returns:
            Whether the export was successful
        """
        if segment not in self.segments:
            logger.warning(f"Segment for range [{segment.min_rating}, {segment.max_rating}] not found")
            return False
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            # Copy segment file
            if segment.file_path and os.path.exists(segment.file_path):
                target_path = os.path.join(export_dir, os.path.basename(segment.file_path))
                shutil.copy2(segment.file_path, target_path)
            
            # Export metadata
            metadata = segment.to_dict()
            metadata_path = os.path.join(export_dir, f"segment_{segment.min_rating:.1f}_{segment.max_rating:.1f}_metadata.json")
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Exported segment for rating range [{segment.min_rating}, {segment.max_rating}] to {export_dir}")
            return True
        except Exception as e:
            logger.error(f"Error exporting segment: {str(e)}")
            return False
    
    def import_segment(self, segment_file: str, metadata_file: str) -> Optional[ParameterSegment]:
        """
        Import a segment from external files.
        
        Args:
            segment_file: Path to segment file
            metadata_file: Path to metadata file
            
        Returns:
            Imported segment or None if import failed
        """
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Create segment
            segment = ParameterSegment.from_dict(metadata)
            
            # Update file path to local storage
            target_path = os.path.join(self.storage_dir, os.path.basename(segment_file))
            
            # Copy segment file
            shutil.copy2(segment_file, target_path)
            
            # Update segment file path
            segment.file_path = target_path
            
            # Add to segments
            self.segments.append(segment)
            self.segments.sort(key=lambda s: s.min_rating)
            
            # Save updated index
            self._save_segment_index()
            
            logger.info(f"Imported segment for rating range [{segment.min_rating}, {segment.max_rating}]")
            return segment
        except Exception as e:
            logger.error(f"Error importing segment: {str(e)}")
            return None
    
    def create_optimized_segments(
        self,
        rating_ranges: List[Tuple[float, float]],
        lora_ranks: Optional[List[int]] = None,
        quantization_bits: int = 8
    ) -> List[ParameterSegment]:
        """
        Create optimized segments for different rating ranges.
        
        Args:
            rating_ranges: List of (min_rating, max_rating) tuples
            lora_ranks: List of LoRA ranks for each range (None for default)
            quantization_bits: Bits for quantization (4 or 8)
            
        Returns:
            List of created segments
        """
        if self.base_model is None:
            logger.error("Cannot create segments: base model not provided")
            return []
        
        if lora_ranks is None:
            # Default ranks: higher rank for middle ranges, lower for extremes
            lora_ranks = []
            for min_rating, max_rating in rating_ranges:
                avg_rating = (min_rating + max_rating) / 2
                
                # Closer to middle (5) gets higher rank
                distance_from_middle = abs(avg_rating - 5.0)
                if distance_from_middle < 1.5:
                    lora_ranks.append(16)  # Middle range
                elif distance_from_middle < 3.0:
                    lora_ranks.append(8)   # Intermediate range
                else:
                    lora_ranks.append(4)   # Extreme range
        
        # Ensure lengths match
        if len(lora_ranks) != len(rating_ranges):
            logger.warning(f"Length mismatch: {len(rating_ranges)} ranges but {len(lora_ranks)} ranks")
            lora_ranks = lora_ranks[:len(rating_ranges)]
            while len(lora_ranks) < len(rating_ranges):
                lora_ranks.append(8)  # Default rank
        
        created_segments = []
        
        # Create segments
        for (min_rating, max_rating), rank in zip(rating_ranges, lora_ranks):
            # Create LoRA config
            lora_config = LoRAConfig(
                r=rank,
                alpha=2 * rank,  # Scale alpha with rank
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if rank >= 8 else ["q_proj", "v_proj"]
            )
            
            # Create quantization config
            quant_config = QuantizationConfig(
                bits=quantization_bits,
                group_size=128 if quantization_bits == 4 else None,
                use_symmetric=True
            )
            
            # Create segment (without model data for now)
            segment = self.add_segment(
                min_rating=min_rating,
                max_rating=max_rating,
                model_type="lora",
                quantized=True,
                lora_config=lora_config.to_dict(),
                quant_config=quant_config.to_dict(),
                metadata={
                    "created_at": pd.Timestamp.now().isoformat(),
                    "lora_rank": rank,
                    "quantization_bits": quantization_bits
                }
            )
            
            created_segments.append(segment)
        
        logger.info(f"Created {len(created_segments)} optimized segments")
        return created_segments
    
    def get_segment_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored segments.
        
        Returns:
            Dictionary with segment statistics
        """
        if not self.segments:
            return {"count": 0, "total_size_bytes": 0, "segments": []}
        
        total_size = 0
        segment_stats = []
        
        for segment in self.segments:
            # Get file size
            size_bytes = 0
            if segment.file_path and os.path.exists(segment.file_path):
                size_bytes = os.path.getsize(segment.file_path)
            
            # Get segment info
            segment_info = {
                "rating_range": [segment.min_rating, segment.max_rating],
                "model_type": segment.model_type,
                "quantized": segment.quantized,
                "size_bytes": size_bytes,
                "size_mb": size_bytes / 1024 / 1024
            }
            
            # Add LoRA info if available
            if segment.lora_config:
                segment_info["lora_rank"] = segment.lora_config.get("r", 0)
            
            segment_stats.append(segment_info)
            total_size += size_bytes
        
        return {
            "count": len(self.segments),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "segments": segment_stats
        }

# Helper functions to create rating segments
def create_default_rating_segments(
    num_segments: int = 3
) -> List[Tuple[float, float]]:
    """
    Create default rating range segments.
    
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