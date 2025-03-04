"""
Dynamic parameter loading system for efficient model inference.
"""
import os
import json
import time
import torch
from torch import nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from transformers import PreTrainedModel
import threading
import queue
from dataclasses import dataclass
import gc

from teacher_tester.optimization.parameter_storage import SegmentedParameterStorage, ParameterSegment
from teacher_tester.optimization.lora_adapter import LoRAModel
from teacher_tester.optimization.quantization import QuantizedModel
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class ParameterLoadRequest:
    """Request to load parameters for a specific rating."""
    
    rating: float
    callback: Optional[Callable] = None
    device: Optional[str] = None
    priority: int = 0  # Higher values = higher priority
    timestamp: float = 0.0  # When this request was made


class ParameterManager:
    """
    Manages loading and caching of parameters for different rating ranges.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        storage_dir: str,
        cache_size: int = 2,
        preload_segments: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the parameter manager.
        
        Args:
            base_model: Base model for adaptation
            storage_dir: Directory with parameter segments
            cache_size: Number of parameter segments to keep in memory
            preload_segments: Whether to preload segments at initialization
            device: Device to load parameters to
        """
        self.base_model = base_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.storage = SegmentedParameterStorage(storage_dir, base_model)
        self.cache_size = cache_size
        
        # Parameter cache
        self.cache = {}  # rating -> (model, timestamp)
        self.cache_lock = threading.Lock()
        
        # Parameter loading queue
        self.load_queue = queue.PriorityQueue()  # (priority, timestamp, request)
        self.loader_thread = None
        self.running = False
        
        # Start loader thread
        self._start_loader_thread()
        
        # Preload segments if requested
        if preload_segments:
            self._preload_segments()
        
        logger.info(f"Initialized parameter manager with cache size {cache_size}")
        logger.info(f"Using device: {self.device}")
    
    def _start_loader_thread(self):
        """Start the parameter loader thread."""
        if self.loader_thread is not None and self.loader_thread.is_alive():
            return
        
        self.running = True
        self.loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self.loader_thread.start()
        
        logger.info("Started parameter loader thread")
    
    def _loader_worker(self):
        """Worker function to load parameters in background."""
        while self.running:
            try:
                # Get next request from queue
                priority, _, request = self.load_queue.get(timeout=1.0)
                
                logger.debug(f"Processing load request for rating {request.rating} (priority: {priority})")
                
                # Load parameters
                model = self._load_parameters_for_rating(request.rating, request.device or self.device)
                
                # Execute callback if provided
                if request.callback is not None:
                    try:
                        request.callback(model)
                    except Exception as e:
                        logger.error(f"Error in load callback: {str(e)}")
                
                # Mark task as done
                self.load_queue.task_done()
            except queue.Empty:
                # No requests, wait for more
                pass
            except Exception as e:
                logger.error(f"Error in parameter loader: {str(e)}")
    
    def _preload_segments(self):
        """Preload parameter segments into cache."""
        # Get all segments
        segments = self.storage.segments
        
        if not segments:
            logger.warning("No segments available for preloading")
            return
        
        # Preload up to cache_size segments
        for i, segment in enumerate(segments[:self.cache_size]):
            # Calculate midpoint of rating range
            rating = (segment.min_rating + segment.max_rating) / 2
            
            # Queue load request with low priority
            self.queue_parameter_load(rating, priority=-100 + i)
        
        logger.info(f"Queued {min(len(segments), self.cache_size)} segments for preloading")
    
    def _load_parameters_for_rating(
        self,
        rating: float,
        device: Optional[str] = None
    ) -> Optional[Union[LoRAModel, QuantizedModel, PreTrainedModel]]:
        """
        Load parameters for a specific rating.
        
        Args:
            rating: Rating to load parameters for
            device: Device to load parameters to
            
        Returns:
            Loaded model or None if loading failed
        """
        device = device or self.device
        
        # Check if parameters are already in cache
        with self.cache_lock:
            for cache_rating, (model, _) in self.cache.items():
                segment = self.storage.find_segment_for_rating(cache_rating)
                if segment and segment.contains_rating(rating):
                    # Update access timestamp
                    self.cache[cache_rating] = (model, time.time())
                    
                    logger.debug(f"Using cached parameters for rating {rating} (from cache rating {cache_rating})")
                    return model
        
        # Not in cache, load from storage
        segment = self.storage.find_segment_for_rating(rating)
        
        if segment is None:
            logger.warning(f"No segment found for rating {rating}")
            return None
        
        # Load segment
        model = self.storage.load_segment(segment, device)
        
        if model is None:
            logger.error(f"Failed to load segment for rating {rating}")
            return None
        
        # Add to cache
        with self.cache_lock:
            # Check if cache is full
            if len(self.cache) >= self.cache_size:
                # Remove least recently used segment
                lru_rating = min(self.cache.keys(), key=lambda r: self.cache[r][1])
                
                # Clear the model to free memory
                del self.cache[lru_rating]
                gc.collect()
                
                logger.debug(f"Removed parameters for rating {lru_rating} from cache")
            
            # Add to cache
            self.cache[rating] = (model, time.time())
        
        logger.info(f"Loaded parameters for rating {rating}")
        return model
    
    def get_parameters(
        self,
        rating: float,
        block: bool = True,
        timeout: Optional[float] = None
    ) -> Optional[Union[LoRAModel, QuantizedModel, PreTrainedModel]]:
        """
        Get parameters for a specific rating.
        
        Args:
            rating: Rating to get parameters for
            block: Whether to block until parameters are loaded
            timeout: Timeout in seconds if blocking
            
        Returns:
            Model with appropriate parameters
        """
        # Check if parameters are already in cache
        with self.cache_lock:
            for cache_rating, (model, _) in self.cache.items():
                segment = self.storage.find_segment_for_rating(cache_rating)
                if segment and segment.contains_rating(rating):
                    # Update access timestamp
                    self.cache[cache_rating] = (model, time.time())
                    
                    logger.debug(f"Using cached parameters for rating {rating} (from cache rating {cache_rating})")
                    return model
        
        if block:
            # Load parameters synchronously
            return self._load_parameters_for_rating(rating)
        else:
            # Queue asynchronous load
            self.queue_parameter_load(rating, priority=10)  # High priority
            return None
    
    def queue_parameter_load(
        self,
        rating: float,
        callback: Optional[Callable] = None,
        device: Optional[str] = None,
        priority: int = 0
    ):
        """
        Queue parameter loading in background.
        
        Args:
            rating: Rating to load parameters for
            callback: Function to call with loaded model
            device: Device to load parameters to
            priority: Priority for loading (higher = more urgent)
        """
        # Create request
        request = ParameterLoadRequest(
            rating=rating,
            callback=callback,
            device=device,
            priority=priority,
            timestamp=time.time()
        )
        
        # Add to queue with inverted priority (lower value = higher priority)
        # and timestamp for tie-breaking
        self.load_queue.put((-priority, request.timestamp, request))
        
        logger.debug(f"Queued parameter load for rating {rating} (priority: {priority})")
    
    def predict_next_rating(
        self,
        current_rating: float,
        confidence: float,
        preload: bool = True
    ) -> float:
        """
        Predict the next rating to be requested and preload parameters.
        
        Args:
            current_rating: Current rating
            confidence: Current confidence in the rating
            preload: Whether to preload predicted parameters
            
        Returns:
            Predicted next rating
        """
        # Simple prediction: move towards extremes based on current rating and confidence
        if current_rating < 5.0:
            # Currently low rating, might go lower
            predicted_rating = max(0.0, current_rating - (1.0 - confidence) * 2.0)
        else:
            # Currently high rating, might go higher
            predicted_rating = min(10.0, current_rating + (1.0 - confidence) * 2.0)
        
        # If current confidence is high, prediction is more likely correct
        priority = int(confidence * 10) - 5  # Range: -5 to 5
        
        # Preload parameters if requested
        if preload:
            self.queue_parameter_load(predicted_rating, priority=priority)
            logger.debug(f"Preloading parameters for predicted rating {predicted_rating}")
        
        return predicted_rating
    
    def stop(self):
        """Stop the parameter manager and release resources."""
        # Stop loader thread
        self.running = False
        if self.loader_thread and self.loader_thread.is_alive():
            self.loader_thread.join(timeout=2.0)
        
        # Clear cache
        with self.cache_lock:
            self.cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Stopped parameter manager and cleared cache")

class AdaptiveModelWrapper:
    """
    Wrapper for a model with dynamic parameter loading based on rating.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        parameter_manager: ParameterManager
    ):
        """
        Initialize model wrapper.
        
        Args:
            base_model: Base model for adaptation
            parameter_manager: Parameter manager for loading parameters
        """
        self.base_model = base_model
        self.parameter_manager = parameter_manager
        self.current_rating = None
        self.current_model = None
        
        logger.info("Initialized adaptive model wrapper")
    
    def adapt_to_rating(
        self,
        rating: float,
        block: bool = True,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Adapt the model to a specific rating.
        
        Args:
            rating: Rating to adapt to
            block: Whether to block until adaptation is complete
            callback: Function to call after adaptation
            
        Returns:
            Whether adaptation was successful (always true if non-blocking)
        """
        # Check if already adapted to this rating
        if self.current_rating is not None:
            segment = self.parameter_manager.storage.find_segment_for_rating(self.current_rating)
            if segment and segment.contains_rating(rating):
                logger.debug(f"Already adapted to rating {rating} (current: {self.current_rating})")
                return True
        
        # Define callback
        def on_parameters_loaded(model):
            self.current_model = model
            self.current_rating = rating
            
            # Call user callback if provided
            if callback is not None:
                callback(model)
            
            logger.info(f"Model adapted to rating {rating}")
        
        if block:
            # Load parameters synchronously
            model = self.parameter_manager.get_parameters(rating, block=True)
            
            if model is None:
                logger.error(f"Failed to adapt model to rating {rating}")
                return False
            
            on_parameters_loaded(model)
            return True
        else:
            # Queue asynchronous load
            self.parameter_manager.queue_parameter_load(
                rating=rating,
                callback=on_parameters_loaded,
                priority=10  # High priority
            )
            return True
    
    def forward(self, *args, **kwargs):
        """
        Forward pass using the current model.
        
        This forwards to the appropriate model based on the current rating.
        """
        if self.current_model is None:
            # Fall back to base model if not adapted
            logger.warning("Using base model for forward pass (not adapted)")
            return self.base_model(*args, **kwargs)
        
        # Check model type and forward appropriately
        if isinstance(self.current_model, LoRAModel):
            # For LoRA models, we need to apply adaptations
            result = self.base_model(*args, **kwargs)
            
            # TODO: Apply LoRA adaptations to result
            # This is simplified and would need actual LoRA forward logic
            
            return result
        elif isinstance(self.current_model, QuantizedModel):
            # For quantized models, use their forward method
            return self.current_model(*args, **kwargs)
        else:
            # For full models, just use directly
            return self.current_model(*args, **kwargs)
    
    def predict_and_preload(self, confidence: float):
        """
        Predict next rating and preload parameters.
        
        Args:
            confidence: Current confidence in the rating
        """
        if self.current_rating is None:
            # Can't predict without current rating
            return
        
        # Predict next rating
        predicted_rating = self.parameter_manager.predict_next_rating(
            current_rating=self.current_rating,
            confidence=confidence,
            preload=True
        )
        
        logger.debug(f"Predicted next rating: {predicted_rating} (current: {self.current_rating})")