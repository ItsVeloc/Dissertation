"""
LoRA fine-tuning pipeline for teacher-tester models.
"""
import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    get_cosine_schedule_with_warmup
)
import pandas as pd
from tqdm import tqdm
import logging
import wandb

from teacher_tester.data.storage import get_storage
from teacher_tester.optimization.lora_adapter import LoRAConfig, LoRAModel
from teacher_tester.optimization.parameter_storage import (
    SegmentedParameterStorage, 
    ParameterSegment,
    create_default_rating_segments
)
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class LoRATrainingArguments:
    """Arguments for LoRA fine-tuning."""
    
    # Output directories
    output_dir: str = field(
        default="data/lora_models",
        metadata={"help": "Directory to save models"}
    )
    logging_dir: str = field(
        default="data/logs",
        metadata={"help": "Directory for logs"}
    )
    
    # Training parameters
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for training"}
    )
    learning_rate: float = field(
        default=5e-4,
        metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Warmup steps"}
    )
    
    # LoRA parameters
    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank for LoRA adaptation"}
    )
    lora_alpha: float = field(
        default=16.0,
        metadata={"help": "Alpha for LoRA adaptation"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout for LoRA layers"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "Modules to apply LoRA"}
    )
    
    # Data parameters
    train_ratio: float = field(
        default=0.8,
        metadata={"help": "Ratio of data to use for training vs validation"}
    )
    min_rating: float = field(
        default=0.0,
        metadata={"help": "Minimum rating for this segment"}
    )
    max_rating: float = field(
        default=10.0,
        metadata={"help": "Maximum rating for this segment"}
    )
    
    # Logging
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X steps"}
    )
    eval_steps: int = field(
        default=50,
        metadata={"help": "Evaluate every X steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X steps"}
    )
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to use Weights & Biases for logging"}
    )
    wandb_project: str = field(
        default="teacher-tester",
        metadata={"help": "W&B project name"}
    )
    
    # Device
    device: str = field(
        default="",
        metadata={"help": "Device to use ('' for auto, 'cpu', 'cuda', 'cuda:0', etc.)"}
    )

class TeacherTesterDataset(Dataset):
    """Dataset for teacher-tester fine-tuning."""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 512,
        min_rating: float = 0.0,
        max_rating: float = 10.0
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory with training examples
            tokenizer: Tokenizer for model
            max_length: Maximum token length
            min_rating: Minimum rating to include
            max_rating: Maximum rating to include
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        storage = get_storage()
        self.examples = storage.load_training_examples()
        
        # Filter by rating
        self.examples = [
            ex for ex in self.examples 
            if min_rating <= ex.true_rating < max_rating
        ]
        
        logger.info(f"Loaded {len(self.examples)} examples for rating range [{min_rating}, {max_rating}]")
    
    def __len__(self):
        """Get dataset length."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get item by index."""
        example = self.examples[idx]
        
        # Get conversation
        conversation = get_storage().load_conversation(example.conversation_id)
        
        if conversation is None:
            # Handle missing conversation
            logger.warning(f"Missing conversation: {example.conversation_id}")
            
            # Create a default item
            input_ids = self.tokenizer.encode(
                "Missing conversation",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(input_ids),
                "rating": torch.tensor(example.true_rating, dtype=torch.float)
            }
        
        # Format conversation for fine-tuning
        formatted_text = self._format_conversation(conversation)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare dataset item
        item = {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "rating": torch.tensor(example.true_rating, dtype=torch.float)
        }
        
        # For language modeling, labels are the same as inputs
        item["labels"] = inputs.input_ids[0].clone()
        
        return item
    
    def _format_conversation(self, conversation):
        """Format conversation for fine-tuning."""
        # This formatting would depend on your specific needs
        # Here's a simple example:
        formatted_text = f"Subject: {conversation.subject}\nRating: {conversation.true_rating}\n\n"
        
        for msg in conversation.messages:
            if msg.role == "teacher":
                formatted_text += f"Teacher: {msg.content}\n"
            else:
                formatted_text += f"Tester: {msg.content}\n"
        
        return formatted_text

class LoRATrainer:
    """Trainer for LoRA fine-tuning."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        args: LoRATrainingArguments
    ):
        """
        Initialize trainer.
        
        Args:
            model: Base model to fine-tune
            tokenizer: Tokenizer for model
            args: Training arguments
        """
        self.base_model = model
        self.tokenizer = tokenizer
        self.args = args
        
        # Determine device
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(args.logging_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        
        # Setup W&B if enabled
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"lora_r{args.lora_rank}_rating_{args.min_rating}-{args.max_rating}"
            )
        
        logger.info(f"Initialized LoRA trainer for rating range [{args.min_rating}, {args.max_rating}]")
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self):
        """Prepare training and validation data."""
        # Create dataset
        full_dataset = TeacherTesterDataset(
            data_dir="data/training", 
            tokenizer=self.tokenizer,
            min_rating=self.args.min_rating,
            max_rating=self.args.max_rating
        )
        
        # Split into train and validation sets
        dataset_size = len(full_dataset)
        train_size = int(self.args.train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        # Use random split
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size
        )
        
        logger.info(f"Prepared data: {train_size} training, {val_size} validation samples")
    
    def setup_model(self):
        """Setup model with LoRA adaptation."""
        # Create LoRA config
        lora_config = LoRAConfig(
            r=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            dropout=self.args.lora_dropout,
            target_modules=self.args.target_modules
        )
        
        # Create LoRA model
        self.lora_model = LoRAModel(
            base_model=self.base_model,
            config=lora_config
        )
        
        # Move model to device
        self.base_model.to(self.device)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Get trainable parameters
        self.trainable_params = []
        for name, param in self.lora_model.lora_layers.items():
            for param_name, weight in param.named_parameters():
                weight.requires_grad = True
                self.trainable_params.append(weight)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Create scheduler
        num_training_steps = len(self.train_loader) * self.args.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        num_params = sum(p.numel() for p in self.trainable_params)
        logger.info(f"Setup model with {num_params} trainable parameters")
    
    def train(self):
        """Train the model."""
        logger.info("Starting training")
        
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            # Training phase
            self.base_model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.base_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"]
                )
                
                # Extract loss
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                
                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
                
                # Track loss
                train_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Logging
                global_step += 1
                if global_step % self.args.logging_steps == 0:
                    avg_train_loss = train_loss / global_step
                    lr = self.scheduler.get_last_lr()[0]
                    
                    logger.info(f"Step {global_step}: loss={avg_train_loss:.4f}, lr={lr:.6f}")
                    
                    if self.args.use_wandb:
                        wandb.log({
                            "train_loss": avg_train_loss,
                            "learning_rate": lr,
                            "epoch": epoch,
                            "step": global_step
                        })
                
                # Evaluation
                if global_step % self.args.eval_steps == 0:
                    val_loss = self.evaluate()
                    
                    # Log validation loss
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    
                    if self.args.use_wandb:
                        wandb.log({"val_loss": val_loss, "step": global_step})
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(os.path.join(self.args.output_dir, "best_model"))
                        logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0:
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                    self.save_model(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # End of epoch
            avg_epoch_loss = train_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1}/{self.args.epochs} completed. Avg loss: {avg_epoch_loss:.4f}")
        
        # Final evaluation
        final_val_loss = self.evaluate()
        logger.info(f"Training completed. Final validation loss: {final_val_loss:.4f}")
        
        # Save final model
        self.save_model(os.path.join(self.args.output_dir, "final_model"))
        logger.info("Saved final model")
        
        # Close W&B
        if self.args.use_wandb:
            wandb.finish()
    
    def evaluate(self):
        """Evaluate the model on validation data."""
        self.base_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.base_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"]
                )
                
                # Extract loss
                val_loss += outputs.loss.item()
        
        # Calculate average loss
        avg_val_loss = val_loss / len(self.val_loader)
        
        return avg_val_loss
    
    def save_model(self, output_dir: str):
        """
        Save LoRA model.
        
        Args:
            output_dir: Directory to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA weights
        self.lora_model.save_lora_weights(os.path.join(output_dir, "lora_weights.pt"))
        
        # Save training arguments
        with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        # Save metadata
        metadata = {
            "min_rating": self.args.min_rating,
            "max_rating": self.args.max_rating,
            "lora_rank": self.args.lora_rank,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_full_training(self):
        """Run the complete training pipeline."""
        # Prepare data
        self.prepare_data()
        
        # Setup model
        self.setup_model()
        
        # Train model
        self.train()
        
        return os.path.join(self.args.output_dir, "final_model")

def train_rating_segment(
    model: PreTrainedModel,
    tokenizer,
    min_rating: float,
    max_rating: float,
    output_dir: str = "data/lora_models",
    lora_rank: int = 8,
    **kwargs
) -> str:
    """
    Train a LoRA model for a specific rating range.
    
    Args:
        model: Base model to fine-tune
        tokenizer: Tokenizer for model
        min_rating: Minimum rating for this segment
        max_rating: Maximum rating for this segment
        output_dir: Directory to save model
        lora_rank: Rank for LoRA adaptation
        **kwargs: Additional training arguments
        
    Returns:
        Path to trained model
    """
    # Create training arguments
    args = LoRATrainingArguments(
        output_dir=os.path.join(output_dir, f"lora_r{lora_rank}_{min_rating:.1f}_{max_rating:.1f}"),
        min_rating=min_rating,
        max_rating=max_rating,
        lora_rank=lora_rank,
        **kwargs
    )
    
    # Create trainer
    trainer = LoRATrainer(model, tokenizer, args)
    
    # Run training
    model_path = trainer.run_full_training()
    
    logger.info(f"Trained LoRA model for rating range [{min_rating}, {max_rating}], saved to {model_path}")
    
    return model_path

def train_all_segments(
    model: PreTrainedModel,
    tokenizer,
    storage: SegmentedParameterStorage,
    num_segments: int = 3,
    output_dir: str = "data/lora_models",
    **kwargs
) -> List[ParameterSegment]:
    """
    Train LoRA models for all rating segments.
    
    Args:
        model: Base model to fine-tune
        tokenizer: Tokenizer for model
        storage: Parameter storage for saving segments
        num_segments: Number of segments to create
        output_dir: Directory to save models
        **kwargs: Additional training arguments
        
    Returns:
        List of created parameter segments
    """
    # Create rating segments
    rating_ranges = create_default_rating_segments(num_segments)
    
    # Train models for each segment
    segments = []
    
    for min_rating, max_rating in rating_ranges:
        # Determine appropriate LoRA rank based on segment
        avg_rating = (min_rating + max_rating) / 2
        distance_from_middle = abs(avg_rating - 5.0)
        
        if distance_from_middle < 1.5:
            lora_rank = 16  # Middle range: more complex
        elif distance_from_middle < 3.0:
            lora_rank = 8   # Intermediate range
        else:
            lora_rank = 4   # Extreme range: simpler adaptation
        
        # Train model
        model_path = train_rating_segment(
            model=model,
            tokenizer=tokenizer,
            min_rating=min_rating,
            max_rating=max_rating,
            output_dir=output_dir,
            lora_rank=lora_rank,
            **kwargs
        )
        
        # Create LoRA config
        lora_config = LoRAConfig(
            r=lora_rank,
            alpha=2 * lora_rank,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if lora_rank >= 8 else ["q_proj", "v_proj"]
        )
        
        # Add segment to storage
        segment = storage.add_segment(
            min_rating=min_rating,
            max_rating=max_rating,
            model_type="lora",
            quantized=True,
            lora_config=lora_config.to_dict(),
            metadata={
                "trained_at": datetime.datetime.now().isoformat(),
                "lora_rank": lora_rank
            }
        )
        
        # Copy model to storage
        shutil.copy(
            os.path.join(model_path, "lora_weights.pt"),
            segment.file_path
        )
        
        segments.append(segment)
        
        logger.info(f"Added segment for rating range [{min_rating}, {max_rating}] to storage")
    
    return segments

if __name__ == "__main__":
    # Example command-line usage
    parser = HfArgumentParser(LoRATrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Placeholder for actual model loading
    # In real usage, you would load your model and tokenizer here
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Train segment
    train_rating_segment(model, tokenizer, args.min_rating, args.max_rating)