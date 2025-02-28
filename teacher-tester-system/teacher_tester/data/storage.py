import os
import json
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import pandas as pd
from .schemas import Conversation, TrainingExample
from config.config_utils import get_config
from teacher_tester.utils.logging import setup_logger

logger = setup_logger(__name__)

class DataStorage:
    """Data storage utilities for conversations and training examples."""
    
    def __init__(self):
        """Initialize the data storage with configured paths."""
        config = get_config()
        self.conversations_dir = config.get("storage.conversations_dir", "data/conversations")
        self.training_examples_dir = config.get("storage.training_examples_dir", "data/training")
        
        # Create directories if they don't exist
        os.makedirs(self.conversations_dir, exist_ok=True)
        os.makedirs(self.training_examples_dir, exist_ok=True)
    
    def _serialize_datetime(self, obj):
        """Helper method to serialize datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def save_conversation(self, conversation: Conversation) -> str:
        """
        Save a conversation to file.
        
        Args:
            conversation: The conversation to save
            
        Returns:
            The file path where the conversation was saved
        """
        filepath = os.path.join(self.conversations_dir, f"{conversation.id}.json")
        
        with open(filepath, 'w') as f:
            json.dump(conversation.dict(), f, default=self._serialize_datetime, indent=2)
            
        logger.info(f"Saved conversation {conversation.id} to {filepath}")
        return filepath
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load a conversation from file.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            The loaded conversation or None if not found
        """
        filepath = os.path.join(self.conversations_dir, f"{conversation_id}.json")
        
        if not os.path.exists(filepath):
            logger.warning(f"Conversation {conversation_id} not found at {filepath}")
            return None
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Convert ISO format strings back to datetime
            if 'start_time' in data and data['start_time']:
                data['start_time'] = datetime.fromisoformat(data['start_time'])
            if 'end_time' in data and data['end_time']:
                data['end_time'] = datetime.fromisoformat(data['end_time'])
                
            for msg in data.get('messages', []):
                if 'timestamp' in msg and msg['timestamp']:
                    msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                    
            conversation = Conversation(**data)
            logger.info(f"Loaded conversation {conversation_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {str(e)}")
            return None
    
    def list_conversations(self) -> List[str]:
        """
        List all available conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        files = os.listdir(self.conversations_dir)
        conversation_ids = [f.replace('.json', '') for f in files if f.endswith('.json')]
        return conversation_ids
    
    def save_training_example(self, example: TrainingExample) -> str:
        """
        Save a training example to file.
        
        Args:
            example: The training example to save
            
        Returns:
            The file path where the example was saved
        """
        filepath = os.path.join(self.training_examples_dir, f"{example.id}.json")
        
        with open(filepath, 'w') as f:
            json.dump(example.dict(), f, default=self._serialize_datetime, indent=2)
            
        logger.info(f"Saved training example {example.id} to {filepath}")
        return filepath
    
    def load_training_examples(self, limit: Optional[int] = None) -> List[TrainingExample]:
        """
        Load training examples from files.
        
        Args:
            limit: Maximum number of examples to load
            
        Returns:
            List of training examples
        """
        files = os.listdir(self.training_examples_dir)
        example_files = [f for f in files if f.endswith('.json')]
        
        if limit:
            example_files = example_files[:limit]
            
        examples = []
        for file in example_files:
            filepath = os.path.join(self.training_examples_dir, file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                example = TrainingExample(**data)
                examples.append(example)
            except Exception as e:
                logger.error(f"Error loading training example {file}: {str(e)}")
                
        logger.info(f"Loaded {len(examples)} training examples")
        return examples
    
    def get_training_dataframe(self) -> pd.DataFrame:
        """
        Load all training examples into a pandas DataFrame.
        
        Returns:
            DataFrame containing all training examples
        """
        examples = self.load_training_examples()
        if not examples:
            return pd.DataFrame()
            
        data = [example.dict() for example in examples]
        df = pd.DataFrame(data)
        return df

# Singleton storage instance
_storage_instance = None

def get_storage() -> DataStorage:
    """Get the global storage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = DataStorage()
    return _storage_instance