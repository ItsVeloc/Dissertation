from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
import uuid

class Message(BaseModel):
    """Represents a single message in a conversation."""
    role: Literal["teacher", "tester"] 
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    confidence: Optional[float] = None  # Only for teacher messages
    
    @validator('confidence')
    def validate_confidence(cls, v, values):
        if values.get('role') == 'teacher' and v is None:
            raise ValueError('Teacher messages must include confidence')
        return v

class Conversation(BaseModel):
    """Represents a full conversation between teacher and tester."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    true_rating: float = Field(..., ge=0.0, le=10.0)  # Ground truth rating of tester
    predicted_rating: Optional[float] = None  # Teacher's final prediction
    messages: List[Message] = []
    final_confidence: Optional[float] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    terminated_reason: Optional[Literal["confidence_threshold", "max_exchanges"]] = None
    
    def add_message(self, role: str, content: str, confidence: Optional[float] = None) -> None:
        """Add a new message to the conversation."""
        msg = Message(role=role, content=content, confidence=confidence)
        self.messages.append(msg)
    
    def is_complete(self, confidence_threshold: float = 0.8, max_exchanges: int = 5) -> bool:
        """Check if conversation should be terminated."""
        teacher_messages = [m for m in self.messages if m.role == "teacher"]
        
        # Check confidence threshold
        if teacher_messages and teacher_messages[-1].confidence >= confidence_threshold:
            self.terminated_reason = "confidence_threshold"
            return True
            
        # Check max exchanges
        if len(teacher_messages) >= max_exchanges:
            self.terminated_reason = "max_exchanges"
            return True
            
        return False
    
    def complete_conversation(self, predicted_rating: float, confidence: float) -> None:
        """Mark conversation as complete with final prediction."""
        self.predicted_rating = predicted_rating
        self.final_confidence = confidence
        self.end_time = datetime.now()

class TrainingExample(BaseModel):
    """A processed training example derived from a conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    subject: str
    true_rating: float
    predicted_rating: float
    prediction_error: float
    confidence: float
    calibration_error: float  # Difference between confidence and accuracy
    conversation_length: int
    features: Dict[str, Any] = {}  # Extracted features for training