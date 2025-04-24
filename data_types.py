from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from PIL import Image


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, Any]
    # Optional fields for multimodal tasks
    image: Optional[Image.Image] = None


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    
    # Fields for Countdown task
    numbers: Optional[List[List[int]]] = None
    target: Optional[List[int]] = None
    
    # Fields for MathVision task
    images: Optional[List[Image.Image]] = None
    answers: Optional[List[str]] = None
    subjects: Optional[List[str]] = None
    levels: Optional[List[int]] = None
    options: Optional[List[List[str]]] = None
    questions: Optional[List[str]] = None
    ids: Optional[List[str]] = None
