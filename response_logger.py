import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

class ResponseLogger:
    """Logger for tracking model responses to image-text prompts"""
    
    def __init__(self, log_dir: str = "logs_responses"):
        """
        Initialize response logger
        
        Args:
            log_dir: Directory to save response logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self.log_file = self.run_dir / "responses.json"
        
        # Initialize responses list
        self.responses = []
        
        # Counter for generating unique image filenames
        self.image_counter = 0
        
        print(f"Response logger initialized. Logs will be saved to: {self.run_dir}")
    
    def log_response(
        self, 
        question: str, 
        response: str, 
        image: Optional[Image.Image] = None, 
        expected_answer: Optional[str] = None,
        extracted_answer: Optional[str] = None,
        reward: Optional[float] = None,
        reward_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a model response
        
        Args:
            question: The question prompt
            response: Model's response text
            image: Optional PIL Image that was shown to the model
            expected_answer: Optional ground truth answer
            extracted_answer: Optional answer extracted from the response
            reward: Optional reward value
            reward_info: Optional reward breakdown information
            metadata: Optional additional metadata to log
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response,
            "image_path": None,
        }
        
        # Add optional fields if provided
        if expected_answer is not None:
            log_entry["expected_answer"] = expected_answer
        
        if extracted_answer is not None:
            log_entry["extracted_answer"] = extracted_answer
            
        if reward is not None:
            log_entry["reward"] = reward
            
        if reward_info is not None:
            log_entry["reward_info"] = reward_info
            
        if metadata is not None:
            log_entry["metadata"] = metadata
        
        # Add to responses list
        self.responses.append(log_entry)
        
        # Write to log file, replacing previous content
        with open(self.log_file, "w") as f:
            f.write(json.dumps(self.responses, indent=2))
            
    def log_batch_responses(self, episodes: List[Any]):
        """
        Log responses from a batch of episodes
        
        Args:
            episodes: List of Episode objects from rollout
        """
        # Clear previous responses list
        self.responses = []
        
        # Log all episodes in the batch
        for episode in episodes:
            self.log_response(
                question=episode.prefix,
                response=episode.text[len(episode.prefix):],
                image=episode.image if hasattr(episode, "image") else None,
                expected_answer=episode.reward_info.get("expected_answer"),
                extracted_answer=episode.reward_info.get("extracted_answer"),
                reward=episode.reward,
                reward_info=episode.reward_info,
                metadata={
                    "is_finished": episode.is_finished,
                    "response_length": len(episode.generated_token_ids)
                }
            ) 