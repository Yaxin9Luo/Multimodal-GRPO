import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer
import sympy

SYSTEM_MESSAGE = (
    "You are a helpful vision-language assistant. You are given an image and a question about it. "
    "ALWAYS analyze the image carefully and provide a DETAILED step-by-step reasoning before giving your final answer. "
    "Be thorough and consider all relevant aspects of the image before reaching a conclusion."
)

USER_TEMPLATE = (
    "IMPORTANT: First think step-by-step to solve this problem {question}, based on the image. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> A </answer>."    
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


class ViRLDataset(Dataset):
    """Dataset loader for ViRL39K vision-language reasoning dataset"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        images_dir: str = None,
        split: str = "train",
        max_samples: int = None,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize the ViRL39K dataset
        
        Args:
            tokenizer: Tokenizer for encoding text
            data_path: Path to the parquet file containing the dataset
            images_dir: Path to the directory containing the images
            split: 'train' or 'val'
            max_samples: Maximum number of samples to load (for debugging)
            test_ratio: Ratio of data to use for validation
            random_seed: Random seed for reproducibility
        """
        # Load the full dataset
        data_file = Path(data_path) / "39Krelease.parquet"
        self.df = pd.read_parquet(data_file)
        
        # Set default images directory if not provided
        if images_dir is None:
            images_dir = os.path.join(data_path, "images")
        self.images_dir = Path(images_dir)
        
        # Shuffle data with fixed seed for reproducibility
        self.df = self.df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Split into train and validation sets
        test_size = int(len(self.df) * test_ratio)
        if split == "train":
            self.df = self.df.iloc[:-test_size]
        else:  # validation
            self.df = self.df.iloc[-test_size:]
        
        # Limit the number of samples if specified
        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples]
        
        self.tokenizer = tokenizer
        print(f"Loaded {len(self.df)} samples for {split} split")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single example from the dataset"""
        row = self.df.iloc[idx]
        
        # Extract question and answer
        question = row['question']
        answer = row['answer']
        
        # Clean boxed answer format from \boxed{X} to the content X
        clean_answer = self._clean_boxed_answer(answer)
        
        # Extract image path from the dataset
        image_path = self._extract_image_path(row['image'])
        image_full_path = os.path.join(self.images_dir, image_path)
        
        try:
            # Load the image
            image = Image.open(image_full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_full_path}: {e}")
            # Return a dummy black image if the actual image can't be loaded
            image = Image.new('RGB', (224, 224), color='black')
        
        # Encode the question as the prefix
        prefix_data = self.encode_prefix(question)
        
        # Return the item with all required fields
        return {
            'image': image,
            'question': question,
            'answer': clean_answer,
            'category': row.get('category', ''),
            'prefix': prefix_data['prefix'],
            'prefix_tokens': prefix_data['prefix_tokens'],
            'prefix_token_ids': prefix_data['prefix_token_ids'],
        }

    def _clean_boxed_answer(self, answer: str) -> str:
        """Clean the boxed answer format from LaTeX notation"""
        # Extract content from \boxed{content}
        boxed_match = re.search(r'\\boxed\{(.*?)\}', answer)
        if boxed_match:
            return boxed_match.group(1)
        return answer

    def _extract_image_path(self, image_data) -> str:
        """Extract the image filename from the dataset's image field"""
        try:
            # Handle numpy arrays
            if hasattr(image_data, 'dtype') and hasattr(image_data, 'shape'):  # Check if it's numpy array
                # Convert numpy array to string
                if image_data.size > 0:
                    image_data = image_data[0]  # Get first element
            
            # Convert bytes to string if needed
            if isinstance(image_data, bytes):
                image_data = image_data.decode('utf-8', errors='replace')
            
            # Handle NoneType
            if image_data is None:
                return "missing_image.jpg"
            
            # Ensure we're working with a string
            image_data = str(image_data)
            
            # Remove quotes, extra brackets and whitespace
            image_data = image_data.strip("'\"[] \t\n\r")
            
            # Extract the filename from paths like "images/filename.jpg"
            if image_data.startswith('images/'):
                return image_data.split('images/')[1]
            
            return image_data
            
        except Exception as e:
            print(f"Error extracting image path: {e}, using fallback")
            return f"fallback_image_{abs(hash(str(image_data))) % 10000}.jpg"

    def encode_prefix(self, question: str):
        """Encode the question as a chat with system and user messages"""
        user_message = USER_TEMPLATE.format(question=question)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch"""
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        images = [item["image"] for item in batch]
        answers = [item["answer"] for item in batch]
        questions = [item["question"] for item in batch]
        return MiniBatch(
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
            images=images,
            answers=answers,
            questions=questions,
        )


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    # Strip end token if present
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0

    reward = 0.0

    if think_match:
        reward += 0.1

    if answer_match:
        reward += 0.5

    return reward

#### Original implementation
# def answer_reward_function(response: str, answer: str = None) -> float:
#     """
#     Compute how well the extracted answer matches the ground truth.
#     """
#     answer_regex = r"<answer>(.*?)<\/answer>"
#     answer_match = re.search(answer_regex, response, re.DOTALL)
#     if not answer_match:
#         return 0.0

#     extracted_answer = answer_match.group(1).strip()
#     if not extracted_answer or not answer:
#         return 0.0
    
#     # Clean the ground truth (removing \boxed{} if present)
#     if '\\boxed{' in answer:
#         answer = re.search(r'\\boxed\{(.*?)\}', answer).group(1)
    
#     # Normalize both answers for comparison
#     extracted = normalize_answer(extracted_answer)
#     ground = normalize_answer(answer)
    
#     # Check if the answer is a multiple choice option (A, B, C, D)
#     multiple_choice_match = re.match(r'^([a-d])$', ground, re.IGNORECASE)
#     if multiple_choice_match:
#         # For multiple choice, only give full credit for exact match
#         choice_match = re.search(r'([a-d])', extracted, re.IGNORECASE)
#         if choice_match and choice_match.group(1).lower() == multiple_choice_match.group(1).lower():
#             return 1.0
#         return 0.0
    
#     # For numeric answers, check for exact match
#     numeric_match = re.match(r'^[\d\.\-\+\/]+$', ground)
#     if numeric_match:
#         # Try to normalize and compare numbers
#         try:
#             # Extract numeric part from extracted answer
#             extracted_num = re.search(r'([\d\.\-\+\/]+)', extracted)
#             if extracted_num and extracted_num.group(1) == ground:
#                 return 1.0
#             return 0.0
#         except:
#             return 0.0
    
#     # Exact match for everything else
#     if extracted == ground:
#         return 1.0
    
#     # Partial credit only if the answer contains some of the same words
#     # Calculate overlap between words
#     extracted_words = set(extracted.split())
#     ground_words = set(ground.split())
#     common_words = extracted_words.intersection(ground_words)
    
#     if not common_words:
#         return 0.0
    
#     # Much stricter partial credit
#     jaccard = len(common_words) / len(ground_words.union(extracted_words))
#     if jaccard > 0.8:  # High overlap required for partial credit
#         return 0.5
    
#     return 0.0


def parse_sympy_expr(expr_str: str) -> sympy.Expr:
    """
    Attempt to parse a string using sympy. 
    This will handle expressions like '1/2', '0.5', '(4 - 3) / 7', etc.
    If parsing fails, it raises a Sympy exception.
    """
    # Basic cleanup: remove latex fraction commands, e.g. \frac{1}{2} -> (1)/(2)
    # You can expand this if you have more complex LaTeX patterns.
    expr_str = expr_str.replace(r'\frac', '')
    expr_str = expr_str.replace('{', '(').replace('}', ')')
    
    # Remove any TeX escaping like '\\'
    expr_str = expr_str.replace(r'\\', '')

    # Try to simplify spaces or newlines
    expr_str = expr_str.strip()

    # Use sympy's parsing
    parsed_expr = sympy.sympify(expr_str, rational=True)
    # The rational=True flag tries to convert floats to exact rational
    return parsed_expr


def are_equivalent_sympy(expr_a: str, expr_b: str, tol=1e-6) -> bool:
    """
    Check if two expressions are equivalent numerically or symbolically.
    Returns True if they simplify to the same value, or if they're 
    numerically close within a tolerance.
    """
    try:
        parsed_a = parse_sympy_expr(expr_a)
        parsed_b = parse_sympy_expr(expr_b)
        # Attempt symbolic simplification
        diff = sympy.simplify(parsed_a - parsed_b)
        if diff == 0:
            return True
        # If not exactly zero, check numeric closeness
        diff_value = float(diff.evalf())
        return abs(diff_value) < tol
    except:
        # If we fail to parse or simplify, we return False
        return False


def answer_reward_function(response: str, answer: str = None) -> float:
    """
    Compute how well the extracted answer matches the ground truth,
    using Sympy-based comparison for numeric expressions.
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    extracted_answer = answer_match.group(1).strip()
    if not extracted_answer or not answer:
        return 0.0
    
    # Clean the ground truth (removing \boxed{} if present)
    if '\\boxed{' in answer:
        box_match = re.search(r'\\boxed\{(.*?)\}', answer)
        if box_match:
            answer = box_match.group(1)
    
    # Normalize user answer text to reduce punctuation or filler
    extracted = normalize_answer(extracted_answer)
    ground = normalize_answer(answer)
    
    # 1) Check if ground is a multiple choice (A, B, C, or D)
    multiple_choice_match = re.match(r'^([a-d])$', ground, re.IGNORECASE)
    if multiple_choice_match:
        # For multiple choice, only give full credit for exact match
        choice_match = re.search(r'([a-d])', extracted, re.IGNORECASE)
        if choice_match and choice_match.group(1).lower() == multiple_choice_match.group(1).lower():
            return 1.0
        return 0.0
    
    # 2) Try a Sympy-based numeric or symbolic match
    # If the expressions simplify to the same or are numerically close, reward = 1.0
    if are_equivalent_sympy(extracted, ground):
        return 1.0
    
    # 3) If not equivalent, check if they're identical as strings 
    # (maybe purely text answers, e.g., "triangle apex" vs "triangle apex")
    if extracted == ground:
        return 1.0
    
    # 4) As a last resort, partial credit if there's high word overlap
    extracted_words = set(extracted.split())
    ground_words = set(ground.split())
    common_words = extracted_words.intersection(ground_words)
    
    if not common_words:
        return 0.0
    
    # Stricter partial credit
    jaccard = len(common_words) / len(ground_words.union(extracted_words))
    if jaccard > 0.8:  # High overlap required for partial credit
        return 0.5
    
    return 0.0


# def normalize_answer(answer: str) -> str:
#     """Normalize the answer for better comparison"""
#     # Convert to lowercase
#     answer = answer.lower()
    
#     # Remove punctuation and extra whitespace
#     answer = re.sub(r'[^\w\s]', '', answer)
#     answer = re.sub(r'\s+', ' ', answer).strip()
    
#     # Remove common articles and filler words
#     answer = re.sub(r'\b(a|an|the|is|are|was|were)\b', '', answer)
    
#     # Remove extra spaces after word removal
#     answer = re.sub(r'\s+', ' ', answer).strip()
    
#     return answer


def normalize_answer(answer: str) -> str:
    """
    Normalize the answer text while preserving most math symbols for Sympy parsing.
    - Converts to lowercase
    - Removes characters that are not letters, digits, whitespace, or basic math symbols
    - Collapses multiple spaces
    - Removes common filler words (a, an, the, is, are, was, were)
    """

    # 1) Convert to lowercase
    answer = answer.lower()

    # 2) Preserve digits, letters, whitespace, and these math-related symbols:
    #    + - * / ^ ( ) { } = and the backslash for LaTeX commands
    #    Anything else is replaced with a blank.
    allowed_chars_pattern = r'[^0-9a-z+\-\*/^()\{\}=\s\\]'
    answer = re.sub(allowed_chars_pattern, '', answer)

    # 3) Collapse multiple whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()

    # 4) Remove common filler words ("a", "an", "the", etc.)
    #    (If you find these filler removals harm numeric expressions, 
    #     you might remove this step or only apply it to non-numeric answers.)
    filler_pattern = r'\b(a|an|the|is|are|was|were)\b'
    answer = re.sub(filler_pattern, '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    return answer


def virl_reward_function(
    response: str,
    answer: str = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """
    Reward function for ViRL39K.
    
    Args:
        response: The model's response
        answer: The ground truth answer
        end_token: Optional end token to remove from response
    
    Returns:
        Dictionary with reward value and additional reward info
    """
    format_reward = format_reward_function(response, end_token)
    answer_reward = answer_reward_function(response, answer)
    
    # Extract the answer for logging purposes
    extracted_answer = ""
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if answer_match:
        extracted_answer = answer_match.group(1).strip()
    
    # Compute weighted total reward
    total_reward = 0.1 * format_reward + 0.9 * answer_reward
    
    return {
        "reward": total_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "extracted_answer": extracted_answer,
            "expected_answer": answer
        }
    }


