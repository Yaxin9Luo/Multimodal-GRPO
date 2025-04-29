"""
Evaluation script for testing Qwen2.5-VL model capabilities on ViRL39K dataset before RL training.
This script evaluates the base model's performance on visual reasoning tasks without any GRPO training.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_types import MiniBatch
from qwen2vl_model import Transformer
from tokenizer import Tokenizer
from virl_dataset import ViRLDataset, virl_reward_function


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base Qwen2.5-VL model on ViRL39K dataset")
    parser.add_argument("--config", type=str, default="config_vl.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Number of samples to evaluate (use 0 for all)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], 
                        help="Dataset split to evaluate on")
    return parser.parse_args()


def generate_response(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    image: Optional[Image.Image] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Generate a response from the model given a prompt and optional image."""
    
    # Tokenize the prompt
    input_ids = tokenizer.tokenize(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Initialize the KV cache
    model.init_kv_cache(
        max_batch_size=1,
        max_seq_len=len(input_ids[0]) + max_new_tokens,
        device=device,
        dtype=dtype,
    )
    
    # Process images if provided
    images = [image] if image is not None else None
    
    # Track generated tokens
    generated_ids = []
    
    try:
        # First token generation with image (if available)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype):
                if image is not None:
                    logits = model.inference(input_ids, images=images, start_pos=0)
                else:
                    logits = model.inference(input_ids, start_pos=0)
                
                # Temperature sampling
                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits[:, -1], dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits[:, -1], dim=-1)
        
        generated_ids.append(next_token.item())
        
        # Continue generating tokens
        for i in range(max_new_tokens - 1):
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=dtype):
                    # Use previously generated token
                    current_token = next_token.view(1, 1)
                    logits = model.inference(current_token, start_pos=len(input_ids[0]) + i)
                    
                    # Temperature sampling
                    if temperature > 0:
                        logits = logits / temperature
                        probs = torch.softmax(logits[:, -1], dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token_id = next_token.item()
            generated_ids.append(next_token_id)
            
            # Stop if we generate an EOS token
            if next_token_id == tokenizer.eos_token_id:
                break
    
    finally:
        # Always clean up KV cache
        model.del_kv_cache()
    
    # Decode the generated tokens
    generated_text = tokenizer.detokenize(generated_ids)
    return generated_text


def evaluate_model(config_path: str, output_dir: str, num_samples: int, split: str):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device and dtype setup
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    
    # Load model
    print(f"Loading model from {config['model']['pretrained_model_path']}...")
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    model = Transformer.from_pretrained(pretrained_model_path, device=device).eval()
    
    # Load dataset
    print(f"Loading {split} dataset from {config['data']['path']}...")
    dataset = ViRLDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        images_dir=config["data"].get("images_dir"),
        split=split,
        max_samples=num_samples if num_samples > 0 else None,
        test_ratio=config["data"].get("test_ratio", 0.1),
        random_seed=config["training"]["random_seed"],
    )
    
    print(f"Loaded {len(dataset)} samples for evaluation")
    
    # Initialize results
    results = []
    correct_count = 0
    total_count = 0
    
    # Evaluate each sample
    for idx in tqdm(range(len(dataset)), desc="Evaluating samples"):
        try:
            sample = dataset[idx]
            
            # Log the question
            print(f"\nQuestion {idx+1}/{len(dataset)}")
            print(f"Subject: {sample.get('subject', 'N/A')}")
            print(f"Question: {sample.get('question', 'N/A')}")
            
            # Generate response
            prompt = sample["prefix"]
            image = sample.get("image")
            
            generated_text = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                image=image,
                max_new_tokens=config["training"]["max_gen_len"],
                device=device,
                dtype=dtype,
            )
            
            # Calculate reward
            reward_result = virl_reward_function(
                response=generated_text,
                answer=sample.get("answer", ""),
                end_token=tokenizer.eos_token,
            )
            
            # Extract answer from response
            reward = reward_result["reward"]
            reward_info = reward_result["reward_info"]
            
            # Track statistics
            total_count += 1
            if reward_info["answer_reward"] > 0:
                correct_count += 1
            
            # Store the result
            result = {
                "id": idx,
                "question_id": sample.get("id", f"sample_{idx}"),
                "subject": sample.get("subject", "unknown"),
                "level": sample.get("level", 0),
                "question": sample.get("question", ""),
                "options": sample.get("options", []),
                "expected_answer": sample.get("answer", ""),
                "model_response": generated_text,
                "extracted_answer": reward_info.get("extracted_answer", ""),
                "is_correct": reward_info["answer_reward"] > 0,
                "reward": reward,
                "reward_info": reward_info,
            }
            results.append(result)
            
            # Display result
            print(f"Response: {generated_text}")
            print(f"Expected: {sample.get('answer', '')}")
            print(f"Extracted: {reward_info.get('extracted_answer', '')}")
            print(f"Correct: {reward_info['answer_reward'] > 0}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate overall metrics
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Save results
    results_file = output_dir / f"vl_base_evaluation_{split}.json"
    with open(results_file, "w") as f:
        json.dump({
            "config": config,
            "split": split,
            "num_samples": len(dataset),
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": results,
        }, f, indent=2)
    
    # Print summary
    print(f"\nEvaluation complete!")
    print(f"Total samples: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.config, args.output_dir, args.num_samples, args.split) 