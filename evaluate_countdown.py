import argparse
import os
import torch
from pathlib import Path

from tokenizer import Tokenizer
from qwen2_model import Transformer as TextModel
from countdown_task import USER_TEMPLATE, SYSTEM_MESSAGE, RESPONSE_PROMPT


def load_model(model_path, original_model_dir, device):
    """Load model from checkpoint with exact same method as training."""
    print(f"Loading model from {model_path}")
    print(f"Using original model directory: {original_model_dir}")
    
    # Initialize model from original weights
    model = TextModel.from_pretrained(original_model_dir, device=device)
    
    # Load fine-tuned checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is valid
    print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")
    
    # Load state dictionary
    model.load_state_dict(checkpoint)
    
    # Verify a few parameters to ensure the model was loaded
    first_param = next(model.parameters())
    print(f"First parameter shape: {first_param.shape}, mean: {first_param.mean().item():.6f}")
    
    model.eval()
    return model


@torch.no_grad()
def generate_answer(model, tokenizer, numbers, target, device="cuda", max_new_tokens=512, temperature=0.7):
    """Generate an answer to the countdown task using the exact training format."""
    # Format the prompt exactly as in training
    user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
    
    # Use the exact method from the CountdownTasksDataset
    prefix = tokenizer.encode_chat_with_response_prompt(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ],
        RESPONSE_PROMPT,
    )
    tokens = tokenizer.tokenize(prefix)
    prefix_token_ids = tokens.ids
    
    print("\nPrompt:")
    print(f"System: {SYSTEM_MESSAGE}")
    print(f"User: {user_message}")
    print(f"Response start: {RESPONSE_PROMPT}")
    
    # Prepare input
    input_ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=device)
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Generate in a manner more similar to the training rollout function
    total_len = len(prefix_token_ids) + max_new_tokens
    tokens = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=device)
    tokens[0, :len(prefix_token_ids)] = torch.tensor(prefix_token_ids, dtype=torch.long, device=device)
    
    # Initialize KV cache
    model.init_kv_cache(
        max_batch_size=1,
        max_seq_len=total_len,
        device=device,
        dtype=torch.bfloat16,
    )
    
    # Generate tokens
    input_text_mask = tokens != pad_token_id
    is_finished = torch.zeros((1,), dtype=torch.bool, device=device)
    prev_pos = 0
    
    try:
        for cur_pos in range(len(prefix_token_ids), total_len):
            # Print progress every 10 tokens
            if (cur_pos - len(prefix_token_ids)) % 10 == 0:
                print(f"\rGenerating: {cur_pos-len(prefix_token_ids)}/{max_new_tokens}", end="", flush=True)
            
            # Get logits with autocast
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature if needed
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Sample from distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Use argmax for very low temperature, multinomial otherwise
            if temperature <= 0.01:
                next_token = torch.argmax(probs).item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for mask
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], torch.tensor([next_token], device=device)
            )[0].item()
            
            # Handle finished sequences
            next_token = torch.where(is_finished, pad_token_id, next_token).item()
            tokens[:, cur_pos] = next_token
            
            # Check for end token
            if end_token_id is not None:
                is_end_token = next_token == end_token_id
                is_generated_token = ~input_text_mask[:, cur_pos]
                is_finished = is_finished | (is_end_token & is_generated_token)
            
            prev_pos = cur_pos
            
            # Early stopping if finished
            if is_finished.all():
                break
    
    finally:
        print("\rGeneration completed                    ")
        # Clean up
        model.del_kv_cache()
    
    # Get the generated text (only the generated part)
    generated_token_ids = tokens[0, len(prefix_token_ids):cur_pos].tolist()
    # Remove padding tokens if any
    if pad_token_id in generated_token_ids:
        generated_token_ids = generated_token_ids[:generated_token_ids.index(pad_token_id)]
    
    generated_text = tokenizer.detokenize(generated_token_ids)
        
    return {
        "generated_text": generated_text,
        "full_text": prefix + generated_text
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on countdown tasks")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--original_model_dir", type=str, default="Qwen2.5-3B-Instruct",
                        help="Path to the original model directory for initialization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Temperature for generation (lower = more deterministic)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    torch.set_default_device(device)
    
    # Load tokenizer from original model
    tokenizer_path = Path(args.original_model_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = Tokenizer(str(tokenizer_path))
    
    # Load model
    model = load_model(
        model_path=args.model_path,
        original_model_dir=args.original_model_dir,
        device=device
    )
    
    print("\nCountdown Task Evaluator")
    print(f"Temperature: {args.temperature}")
    print("Type 'exit' to quit")
    
    while True:
        try:
            print("\nEnter numbers separated by commas (e.g., 97,31,25):")
            numbers_input = input("> ")
            
            if numbers_input.lower() == 'exit':
                break
                
            # Parse numbers
            try:
                numbers = [int(n.strip()) for n in numbers_input.split(',')]
                if len(numbers) != 3 and len(numbers) != 4:
                    print("Please enter 3 or 4 numbers")
                    continue
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas")
                continue
            
            # Get target
            print("Enter target number:")
            target_input = input("> ")
            
            if target_input.lower() == 'exit':
                break
                
            # Parse target
            try:
                target = int(target_input.strip())
            except ValueError:
                print("Invalid input. Please enter a number")
                continue
            
            # Generate answer
            print("\nGenerating solution...")
            result = generate_answer(
                model=model,
                tokenizer=tokenizer,
                numbers=numbers,
                target=target,
                device=device,
                temperature=args.temperature
            )
            
            print("\nModel response:")
            print(result["generated_text"])
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 