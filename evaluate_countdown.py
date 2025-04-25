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
    model.load_state_dict(checkpoint)
    
    model.eval()
    return model


@torch.no_grad()
def generate_answer(model, tokenizer, numbers, target, device="cuda", max_new_tokens=128):
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
    
    # Initialize KV cache
    model.init_kv_cache(
        max_batch_size=1,
        max_seq_len=len(prefix_token_ids) + max_new_tokens,
        device=device,
        dtype=torch.bfloat16,
    )
    
    # Generate tokens
    generated_ids = []
    
    try:
        curr_ids = input_ids
        position = 0
        
        for i in range(max_new_tokens):
            # Get next token logits
            logits = model.inference(curr_ids, start_pos=position)
            next_token_logits = logits[0, -1, :]
            
            # Sample next token
            probs = torch.nn.functional.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to generated sequence
            generated_ids.append(next_token)
            
            # Stop if end token
            if next_token == tokenizer.eos_token_id:
                break
            
            # Prepare for next iteration
            curr_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            position += 1
    
    finally:
        # Clean up
        model.del_kv_cache()
    
    # Get the generated text
    generated_text = tokenizer.detokenize(generated_ids)
    full_text = prefix + generated_text
    
    return {
        "generated_text": generated_text,
        "full_text": full_text
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on countdown tasks")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--original_model_dir", type=str, default="Qwen2.5-3B-Instruct",
                        help="Path to the original model directory for initialization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    
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
                device=device
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