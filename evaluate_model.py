import argparse
import os
import re
import torch
from pathlib import Path
from PIL import Image

from tokenizer import Tokenizer
from qwen2_model import Transformer as TextModel
from qwen2vl_model import Transformer as VLModel


def load_model(model_path, model_type, device, dtype, original_model_dir=None):
    """Load the specified model type from checkpoint."""
    model_path = Path(model_path)
    
    # If we're loading a checkpoint file and not a model directory
    if os.path.isfile(model_path):
        if original_model_dir is None:
            # Default model dirs based on model type
            if model_type == "text":
                original_model_dir = "Qwen2.5-3B-Instruct"
            else:  # vl
                original_model_dir = "Qwen2.5-VL-3B-Instruct"
                
        # Use the original model directory for initialization
        print(f"Using original model directory: {original_model_dir}")
        
        if model_type == "text":
            model = TextModel.from_pretrained(original_model_dir, device=device)
        elif model_type == "vl":
            model = VLModel.from_pretrained(original_model_dir, device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load the weights from checkpoint
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        # It's a model directory, not a checkpoint file
        if model_type == "text":
            model = TextModel.from_pretrained(model_path, device=device)
        elif model_type == "vl":
            model = VLModel.from_pretrained(model_path, device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()  # Set to evaluation mode
    print(f"Model loaded successfully: {model_type} model")
    return model


def is_countdown_task(prompt):
    """Detect if the prompt is a countdown task."""
    # Check for numbers and target pattern
    numbers_pattern = r"numbers\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?"
    target_pattern = r"equation that equals\s*\[?\s*(\d+)\s*\]?"
    
    has_numbers = re.search(numbers_pattern, prompt)
    has_target = re.search(target_pattern, prompt)
    
    return has_numbers and has_target


def format_countdown_prompt(prompt):
    """Format the prompt as a countdown task."""
    # Extract numbers and target
    numbers_match = re.search(r"numbers\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?", prompt)
    target_match = re.search(r"equation that equals\s*\[?\s*(\d+)\s*\]?", prompt)
    
    if numbers_match and target_match:
        numbers = [int(numbers_match.group(1)), int(numbers_match.group(2)), int(numbers_match.group(3))]
        target = int(target_match.group(1))
        
        # Format using the exact template from the training set
        formatted_prompt = (
            f"Using the numbers {numbers}, create an equation that equals {target}. "
            "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
            "Show your work in <think> </think> tags. "
            "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
        )
        return formatted_prompt
    
    return prompt


def encode_prompt(tokenizer, prompt, system_message=None):
    """Encode the prompt with optional system message."""
    # Check if this is a countdown task and format it properly
    if is_countdown_task(prompt):
        prompt = format_countdown_prompt(prompt)
        print(f"Detected countdown task. Formatted prompt: {prompt}")
        
        # Use specific system message for countdown task
        system_message = (
            "You are a helpful assistant. You first think about the reasoning process "
            "in your mind and then provide the user with the answer."
        )
    
    if system_message:
        # Use chat format
        return tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "Let me solve this step by step.\n<think>"
        )
    else:
        # Simple prompt
        return tokenizer.encode(prompt)


def load_image(image_path):
    """Load and preprocess an image."""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


@torch.no_grad()
def generate_response(model, tokenizer, prompt, model_type, system_message=None, image_path=None, device='cuda', 
                     max_new_tokens=512, temperature=0.7, top_p=0.9, dtype=torch.bfloat16):
    """Generate a response from the model."""
    # Encode the prompt
    prefix = encode_prompt(tokenizer, prompt, system_message)
    prefix_tokens = tokenizer.tokenize(prefix)
    prefix_token_ids = prefix_tokens.ids
    
    # Prepare the input tokens
    input_ids = torch.tensor([prefix_token_ids], dtype=torch.long, device=device)
    
    # Load image if provided and model is vision-language
    image = None
    if model_type == "vl" and image_path:
        image = load_image(image_path)
        if image is None:
            print("Warning: Failed to load image, proceeding without it")
        else:
            print(f"Image loaded from {image_path}")
    
    # Prepare KV cache for generation with the same dtype as the model
    with torch.autocast(device_type=device.type, dtype=dtype):
        model.init_kv_cache(
            max_batch_size=1,
            max_seq_len=len(prefix_token_ids) + max_new_tokens,
            device=device,
            dtype=dtype,
        )
    
    # Generate the response
    generated_ids = []
    
    try:
        # First forward pass (includes the prefix)
        curr_ids = input_ids
        position = 0
        
        for i in range(max_new_tokens):
            # Forward pass with autocast to ensure consistent dtype
            with torch.autocast(device_type=device.type, dtype=dtype):
                if model_type == "vl" and image is not None and i == 0:
                    # For the first pass in VL model, include the image
                    logits = model.inference(curr_ids, images=[image], start_pos=position)
                else:
                    # For subsequent passes or non-VL models
                    logits = model.inference(curr_ids, start_pos=position)
                
                # Get the logits for the last token
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature and top-p sampling
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                if top_p > 0:
                    # Apply top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=0), dim=0)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[0] = False  # Keep at least one token
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample from the distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=0)
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to generated tokens
            generated_ids.append(next_token)
            
            # Check if we hit the end token
            if next_token == tokenizer.eos_token_id:
                break
            
            # Prepare for next iteration
            curr_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            position += 1
    
    finally:
        # Clean up KV cache
        model.del_kv_cache()
    
    # Detokenize the response
    full_ids = prefix_token_ids + generated_ids
    generated_text = tokenizer.detokenize(generated_ids)
    
    return {
        "prompt": prefix,
        "response": generated_text,
        "full_text": tokenizer.detokenize(full_ids),
    }


def interactive_mode(model, tokenizer, model_type, device, system_message=None, dtype=torch.bfloat16):
    """Run interactive mode with the model."""
    print(f"\nEntering interactive mode with {model_type} model")
    print("Type 'exit' to quit, 'image:/path/to/image.jpg' to load an image (VL model only)")
    print("For countdown tasks, use format: 'Using the numbers [1,2,3], create an equation that equals 10'")
    
    current_image_path = None
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Exiting interactive mode")
                break
            
            # Check if the input is an image path command
            if user_input.startswith("image:"):
                if model_type != "vl":
                    print("Image input is only supported for vision-language models")
                    continue
                
                image_path = user_input[6:].strip()
                current_image_path = image_path
                print(f"Image set to: {current_image_path}")
                continue
            
            # Generate a response
            result = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                model_type=model_type,
                system_message=system_message,
                image_path=current_image_path,
                device=device,
                dtype=dtype,
            )
            
            print(f"\nModel: {result['response']}")
            
        except KeyboardInterrupt:
            print("\nExiting due to user interrupt")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GRPO model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model checkpoint or directory")
    parser.add_argument("--model_type", type=str, choices=["text", "vl"], required=True,
                        help="Type of model: 'text' for text-only, 'vl' for vision-language")
    parser.add_argument("--original_model_dir", type=str, default=None,
                        help="Path to the original model directory with config files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--system_message", type=str, default=None,
                        help="Optional system message for the prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type to use for the model")
    
    args = parser.parse_args()
    
    # Set up the device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    torch.set_default_device(device)
    
    # Set default system messages based on model type
    if args.system_message is None:
        if args.model_type == "text":
            args.system_message = (
                "You are a helpful assistant. You first think about the reasoning process "
                "in your mind and then provide the user with the answer."
            )
        else:  # vl
            args.system_message = (
                "You are a helpful vision-language assistant. You are given an image and a question about it. "
                "ALWAYS analyze the image carefully and provide a DETAILED step-by-step reasoning before giving your final answer. "
                "Be thorough and consider all relevant aspects of the image before reaching a conclusion."
            )
    
    # Determine original model directory if needed and not provided
    model_path = Path(args.model_path)
    if os.path.isfile(model_path) and args.original_model_dir is None:
        # Set default model paths based on model type
        if args.model_type == "text":
            args.original_model_dir = "Qwen2.5-3B-Instruct"
        else:  # vl
            args.original_model_dir = "Qwen2.5-VL-3B-Instruct"
    
    # Load tokenizer from the original model directory
    if os.path.isfile(model_path):
        tokenizer_path = Path(args.original_model_dir) / "tokenizer.json"
    else:
        tokenizer_path = model_path / "tokenizer.json"
    
    if not tokenizer_path.exists():
        raise ValueError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = Tokenizer(str(tokenizer_path))
    
    # Load the model
    model = load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device,
        dtype=dtype,
        original_model_dir=args.original_model_dir
    )
    
    # Run interactive mode
    interactive_mode(
        model=model,
        tokenizer=tokenizer,
        model_type=args.model_type,
        device=device,
        system_message=args.system_message,
        dtype=dtype,
    )


if __name__ == "__main__":
    main() 