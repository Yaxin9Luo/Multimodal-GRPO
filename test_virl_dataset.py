import os
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tokenizer import Tokenizer
from virl_dataset import ViRLDataset, virl_reward_function

def display_sample(sample):
    """Display a sample from the dataset."""
    try:
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Category: {sample['category']}")
        print(f"Prefix length: {len(sample['prefix_token_ids'])}")
        
        # Check image
        if 'image' in sample and sample['image'] is not None:
            img = sample['image']
            print(f"Image size: {img.size}, mode: {img.mode}")
            # Optionally save image
            img.save("sample_image.png")
            print("Image saved as sample_image.png")
        else:
            print("No image found in sample")
    except Exception as e:
        print(f"Error displaying sample: {e}")
        print(f"Sample keys: {list(sample.keys())}")

def test_reward_function():
    """Test the reward function with different responses."""
    ground_truth = "A"
    
    # Test a short response
    response = "Looking at the image, I can see that the answer is A."
    reward_info = virl_reward_function(response, ground_truth)
    print(f"Short response: '{response}'")
    print(f"Ground truth: '{ground_truth}'")
    print(f"Extracted answer: '{reward_info['reward_info']['extracted_answer']}'")
    print(f"Answer reward: {reward_info['reward_info']['answer_reward']}")
    print(f"Format reward: {reward_info['reward_info']['format_reward']}")
    print(f"Total reward: {reward_info['reward']}")
    print()
    
    # Test minimal tags without detailed reasoning
    response = "<think>The answer is A</think>\n<answer>A</answer>"
    reward_info = virl_reward_function(response, ground_truth)
    print(f"Minimal tags without details: '{response}'")
    print(f"Format reward: {reward_info['reward_info']['format_reward']}")
    print(f"Total reward: {reward_info['reward']}")
    print()
    
    # Test detailed step-by-step reasoning
    detailed_response = """<think>
First, I need to analyze the image carefully. I can see this is a multiple-choice question with options labeled A through D.

Step 1: I notice that the image shows a geometric shape with a dotted line running through it.
Step 2: Looking at the shape, it appears to be symmetrical across the dotted line.
Step 3: When I check both sides of the dotted line, the elements on each side are mirror images of each other.
Step 4: This is the definition of a line of symmetry - a line that divides a shape into two mirror-image halves.

Based on my observation, the dotted line does indeed create identical mirror halves, meeting the criteria for a line of symmetry.
</think>

<answer>A</answer>"""
    
    reward_info = virl_reward_function(detailed_response, ground_truth)
    print(f"Detailed step-by-step reasoning (truncated):")
    print(f"Format reward: {reward_info['reward_info']['format_reward']}")
    print(f"Total reward: {reward_info['reward']}")
    print()
    
    # Test wrong answer but good reasoning
    wrong_but_detailed = """<think>
First, I need to analyze the image carefully. I can see this is a multiple-choice question with options labeled A through D.

Step 1: I notice that the image shows a geometric shape with a dotted line running through it.
Step 2: Looking at the shape, it appears to be asymmetrical across the dotted line.
Step 3: When I check both sides of the dotted line, the elements on each side are different from each other.
Step 4: For a line of symmetry, both sides must be mirror images, which is not the case here.

Based on my observation, the dotted line does not create identical mirror halves, so it's not a line of symmetry.
</think>

<answer>B</answer>"""
    
    reward_info = virl_reward_function(wrong_but_detailed, ground_truth)
    print(f"Wrong answer but good reasoning (truncated):")
    print(f"Answer reward: {reward_info['reward_info']['answer_reward']}")
    print(f"Format reward: {reward_info['reward_info']['format_reward']}")
    print(f"Total reward: {reward_info['reward']}")
    print()

def main():
    # Initialize tokenizer
    tokenizer_path = "Qwen2.5-VL-3B-Instruct/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}. Please update the path.")
        return
    
    tokenizer = Tokenizer(tokenizer_path)
    
    # Load a small subset of the dataset
    print("Loading ViRL39K dataset...")
    try:
        dataset = ViRLDataset(
            data_path="ViRL39K",
            images_dir="/data/yaxin/GRPO-Zero/ViRL39K/images",
            tokenizer=tokenizer,
            split="train",
            max_samples=5,  # Only load 5 samples for testing
        )
        
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Let's examine raw image values
        print("\nExamining raw image data values:")
        for i in range(min(5, len(dataset.df))):
            raw_image_data = dataset.df.iloc[i]['image']
            print(f"Row {i}, Raw image data: {type(raw_image_data)}, Value: {raw_image_data}")
            # Try to extract the path
            extracted_path = dataset._extract_image_path(raw_image_data)
            print(f"Extracted path: {extracted_path}")
            print(f"Full image path: {os.path.join(dataset.images_dir, extracted_path)}")
            print(f"File exists: {os.path.exists(os.path.join(dataset.images_dir, extracted_path))}")
            print()
        
        # Display a few samples
        for i in range(min(3, len(dataset))):
            try:
                print(f"\nSample {i+1}:")
                sample = dataset[i]
                display_sample(sample)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test the dataloader
    print("\nTesting DataLoader...")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=ViRLDataset.collate_fn,
        )
        
        batch = next(iter(dataloader))
        print(f"Batch size: {len(batch.prefix)}")
        print(f"Number of images: {len(batch.images)}")
    except Exception as e:
        print(f"Error testing dataloader: {e}")
    
    # Test the reward function
    print("\nTesting reward function...")
    test_reward_function()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 