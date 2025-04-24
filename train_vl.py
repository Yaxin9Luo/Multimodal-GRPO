import argparse
import html
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

from data_types import MiniBatch
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2vl_model import Transformer
from response_logger import ResponseLogger
from tokenizer import Tokenizer
from virl_dataset import ViRLDataset, virl_reward_function


def evaluate(model, tokenizer, device, dtype, config, response_logger=None):
    """Evaluate model on validation set."""
    with torch.no_grad():
        model.eval()
        print("\nStarting evaluation...")
        try:
            val_dataset = ViRLDataset(
                data_path=config["data"]["path"],
                tokenizer=tokenizer,
                images_dir=config["data"].get("images_dir"),
                split="val",
                max_samples=config["data"].get("max_val_samples"),
                test_ratio=config["data"].get("test_ratio", 0.1),
                random_seed=config["training"]["random_seed"],
            )
            val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                collate_fn=ViRLDataset.collate_fn,
                batch_size=config["training"]["num_questions_per_batch"],
            )
            
            print(f"Loaded {len(val_dataset)} validation samples")
            num_correct = 0
            num_total = 0
            
            # Add timeout for evaluation
            max_batches = len(val_dataloader)
            batch_count = 0
            max_eval_batches = config["training"].get("max_eval_batches", max_batches)
            max_eval_batches = min(max_eval_batches, max_batches)
            
            print(f"Processing {max_eval_batches} batches out of {max_batches} total")
            
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= max_eval_batches:
                    print(f"Reached max eval batches ({max_eval_batches}), stopping evaluation")
                    break
                    
                batch_count += 1
                print(f"\nEvaluating batch {batch_idx+1}/{max_eval_batches}")
                
                try:
                    episodes = rollout(
                        model=model,
                        tokenizer=tokenizer,
                        batch=batch,
                        max_gen_len=config["training"]["max_gen_len"],
                        # only generate one answer per question during evaluation
                        num_answer_per_question=1,
                        reward_function=virl_reward_function,
                        device=device,
                        dtype=dtype,
                        is_multimodal=True,
                    )
                    
                    # Log evaluation responses if logger is provided
                    if response_logger is not None:
                        # Log with evaluation metadata
                        for episode in episodes:
                            # Add evaluation tag to metadata
                            metadata = {
                                "is_finished": episode.is_finished,
                                "response_length": len(episode.generated_token_ids),
                                "split": "eval"
                            }
                            response_logger.log_response(
                                question=episode.prefix,
                                response=episode.text[len(episode.prefix):],
                                image=episode.image if hasattr(episode, "image") else None,
                                expected_answer=episode.reward_info.get("expected_answer"),
                                extracted_answer=episode.reward_info.get("extracted_answer"),
                                reward=episode.reward,
                                reward_info=episode.reward_info,
                                metadata=metadata
                            )
                    
                    for episode in episodes:
                        num_total += 1
                        # Only count the answer part of the reward
                        num_correct += episode.reward_info["answer_reward"]
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Guard against division by zero
            if num_total == 0:
                print("Warning: No samples were successfully evaluated!")
                success_rate = 0.0
            else:
                success_rate = num_correct / num_total
                
            print(f"Evaluation completed. Success rate: {success_rate:.4f} ({num_correct}/{num_total})")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            success_rate = 0.0

    # Make sure to return to training mode
    model.train()
    return success_rate


def init_wandb(config):
    """Initialize Weights & Biases for experiment tracking"""
    if not wandb_available:
        print("wandb not available, skipping wandb initialization")
        return False
    
    try:
        # Initialize wandb with the configuration from the config file
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["name"],
            tags=config["wandb"]["tags"],
            notes=config["wandb"]["notes"],
            mode=config["wandb"]["mode"],
            config=config,
        )
        print(f"Initialized wandb run: {wandb.run.name}")
        return True
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return False


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb if available and enabled in config
    wandb_enabled = wandb_available and config["training"].get("wandb_enabled", False)
    if wandb_enabled:
        init_wandb(config)
    
    # Set up directories
    Path(config["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["training"]["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Initialize response logger
    response_logger = ResponseLogger(log_dir=config["training"].get("response_log_dir", "logs_responses"))
    
    # Initialize model and training environment
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH
    MAX_STEPS = config["training"].get("max_steps", 0)  # Get max_steps or default to 0 (unlimited)

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/virl_{current_time}")
    
    # Log the directory for TensorBoard logs to wandb
    if wandb_enabled:
        wandb.config.update({"tensorboard_log_dir": f"{config['training']['log_dir']}/virl_{current_time}"})
    
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = ViRLDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        images_dir=config["data"].get("images_dir"),
        split="train",
        max_samples=config["data"].get("max_train_samples"),
        test_ratio=config["data"].get("test_ratio", 0.1),
        random_seed=config["training"]["random_seed"],
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=ViRLDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    # Initialize the Qwen2.5-VL model
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    # Freeze the vision encoder
    model.freeze_vision_encoder()

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Watch the model in wandb to track gradients, parameters, etc.
    if wandb_enabled:
        wandb.watch(model, log="all", log_freq=config["training"]["eval_interval"])

    # Training loop
    step = 1
    epoch = 1
    
    print(f"Starting training for {MAX_STEPS} steps")
    
    while step <= MAX_STEPS or MAX_STEPS == 0:
        print(f"Epoch {epoch} started")
        
        for batch in train_dataloader:
            if MAX_STEPS > 0 and step > MAX_STEPS:
                break
                
            print(f"Processing step {step}/{MAX_STEPS}")
            
            episodes = rollout(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_gen_len=config["training"]["max_gen_len"],
                num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                reward_function=virl_reward_function,
                device=device,
                dtype=dtype,
                is_multimodal=True,
            )
            
            # Log model responses and images
            print(f"Logging responses for batch {step}...")
            response_logger.log_batch_responses(episodes)
            
            if config["training"]["skip_unfinished_episodes"]:
                episodes = [episode for episode in episodes if episode.is_finished]
            results = update_policy(
                model=model,
                optimizer=optimizer,
                episodes=episodes,
                micro_batch_size=config["training"]["micro_batch_size"],
                pad_token_id=tokenizer.pad_token_id,
                max_grad_norm=config["training"]["max_grad_norm"],
                device=device,
                dtype=dtype,
                is_multimodal=True,
            )
            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            # compute and log important metrics
            reward = [episode.reward for episode in episodes]
            answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
            format_reward = [episode.reward_info["format_reward"] for episode in episodes]
            num_finished_episodes = sum(episode.is_finished for episode in episodes)
            mean_reward = np.mean(reward)
            std_reward = np.std(reward)
            success_rate = np.mean(answer_reward)
            format_score = np.mean(format_reward)
            grad_norm = results["grad_norm"]
            entropy = results["entropy"]
            lr = optimizer.param_groups[0]["lr"]
            loss = results["loss"]
            mean_response_len = np.mean(
                [len(episode.generated_token_ids) for episode in episodes]
            )
            
            # Log metrics to terminal
            print(
                f"\rStep {step}/{MAX_STEPS}, mean_reward: {mean_reward:.2f}, "
                f"train success_rate: {success_rate:.2f}, "
                f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
                f"num_finished_episodes: {num_finished_episodes}, "
                f"mean_response_len: {mean_response_len:.2f}, "
                f"entropy: {entropy:.2f}"
            )
            
            # Evaluate if needed
            if step % config["training"]["eval_interval"] == 0:
                eval_success_rate = evaluate(model, tokenizer, device, dtype, config, response_logger)
                print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
                tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)
                
                # Log evaluation metrics to wandb
                if wandb_enabled:
                    wandb.log({"success_rate/eval": eval_success_rate}, step=step)

            # Log metrics to TensorBoard
            tb_writer.add_scalar("loss", loss, step)
            tb_writer.add_scalar("mean_reward", mean_reward, step)
            tb_writer.add_scalar("std_reward", std_reward, step)
            tb_writer.add_scalar("success_rate/train", success_rate, step)
            tb_writer.add_scalar("format_reward", format_score, step)
            tb_writer.add_scalar("grad_norm", grad_norm, step)
            tb_writer.add_scalar("duration", duration, step)
            tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
            tb_writer.add_scalar("learning_rate", lr, step)
            tb_writer.add_scalar("mean_response_len", mean_response_len, step)
            tb_writer.add_scalar("entropy", entropy, step)
            
            # Log metrics to wandb
            if wandb_enabled:
                wandb_log = {
                    "loss": loss,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "success_rate/train": success_rate,
                    "format_reward": format_score,
                    "grad_norm": grad_norm,
                    "duration": duration,
                    "num_finished_episodes": num_finished_episodes,
                    "learning_rate": lr,
                    "mean_response_len": mean_response_len,
                    "entropy": entropy,
                    "epoch": epoch,
                    "progress": step / MAX_STEPS if MAX_STEPS > 0 else 0
                }
                wandb.log(wandb_log, step=step)
                
                # Log some example generated texts to wandb
                if step % config["training"]["eval_interval"] == 0:
                    wandb_examples = []
                    # Log a few examples (limit to 5 to avoid too much data)
                    for i, episode in enumerate(episodes[:5]):
                        # Can't log images directly in table, so log just text
                        wandb_examples.append(
                            wandb.Table(
                                columns=["question", "response", "reward", "extracted_answer", "expected_answer"],
                                data=[[
                                    episode.prefix,
                                    episode.text[len(episode.prefix):],  # Just the generated part
                                    episode.reward,
                                    episode.reward_info["extracted_answer"],
                                    episode.reward_info["expected_answer"]
                                ]]
                            )
                        )
                    wandb.log({"examples": wandb_examples}, step=step)
            
            # Log text samples to TensorBoard
            for i, episode in enumerate(episodes[:5]):  # Log only 5 examples to save space
                # TensorBoard treats text as markdown.
                text = html.escape(episode.text)
                tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

            # save checkpoint
            if step % config["training"]["ckpt_save_interval"] == 0:
                output_file = ckpt_dir / f"virl_ckpt_{step:06d}.pt"
                torch.save(model.state_dict(), output_file)
                print(f"Saved checkpoint to {output_file}")
                
                # Log checkpoint to wandb
                if wandb_enabled:
                    # Save the checkpoint to wandb artifacts
                    artifact = wandb.Artifact(
                        name=f"virl-model-checkpoint-{step}", 
                        type="model",
                        description=f"ViRL model checkpoint at step {step}"
                    )
                    artifact.add_file(str(output_file))
                    wandb.log_artifact(artifact)
            
            step += 1
            
        epoch += 1
        
        # Save final checkpoint if we've reached max steps
        if MAX_STEPS > 0 and step > MAX_STEPS:
            output_file = ckpt_dir / f"virl_ckpt_final.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved final checkpoint to {output_file}")
            if wandb_enabled:
                artifact = wandb.Artifact(
                    name="virl-model-checkpoint-final", 
                    type="model",
                    description="Final ViRL model checkpoint"
                )
                artifact.add_file(str(output_file))
                wandb.log_artifact(artifact)
            break

    # Finalize wandb run
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_vl.yaml")
    args = parser.parse_args()
    main(args.config) 