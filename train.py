import html
import json
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def save_responses_to_json(episodes, step, log_file_path):
    """Save model responses to a JSON file.
    
    Args:
        episodes: List of episodes from rollout.
        step: Current training step.
        log_file_path: Path to the JSON log file.
    """
    responses_data = []
    for i, episode in enumerate(episodes):
        response_data = {
            "step": step,
            "episode_id": i,
            "prompt": episode.prefix,
            "response": episode.text[len(episode.prefix):],  # Just the generated part
            "full_text": episode.text,
            "reward": float(episode.reward),
            "format_reward": float(episode.reward_info["format_reward"]),
            "answer_reward": float(episode.reward_info["answer_reward"]),
            "is_finished": bool(episode.is_finished),
            "numbers": [],  # Default empty list
            "target": None,  # Default None
            "timestamp": datetime.now().isoformat()
        }
        responses_data.append(response_data)
    
    # Instead of appending, replace the entire contents with just the current step's responses
    with open(log_file_path, "w") as f:
        json.dump(responses_data, f, indent=2)


def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


def init_wandb(config):
    """Initialize wandb if enabled in config."""
    if "wandb" not in config or not config["wandb"].get("enabled", False):
        return False
    
    wandb_config = config["wandb"]
    run_name = wandb_config.get("name")
    if run_name is None:
        # Auto-generate a name if not provided
        model_name = Path(config["model"]["pretrained_model_path"]).name
        run_name = f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Initialize wandb
    wandb.init(
        project=wandb_config.get("project", "GRPO-Zero"),
        name=run_name,
        config=config,
        tags=wandb_config.get("tags", []),
        notes=wandb_config.get("notes", ""),
        mode=wandb_config.get("mode", "online"),
    )
    
    # Log the configuration parameters
    wandb.config.update({
        "model_path": config["model"]["pretrained_model_path"],
        "device": config["model"]["device"],
        "dtype": config["model"]["dtype"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "memory_efficient_adamw": config["training"]["memory_efficient_adamw"],
    })
    
    return True


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize wandb if enabled
    wandb_enabled = init_wandb(config)

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

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    
    # Create a responses log file
    log_dir = Path(config["training"].get("responses_log_dir", config["training"]["log_dir"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    responses_log_file = log_dir / f"responses_{current_time}.json"
    
    # Log the directory for TensorBoard logs to wandb
    if wandb_enabled:
        wandb.config.update({
            "tensorboard_log_dir": f"{config['training']['log_dir']}/{current_time}",
            "responses_log_file": str(responses_log_file)
        })
    
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

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

    for step, batch in enumerate(train_dataloader, start=1):
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        
        # Save model responses to JSON file
        log_interval = config["training"].get("response_log_interval", config["training"]["eval_interval"])
        if step % log_interval == 0:
            save_responses_to_json(episodes, step, responses_log_file)
        
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # compute and log important metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )
        
        # Log metrics to terminal
        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )
        
        # Evaluate if needed
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)
            
            # Log evaluation metrics to wandb
            if wandb_enabled:
                wandb.log({"success_rate/eval": eval_success_rate}, step=step)
                
            # Log eval responses to JSON as well
            test_dataset = CountdownTasksDataset(
                data_path=config["data"]["path"],
                tokenizer=tokenizer,
                split="test",
                test_size=config["data"]["test_size"],
            )
            generator = torch.Generator(device=device)
            eval_dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                collate_fn=CountdownTasksDataset.collate_fn,
                generator=generator,
                batch_size=min(5, config["data"]["test_size"]),  # Log a small subset of eval examples
                drop_last=False,
            )
            
            for eval_batch in eval_dataloader:
                eval_episodes = rollout(
                    model=model,
                    tokenizer=tokenizer,
                    batch=eval_batch,
                    max_gen_len=config["training"]["max_gen_len"] * 2,
                    num_answer_per_question=1, 
                    reward_function=reward_function,
                    device=device,
                    dtype=dtype,
                )
                eval_log_file = log_dir / f"eval_responses_{current_time}.json"
                save_responses_to_json(eval_episodes, step, eval_log_file)
                break  # Just log one batch of eval examples

        # Log metrics to TensorBoard
        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
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
                "format_reward": format_reward,
                "grad_norm": grad_norm,
                "duration": duration,
                "num_finished_episodes": num_finished_episodes,
                "learning_rate": lr,
                "mean_response_len": mean_response_len,
                "entropy": entropy,
            }
            wandb.log(wandb_log, step=step)
            
            # Log some example generated texts to wandb
            if step % config["training"]["eval_interval"] == 0:
                wandb_examples = []
                # Log a few examples (limit to 5 to avoid too much data)
                for i, episode in enumerate(episodes[:5]):
                    wandb_examples.append(
                        wandb.Table(
                            columns=["prompt", "response", "reward", "format_reward", "answer_reward"],
                            data=[[
                                episode.prefix,
                                episode.text[len(episode.prefix):],  # Just the generated part
                                episode.reward,
                                episode.reward_info["format_reward"],
                                episode.reward_info["answer_reward"]
                            ]]
                        )
                    )
                wandb.log({"examples": wandb_examples}, step=step)
        
        # Log text samples to TensorBoard
        for i, episode in enumerate(episodes):
            # TensorBoard treats text as markdown.
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")
            
            # Log checkpoint to wandb
            if wandb_enabled:
                # Save the checkpoint to wandb artifacts
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-{step}", 
                    type="model",
                    description=f"Model checkpoint at step {step}"
                )
                artifact.add_file(str(output_file))
                wandb.log_artifact(artifact)

    # Finalize wandb run
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
