import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List, Optional, Dict, Any, Union

import numpy as np
import torch
from PIL import Image

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer


@torch.no_grad()
def rollout(
    model: Union[Transformer, Any],
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    is_multimodal: bool = False,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    # Process images if working with multimodal data
    images = None
    if is_multimodal and hasattr(batch, "images") and batch.images is not None:
        # Duplicate each image based on num_answer_per_question
        duplicated_images = []
        for image in batch.images:
            for _ in range(num_answer_per_question):
                duplicated_images.append(image)
        images = duplicated_images

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    try:
        for cur_pos in range(min_prompt_len, total_len):
            # Only print progress every 10 tokens to reduce console output
            if (cur_pos - min_prompt_len) % 10 == 0:
                print(
                    f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
                    flush=True,
                    end="",
                )
            with torch.autocast(device_type=device.type, dtype=dtype):
                if is_multimodal and hasattr(model, "inference") and images is not None:
                    # For multimodal inference
                    current_tokens = tokens[:, prev_pos:cur_pos]
                    if prev_pos == 0 and cur_pos == min_prompt_len:
                        # First forward pass with images
                        logits = model.inference(current_tokens, images=images, start_pos=prev_pos)
                    else:
                        # Subsequent passes without images (using KV cache)
                        logits = model.inference(current_tokens, start_pos=prev_pos)
                else:
                    # Original text-only inference
                    logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
                
            probs = torch.softmax(logits[:, -1], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # if an rollout is finished, we fill the rest of the tokens with pad_token_id
            next_token = torch.where(is_finished, pad_token_id, next_token)
            tokens[:, cur_pos] = next_token
            if end_token_id is not None:
                is_end_token = next_token == end_token_id
                is_generated_token = ~input_text_mask[:, cur_pos]
                is_finished = is_finished | (is_end_token & is_generated_token)
            prev_pos = cur_pos
            
            # Add early stopping condition for efficiency
            if is_finished.all():
                break
                
            # # Add timeout based on length (avoid getting stuck in long generations)
            # if cur_pos >= min_prompt_len + max_gen_len // 2 and is_finished.sum() / bsz >= 0.8:
            #     print("\nEarly stopping at 80% completion")
            #     break
                
    except Exception as e:
        print(f"\nException during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.del_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\rGeneration completed                   ")
    
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            
            # Call the reward function based on the task type
            if is_multimodal and hasattr(batch, "answers") and batch.answers is not None:
                # MathVision reward function
                rewards = reward_function(
                    response=generated_text,
                    answer=batch.answers[i],
                    end_token=end_token,
                )
                # Create episode with image
                episode = Episode(
                    prefix=batch.prefix[i],
                    text=batch.prefix[i] + generated_text,
                    prefix_token_ids=batch.prefix_token_ids[i],
                    prefix_tokens=batch.prefix_tokens[i],
                    generated_token_ids=generated_token_ids,
                    is_finished=is_finished_list[idx],
                    reward=rewards["reward"],
                    reward_info=rewards["reward_info"],
                    image=batch.images[i] if batch.images is not None else None,
                )
            else:
                # Countdown task reward function
                rewards = reward_function(
                    response=generated_text,
                    numbers=batch.numbers[i] if batch.numbers is not None else None,
                    target=batch.target[i] if batch.target is not None else None,
                    end_token=end_token,
                )
                episode = Episode(
                    prefix=batch.prefix[i],
                    text=batch.prefix[i] + generated_text,
                    prefix_token_ids=batch.prefix_token_ids[i],
                    prefix_tokens=batch.prefix_tokens[i],
                    generated_token_ids=generated_token_ids,
                    is_finished=is_finished_list[idx],
                    reward=rewards["reward"],
                    reward_info=rewards["reward_info"],
                )
            episodes.append(episode)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    is_multimodal: bool = False,
):
    """Update the policy using the GRPO algorithm."""
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )
        
        # Process images if needed
        batch_images = None
        if is_multimodal:
            batch_images = [episode.image for episode in batch_episodes if episode.image is not None]
            if not batch_images:
                batch_images = None

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            
            # Forward pass with images for multimodal or without for text-only
            if is_multimodal and batch_images and hasattr(model, "forward"):
                logits = model.forward(input_token_ids, images=batch_images).float()
            else:
                logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
